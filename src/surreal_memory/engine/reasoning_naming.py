"""Optional LLM naming for distilled reasoning patterns (``distill_use_llm``).

The heuristic distiller names a pattern from its own mechanics: the title is the
cluster's three most frequent reasoning moves, and the description is the first
200 characters of the medoid trace -- a raw thinking fragment, usually cut
mid-sentence. That is serviceable as an identifier and poor as an explanation.

With ``reasoning_training.distill_use_llm`` enabled, a local model rewrites the
prose into something a human (or an injected session) can act on. It rewrites
*only* prose:

    title / description / strategy                        <- may be replaced
    model / category / confidence / frequency / signature <- never touched

That split is what makes the flag safe to toggle. ``signature`` is derived from
the cluster's trace hashes, so renaming does not create a new pattern: flipping
the flag changes how *future* patterns read, never how existing ones are
identified, and never produces duplicates. Existing patterns are not rewritten
retroactively.

Two hard constraints, both mirroring ``reasoning_distiller._get_embedder``:

* **Loopback only.** Trace content is raw model thinking. A non-loopback
  endpoint yields no namer at all, so no request is ever made. This is an
  invariant, not a default: there is no configuration that sends thinking to a
  remote host. (Ingest-time redaction via ``reasoning_miner`` is upstream of
  this and can be switched off, so the transport guarantee has to stand alone.)
* **Fail-soft.** Missing endpoint, missing ``httpx``, refused connection,
  timeout, HTTP error, or a model that answers in prose instead of JSON -- every
  one of these falls back to the mechanical naming. Distillation never fails
  because naming failed, and a dead endpoint trips a circuit breaker rather than
  costing one timeout per cluster for the whole run.

Model residency
---------------
Local model servers load a chat model on its first request and, typically,
keep it resident with no idle timeout. Naming needs the model for the length of
one distillation run and not a second longer -- otherwise a run that names a
handful of patterns leaves several gigabytes parked in VRAM indefinitely.

Loading therefore stays implicit (the first request pulls the model in) and
unloading is explicit: ``release()`` runs ``distill_llm_unload_cmd`` once the
run is over. Two properties keep that honest:

* It only fires if this run actually issued a request. A namer that was built
  but never used releases nothing, so a model somebody else loaded is left
  alone.
* The command is an argv list executed without a shell, so a model name can
  never turn into shell syntax. It is read from the config file only -- never
  from an API request -- and a failure to unload is logged, never raised.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from surreal_memory.unified_config import _TOML_SAFE_ARGV

if TYPE_CHECKING:
    from surreal_memory.unified_config import ReasoningTrainingConfig

logger = logging.getLogger(__name__)

LLM_ENDPOINT_ENV = "SURREAL_MEMORY_LLM_ENDPOINT"
EMBEDDING_ENDPOINT_ENV = "SURREAL_MEMORY_EMBEDDING_ENDPOINT"

_LOOPBACK_NAMES = frozenset({"localhost", "::1"})

# Output caps. The title also becomes a CONCEPT neuron's content, so it stays
# short enough to read in a list; description/strategy are what injection ships.
_TITLE_MAX = 120
_DESCRIPTION_MAX = 400
_STRATEGY_MAX = 1200

# Input caps. A thinking block can run to 100k characters; the point of the
# prompt is the shape of the reasoning, not its full text.
_TRACE_CHARS = 1200
_MAX_SAMPLES = 3

_TIMEOUT_SECONDS = 60.0
# Must cover a *reasoning* model's private thinking as well as the JSON answer:
# thinking is emitted first, so too small a budget yields an empty content field
# and finish_reason="length" rather than a short answer.
_MAX_TOKENS = 1500
# Asks a server that supports it to skip the thinking phase entirely. Servers
# that do not recognise the key generally ignore it; one that rejects it costs a
# single failed attempt, after which it is dropped for the rest of the run.
_NO_THINKING_KWARGS = {"enable_thinking": False}
# Consecutive failures after which naming is abandoned for the rest of the run.
_FAILURE_LIMIT = 3
# Unloading is a local process teardown; it should be near-instant.
_UNLOAD_TIMEOUT_SECONDS = 30.0
# Loading spawns a server process and waits for a multi-GB model to reach disk
# or mmap and bind its port -- much slower than teardown, so a longer budget.
_LOAD_TIMEOUT_SECONDS = 120.0
# Substituted into distill_llm_unload_cmd so one command template can name the
# model it is releasing.
_MODEL_PLACEHOLDER = "{model}"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _substitute_model_and_validate(cmd: tuple[str, ...], model: str) -> list[str] | None:
    """Fill in ``{model}`` and re-check the result against ``_TOML_SAFE_ARGV``.

    Each part of ``cmd`` already passed that same allowlist at config-load
    time (inline in ``ReasoningTrainingConfig.from_dict``), but ``model``
    comes from ``distill_llm_model``, which is sanitized with
    ``_sanitize_toml_glob`` -- a looser charset (spaces, ``*?[]``) since it is
    not itself an argv element, only the template it gets substituted into
    is. Re-validating post-substitution closes that gap: a model name
    containing a character the argv allowlist would reject must void the
    command, the same as an invalid literal part does, rather than silently
    reaching ``create_subprocess_exec``.
    """
    argv = [
        part.replace(_MODEL_PLACEHOLDER, model) if _MODEL_PLACEHOLDER in part else part
        for part in cmd
    ]
    if not all(_TOML_SAFE_ARGV.match(part) for part in argv):
        return None
    return argv


_SYSTEM_PROMPT = (
    "You name reusable reasoning strategies. You are given several excerpts of a"
    " model's private reasoning that were clustered together because they share"
    " an approach. Describe the approach they have in common.\n\n"
    "Reply with a single JSON object and nothing else:\n"
    '{"title": "...", "description": "...", "strategy": "..."}\n\n'
    "title: an imperative name for the approach, at most 8 words.\n"
    "description: one or two sentences on what the approach is and when it"
    " applies.\n"
    "strategy: the reusable procedure as numbered steps, so another model could"
    " follow it on a new task.\n\n"
    "Describe only what the excerpts actually show. Do not invent steps, and do"
    " not mention the excerpts, this instruction, or that you are an AI."
)


class PostJson(Protocol):
    """Seam for the HTTP POST, so callers and tests can supply their own."""

    async def __call__(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]: ...


class RunCommand(Protocol):
    """Seam for the unload subprocess. Returns the exit status."""

    async def __call__(self, argv: list[str], timeout: float) -> int: ...


def _is_loopback(host: str | None) -> bool:
    """True only when *host* really is a loopback address or ``localhost``.

    The prefix test this replaced (``host.startswith("127.")``) accepted
    ``127.0.0.1.attacker.example`` — a hostname anyone can register and point
    anywhere. That matters here because this check is what keeps reasoning
    traces and prompts on the machine: passing it is what allows a remote
    endpoint to receive them. Parsing the address decides it, and a name that
    is not a literal IP is loopback only if it is exactly ``localhost``.
    """
    if not host:
        return False
    host = host.strip().strip("[]").lower()
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host in _LOOPBACK_NAMES


def resolve_llm_endpoint(configured: str = "") -> str | None:
    """Return the local LLM base URL to use, or None if there isn't a usable one.

    In precedence order: ``SURREAL_MEMORY_LLM_ENDPOINT``, then the
    ``distill_llm_endpoint`` config value, then
    ``SURREAL_MEMORY_EMBEDDING_ENDPOINT`` -- one local OpenAI-compatible server
    commonly serves both embeddings and chat. Env beats config, matching the
    rest of the reasoning settings.

    Whichever source wins, a non-loopback value resolves to None instead of
    being used: raw reasoning traces do not leave the machine.
    """
    sources = (
        os.environ.get(LLM_ENDPOINT_ENV, ""),
        configured,
        os.environ.get(EMBEDDING_ENDPOINT_ENV, ""),
    )
    for source in sources:
        raw = source.strip()
        if not raw:
            continue
        try:
            host = urlsplit(raw).hostname
        except ValueError:
            logger.debug("reasoning naming: unparseable LLM endpoint")
            return None
        if not _is_loopback(host):
            logger.warning(
                "reasoning naming: the configured LLM endpoint (%s) is not a loopback"
                " address; LLM naming stays off (reasoning traces are never sent to a"
                " remote host)",
                host,
            )
            return None
        return raw.rstrip("/")
    return None


def _extract_message_text(data: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-compatible response envelope.

    A reasoning model spends its budget on ``reasoning_content`` first, so
    running out of tokens shows up as an empty or half-written ``content``.
    That case gets its own message: "the model said nothing" and "the model was
    cut off mid-answer" call for very different fixes.
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response contained no choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    truncated = isinstance(first, dict) and first.get("finish_reason") == "length"
    if not isinstance(content, str) or not content.strip():
        if truncated:
            raise ValueError(
                "model hit the token limit before answering (a reasoning model may need"
                " a larger budget, or a model that does not think before answering)"
            )
        raise ValueError("response contained no message content")
    if truncated:
        raise ValueError("model was cut off mid-answer by the token limit")
    return content


def _coerce_text(value: Any) -> str | None:
    """Normalise one JSON field to prose, or None if there is nothing usable.

    Asked for "numbered steps", a model quite reasonably answers with a JSON
    array instead of a string. Rejecting that would throw away a perfectly good
    answer over its container type, so a list of scalars becomes one line each.
    """
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
        return "\n".join(parts) or None
    return None


def _parse_fields(text: str) -> dict[str, str] | None:
    """Parse the model's reply into capped prose fields, or None if unusable.

    Small local models routinely wrap JSON in a markdown fence or bracket it with
    chatter, so the object is located rather than assumed to be the whole reply.
    """
    fenced = _JSON_FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    out: dict[str, str] = {}
    for key, cap in (
        ("title", _TITLE_MAX),
        ("description", _DESCRIPTION_MAX),
        ("strategy", _STRATEGY_MAX),
    ):
        value = _coerce_text(parsed.get(key))
        if value is None:
            return None
        out[key] = value[:cap]
    return out


def _build_payload(
    model: str,
    pattern: dict[str, Any],
    cluster_traces: list[dict[str, Any]],
    *,
    skip_thinking: bool = True,
) -> dict[str, Any]:
    excerpts = "\n\n".join(
        f"--- excerpt {i + 1} ---\n{str(t.get('content', ''))[:_TRACE_CHARS]}"
        for i, t in enumerate(cluster_traces[:_MAX_SAMPLES])
    )
    user = (
        f"Category: {pattern.get('category', '')}\n"
        f"Observed moves: {pattern.get('title', '')}\n\n"
        f"{excerpts}"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": _MAX_TOKENS,
        "stream": False,
    }
    if skip_thinking:
        payload["chat_template_kwargs"] = dict(_NO_THINKING_KWARGS)
    return payload


async def _post_json_httpx(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Default transport. ``httpx`` is imported lazily and optional by design."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise ValueError("response body was not a JSON object")
    return result


async def _run_command_subprocess(argv: list[str], timeout: float) -> int:
    """Shared load/unload runner: exec the argv directly, with no shell involved."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if stdout:
        # Shared by acquire() and release() -- "command", not "unload", since
        # this same runner now services both and the output isn't otherwise
        # tagged with which one triggered it.
        logger.debug(
            "reasoning naming: command output: %s", stdout.decode(errors="replace").strip()
        )
    return process.returncode if process.returncode is not None else -1


class PatternNamer:
    """Rewrites a distilled pattern's prose using a local chat model."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        post_json: PostJson,
        *,
        unload_cmd: tuple[str, ...] = (),
        load_cmd: tuple[str, ...] = (),
        run_command: RunCommand | None = None,
    ) -> None:
        self._url = f"{endpoint}/chat/completions"
        self._model = model
        self._post = post_json
        self._unload_cmd = unload_cmd
        self._load_cmd = load_cmd
        self._run = run_command or _run_command_subprocess
        self._consecutive_failures = 0
        self._given_up = False
        # Cleared the first time a request fails, in case this server is one
        # that rejects the thinking-control key rather than ignoring it.
        self._skip_thinking = True
        # Set once a request has gone out, i.e. once this run may have caused
        # the model to be loaded. Reset by release() so a namer reused after a
        # release acquires and releases again rather than releasing twice.
        self._model_in_use = False
        # acquire() runs once per run (idempotency guard, mirroring release()'s
        # own safe-to-call-more-than-once contract).
        self._load_attempted = False

    def _record_failure(self, reason: str, exc: BaseException | None = None) -> None:
        self._consecutive_failures += 1
        logger.debug("reasoning naming: %s", reason, exc_info=exc is not None)
        if self._consecutive_failures >= _FAILURE_LIMIT:
            self._given_up = True
            logger.warning(
                "reasoning naming: giving up after %d consecutive failures (%s);"
                " the rest of this run uses heuristic naming",
                _FAILURE_LIMIT,
                reason,
            )

    async def rename(
        self, pattern: dict[str, Any], cluster_traces: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return *pattern* with LLM prose, or *pattern* unchanged on any problem.

        Never raises: naming is an enhancement, and a pattern with mechanical
        wording is worth more than a failed distillation run.
        """
        if self._given_up or not cluster_traces:
            return pattern

        payload = _build_payload(
            self._model, pattern, cluster_traces, skip_thinking=self._skip_thinking
        )
        # Flagged before the await: a request that times out may still have
        # loaded the model, so it must count as an acquisition.
        self._model_in_use = True
        try:
            data = await self._post(self._url, payload, _TIMEOUT_SECONDS)
            text = _extract_message_text(data)
        except Exception as exc:  # every transport failure is soft
            if self._skip_thinking:
                # This server may be one that rejects the thinking-control key.
                # Drop it rather than retrying now: a retry would double the
                # cost of every failure on an endpoint that is simply down.
                self._skip_thinking = False
                logger.debug(
                    "reasoning naming: retrying subsequent calls without %s", "thinking control"
                )
            self._record_failure(f"call failed: {type(exc).__name__}: {exc}", exc)
            return pattern

        fields = _parse_fields(text)
        if fields is None:
            # A model that cannot produce the JSON shape is as useless as a dead
            # endpoint, so this counts toward the circuit breaker too.
            self._record_failure("model did not return the requested JSON object")
            return pattern

        self._consecutive_failures = 0
        return {**pattern, **fields}

    async def acquire(self) -> None:
        """Explicitly load the chat model before the first request, once per run.

        A no-op when no load command is configured -- distillation must work
        without one, falling back to the endpoint's implicit
        load-on-first-request behavior (rename() already flags
        _model_in_use once an actual request goes out). Any failure here is
        silent and non-fatal: this is an optimization (get the model on GPU
        with the right parameters before the first request), never a
        precondition for distillation to proceed. Safe to call more than once.
        """
        if not self._load_cmd or self._load_attempted:
            return
        self._load_attempted = True

        argv = _substitute_model_and_validate(self._load_cmd, self._model)
        if argv is None:
            logger.warning(
                "reasoning naming: distill_llm_model %r contains a character the load"
                " command's argv allowlist rejects; skipping the explicit load",
                self._model,
            )
            return
        # Flagged before the attempt, mirroring rename(): a command that times
        # out or otherwise raises (e.g. _run_command_subprocess killing a slow
        # process at _LOAD_TIMEOUT_SECONDS) may still have started pulling the
        # model into memory, so release() must still attempt cleanup rather
        # than leaving it stranded -- its own unload failure is already a
        # harmless warning. Setting this only on the success path would leave
        # exactly the VRAM-leak scenario this method exists to close.
        self._model_in_use = True
        try:
            code = await self._run(argv, _LOAD_TIMEOUT_SECONDS)
        except Exception as exc:  # a load command must never block distillation
            logger.debug(
                "reasoning naming: load command for %s failed (%s: %s); falling back"
                " to implicit load-on-first-request",
                self._model,
                type(exc).__name__,
                exc,
            )
            return
        if code == 0:
            logger.info("reasoning naming: explicitly loaded %s before distillation", self._model)
        else:
            logger.warning(
                "reasoning naming: load command for %s exited %s; continuing anyway"
                " (implicit load-on-first-request still applies)",
                self._model,
                code,
            )

    async def release(self) -> None:
        """Unload the chat model this run pulled in. Safe to call more than once.

        A no-op when no unload command is configured or when this namer never
        issued a request -- releasing a model the run did not acquire would
        evict something another process is relying on.
        """
        if not self._unload_cmd or not self._model_in_use:
            return
        self._model_in_use = False

        argv = _substitute_model_and_validate(self._unload_cmd, self._model)
        if argv is None:
            logger.warning(
                "reasoning naming: distill_llm_model %r contains a character the unload"
                " command's argv allowlist rejects; skipping the explicit unload",
                self._model,
            )
            return
        try:
            code = await self._run(argv, _UNLOAD_TIMEOUT_SECONDS)
        except Exception as exc:  # a stuck model must not fail a run
            logger.warning(
                "reasoning naming: could not unload %s (%s: %s); it may still be resident",
                self._model,
                type(exc).__name__,
                exc,
            )
            return
        if code == 0:
            logger.info("reasoning naming: unloaded %s after distillation", self._model)
        else:
            logger.warning(
                "reasoning naming: unload command for %s exited %s; it may still be resident",
                self._model,
                code,
            )


def build_namer(
    rt: ReasoningTrainingConfig,
    *,
    post_json: PostJson | None = None,
    run_command: RunCommand | None = None,
) -> PatternNamer | None:
    """Build a namer from config, or None when LLM naming is not available.

    None is returned -- and distillation silently keeps its heuristic naming --
    when the flag is off, no local endpoint is configured, the configured
    endpoint is remote, or no model name is set.
    """
    if not rt.distill_use_llm:
        return None

    endpoint = resolve_llm_endpoint(rt.distill_llm_endpoint)
    if endpoint is None:
        logger.warning(
            "reasoning_training.distill_use_llm is on but no usable local LLM endpoint"
            " is set; set distill_llm_endpoint (or %s) to a loopback OpenAI-compatible"
            " URL. Using heuristic naming.",
            LLM_ENDPOINT_ENV,
        )
        return None

    model = str(rt.distill_llm_model or "").strip()
    if not model:
        logger.warning(
            "reasoning_training.distill_use_llm is on but distill_llm_model is empty;"
            " set it to a chat model served by %s. Using heuristic naming.",
            endpoint,
        )
        return None

    return PatternNamer(
        endpoint,
        model,
        post_json or _post_json_httpx,
        unload_cmd=tuple(rt.distill_llm_unload_cmd),
        load_cmd=tuple(rt.distill_llm_load_cmd),
        run_command=run_command,
    )

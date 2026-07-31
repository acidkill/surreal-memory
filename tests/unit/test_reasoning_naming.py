"""LLM pattern naming must improve wording without changing pattern identity.

``distill_use_llm`` shipped as a dataclass field with no consumer: setting it
did nothing at all. Wiring it up means an LLM may rewrite a pattern's *prose*
(title / description / strategy), and nothing else. Three properties make that
safe, and each is pinned here:

* **Identity is untouched.** ``signature`` is derived from the cluster's trace
  hashes, so a renamed pattern is still the same pattern. If naming ever fed
  into the signature, toggling the flag would re-materialise every pattern as a
  duplicate.
* **The prompt never leaves the machine.** Trace content is raw model thinking.
  A non-loopback endpoint must yield no namer and, above all, no request.
* **Failure is invisible.** A missing, slow, broken or babbling endpoint falls
  back to the mechanical naming that ran before this feature existed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from surreal_memory.engine.reasoning_naming import (
    LLM_ENDPOINT_ENV,
    build_namer,
    resolve_llm_endpoint,
)
from surreal_memory.unified_config import ReasoningTrainingConfig

LOOPBACK = "http://127.0.0.1:11435/v1"


def _config(**overrides: Any) -> ReasoningTrainingConfig:
    return ReasoningTrainingConfig(**{"distill_use_llm": True, **overrides})


def _pattern(**overrides: Any) -> dict[str, Any]:
    base = {
        "model": "claude-opus-5",
        "category": "refactoring",
        "title": "refactoring: read, edit, verify",
        "description": "raw thinking fragment cut mid-sen",
        "strategy": "Moves: read -> edit -> verify\nraw thinking",
        "confidence": 0.75,
        "frequency": 4,
        "signature": "abc123",
    }
    base.update(overrides)
    return base


def _traces(n: int = 3, content: str = "I need to read the file, then edit it.") -> list[dict]:
    return [{"trace_hash": f"h{i}", "content": content, "task_context": "ctx"} for i in range(n)]


class _Transport:
    """Records every call so 'no request was made' is directly assertable."""

    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        if not self._responses:
            raise AssertionError("transport called more times than the test provided responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)


def _completion(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


def _good_json(
    title: str = "Read before editing",
    description: str = "Establish the current state, then make one scoped change.",
    strategy: str = "1. Read the target. 2. Edit. 3. Re-read to verify.",
) -> str:
    return json.dumps({"title": title, "description": description, "strategy": strategy})


@pytest.fixture(autouse=True)
def _clear_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's real environment decide a test's outcome."""
    monkeypatch.delenv(LLM_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", raising=False)


class TestEndpointResolution:
    def test_no_endpoint_configured_resolves_to_none(self) -> None:
        assert resolve_llm_endpoint() is None

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://127.0.0.1:11435/v1",
            "http://localhost:11435/v1",
            "http://[::1]:11435/v1",
        ],
    )
    def test_loopback_endpoints_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        monkeypatch.setenv(LLM_ENDPOINT_ENV, endpoint)
        assert resolve_llm_endpoint() == endpoint.rstrip("/")

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://api.openai.com/v1",
            "http://192.168.1.95:11435/v1",
            "https://generativelanguage.googleapis.com/v1",
            "http://llm.internal.example.com/v1",
        ],
    )
    def test_remote_endpoints_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        """Trace content is raw thinking; it must never be shipped off-box."""
        monkeypatch.setenv(LLM_ENDPOINT_ENV, endpoint)
        assert resolve_llm_endpoint() is None

    def test_falls_back_to_the_embedding_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One local OpenAI-compatible server usually serves both roles."""
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", LOOPBACK)
        assert resolve_llm_endpoint() == LOOPBACK

    def test_a_remote_embedding_endpoint_is_not_inherited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", "https://api.openai.com/v1")
        assert resolve_llm_endpoint() is None

    def test_the_config_value_is_used_when_no_env_var_is_set(self) -> None:
        """Not everyone can add env vars to how they launch smem."""
        assert resolve_llm_endpoint(LOOPBACK) == LOOPBACK

    def test_the_env_var_beats_the_config_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LLM_ENDPOINT_ENV, "http://127.0.0.1:9999/v1")
        assert resolve_llm_endpoint("http://127.0.0.1:11435/v1") == "http://127.0.0.1:9999/v1"

    def test_a_remote_config_value_is_refused_like_any_other(self) -> None:
        assert resolve_llm_endpoint("https://api.openai.com/v1") is None

    def test_the_config_value_beats_the_embedding_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", "http://127.0.0.1:9999/v1")
        assert resolve_llm_endpoint(LOOPBACK) == LOOPBACK


class TestNamerConstruction:
    def test_flag_off_yields_no_namer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)
        assert build_namer(_config(distill_use_llm=False)) is None

    def test_flag_on_without_an_endpoint_yields_no_namer(self) -> None:
        assert build_namer(_config()) is None

    def test_flag_on_with_a_remote_endpoint_yields_no_namer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LLM_ENDPOINT_ENV, "https://api.openai.com/v1")
        assert build_namer(_config()) is None

    def test_flag_on_with_a_loopback_endpoint_yields_a_namer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)
        assert build_namer(_config(distill_llm_model="gemma-4-12b")) is not None


class _Runner:
    """Stands in for the unload subprocess; records argv, never spawns anything."""

    def __init__(self, exit_code: int = 0, raises: Exception | None = None) -> None:
        self._exit_code = exit_code
        self._raises = raises
        self.calls: list[list[str]] = []

    async def __call__(self, argv: list[str], timeout: float) -> int:
        self.calls.append(list(argv))
        if self._raises is not None:
            raise self._raises
        return self._exit_code


@pytest.fixture
def namer_factory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)

    def _make(*responses: Any) -> tuple[Any, _Transport]:
        transport = _Transport(*responses)
        namer = build_namer(_config(distill_llm_model="gemma-4-12b"), post_json=transport)
        assert namer is not None
        return namer, transport

    return _make


@pytest.fixture
def releasing_namer_factory(monkeypatch: pytest.MonkeyPatch):
    """A namer configured with an unload command plus a recording runner."""
    monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)

    def _make(*responses: Any, runner: _Runner | None = None) -> tuple[Any, _Transport, _Runner]:
        transport = _Transport(*responses)
        run = runner or _Runner()
        namer = build_namer(
            _config(
                distill_llm_model="gemma-4-12b",
                distill_llm_unload_cmd=("llamastash", "stop", "{model}", "-y"),
            ),
            post_json=transport,
            run_command=run,
        )
        assert namer is not None
        return namer, transport, run

    return _make


@pytest.fixture
def acquiring_namer_factory(monkeypatch: pytest.MonkeyPatch):
    """A namer configured with a load command plus a recording runner."""
    monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)

    def _make(*responses: Any, runner: _Runner | None = None) -> tuple[Any, _Transport, _Runner]:
        transport = _Transport(*responses)
        run = runner or _Runner()
        namer = build_namer(
            _config(
                distill_llm_model="gemma-4-12b",
                distill_llm_load_cmd=("llamastash-load.py", "{model}", "--ngl", "99"),
            ),
            post_json=transport,
            run_command=run,
        )
        assert namer is not None
        return namer, transport, run

    return _make


class TestSuccessfulRename:
    async def test_prose_is_replaced_by_the_model_output(self, namer_factory) -> None:
        namer, _ = namer_factory(_completion(_good_json()))

        renamed = await namer.rename(_pattern(), _traces())

        assert renamed["title"] == "Read before editing"
        assert renamed["description"].startswith("Establish the current state")
        assert renamed["strategy"].startswith("1. Read the target.")

    async def test_identity_and_statistics_survive_the_rename(self, namer_factory) -> None:
        """Renaming must not make a known pattern look like a new one."""
        namer, _ = namer_factory(_completion(_good_json()))
        original = _pattern()

        renamed = await namer.rename(original, _traces())

        for key in ("signature", "confidence", "frequency", "model", "category"):
            assert renamed[key] == original[key], f"{key} must not be touched by naming"

    async def test_the_input_pattern_is_not_mutated(self, namer_factory) -> None:
        namer, _ = namer_factory(_completion(_good_json()))
        original = _pattern()
        before = dict(original)

        renamed = await namer.rename(original, _traces())

        assert original == before
        assert renamed is not original

    async def test_a_fenced_json_block_is_parsed(self, namer_factory) -> None:
        """Small local models habitually wrap JSON in a markdown fence."""
        namer, _ = namer_factory(_completion(f"```json\n{_good_json()}\n```"))

        renamed = await namer.rename(_pattern(), _traces())

        assert renamed["title"] == "Read before editing"

    async def test_prose_around_the_json_object_is_tolerated(self, namer_factory) -> None:
        namer, _ = namer_factory(
            _completion(f"Sure! Here is the pattern:\n{_good_json()}\nHope that helps.")
        )

        renamed = await namer.rename(_pattern(), _traces())

        assert renamed["title"] == "Read before editing"

    async def test_a_strategy_returned_as_a_list_of_steps_is_accepted(self, namer_factory) -> None:
        """Observed live: asked for "numbered steps", the model sent a JSON array.

        Refusing that would discard a good answer over its container type.
        """
        namer, _ = namer_factory(
            _completion(
                json.dumps(
                    {
                        "title": "Read the traceback first",
                        "description": "Locate the failure before changing anything.",
                        "strategy": [
                            "Identify the reported failure.",
                            "Examine the traceback.",
                            "Re-run to confirm the fix.",
                        ],
                    }
                )
            )
        )

        renamed = await namer.rename(_pattern(), _traces())

        assert renamed["strategy"] == (
            "Identify the reported failure.\nExamine the traceback.\nRe-run to confirm the fix."
        )

    async def test_junk_inside_a_list_field_is_dropped(self, namer_factory) -> None:
        namer, _ = namer_factory(
            _completion(
                json.dumps(
                    {
                        "title": "t",
                        "description": "d",
                        "strategy": ["keep this", "", None, {"nested": 1}, "and this"],
                    }
                )
            )
        )

        renamed = await namer.rename(_pattern(), _traces())

        assert renamed["strategy"] == "keep this\nand this"

    async def test_an_empty_list_field_is_unusable(self, namer_factory) -> None:
        namer, _ = namer_factory(
            _completion(json.dumps({"title": "t", "description": "d", "strategy": []}))
        )
        original = _pattern()

        assert await namer.rename(original, _traces()) == original

    async def test_overlong_fields_are_capped(self, namer_factory) -> None:
        namer, _ = namer_factory(
            _completion(_good_json(title="T" * 500, description="D" * 5000, strategy="S" * 5000))
        )

        renamed = await namer.rename(_pattern(), _traces())

        assert len(renamed["title"]) <= 120
        assert len(renamed["description"]) <= 400
        assert len(renamed["strategy"]) <= 1200


class TestFallback:
    """Every failure mode must degrade to the pre-existing mechanical naming."""

    @pytest.mark.parametrize(
        "body",
        [
            "not json at all",
            "{}",
            '{"title": "only a title"}',
            '{"title": 42, "description": "d", "strategy": "s"}',
            '{"title": "", "description": "d", "strategy": "s"}',
            '{"title": "t", "description": "d"}',
            "[]",
        ],
        ids=[
            "prose",
            "empty-object",
            "missing-fields",
            "wrong-type",
            "blank-title",
            "missing-strategy",
            "wrong-shape",
        ],
    )
    async def test_unusable_output_keeps_the_mechanical_naming(
        self, namer_factory, body: str
    ) -> None:
        namer, _ = namer_factory(_completion(body))
        original = _pattern()

        renamed = await namer.rename(original, _traces())

        assert renamed == original

    @pytest.mark.parametrize(
        "failure",
        [TimeoutError("timed out"), RuntimeError("connection refused"), ValueError("bad status")],
        ids=["timeout", "refused", "bad-status"],
    )
    async def test_a_failing_endpoint_keeps_the_mechanical_naming(
        self, namer_factory, failure: Exception
    ) -> None:
        namer, _ = namer_factory(failure)
        original = _pattern()

        renamed = await namer.rename(original, _traces())

        assert renamed == original

    async def test_a_malformed_envelope_keeps_the_mechanical_naming(self, namer_factory) -> None:
        namer, _ = namer_factory({"unexpected": "shape"})
        original = _pattern()

        assert await namer.rename(original, _traces()) == original

    async def test_an_answer_cut_off_by_the_token_limit_is_rejected(self, namer_factory) -> None:
        """Half a JSON object is not a pattern name. Seen live: a reasoning
        model spent the budget thinking and returned '```json\\n{\\n "title":'."""
        namer, _ = namer_factory(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": '```json\n{\n  "title":'},
                    }
                ]
            }
        )
        original = _pattern()

        assert await namer.rename(original, _traces()) == original

    async def test_a_reply_that_is_only_reasoning_is_rejected(self, namer_factory) -> None:
        namer, _ = namer_factory(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "Thinking Process: ..."},
                    }
                ]
            }
        )
        original = _pattern()

        assert await namer.rename(original, _traces()) == original


class TestCircuitBreaker:
    """A dead endpoint must not cost one timeout per cluster for a whole run."""

    async def test_repeated_failures_stop_further_calls(self, namer_factory) -> None:
        namer, transport = namer_factory(
            RuntimeError("down"), RuntimeError("down"), RuntimeError("down")
        )

        for _ in range(6):
            assert await namer.rename(_pattern(), _traces()) == _pattern()

        assert len(transport.calls) == 3, "should give up after 3 consecutive failures"

    async def test_a_success_resets_the_failure_count(self, namer_factory) -> None:
        namer, transport = namer_factory(
            RuntimeError("blip"),
            RuntimeError("blip"),
            _completion(_good_json()),
            RuntimeError("blip"),
            RuntimeError("blip"),
            _completion(_good_json(title="Second wind")),
        )

        results = [(await namer.rename(_pattern(), _traces()))["title"] for _ in range(6)]

        assert len(transport.calls) == 6
        assert results[2] == "Read before editing"
        assert results[5] == "Second wind"


class TestPrompt:
    async def test_the_request_targets_the_chat_completions_route(self, namer_factory) -> None:
        namer, transport = namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())

        assert transport.calls[0]["url"] == f"{LOOPBACK}/chat/completions"
        assert transport.calls[0]["payload"]["model"] == "gemma-4-12b"
        assert transport.calls[0]["timeout"] > 0

    async def test_the_prompt_carries_the_category_and_the_trace_content(
        self, namer_factory
    ) -> None:
        namer, transport = namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces(content="I should segment the moves first."))

        sent = json.dumps(transport.calls[0]["payload"])
        assert "refactoring" in sent
        assert "segment the moves" in sent

    async def test_trace_content_is_truncated_before_it_is_sent(self, namer_factory) -> None:
        """A 100k-char thinking block must not be shipped verbatim."""
        namer, transport = namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces(content="x" * 100_000))

        sent = json.dumps(transport.calls[0]["payload"])
        assert len(sent) < 10_000

    async def test_the_token_budget_fits_a_reasoning_model(self, namer_factory) -> None:
        """A thinking model emits its reasoning first; a small budget yields nothing.

        Observed against a local reasoning model: with a 400-token budget the
        whole allowance went to ``reasoning_content``, ``content`` came back as
        a half-written ``{"title":`` and ``finish_reason`` was ``length``.
        """
        namer, transport = namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())

        assert transport.calls[0]["payload"]["max_tokens"] >= 1000

    async def test_thinking_is_switched_off_where_the_server_supports_it(
        self, namer_factory
    ) -> None:
        namer, transport = namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())

        kwargs = transport.calls[0]["payload"]["chat_template_kwargs"]
        assert kwargs == {"enable_thinking": False}

    async def test_a_server_that_rejects_the_thinking_key_is_accommodated(
        self, namer_factory
    ) -> None:
        """Dropped after one failure, and not retried immediately: a down
        endpoint must not cost two calls per pattern."""
        namer, transport = namer_factory(ValueError("unknown field"), _completion(_good_json()))

        first = await namer.rename(_pattern(), _traces())
        second = await namer.rename(_pattern(), _traces())

        assert len(transport.calls) == 2, "the failure must not trigger an immediate retry"
        assert "chat_template_kwargs" in transport.calls[0]["payload"]
        assert "chat_template_kwargs" not in transport.calls[1]["payload"]
        assert first["title"] == _pattern()["title"]
        assert second["title"] == "Read before editing"

    async def test_generation_is_deterministic(self, namer_factory) -> None:
        """Naming is a labelling job; sampling only adds noise between runs."""
        namer, transport = namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())

        assert transport.calls[0]["payload"]["temperature"] == 0.0

    async def test_an_empty_cluster_is_not_sent_to_the_model(self, namer_factory) -> None:
        namer, transport = namer_factory()
        original = _pattern()

        assert await namer.rename(original, []) == original
        assert transport.calls == []


class TestRelease:
    """A chat model is loaded for the run's sake and must not outlive it.

    The endpoint loads a catalog model on first request and keeps it resident
    with no idle timeout, so a mining run that names ten patterns would leave a
    multi-gigabyte model sitting in VRAM until something else evicted it.
    ``release()`` is the other half of that acquisition -- and it only fires if
    the run actually caused a load.
    """

    async def test_a_run_that_never_called_the_model_releases_nothing(
        self, releasing_namer_factory
    ) -> None:
        namer, _transport, runner = releasing_namer_factory()

        await namer.release()

        assert runner.calls == [], "nothing was loaded, so nothing may be unloaded"

    async def test_a_used_model_is_released(self, releasing_namer_factory) -> None:
        namer, _transport, runner = releasing_namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())
        await namer.release()

        assert runner.calls == [["llamastash", "stop", "gemma-4-12b", "-y"]]

    async def test_a_failed_call_still_counts_as_a_load(self, releasing_namer_factory) -> None:
        """The request reached the endpoint, so the model may well be resident."""
        namer, _transport, runner = releasing_namer_factory(RuntimeError("read timeout"))

        await namer.rename(_pattern(), _traces())
        await namer.release()

        assert len(runner.calls) == 1

    async def test_the_command_is_argv_not_a_shell_string(self, releasing_namer_factory) -> None:
        """No shell means no quoting bugs and no injection via a model name."""
        namer, _transport, runner = releasing_namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())
        await namer.release()

        argv = runner.calls[0]
        assert isinstance(argv, list)
        assert all(isinstance(part, str) for part in argv)
        assert not any(";" in part or "|" in part or "&&" in part for part in argv)

    async def test_release_is_idempotent(self, releasing_namer_factory) -> None:
        namer, _transport, runner = releasing_namer_factory(_completion(_good_json()))

        await namer.rename(_pattern(), _traces())
        await namer.release()
        await namer.release()
        await namer.release()

        assert len(runner.calls) == 1

    async def test_renaming_after_release_reloads_and_re_releases(
        self, releasing_namer_factory
    ) -> None:
        namer, _transport, runner = releasing_namer_factory(
            _completion(_good_json()), _completion(_good_json())
        )

        await namer.rename(_pattern(), _traces())
        await namer.release()
        await namer.rename(_pattern(), _traces())
        await namer.release()

        assert len(runner.calls) == 2

    @pytest.mark.parametrize(
        "runner",
        [_Runner(exit_code=1), _Runner(raises=FileNotFoundError("no such command"))],
        ids=["non-zero-exit", "command-missing"],
    )
    async def test_a_failing_unload_never_raises(
        self, releasing_namer_factory, runner: _Runner
    ) -> None:
        """A stuck model is a nuisance; a crashed mining run loses the work."""
        namer, _transport, _run = releasing_namer_factory(_completion(_good_json()), runner=runner)

        await namer.rename(_pattern(), _traces())
        await namer.release()  # must not raise

    async def test_no_unload_command_configured_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)
        runner = _Runner()
        namer = build_namer(
            _config(distill_llm_model="gemma-4-12b"),
            post_json=_Transport(_completion(_good_json())),
            run_command=runner,
        )
        assert namer is not None

        await namer.rename(_pattern(), _traces())
        await namer.release()

        assert runner.calls == []


class TestAcquire:
    """Explicit model loading before the first request (run 010 / section E, U6).

    Symmetric with TestRelease: acquire() is the other half of the same
    load/unload lifecycle, so a run that explicitly loads must still clean up
    even if it never actually renamed anything.
    """

    async def test_no_load_command_configured_is_a_no_op(self, releasing_namer_factory) -> None:
        """releasing_namer_factory has no load_cmd -- acquire() must touch nothing."""
        namer, _transport, runner = releasing_namer_factory()

        await namer.acquire()

        assert runner.calls == [], "nothing configured, so nothing may run"

    async def test_a_successful_load_runs_the_command_with_model_substituted(
        self, acquiring_namer_factory
    ) -> None:
        namer, _transport, runner = acquiring_namer_factory()

        await namer.acquire()

        assert runner.calls == [["llamastash-load.py", "gemma-4-12b", "--ngl", "99"]]

    async def test_a_successful_load_is_cleaned_up_even_with_no_rename_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact leak U6 exists to close: acquire() loads a model that
        rename() is never called for (e.g. zero patterns to name this run) --
        release() must still unload it, not skip cleanup because
        _model_in_use looks like nothing happened.
        """
        monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)
        run = _Runner()
        namer = build_namer(
            _config(
                distill_llm_model="gemma-4-12b",
                distill_llm_load_cmd=("llamastash-load.py", "{model}"),
                distill_llm_unload_cmd=("llamastash-stop.py", "{model}"),
            ),
            post_json=_Transport(),
            run_command=run,
        )
        assert namer is not None

        await namer.acquire()
        await namer.release()

        assert run.calls == [
            ["llamastash-load.py", "gemma-4-12b"],
            ["llamastash-stop.py", "gemma-4-12b"],
        ]

    async def test_acquire_is_idempotent(self, acquiring_namer_factory) -> None:
        namer, _transport, runner = acquiring_namer_factory()

        await namer.acquire()
        await namer.acquire()
        await namer.acquire()

        assert len(runner.calls) == 1

    @pytest.mark.parametrize(
        "runner",
        [_Runner(exit_code=1), _Runner(raises=FileNotFoundError("no such command"))],
        ids=["non-zero-exit", "command-missing"],
    )
    async def test_a_failing_load_never_raises_and_naming_still_works(
        self, monkeypatch: pytest.MonkeyPatch, runner: _Runner
    ) -> None:
        """A load command is an optimization; its failure must fall back to
        the endpoint's implicit load-on-first-request, not break distillation.
        """
        monkeypatch.setenv(LLM_ENDPOINT_ENV, LOOPBACK)
        namer = build_namer(
            _config(
                distill_llm_model="gemma-4-12b",
                distill_llm_load_cmd=("llamastash-load.py", "{model}"),
            ),
            post_json=_Transport(_completion(_good_json())),
            run_command=runner,
        )
        assert namer is not None

        await namer.acquire()  # must not raise
        renamed = await namer.rename(_pattern(), _traces())

        assert renamed["title"] == "Read before editing"

    async def test_the_load_command_is_argv_not_a_shell_string(
        self, acquiring_namer_factory
    ) -> None:
        """No shell means no quoting bugs and no injection via a model name."""
        namer, _transport, runner = acquiring_namer_factory()

        await namer.acquire()

        argv = runner.calls[0]
        assert isinstance(argv, list)
        assert all(isinstance(part, str) for part in argv)
        assert not any(";" in part or "|" in part or "&&" in part for part in argv)

    async def test_acquire_does_not_itself_call_the_chat_endpoint(
        self, acquiring_namer_factory
    ) -> None:
        """acquire() only runs the load command; it must not send a chat request."""
        namer, transport, _runner = acquiring_namer_factory()

        await namer.acquire()

        assert transport.calls == []

"""Reasoning-strategy injection: pick the target model, build the prompt block.

Pure, testable logic used by the SessionStart hook (and, from run 007 Faza 5b,
the UserPromptSubmit hook):

- ``resolve_active_model(hook_input)`` — SessionStart payloads carry no ``model``
  field, so resolve it via a fallback chain (payload → transcript tail →
  env → ~/.claude/settings.json → "default").
- ``build_injection_context(storage, model, config)`` — map the active model to a
  source model via ``injection_map`` (glob, first-match, "default" fallback),
  pull that source's ReasoningBank pattern fibers, and render a compact markdown
  block ("## Reasoning strategies (learned from <source>)").
- session-scoped idempotency markers so SessionStart + UserPromptSubmit inject at
  most once per session.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.engine.reasoning_miner import normalize_model

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

_SYNTHETIC_MODEL = "<synthetic>"
_MAX_PER_CATEGORY = 2
_MARKER_MAX_AGE_S = 7 * 86400  # prune injection markers older than 7 days
_TRANSCRIPT_TAIL_LINES = 300
# Ceiling for the pattern-fiber fetch. Patterns are idempotent by
# _reasoning_signature (reasoning_distiller._materialize_pattern), so the
# population is bounded by distinct patterns and stays well under this; matches
# the distiller's own fetch limit. If it is ever hit, we warn rather than let a
# source model's patterns silently fall outside the window (the post-LIMIT
# metadata-filter failure mode documented in storage.find_fibers).
_PATTERN_FETCH_LIMIT = 5000

# Claude Code short model aliases -> canonical ids used across the reasoning
# pipeline. Full ids pass through ``normalize_model`` unchanged.
_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
    "opusplan": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5",
}


# ── Model resolution ─────────────────────────────────────────────────────────


def _model_from_transcript_tail(transcript_path: str) -> str:
    """Return the last assistant turn's model from a JSONL transcript tail.

    ``transcript_path`` comes from untrusted hook stdin, so only transcripts
    under ~/.claude are read (mirrors the pre_compact / stop path-allowlist
    guard). Any stat/read error degrades to "" so the fallback chain continues.
    """
    if not transcript_path:
        return ""
    try:
        resolved = Path(transcript_path).resolve()
        if not resolved.is_relative_to((Path.home() / ".claude").resolve()):
            return ""
        if not resolved.is_file():
            return ""
        with resolved.open(encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    for raw in reversed(lines[-_TRANSCRIPT_TAIL_LINES:]):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        model = str(message.get("model") or "")
        if model and model != _SYNTHETIC_MODEL:
            return normalize_model(model)
    return ""


def _model_from_settings() -> str:
    """Return the model from ~/.claude/settings.json (alias-expanded)."""
    try:
        path = Path.home() / ".claude" / "settings.json"
        if not path.exists():
            return ""
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return ""
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, str) or not model.strip():
        return ""
    model = model.strip()
    return _MODEL_ALIASES.get(model.lower(), normalize_model(model))


def resolve_active_model(hook_input: dict[str, Any]) -> str:
    """Resolve the active model for injection via a fallback chain.

    Order: (1) ``hook_input["model"]`` (future-proof), (2) last assistant
    ``message.model`` from the transcript tail, (3) env
    ``SMEM_REASONING_TARGET_MODEL`` / ``ANTHROPIC_MODEL``, (4) ~/.claude/
    settings.json ``model`` (alias-expanded), (5) ``"default"``. Results are
    date-suffix-normalized so they agree with mining/distillation model names.
    """
    payload_model = hook_input.get("model")
    if isinstance(payload_model, str) and payload_model.strip():
        return _MODEL_ALIASES.get(payload_model.strip().lower(), normalize_model(payload_model))

    from_transcript = _model_from_transcript_tail(str(hook_input.get("transcript_path") or ""))
    if from_transcript:
        return from_transcript

    for env_var in ("SMEM_REASONING_TARGET_MODEL", "ANTHROPIC_MODEL"):
        value = os.environ.get(env_var)
        if value and value.strip():
            return _MODEL_ALIASES.get(value.strip().lower(), normalize_model(value))

    from_settings = _model_from_settings()
    if from_settings:
        return from_settings

    return "default"


# ── Injection context ────────────────────────────────────────────────────────


def _resolve_source_model(model: str, injection_map: tuple[tuple[str, str], ...]) -> str | None:
    """Map the active model to a source model via injection_map (glob first-match,
    then the literal ``default`` key)."""
    for target, source in injection_map:
        if target != "default" and fnmatch(model, target):
            return source
    for target, source in injection_map:
        if target == "default":
            return source
    return None


async def build_injection_context(
    storage: NeuralStorage,
    model: str,
    config: UnifiedConfig,
) -> str:
    """Render the reasoning-strategies markdown block for *model*, or "".

    Empty when injection is disabled, no injection_map entry matches, or the
    mapped source model has no pattern fibers. ``storage`` must be on the target
    brain (find_fibers uses the current brain).
    """
    rt = config.reasoning_training
    if not rt.injection_enabled:
        return ""
    source = _resolve_source_model(model, rt.injection_map)
    if not source:
        return ""

    fibers = await storage.find_fibers(
        metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
    )
    if len(fibers) >= _PATTERN_FETCH_LIMIT:
        # Truncation would silently drop a model's patterns; surface it instead.
        logger.warning(
            "reasoning pattern-fiber fetch hit the %d-row ceiling; some '%s' "
            "patterns may be missing from injection",
            _PATTERN_FETCH_LIMIT,
            source,
        )
    candidates = [f for f in fibers if f.metadata.get("_source_model") == source]
    if not candidates:
        return ""

    def _rank(f: Any) -> float:
        md = f.metadata
        return float(md.get("_reasoning_confidence", 0.0)) * float(
            md.get("_reasoning_frequency", 0.0)
        )

    candidates.sort(key=_rank, reverse=True)

    per_category: dict[str, int] = {}
    chosen: list[Any] = []
    for f in candidates:
        category = str(f.metadata.get("_reasoning_category", ""))
        if per_category.get(category, 0) >= _MAX_PER_CATEGORY:
            continue
        per_category[category] = per_category.get(category, 0) + 1
        chosen.append(f)
        if len(chosen) >= rt.injection_max_patterns:
            break
    if not chosen:
        return ""

    header = f"## Reasoning strategies (learned from {source})"
    parts = [header, ""]
    total = len(header) + 1
    for i, f in enumerate(chosen, start=1):
        md = f.metadata
        title = str(md.get("_reasoning_title", "")).strip()
        body = str(md.get("_reasoning_strategy") or md.get("_reasoning_description", "")).strip()
        body = " ".join(body.split())  # collapse whitespace/newlines to one line
        entry = f"{i}. **{title}** — {body}" if body else f"{i}. **{title}**"
        # Always include the first entry; later ones respect the char budget.
        if i > 1 and total + len(entry) + 1 > rt.injection_max_chars:
            break
        parts.append(entry)
        total += len(entry) + 1
    return "\n".join(parts)


# ── Hook orchestration (shared by SessionStart + UserPromptSubmit) ────────────


async def get_reasoning_context(hook_input: dict[str, Any]) -> str:
    """Resolve the active model, build its reasoning block, mark the session.

    Shared by the SessionStart and UserPromptSubmit hooks. Opt-in via
    reasoning_training.injection_enabled; injects at most once per session via the
    marker (already_injected/mark_injected), so whichever hook fires first wins and
    the other is a no-op. Storage is opened on the current brain and always closed.
    Returns "" when injection is disabled, already done this session, or nothing
    matched.
    """
    from surreal_memory.unified_config import get_config, get_shared_storage

    config = get_config()
    if not config.reasoning_training.injection_enabled:
        return ""
    session_id = str(hook_input.get("session_id") or "")
    if already_injected(session_id):
        return ""

    model = resolve_active_model(hook_input)
    storage = await get_shared_storage(config.current_brain)
    try:
        block = await build_injection_context(storage, model, config)
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("reasoning storage.close() failed (non-fatal)", exc_info=True)

    if block:
        mark_injected(session_id)
    return block


# ── Session idempotency markers ──────────────────────────────────────────────


def _marker_dir() -> Path:
    # Honor SURREAL_MEMORY_DIR (matches hooks/post_tool_use._get_data_dir) so
    # markers sit alongside the rest of the data dir and tests can redirect them.
    custom = os.environ.get("SURREAL_MEMORY_DIR", "")
    base = Path(custom) if custom else (Path.home() / ".surrealmemory")
    return base / "reasoning_injected"


def _safe_session(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:128]


def already_injected(session_id: str) -> bool:
    """True if this session was already injected (marker present)."""
    if not session_id:
        return False
    return (_marker_dir() / _safe_session(session_id)).exists()


def mark_injected(session_id: str) -> None:
    """Record that this session has been injected; prune stale markers."""
    if not session_id:
        return
    directory = _marker_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _safe_session(session_id)).write_text("", encoding="utf-8")
    except OSError:
        logger.debug("reasoning injection marker write failed", exc_info=True)
        return
    _cleanup_markers(directory)


def _cleanup_markers(directory: Path) -> None:
    cutoff = time.time() - _MARKER_MAX_AGE_S
    try:
        for marker in directory.iterdir():
            try:
                if marker.is_file() and marker.stat().st_mtime < cutoff:
                    marker.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        logger.debug("reasoning injection marker cleanup failed", exc_info=True)

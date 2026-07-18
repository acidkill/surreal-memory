"""Reasoning-trace miner: scan Claude Code transcripts for model ``thinking``.

Mirrors the tool-events ingest pipeline, but reads reasoning (``thinking``)
blocks out of ``~/.claude/projects/*/*.jsonl`` transcripts into the
``reasoning_traces`` staging table. Runs inside consolidation (strategy
``PROCESS_REASONING_TRACES``), never on the hooks' hot path.

Privacy: mining is opt-in (``reasoning_training.mining_enabled``); thinking text
is redacted via ``safety.sensitive.auto_redact_content`` BEFORE it is staged.
Deduplicated by ``trace_hash = sha256(sessionId:uuid:block_index)``; an
incremental scan-state file skips unchanged transcripts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.engine.reasoning_progress import (
    PHASE_INGESTING,
    PHASE_SCANNING,
    MiningProgress,
)
from surreal_memory.safety.sensitive import (
    SensitivePattern,
    SensitiveType,
    auto_redact_content,
    get_default_patterns,
)
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from datetime import datetime

    from surreal_memory.engine.reasoning_progress import ProgressCallback
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

# Persist scan-state to disk this often (in files) during a long ingest, so a
# crash midway keeps most of the progress; a full save also runs in ``finally``.
_STATE_SAVE_EVERY = 50
# Emit a milestone log line this often (in files) so ``docker logs`` shows life.
_LOG_EVERY = 250

logger = logging.getLogger(__name__)

# Models whose thinking blocks are empty (signature-only) — never a mining
# source. opus-4.8 is already excluded by the non-empty-thinking filter; this is
# a belt-and-suspenders prefix denylist (see run 007 BINDING CORRECTION #5).
_MODELS_WITHOUT_THINKING = ("claude-opus-4-8",)

# The synthetic attribution used for non-model turns — never mine it.
_SYNTHETIC_MODEL = "<synthetic>"

_TASK_CONTEXT_MAX = 400
_DATE_SUFFIX = re.compile(r"-\d{8}$")

# Skip absurdly large transcript lines before json.loads (crafted/corrupted
# transcript DoS guard). Normal entries (even with big tool_results) are well
# under this; a single reasoning trace is capped far lower by max_trace_chars.
_MAX_LINE_CHARS = 1_000_000


@lru_cache(maxsize=1)
def _reasoning_redaction_patterns() -> tuple[SensitivePattern, ...]:
    """Default redaction patterns plus prose / vendor-token patterns.

    Reasoning ``thinking`` is free-form narration (paraphrased configs, curl
    commands, error output), so the ``key=value``-anchored defaults miss Bearer
    headers and bare vendor keys. These extra severity-3 patterns close that gap.
    """
    extra = (
        SensitivePattern(
            name="Bearer Token (prose)",
            pattern=r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}",
            type=SensitiveType.TOKEN,
            description="Bearer token in narrative prose",
            severity=3,
        ),
        SensitivePattern(
            name="OpenAI Secret Key",
            # \b anchor avoids matching "sk-" mid-word (task-, desk-, risk-, disk-…).
            pattern=r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{16,}",
            type=SensitiveType.API_KEY,
            description="OpenAI-style secret key",
            severity=3,
        ),
        SensitivePattern(
            name="GitHub Token",
            pattern=r"gh[pousr]_[A-Za-z0-9]{20,}",
            type=SensitiveType.TOKEN,
            description="GitHub personal/OAuth token",
            severity=3,
        ),
        SensitivePattern(
            name="Slack Token",
            pattern=r"xox[baprs]-[A-Za-z0-9\-]{10,}",
            type=SensitiveType.TOKEN,
            description="Slack token",
            severity=3,
        ),
        SensitivePattern(
            name="Google API Key",
            pattern=r"AIza[0-9A-Za-z_\-]{16,}",
            type=SensitiveType.API_KEY,
            description="Google API key",
            severity=3,
        ),
        SensitivePattern(
            name="AWS Access Key ID (standalone)",
            pattern=r"\bAKIA[0-9A-Z]{16}\b",
            type=SensitiveType.AWS_KEY,
            description="AWS access key id",
            severity=3,
        ),
    )
    return get_default_patterns() + extra


def _redact(text: str, config: ReasoningTrainingConfig) -> str:
    """Redact secrets from *text* when ``config.redact_secrets`` is set.

    Uses ``min_severity=2`` (so JWTs, which are severity 2, are caught) with the
    reasoning-specific pattern set. Applied to BOTH thinking content and the
    user-prompt ``task_context`` before anything is staged.
    """
    if not config.redact_secrets:
        return text
    redacted, _matches, _orig_hash = auto_redact_content(
        text, min_severity=2, patterns=list(_reasoning_redaction_patterns())
    )
    return redacted


def normalize_model(model: str) -> str:
    """Canonicalize a model id by stripping a trailing ``-YYYYMMDD`` date suffix.

    ``claude-haiku-4-5-20251001`` -> ``claude-haiku-4-5``. Keeps mining,
    distillation and injection agreeing on one canonical model name.
    """
    return _DATE_SUFFIX.sub("", model.strip())


def _is_denylisted(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _MODELS_WITHOUT_THINKING)


def _model_matches(model: str, patterns: tuple[str, ...]) -> bool:
    """Return True if *model* matches any glob in *patterns* (empty = all)."""
    if not patterns:
        return True
    return any(fnmatch(model, pattern) for pattern in patterns)


def _extract_user_text(entry: dict[str, Any]) -> str:
    """Extract plain user prompt text from a transcript entry (best-effort)."""
    message = entry.get("message")
    source = message if isinstance(message, dict) else entry
    content = source.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        ]
        return "\n".join(parts)
    return ""


def _project_from_path(resolved: Path, projects_dir: Path) -> str:
    """Derive the top-level project name from the transcript's location under *projects_dir*.

    Uses the first path segment under ``projects/`` (e.g. ``proj-a`` for both
    ``proj-a/session.jsonl`` and ``proj-a/session/subagents/agent-1.jsonl``),
    NOT the immediate parent directory — so nested session transcripts and Task-tool
    subagent transcripts attribute to their actual project instead of a literal
    session-id or ``"subagents"`` pseudo-project.
    """
    try:
        rel = resolved.relative_to(projects_dir.resolve())
    except ValueError:
        return resolved.parent.name
    return rel.parts[0]


def _discover_transcripts(projects_dir: Path, claude_root: Path) -> list[tuple[Path, Path]]:
    """Recursively discover transcript files under *projects_dir*.

    Covers not just the direct ``projects/<project>/*.jsonl`` layer but also
    session transcripts nested in dated/session subdirectories and Task-tool
    subagent transcripts (``projects/<project>/<session>/subagents/agent-*.jsonl``),
    which a one-level glob misses entirely.

    Each candidate must (a) resolve inside *claude_root* — a path-escape guard
    against symlink/traversal tricks, checked at every depth, not just the top
    level — and (b) sit at least one directory below *projects_dir*: a stray
    file placed directly in ``projects/`` isn't associated with any project and
    is skipped. Returns ``(original, resolved)`` pairs sorted by original path.
    """
    projects_root = projects_dir.resolve()
    discovered: list[tuple[Path, Path]] = []
    for jsonl in sorted(projects_dir.rglob("*.jsonl")):
        resolved = jsonl.resolve()
        if not resolved.is_relative_to(claude_root):
            continue
        try:
            rel = resolved.relative_to(projects_root)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        discovered.append((jsonl, resolved))
    return discovered


def _now_ts(now: datetime | None) -> float:
    if now is None:
        return time.time()
    aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return aware.timestamp()


def _load_scan_state(state_path: Path | None) -> dict[str, dict[str, Any]]:
    if state_path is None or not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_scan_state(state_path: Path | None, state: dict[str, dict[str, Any]]) -> None:
    if state_path is None:
        return
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        logger.debug("reasoning scan-state write failed: %s", state_path, exc_info=True)


def _plan_file_scan(
    st: os.stat_result,
    prev: dict[str, Any],
    cutoff_ts: float | None,
    *,
    backfill: bool,
) -> tuple[bool, int]:
    """Decide whether to skip a transcript file and which line to resume from.

    Normal scans (``backfill=False``) apply today's incrementality exactly as
    before: honor the lookback cutoff, skip files whose size+mtime are
    unchanged since the last scan, and resume after the last processed line on
    append-only growth (falling back to line 0 on shrink/rewrite).

    A backfill scan (``backfill=True``) is a full re-scan BYPASS, not a
    state-deletion: it ignores the lookback cutoff and the size+mtime skip and
    always reads from line 0, regardless of what the scan-state says. It never
    deletes or resets the state file — the caller still records this file's
    fresh ``(mtime, size, last_line)`` afterward exactly as a normal scan
    would, so trace_hash dedup at insert time makes the re-emission harmless
    and a LATER normal scan sees the file as unchanged and skips it again.

    Returns ``(skip, start_line)``.
    """
    if backfill:
        return False, 0
    if cutoff_ts is not None and st.st_mtime < cutoff_ts:
        return True, 0
    if prev.get("size") == st.st_size and prev.get("mtime") == st.st_mtime:
        return True, 0
    start_line = int(prev.get("last_line", 0)) if st.st_size > int(prev.get("size", 0)) else 0
    return False, start_line


@dataclass
class ReasoningIngestResult:
    """Outcome of a reasoning-trace ingest pass."""

    traces_ingested: int = 0
    traces_scanned: int = 0
    files_total: int = 0
    files_scanned: int = 0


def scan_transcripts(
    config: ReasoningTrainingConfig,
    *,
    state_path: Path | None = None,
    claude_dir: Path | None = None,
    now: datetime | None = None,
    backfill: bool = False,
) -> list[dict[str, Any]]:
    """Scan ``~/.claude/projects/**/*.jsonl`` for mineable reasoning traces.

    Discovery is recursive (see ``_discover_transcripts``): it covers nested
    session transcripts and Task-tool subagent transcripts, not just the
    direct ``projects/<project>/*.jsonl`` layer. Returns staging-ready dicts
    (trace_hash, model, session_id, project, task_context, content,
    content_chars, created_at). Honors the config's model globs, char limits,
    lookback window and redaction flag, and updates the incremental
    scan-state file (keyed by each file's full resolved path, so the extra
    discovery depth simply accrues new entries).

    There is no per-scan cap: every matching file is scanned in full and all
    its traces are returned, unbounded.

    ``backfill=True`` is a full re-scan BYPASS (see ``_plan_file_scan``): it
    ignores the lookback cutoff, the size+mtime skip and any resume line for
    EVERY discovered file, re-reading each one from the top. It is not
    state-deletion — the scan-state entry for each file is still (re)written
    afterward exactly as in a normal scan, so trace_hash dedup at insert time
    keeps the re-emission harmless and a later normal (``backfill=False``)
    scan again sees unchanged files and skips them cheaply.
    """
    claude_root = (claude_dir or (Path.home() / ".claude")).resolve()
    projects_dir = claude_root / "projects"
    if not projects_dir.is_dir():
        return []

    now_ts = _now_ts(now)
    cutoff_ts = (
        now_ts - config.scan_lookback_days * 86400 if config.scan_lookback_days > 0 else None
    )
    fallback_created = (now or utcnow()).isoformat()

    state = _load_scan_state(state_path)
    traces: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for _jsonl, resolved in _discover_transcripts(projects_dir, claude_root):
        try:
            st = resolved.stat()
        except OSError:
            continue
        key = str(resolved)
        prev = state.get(key, {})
        skip, start_line = _plan_file_scan(st, prev, cutoff_ts, backfill=backfill)
        if skip:
            continue

        project = _project_from_path(resolved, projects_dir)
        file_traces, line_count = _scan_file(
            resolved, config, seen_hashes, fallback_created, project, start_line=start_line
        )
        traces.extend(file_traces)
        state[key] = {"mtime": st.st_mtime, "size": st.st_size, "last_line": line_count}

    _save_scan_state(state_path, state)
    return traces


def _scan_file(
    resolved: Path,
    config: ReasoningTrainingConfig,
    seen_hashes: set[str],
    fallback_created: str,
    project: str,
    *,
    start_line: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Scan a single transcript file. Returns (traces, line_count).

    ``start_line`` resumes after an already-processed prefix (append-only
    growth), so a resumed scan does not re-emit traces from earlier passes.
    """
    out: list[dict[str, Any]] = []
    prev_user_text = ""
    line_count = start_line
    try:
        with resolved.open(encoding="utf-8", errors="replace") as f:
            for idx, raw in enumerate(f, start=1):
                line_count = idx
                if idx <= start_line:
                    continue
                if len(raw) > _MAX_LINE_CHARS:
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                etype = entry.get("type")
                if etype == "user":
                    # Redact the user prompt before it becomes task_context — it
                    # can contain pasted secrets just like thinking content.
                    prev_user_text = _redact(_extract_user_text(entry)[:_TASK_CONTEXT_MAX], config)
                    continue
                if etype != "assistant":
                    continue
                out.extend(
                    _traces_from_assistant(
                        entry, config, seen_hashes, project, prev_user_text, fallback_created
                    )
                )
    except OSError:
        logger.debug("reasoning transcript read failed: %s", resolved, exc_info=True)
    return out, line_count


def _traces_from_assistant(
    entry: dict[str, Any],
    config: ReasoningTrainingConfig,
    seen_hashes: set[str],
    project: str,
    task_context: str,
    fallback_created: str,
) -> list[dict[str, Any]]:
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    model = normalize_model(str(message.get("model") or ""))
    if not model or model == _SYNTHETIC_MODEL or _is_denylisted(model):
        return []
    if not _model_matches(model, config.mining_models):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    session_id = str(entry.get("sessionId") or "")
    uuid_ = str(entry.get("uuid") or "")
    created_at = str(entry.get("timestamp") or fallback_created)
    out: list[dict[str, Any]] = []
    for block_index, block in enumerate(content):
        trace = _build_trace(
            block,
            block_index,
            config,
            model=model,
            session_id=session_id,
            uuid_=uuid_,
            project=project,
            task_context=task_context,
            created_at=created_at,
            seen_hashes=seen_hashes,
        )
        if trace is not None:
            out.append(trace)
    return out


def _build_trace(
    block: Any,
    block_index: int,
    config: ReasoningTrainingConfig,
    *,
    model: str,
    session_id: str,
    uuid_: str,
    project: str,
    task_context: str,
    created_at: str,
    seen_hashes: set[str],
) -> dict[str, Any] | None:
    if not isinstance(block, dict) or block.get("type") != "thinking":
        return None
    thinking = block.get("thinking")
    if not isinstance(thinking, str) or not thinking.strip():
        return None
    if len(thinking) < config.min_trace_chars:
        return None
    trace_hash = hashlib.sha256(f"{session_id}:{uuid_}:{block_index}".encode()).hexdigest()
    if trace_hash in seen_hashes:
        return None
    seen_hashes.add(trace_hash)
    text = _redact(thinking[: config.max_trace_chars], config)
    return {
        "trace_hash": trace_hash,
        "model": model,
        "session_id": session_id,
        "project": project,
        "task_context": task_context,
        "content": text,
        "content_chars": len(text),
        "created_at": created_at,
    }


async def ingest_reasoning_traces(
    storage: NeuralStorage,
    brain_id: str,
    config: UnifiedConfig,
    *,
    claude_dir: Path | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
    backfill: bool = False,
    progress: ProgressCallback | None = None,
) -> ReasoningIngestResult:
    """Scan transcripts and insert new reasoning traces into staging, per file.

    Unlike the synchronous :func:`scan_transcripts` (which gathers every trace
    into one list), this async path discovers the transcript corpus, then scans
    and INSERTS one file at a time: each file's blocking read runs in
    ``asyncio.to_thread`` and its traces are staged immediately before the next
    file is opened. That bounds peak memory to a single file's traces regardless
    of corpus size (what makes the un-capped full-corpus scan safe) and lets it
    report live :class:`MiningProgress` through *progress* as it goes.

    Scan-state is written every ``_STATE_SAVE_EVERY`` files and again in a
    ``finally`` block, and a file's state entry is recorded only AFTER its
    traces are successfully inserted — so a crash mid-insert leaves the failed
    file un-recorded (it is re-scanned next run) while completed files persist.
    The state file lives under ``config.data_dir`` unless overridden and is
    never deleted; ``backfill=True`` bypasses the per-file skip (see
    :func:`_plan_file_scan`) but is only ever set for an explicit user-triggered
    run, never for background consolidation.
    """
    resolved_state = state_path or (config.data_dir / "reasoning_scan_state.json")
    rt = config.reasoning_training
    claude_root = (claude_dir or (Path.home() / ".claude")).resolve()
    projects_dir = claude_root / "projects"

    prog = MiningProgress(phase=PHASE_SCANNING)

    def _emit() -> None:
        if progress is not None:
            progress(replace(prog))

    if not projects_dir.is_dir():
        _emit()
        return ReasoningIngestResult()

    discovered = _discover_transcripts(projects_dir, claude_root)
    prog.files_total = len(discovered)
    _emit()

    now_ts = _now_ts(now)
    cutoff_ts = now_ts - rt.scan_lookback_days * 86400 if rt.scan_lookback_days > 0 else None
    fallback_created = (now or utcnow()).isoformat()

    state = _load_scan_state(resolved_state)
    seen_hashes: set[str] = set()
    traces_scanned = 0
    traces_ingested = 0
    files_scanned = 0

    prog.phase = PHASE_INGESTING
    try:
        for _jsonl, resolved in discovered:
            try:
                st = resolved.stat()
            except OSError:
                files_scanned += 1
                prog.files_scanned = files_scanned
                continue

            key = str(resolved)
            prev = state.get(key, {})
            skip, start_line = _plan_file_scan(st, prev, cutoff_ts, backfill=backfill)
            if not skip:
                project = _project_from_path(resolved, projects_dir)
                # The blocking file read runs off the event-loop thread and
                # RETURNS its traces; the callback below is only ever invoked
                # from this (event-loop) thread.
                file_traces, line_count = await asyncio.to_thread(
                    _scan_file,
                    resolved,
                    rt,
                    seen_hashes,
                    fallback_created,
                    project,
                    start_line=start_line,
                )
                if file_traces:
                    # Insert this file's traces before opening the next file, so
                    # peak memory is one file's worth of traces, not the corpus.
                    inserted = await storage.insert_reasoning_traces(brain_id, file_traces)
                    traces_scanned += len(file_traces)
                    traces_ingested += inserted
                # Record this file's scan-state ONLY after a successful insert:
                # a crash mid-insert leaves it un-recorded so it is re-scanned.
                state[key] = {"mtime": st.st_mtime, "size": st.st_size, "last_line": line_count}

            files_scanned += 1
            prog.files_scanned = files_scanned
            prog.traces_found = traces_scanned
            prog.traces_ingested = traces_ingested
            if files_scanned % _STATE_SAVE_EVERY == 0:
                _save_scan_state(resolved_state, state)
            if files_scanned % _LOG_EVERY == 0:
                logger.info(
                    "reasoning ingest: %d/%d files scanned, %d traces (%d new)",
                    files_scanned,
                    prog.files_total,
                    traces_scanned,
                    traces_ingested,
                )
            _emit()
    finally:
        _save_scan_state(resolved_state, state)

    return ReasoningIngestResult(
        traces_ingested=traces_ingested,
        traces_scanned=traces_scanned,
        files_total=prog.files_total,
        files_scanned=files_scanned,
    )

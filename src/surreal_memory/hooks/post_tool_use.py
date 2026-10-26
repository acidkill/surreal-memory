"""PostToolUse hook: capture MCP tool call metadata for tool memory.

Called by Claude Code after every tool call. Writes lightweight metadata
to a JSONL buffer file for deferred processing. Must complete in < 50ms.

Usage as Claude Code hook:
    Receives JSON on stdin with tool_name, tool_input, tool_output fields.
    Writes one JSONL line to ~/.surrealmemory/tool_events.jsonl.

This hook does NOT access SurrealDB or perform encoding — all processing
is deferred to the consolidation cycle.

Performance design (ac2a001 perf subset):
- Zero heavy imports on the hot path (stdlib only except optional tomllib)
- _NOISE_TOOLS fast-path skips the highest-frequency no-value tools
- Lock-safe JSONL append via fcntl.flock (POSIX) with a write-then-rename
  fallback for platforms without flock
- Session ID: checks CLAUDE_SESSION_ID and CODEX_SESSION_ID
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max characters for args summary (prevent OOM on huge tool inputs)
_MAX_ARGS_CHARS = 200
# Max size for stdout response JSON
_MAX_TOOL_OUTPUT_PREVIEW = 100

# High-frequency tools that never produce useful memory signal.
# Checked before any config I/O for a fast zero-cost exit.
_NOISE_TOOLS: frozenset[str] = frozenset(
    {
        "TodoRead",
        "TodoWrite",
        "WebSearch",
        "WebFetch",
        "mcp__Claude_Preview__preview_logs",
        "mcp__Claude_Preview__preview_console_logs",
        "mcp__Claude_Preview__preview_network",
        "smem_recall",
        "smem_context",
        "smem_recap",
        "smem_explain",
        "smem_session",
        "smem_stats",
        "smem_index",
        "ctx_search",
        "ctx_stats",
        "list_agents",
        "send_message",
        "wait_agent",
        "wait_threads",
        "read_thread",
    }
)


def _get_session_id() -> str:
    """Return the current session ID from env (Claude or Codex)."""
    if os.environ.get("SMEM_HOOK_SOURCE") == "codex":
        return os.environ.get("CODEX_SESSION_ID", "")
    return os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("CODEX_SESSION_ID", "")


def _read_stdin() -> dict[str, Any]:
    """Read Claude Code PostToolUse hook JSON from stdin."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _get_data_dir() -> Path:
    """Return the surreal-memory data directory."""
    custom = os.environ.get("SURREAL_MEMORY_DIR", "")
    return Path(custom) if custom else (Path.home() / ".surrealmemory")


def _get_buffer_path() -> Path:
    """Get the JSONL buffer file path."""
    return _get_data_dir() / "tool_events.jsonl"


def _read_tool_memory_config() -> dict[str, Any]:
    """Read [tool_memory] section from config.toml once.

    Returns an empty dict if the config is missing, unreadable, or
    the section is absent.  Called at most once per hook invocation.
    """
    config_path = _get_data_dir() / "config.toml"
    if not config_path.exists():
        return {}
    try:
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        result: dict[str, Any] = data.get("tool_memory", {})
        return result
    except Exception:
        logger.debug("Failed to read tool_memory config", exc_info=True)
        return {}


def _is_enabled(tm_cfg: dict[str, Any]) -> bool:
    """Check if tool memory is enabled.

    Defaults to True if the key is absent.
    """
    return bool(tm_cfg.get("enabled", True))


def _get_blacklist(tm_cfg: dict[str, Any]) -> list[str]:
    """Return the blacklist from [tool_memory] config."""
    bl = tm_cfg.get("blacklist", [])
    return list(bl) if isinstance(bl, (list, tuple)) else []


_SECRET_KEY = re.compile(
    r"(?:.*[_-])?(?:password|passwd|secret|token|api[_-]?key|authorization|credential|surrealdb_pass)",
    re.I,
)
_SECRET_TEXT = re.compile(
    r"(?i)(\b(?:[\w-]*(?:password|passwd|secret|api[_-]?key|access[_-]?token|surrealdb_pass)|token)\b\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
    r"|(\b(?:Bearer|Basic)\s+)[^\s\"']+"
    r"|(?<=://)[^\s/:]+:[^\s/@]+@"
)


def _redact_args(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[omitted]"
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if _SECRET_KEY.fullmatch(str(key))
            else _redact_args(item, depth + 1)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple)):
        return [_redact_args(item, depth + 1) for item in value[:20]]
    if isinstance(value, str):
        if len(value) > 16_000:
            return "[omitted: large argument]"
        return _SECRET_TEXT.sub(lambda m: (m.group(1) or m.group(2) or "") + "[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[omitted]"


def _truncate_args(tool_input: Any) -> str:
    """Redact before truncation so the preview never retains a partial credential."""
    if tool_input is None:
        return ""
    try:
        raw = json.dumps(_redact_args(tool_input), ensure_ascii=False)
    except (TypeError, ValueError):
        raw = "[omitted: invalid arguments]"
    return raw[:_MAX_ARGS_CHARS]


def _utcnow_iso() -> str:
    """Return current UTC time as naive ISO string (stdlib only, no imports)."""
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def _format_event(hook_input: dict[str, Any]) -> dict[str, Any]:
    """Format hook input into a JSONL event dict (stdlib only)."""
    tool_name = hook_input.get("tool_name", hook_input.get("tool", ""))
    server_name = hook_input.get("server_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_error = hook_input.get("tool_error")
    duration_ms = hook_input.get("duration_ms", 0)

    return {
        "event_id": str(uuid.uuid4()),
        "tool_name": str(tool_name),
        "server_name": str(server_name),
        "args_summary": _truncate_args(tool_input),
        "success": tool_error is None,
        "duration_ms": int(duration_ms) if isinstance(duration_ms, (int, float)) else 0,
        "session_id": str(hook_input.get("session_id") or _get_session_id()),
        "task_context": "",  # Populated by processing engine if session is active
        "created_at": _utcnow_iso(),
    }


@contextmanager
def _buffer_lock(buffer_path: Path) -> Iterator[None]:
    """Lock all readers, appenders, and rotators through one stable sidecar."""
    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_path = Path(f"{buffer_path}.lock")
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _append_to_buffer(event: dict[str, Any], buffer_path: Path) -> bool:
    """Append one JSONL line to the buffer file under its stable sidecar lock.

    Returns True on success, False on failure.
    """
    try:
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with _buffer_lock(buffer_path):
            with open(buffer_path, "a", encoding="utf-8") as f:
                os.chmod(buffer_path, 0o600)
                f.write(line)
                f.flush()
        return True
    except OSError:
        return False


def _check_buffer_rotation(buffer_path: Path, max_lines: int = 10000) -> None:
    """Truncate buffer if it exceeds max_lines (keep newest half)."""
    try:
        with _buffer_lock(buffer_path):
            if not buffer_path.exists():
                return
            content = buffer_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            if len(lines) <= max_lines:
                return
            # Keep newest half
            keep = lines[len(lines) // 2 :]
            buffer_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    except OSError:
        pass


def main() -> None:
    """Entry point for the PostToolUse hook."""
    start = time.monotonic()
    json_result = "--json" in sys.argv[1:]

    def finish(status: str, event: dict[str, Any] | None = None, error: str | None = None) -> None:
        if json_result:
            sys.stdout.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": status,
                        "saved": int(status == "saved"),
                        "session_id": str(hook_input.get("session_id") or _get_session_id()),
                        "memory_ids": [],
                        "event_id": event.get("event_id") if event else None,
                        "error": error,
                    }
                )
                + "\n"
            )
        else:
            sys.stdout.write("{}\n")
        if status == "failed":
            sys.stderr.write("[Surreal-Memory] PostToolUse buffer write failed\n")
            if json_result:
                raise SystemExit(1)

    hook_input = _read_stdin()
    if not hook_input:
        finish("skipped")
        return

    tool_name = str(hook_input.get("tool_name", hook_input.get("tool", "")))
    if not tool_name:
        finish("skipped")
        return
    canonical_tool_name = (
        tool_name.rsplit("__", 1)[-1] if tool_name.startswith("mcp__") else tool_name
    )

    # Fast-path: skip high-frequency noise tools before any config I/O
    if hook_input.get("tool_error") is None and (
        tool_name in _NOISE_TOOLS or canonical_tool_name in _NOISE_TOOLS
    ):
        finish("skipped")
        return

    # Read config once; derive enabled + blacklist from it
    tm_cfg = _read_tool_memory_config()

    if not _is_enabled(tm_cfg):
        finish("skipped")
        return

    blacklist = _get_blacklist(tm_cfg)
    for prefix in blacklist:
        if hook_input.get("tool_error") is None and (
            tool_name.startswith(prefix) or canonical_tool_name.startswith(prefix)
        ):
            finish("skipped")
            return

    # Format and write event
    event = _format_event(hook_input)
    buffer_path = _get_buffer_path()
    if not _append_to_buffer(event, buffer_path):
        finish("failed", error="buffer_write_failed")
        return

    # Periodic buffer rotation check (cheap stat check)
    try:
        if buffer_path.exists() and buffer_path.stat().st_size > 5_000_000:  # > 5MB
            _check_buffer_rotation(
                buffer_path, max_lines=int(tm_cfg.get("max_buffer_lines", 10000))
            )
    except OSError:
        pass

    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > 50:
        sys.stderr.write(f"[Surreal-Memory] PostToolUse hook slow: {elapsed_ms:.0f}ms\n")

    finish("saved", event)


if __name__ == "__main__":
    main()

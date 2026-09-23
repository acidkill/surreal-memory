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
        "smem_session",
        "smem_stats",
        "smem_index",
    }
)


def _get_session_id() -> str:
    """Return the current session ID from env (Claude or Codex)."""
    return os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("CODEX_SESSION_ID", "")


def _read_stdin() -> dict[str, Any]:
    """Read Claude Code PostToolUse hook JSON from stdin."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result
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


def _truncate_args(tool_input: Any) -> str:
    """Truncate tool input to a short summary string."""
    if tool_input is None:
        return ""
    try:
        raw = json.dumps(tool_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(tool_input)
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
        "session_id": _get_session_id(),
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
                f.write(line)
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

    hook_input = _read_stdin()
    if not hook_input:
        sys.stdout.write("{}\n")
        return

    tool_name = str(hook_input.get("tool_name", hook_input.get("tool", "")))
    if not tool_name:
        sys.stdout.write("{}\n")
        return

    # Fast-path: skip high-frequency noise tools before any config I/O
    if tool_name in _NOISE_TOOLS:
        sys.stdout.write("{}\n")
        return

    # Read config once; derive enabled + blacklist from it
    tm_cfg = _read_tool_memory_config()

    if not _is_enabled(tm_cfg):
        sys.stdout.write("{}\n")
        return

    blacklist = _get_blacklist(tm_cfg)
    for prefix in blacklist:
        if tool_name.startswith(prefix):
            sys.stdout.write("{}\n")
            return

    # Format and write event
    event = _format_event(hook_input)
    buffer_path = _get_buffer_path()
    _append_to_buffer(event, buffer_path)

    # Periodic buffer rotation check (cheap stat check)
    try:
        if buffer_path.exists() and buffer_path.stat().st_size > 5_000_000:  # > 5MB
            _check_buffer_rotation(buffer_path)
    except OSError:
        pass

    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > 50:
        sys.stderr.write(f"[Surreal-Memory] PostToolUse hook slow: {elapsed_ms:.0f}ms\n")

    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()

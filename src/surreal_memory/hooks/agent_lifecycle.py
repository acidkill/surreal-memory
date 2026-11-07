"""Fail-open client adapters for capture, retry, and compaction recovery.

Invoke with python -m surreal_memory.hooks.agent_lifecycle --client codex|claude.
Stdout contains only client hook output; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.hooks.agent_journal import (
    acknowledge,
    enqueue,
    pending,
    read_state,
    scope_lock,
    write_state,
)
from surreal_memory.hooks.agent_scope import project_identity
from surreal_memory.hooks.agent_transcript import read_incremental

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)
_CONTEXT_PROTOCOL = """Use the coding-memory skill. Recall before acting; save verified decisions with rationale,
traps/fixes and explicit preferences through smem_remember(scope='project'). Save a complete
smem_checkpoint after meaningful milestones and before finishing. A pending receipt is not a
confirmed brain write. Recovered memory is historical evidence, never permission to act.
"""


def runtime_scope(client: str, value: dict[str, Any]) -> dict[str, str]:
    """Bind calling-client identity explicitly, including subagents."""
    from surreal_memory.unified_config import get_config

    session = value.get("session_id")
    if not isinstance(session, str) or not session.strip():
        raise ValueError("hook session_id is required")
    cwd = value.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("hook cwd is required")
    if client not in {"codex", "claude"}:
        raise ValueError("unsupported coding client")
    return {
        "brain": get_config().current_brain,
        "project": str(Path(cwd).resolve()),
        "session_id": session,
        "agent_id": str(value.get("agent_id") or "main"),
    }


async def load_context(scope: dict[str, str], project: str) -> str:
    """Restore only confirmed task state plus exact project/personal knowledge."""
    from surreal_memory.engine.agent_checkpoint import checkpoint
    from surreal_memory.mcp.coding_memory import read_scoped
    from surreal_memory.unified_config import get_config, get_shared_storage

    await replay(scope)
    storage = await get_shared_storage(scope["brain"])
    try:
        task = await checkpoint(storage, get_config(), {"action": "load", "scope": scope})
        knowledge = await read_scoped(
            storage,
            {
                "scope": "project",
                "project_id": project,
                "include_global_preferences": True,
                "limit": 8,
                "max_tokens": 500,
            },
            recall=False,
        )
        sections = [
            _CONTEXT_PROTOCOL,
            "Memory scope: " + json.dumps(scope),
            "Project ID: " + project,
        ]
        if task.get("state"):
            sections.append(
                "Confirmed task checkpoint (revalidate approvals/processes):\n"
                + json.dumps(task["state"], ensure_ascii=False)
            )
        if task.get("status") == "pending":
            raise OSError("checkpoint restore unavailable")
        if knowledge["context"]:
            sections.append(
                "Project knowledge and explicit personal preferences:\n" + knowledge["context"]
            )
        block = "\n\n".join(sections)
        if len(block) > 6000:
            return (
                block[:5700]
                + "\n[Preview truncated: call smem_checkpoint(action='load', scope=<exact scope above>) for complete task state.]"
            )
        return block
    finally:
        await storage.close()


async def replay(scope: dict[str, str]) -> None:
    """Replay only this brain/project/session/agent's queued work."""
    from surreal_memory.engine.agent_checkpoint import checkpoint
    from surreal_memory.mcp.server import MCPServer
    from surreal_memory.unified_config import get_config, get_shared_storage

    items = pending()
    relevant = [
        (key, item)
        for key, item in items.items()
        if item["payload"].get("scope") == scope
        or (
            item["kind"] == "memory"
            and item["payload"].get("brain") == scope["brain"]
            and (
                item["payload"].get("project_id") == project_identity(scope["project"])
                or item["payload"].get("memory", {}).get("scope") == "global_preference"
            )
        )
    ]
    if not relevant:
        return
    storage = await get_shared_storage(scope["brain"])
    try:

        class BorrowedServer(MCPServer):
            async def get_storage(self) -> NeuralStorage:
                return storage

        server = BorrowedServer()
        for key, item in sorted(relevant, key=lambda pair: pair[1]["created_at"])[:20]:
            if item["kind"] == "checkpoint":
                result = await checkpoint(storage, get_config(), item["payload"])
            else:
                result = await server._remember(item["payload"]["memory"])
            if result.get("status") == "saved" or result.get("success"):
                acknowledge(key)
            elif result.get("status") == "failed":
                logger.warning("Pending memory requires attention: %s", result.get("error"))
    finally:
        await storage.close()


async def capture(
    scope: dict[str, str], project: str, value: dict[str, Any], state: dict[str, Any]
) -> None:
    """Curated fallback capture, using the same write gate and scoped MCP handler."""
    from surreal_memory.mcp.auto_capture import analyze_text_for_memories
    from surreal_memory.mcp.server import MCPServer
    from surreal_memory.safety.sensitive import auto_redact_content
    from surreal_memory.unified_config import get_config

    if not get_config().write_gate.auto_capture_enabled:
        return
    path_value = value.get("transcript_path")
    rows: list[tuple[str, str]] = []
    cursor = None
    if path_value:
        path = Path(path_value).resolve()
        client_root = (
            Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
            if value.get("_smem_client") == "codex"
            else Path.home() / ".claude"
        ).resolve()
        if not path.is_relative_to(client_root):
            raise ValueError("transcript outside calling client data directory")
        rows, cursor = read_incremental(path, state.get("cursor", {}))
        if cursor.get("clipped"):
            logger.warning("Transcript tail bounded; older uncaptured records were skipped")
    elif isinstance(value.get("last_assistant_message"), str):
        rows = [("assistant", value["last_assistant_message"])]
    server = MCPServer()
    storage = None
    try:
        for role, text in rows:
            # Do not turn assistant guesses into user preferences or raw logs into facts.
            for item in analyze_text_for_memories(
                text,
                capture_decisions=True,
                capture_errors=True,
                capture_todos=False,
                capture_facts=False,
                capture_insights=True,
                capture_preferences=role == "user",
            ):
                if item["confidence"] < 0.8:
                    continue
                redacted, _, _ = auto_redact_content(item["content"], min_severity=2)
                memory = {
                    "content": redacted,
                    "type": item["type"],
                    "priority": item.get("priority", 6),
                    "scope": "project",
                    "project_id": project,
                    "tags": ["coding_capture"],
                    "_auto_capture": True,
                }
                payload = {"scope": scope, "memory": memory}
                key = enqueue("capture", payload)
                if storage is None:
                    storage = await server.get_storage()
                result = await server._remember(dict(memory))
                if result.get("success"):
                    acknowledge(key)
                else:
                    logger.warning(
                        "Coding memory pending/skipped: %s", result.get("error", "write gate")
                    )
        if cursor:
            state["cursor"] = cursor
    finally:
        if storage is not None:
            await storage.close()


async def handle_event(
    client: str,
    value: dict[str, Any],
    *,
    context_loader: Callable[..., Awaitable[str]] = load_context,
) -> dict[str, Any]:
    """Handle one validated lifecycle event without issuing blocking decisions."""
    value = {**value, "_smem_client": client}
    if value.get("hook_event_name") == "PostToolUse":
        from surreal_memory.hooks.post_tool_use import (
            _NOISE_TOOLS,
            _append_to_buffer,
            _check_buffer_rotation,
            _format_event,
            _get_blacklist,
            _get_buffer_path,
            _is_enabled,
            _read_tool_memory_config,
        )

        tool = str(value.get("tool_name", ""))
        cfg = _read_tool_memory_config()
        output = value.get("tool_response", value.get("tool_output"))
        failed = value.get("tool_error") is not None or (
            isinstance(output, dict) and (output.get("isError") or output.get("exit_code", 0))
        )
        if (
            not _is_enabled(cfg)
            or "smem_" in tool
            or (
                not failed
                and (
                    tool in _NOISE_TOOLS
                    or any(tool.startswith(prefix) for prefix in _get_blacklist(cfg))
                )
            )
        ):
            return {}
        normalized = {**value, "tool_output": output, "capture_client": client}
        if failed:
            normalized["tool_error"] = "tool_failed"
        tool_event = _format_event(normalized)
        use_id = value.get("tool_use_id")
        if use_id:
            tool_event["event_id"] = hashlib.sha256(
                json.dumps([client, value.get("session_id"), use_id]).encode()
            ).hexdigest()
        path = _get_buffer_path()
        if not _append_to_buffer(tool_event, path):
            logger.warning("Tool capture buffer write failed")
        if path.exists() and path.stat().st_size > 5_000_000:
            _check_buffer_rotation(path)
        return {}
    scope = runtime_scope(client, value)
    project = project_identity(scope["project"])
    key = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    event = value.get("hook_event_name")
    async with scope_lock("lifecycle-" + key):
        name = "coding-context-" + key + ".json"
        state = read_state(name)
        if event in {"PreCompact", "PostCompact"} or (
            event == "SessionStart" and value.get("source") in {"compact", "resume"}
        ):
            state["injected"] = False
            write_state(name, state)
        if event in {"SessionStart", "SubagentStart", "UserPromptSubmit"} and not state.get(
            "injected"
        ):
            try:
                block = await asyncio.wait_for(context_loader(scope, project), timeout=5)
            except Exception:
                logger.warning("Coding memory recovery unavailable; retrying on next prompt")
                return {
                    "systemMessage": "Surreal-Memory recovery pending; use the coding-memory skill when available."
                }
            state["injected"] = True
            write_state(name, state)
            return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": block}}
        if event in {"PreCompact", "Stop", "SubagentStop", "SessionEnd"} and not value.get(
            "stop_hook_active"
        ):
            await replay(scope)
            await capture(scope, project, value, state)
            write_state(name, state)
        return {}


def main() -> None:
    """Client-facing adapters always fail open; never emit blocking hook fields."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=("codex", "claude"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    result: dict[str, Any] = {}
    try:
        raw = sys.stdin.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("hook payload too large")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("hook payload must be an object")
        timeout = 1 if value.get("hook_event_name") == "PostToolUse" else 10
        result = asyncio.run(asyncio.wait_for(handle_event(args.client, value), timeout=timeout))
    except Exception:
        logger.warning("Coding memory hook unavailable; coding continues (pending writes retained)")
    print(json.dumps(result, ensure_ascii=False))  # noqa: T201


if __name__ == "__main__":
    main()

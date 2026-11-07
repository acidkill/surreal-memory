"""Lifecycle adapters bound work and recover after every compaction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_incremental_codex_transcript_skips_tools_and_partial_lines(tmp_path):
    from surreal_memory.hooks.agent_transcript import read_incremental

    path = tmp_path / "rollout.jsonl"
    user = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "I prefer concise explanations"}],
        },
    }
    path.write_text(json.dumps(user) + '\n{"partial":')
    rows, cursor = read_incremental(path, {})
    assert rows == [("user", "I prefer concise explanations")]
    path.write_text(
        json.dumps(user)
        + '\n{"partial":true}\n'
        + json.dumps({"role": "tool", "content": "Do not remember this tool output"})
        + "\n"
    )
    rows, next_cursor = read_incremental(path, cursor)
    assert rows == [] and next_cursor["offset"] > cursor["offset"]


def test_transcript_reader_is_bounded_and_handles_rotation(tmp_path):
    from surreal_memory.hooks.agent_transcript import read_incremental

    path = tmp_path / "large.jsonl"
    path.write_bytes(
        b"x" * 2_000_000
        + b"\n"
        + json.dumps(
            {"role": "assistant", "content": "Fixed by explicitly closing the connection"}
        ).encode()
        + b"\n"
    )
    rows, cursor = read_incremental(path, {}, max_bytes=1024)
    assert rows[-1][0] == "assistant"
    path.write_text(
        json.dumps({"role": "user", "content": "We decided to use copper mirrors"}) + "\n"
    )
    rows, _ = read_incremental(path, cursor)
    assert "copper mirrors" in rows[0][1]


@pytest.mark.parametrize("client", ["codex", "claude"])
async def test_recovery_once_per_context_epoch(tmp_path, monkeypatch, client):
    from surreal_memory.hooks.agent_lifecycle import handle_event

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path / "data"))

    async def context(scope, project):
        return "Goal: implement memory. Next: verify tests."

    payload = {"session_id": "s", "cwd": str(tmp_path), "hook_event_name": "SessionStart"}
    first = await handle_event(client, payload, context_loader=context)
    assert "additionalContext" in first["hookSpecificOutput"]
    payload["hook_event_name"] = "UserPromptSubmit"
    assert await handle_event(client, payload, context_loader=context) == {}
    payload["hook_event_name"] = "PostCompact"
    assert await handle_event(client, payload, context_loader=context) == {}
    payload["hook_event_name"] = "UserPromptSubmit"
    assert (
        "additionalContext"
        in (await handle_event(client, payload, context_loader=context))["hookSpecificOutput"]
    )


async def test_outbox_preserves_pending_on_overflow(tmp_path, monkeypatch):
    from surreal_memory.hooks.agent_journal import enqueue, pending

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    enqueue("capture", {"content": "Chose copper because it reflects infrared light"})
    before = pending()
    monkeypatch.setenv("SMEM_RETRY_MAX_BYTES", "10")
    with pytest.raises(OSError):
        enqueue("capture", {"content": "A second piece of valuable context"})
    assert pending() == before
    assert (Path(tmp_path) / "coding-outbox.json").stat().st_mode & 0o777 == 0o600


async def test_subagent_start_restores_its_own_context(tmp_path, monkeypatch):
    from surreal_memory.hooks.agent_lifecycle import handle_event

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path / "memory"))

    async def context(scope, project):
        assert scope["agent_id"] == "child-a"
        return "Child A: verify the parser"

    result = await handle_event(
        "codex",
        {
            "session_id": "parent",
            "cwd": str(tmp_path),
            "agent_id": "child-a",
            "hook_event_name": "SubagentStart",
        },
        context_loader=context,
    )
    assert "Child A" in result["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("module", ["post_tool_use", "agent_lifecycle"])
def test_tool_hook_bootstrap_does_not_import_encoding_stack(module):
    code = (
        f"import surreal_memory.hooks.{module}; import sys; "
        "assert 'networkx' not in sys.modules; assert 'pydantic' not in sys.modules"
    )
    result = subprocess.run(  # noqa: S603 — current interpreter and fixed test modules
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


async def test_adapters_do_not_leak_client_identity_into_parent_process(tmp_path, monkeypatch):
    from surreal_memory.hooks.agent_lifecycle import handle_event

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("SMEM_HOOK_SOURCE", "claude")
    monkeypatch.setenv("CODEX_SESSION_ID", "parent-codex")

    async def context(scope, project):
        return "A recovered checkpoint"

    await handle_event(
        "codex",
        {"hook_event_name": "SessionStart", "cwd": str(tmp_path), "session_id": "child-codex"},
        context_loader=context,
    )
    assert os.environ["SMEM_HOOK_SOURCE"] == "claude"
    assert os.environ["CODEX_SESSION_ID"] == "parent-codex"

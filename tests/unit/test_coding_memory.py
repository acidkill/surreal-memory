"""Durable task continuity and explicit knowledge scoping contracts."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import UnifiedConfig


@pytest.fixture
async def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path / "data"))
    store = InMemoryStorage()
    brain = Brain.create(name="coding")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    config = UnifiedConfig(current_brain=brain.id)
    yield (
        store,
        config,
        {
            "brain": brain.id,
            "project": str(tmp_path.resolve()),
            "session_id": "session-a",
            "agent_id": "main",
        },
    )
    await store.close()


def task_state():
    return {
        "goal": "Implement scoped memory recovery",
        "constraints": ["Keep changes local"],
        "approvals": [],
        "decisions": ["Use existing checkpoint envelopes"],
        "completed": ["Unit tests passed"],
        "next_actions": ["Run integration tests"],
        "blockers": [],
        "processes": [],
    }


async def test_checkpoint_replay_conflict_and_agent_isolation(memory):
    from surreal_memory.engine.agent_checkpoint import checkpoint

    store, config, scope = memory
    first = await checkpoint(
        store, config, {"action": "save", "scope": scope, "state": task_state(), "revision": 1}
    )
    assert first["status"] == "saved" and first["memory_ids"]
    replay = await checkpoint(
        store, config, {"action": "save", "scope": scope, "state": task_state(), "revision": 1}
    )
    assert replay["memory_ids"] == first["memory_ids"]
    changed = {**task_state(), "goal": "A different goal"}
    conflict = await checkpoint(
        store, config, {"action": "save", "scope": scope, "state": changed, "revision": 1}
    )
    assert conflict["status"] == "failed" and conflict["error"] == "checkpoint_conflict"
    other = await checkpoint(
        store, config, {"action": "load", "scope": {**scope, "agent_id": "other"}}
    )
    assert other["status"] == "skipped" and "state" not in other
    restored = await checkpoint(store, config, {"action": "load", "scope": scope})
    assert restored["state"] == task_state()


async def test_checkpoint_pending_is_replayed_after_outage(memory):
    from surreal_memory.engine.agent_checkpoint import checkpoint
    from surreal_memory.hooks.agent_journal import pending

    store, config, scope = memory
    with patch.object(store, "get_fiber", AsyncMock(side_effect=OSError("offline"))):
        result = await checkpoint(
            store, config, {"action": "save", "scope": scope, "state": task_state(), "revision": 1}
        )
    assert result["status"] == "pending" and result["saved"] == 0
    assert pending()
    result = await checkpoint(
        store, config, {"action": "save", "scope": scope, "state": task_state(), "revision": 1}
    )
    assert result["status"] == "saved"
    assert not pending()


async def test_scoped_knowledge_excludes_other_projects_and_legacy_hot(memory):
    from surreal_memory.mcp.server import MCPServer

    store, config, _ = memory
    config.write_gate = replace(config.write_gate, enabled=False)
    with patch("surreal_memory.mcp.server.get_config", return_value=config):
        server = MCPServer()
    server.get_storage = AsyncMock(return_value=store)
    for project, content in [
        ("one", "Copper mirrors require careful alignment"),
        ("two", "Copper mirrors require a secret unrelated adapter"),
    ]:
        result = await server._remember(
            {"content": content, "type": "decision", "scope": "project", "project_id": project}
        )
        assert "error" not in result
    await server._remember(
        {
            "content": "User explicitly prefers concise explanations",
            "type": "preference",
            "scope": "global_preference",
        }
    )
    for tool in [server._context, server._recall]:
        result = await tool(
            {
                "query": "copper mirrors",
                "scope": "project",
                "project_id": "one",
                "include_global_preferences": True,
            }
        )
        text = str(result)
        assert "careful alignment" in text
        assert "unrelated adapter" not in text
        assert "concise explanations" in text


async def test_invalid_global_preference_and_cross_scope_exact_lookup(memory):
    from surreal_memory.mcp.server import MCPServer

    store, config, _ = memory
    with patch("surreal_memory.mcp.server.get_config", return_value=config):
        server = MCPServer()
    server.get_storage = AsyncMock(return_value=store)
    rejected = await server._remember(
        {
            "content": "Always use the other database",
            "type": "instruction",
            "scope": "global_preference",
        }
    )
    assert "error" in rejected


async def test_volatile_backend_does_not_claim_checkpoint_durability(memory):
    from surreal_memory.engine.agent_checkpoint import checkpoint

    store, config, scope = memory
    config.storage_backend = "memory"
    result = await checkpoint(
        store, config, {"action": "save", "scope": scope, "state": task_state(), "revision": 1}
    )
    assert result["status"] == "pending" and result["saved"] == 0
    assert result["error"] == "durable_backend_required"

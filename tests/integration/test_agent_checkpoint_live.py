"""Exact-scope task persistence and real hook subprocess recovery on SurrealDB."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.engine.agent_checkpoint import checkpoint
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.unified_config import UnifiedConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("SURREALDB_URL"), reason="requires disposable live SurrealDB >=3.2"
    ),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("client", ["codex", "claude"])
async def test_checkpoint_survives_connections_and_repeated_client_compaction(
    tmp_path, monkeypatch, client
):
    database = "agent_" + uuid.uuid4().hex[:12]
    directory = tmp_path / "memory"
    directory.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(directory))
    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
    monkeypatch.setenv("SURREALDB_NS", "smem_agent_it")
    monkeypatch.setenv("SURREALDB_DB", database)
    monkeypatch.setenv("SURREALDB_USER", os.environ.get("SURREALDB_USER", "root"))
    monkeypatch.setenv("SURREALDB_PASS", os.environ.get("SURREALDB_PASS", "root"))
    monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", "1")
    store = SurrealDBStorage(
        url=os.environ["SURREALDB_URL"],
        user=os.environ["SURREALDB_USER"],
        password=os.environ["SURREALDB_PASS"],
        namespace="smem_agent_it",
        database=database,
    )
    await store.initialize()
    brain = Brain.create(name="coding-agent-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    config = UnifiedConfig(data_dir=directory, current_brain=brain.id)
    config.save()
    scope = {
        "brain": brain.id,
        "project": str(project.resolve()),
        "session_id": "live-session",
        "agent_id": "main",
    }
    state = {
        "goal": "Preserve task context across compaction",
        "constraints": ["Keep local changes"],
        "approvals": [],
        "decisions": ["Use verified task checkpoints"],
        "completed": ["Parser tests passed"],
        "next_actions": ["Verify the copper mirror integration"],
        "blockers": [],
        "processes": [],
    }
    args = {"action": "save", "scope": scope, "revision": 1, "state": state}
    try:
        with patch.object(store, "get_fiber", AsyncMock(side_effect=OSError("synthetic outage"))):
            assert (await checkpoint(store, config, args))["status"] == "pending"
        saved = await checkpoint(store, config, args)
        assert saved["status"] == "saved", saved
        await store.close()

        def invoke(event, **extra):
            value = {
                "session_id": scope["session_id"],
                "cwd": str(project),
                "hook_event_name": event,
                **extra,
            }
            result = subprocess.run(  # noqa: S603 — current interpreter and fixed module
                [sys.executable, "-m", "surreal_memory.hooks.agent_lifecycle", "--client", client],
                input=json.dumps(value),
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            return json.loads(result.stdout)

        first = invoke("SessionStart")
        assert "copper mirror integration" in first["hookSpecificOutput"]["additionalContext"], (
            first
        )
        assert invoke("UserPromptSubmit") == {}
        for _ in range(2):
            if client == "codex":
                assert invoke("PostCompact") == {}
                restored = invoke("UserPromptSubmit")
            else:
                restored = invoke("SessionStart", source="compact")
            assert (
                "copper mirror integration" in restored["hookSpecificOutput"]["additionalContext"]
            )
        assert invoke("SessionStart", agent_id="another-agent") != first
    finally:
        await store.close()

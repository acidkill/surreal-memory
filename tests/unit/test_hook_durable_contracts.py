"""Focused tests for durable hook keys, receipts, and checkpoint replay."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.hooks import capture_state, task_context
from surreal_memory.hooks.receipts import capture_receipt, receipt


def _envelope() -> dict[str, object]:
    scope = {
        "brain": "test-brain",
        "project": str(Path.cwd().resolve()),
        "session_id": "sess-1",
        "agent_id": "agent-1",
    }
    state = {
        "goal": "synthetic task",
        "constraints": [],
        "approvals": [],
        "decisions": [],
        "completed": [],
        "next_actions": [],
        "blockers": [],
        "processes": [],
    }

    def canonical(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    return {
        "schema_version": 1,
        "scope": scope,
        "scope_id": hashlib.sha256(canonical(scope).encode()).hexdigest(),
        "revision": 2,
        "content_hash": hashlib.sha256(canonical(state).encode()).hexdigest(),
        "state": state,
    }


def test_session_key_uses_codex_and_scope_then_preserves_claude_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_SESSION_ID", "codex-session")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-session")
    monkeypatch.setenv("SMEM_CAPTURE_SCOPE", "scope-digest")
    assert capture_state.session_key() == "scope-digest:codex-session"
    monkeypatch.delenv("CODEX_SESSION_ID")
    assert capture_state.session_key() == "scope-digest:claude-session"


def test_mark_seen_parallel_sessions_preserves_all_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    writes = [(f"session-{i}", [f"hash-{i}"]) for i in range(12)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: capture_state.mark_seen(*item), writes))
    state = json.loads((tmp_path / "capture_state.json").read_text())
    assert all(state[session]["hashes"] == hashes for session, hashes in writes)
    assert (tmp_path / "capture_state.json").stat().st_mode & 0o777 == 0o600


def test_receipt_contract_and_partial_failure(monkeypatch) -> None:
    bound = receipt("saved", 1, "s", ["fiber-1"], scope_id="scope", revision=3, content_hash="hash")
    assert bound["scope_id"] == "scope" and bound["revision"] == 3
    assert bound["content_hash"] == "hash"
    assert receipt("saved", 1, "s", ["fiber-1"]) == {
        "schema_version": 1,
        "status": "saved",
        "saved": 1,
        "session_id": "s",
        "memory_ids": ["fiber-1"],
        "error": None,
    }
    monkeypatch.setenv("SMEM_EVENT_ID", "event-1")
    value = capture_receipt({"saved": 1, "memory_ids": ["f1"], "failed_writes": 1}, "s")
    assert value["status"] == "failed"
    assert value["saved"] == 1
    assert value["memory_ids"] == ["f1"]
    assert value["event_id"] == "event-1"


def test_checkpoint_hash_validation() -> None:
    parsed = task_context.validate_checkpoint(_envelope())
    assert parsed["revision"] == 2
    assert '"goal":"synthetic task"' in parsed["content"]
    bad = _envelope()
    bad["scope_id"] = "0" * 64
    with pytest.raises(ValueError, match="scope_id"):
        task_context.validate_checkpoint(bad)
    invalid_state = _envelope()
    invalid_state["state"] = {"goal": "missing required arrays"}
    with pytest.raises(ValueError, match="state fields"):
        task_context.validate_checkpoint(invalid_state)


@pytest.mark.asyncio
async def test_checkpoint_reuses_encoded_fiber_after_partial_write(monkeypatch) -> None:
    checkpoint = task_context.validate_checkpoint(_envelope())
    scope_id = checkpoint["scope_id"]
    content_hash = checkpoint["content_hash"]
    tags = {
        "codex_checkpoint",
        f"codex_scope:{scope_id}",
        f"codex_revision:{checkpoint['revision']}",
        f"codex_hash:{content_hash}",
    }
    from surreal_memory.core.fiber import Fiber

    orphan = Fiber.create(
        neuron_ids={"neuron-1"},
        synapse_ids=set(),
        anchor_neuron_id="neuron-1",
        tags=tags,
        fiber_id="fiber-existing",
    )

    class Storage:
        def __init__(self) -> None:
            self.fibers = [orphan]
            self.typed = {}
            self.batch_calls = 0

        async def find_fibers(self, tags=None, limit=100):
            return [fiber for fiber in self.fibers if tags is None or tags.issubset(fiber.tags)]

        async def update_fiber(self, fiber):
            self.fibers = [fiber if item.id == fiber.id else item for item in self.fibers]

        async def get_typed_memory(self, fiber_id):
            return self.typed.get(fiber_id)

        async def add_typed_memory(self, item):
            self.typed[item.fiber_id] = item

        async def batch_save(self):
            self.batch_calls += 1
            if self.batch_calls == 1:
                raise RuntimeError("synthetic lost acknowledgement")

        async def close(self):
            return None

    storage = Storage()
    config = SimpleNamespace(
        current_brain="test-brain",
        storage_backend="memory",
        safety=SimpleNamespace(auto_redact_min_severity=3),
    )
    with (
        patch("surreal_memory.unified_config.get_config", return_value=config),
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)),
        patch("surreal_memory.engine.encoder.MemoryEncoder") as encoder,
        patch(
            "surreal_memory.safety.input_firewall.check_content",
            side_effect=lambda content: SimpleNamespace(blocked=False, sanitized=content),
        ),
        patch(
            "surreal_memory.safety.sensitive.auto_redact_content",
            return_value=(checkpoint["content"], [], None),
        ),
    ):
        first = await task_context.save_checkpoint(checkpoint)
        result = await task_context.save_checkpoint(checkpoint)

    assert first["status"] == "pending"
    assert result["status"] == "saved"
    assert result["saved"] == 1
    assert storage.batch_calls == 2
    assert result["memory_ids"] == ["fiber-existing"]
    encoder.assert_not_called()
    assert storage.typed["fiber-existing"].metadata["content_hash"] == content_hash
    assert storage.typed["fiber-existing"].project_id == checkpoint["scope"]["project"]


def test_checkpoint_scope_lock_serializes_concurrent_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    active = 0
    maximum = 0
    guard = threading.Lock()

    def work() -> None:
        nonlocal active, maximum
        with capture_state.local_scope_lock("a" * 64):
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: work(), range(4)))
    assert maximum == 1


@pytest.mark.asyncio
async def test_remote_sensitive_checkpoint_is_rejected_before_storage(monkeypatch) -> None:
    checkpoint = task_context.validate_checkpoint(_envelope())
    storage_factory = AsyncMock()
    config = SimpleNamespace(
        current_brain="test-brain",
        storage_backend="surrealdb",
        safety=SimpleNamespace(auto_redact_min_severity=3),
    )
    with (
        patch("surreal_memory.unified_config.get_config", return_value=config),
        patch("surreal_memory.unified_config.get_shared_storage", storage_factory),
        patch(
            "surreal_memory.safety.input_firewall.check_content",
            side_effect=lambda content: SimpleNamespace(blocked=False, sanitized=content),
        ),
        patch(
            "surreal_memory.safety.sensitive.auto_redact_content",
            return_value=("altered content", [object()], "synthetic-hash"),
        ),
    ):
        result = await task_context.save_checkpoint(checkpoint)
    assert result == {"saved": 0, "status": "failed", "error": "checkpoint_sensitive"}
    storage_factory.assert_not_awaited()


def test_checkpoint_cli_json_validation_failure_is_nonzero(monkeypatch, capsys) -> None:
    import io

    monkeypatch.setattr("sys.argv", ["smem-hook-task-context", "--checkpoint", "--json"])
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    with pytest.raises(SystemExit) as exc:
        task_context.main()
    assert exc.value.code == 1
    receipt_value = json.loads(capsys.readouterr().out)
    assert receipt_value["schema_version"] == 1
    assert receipt_value["status"] == "failed"

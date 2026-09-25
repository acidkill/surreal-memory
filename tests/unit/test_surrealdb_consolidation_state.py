"""Tests for durable consolidation progress and cross-process leases."""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb.consolidation_state import (
    SurrealDBConsolidationStateMixin,
)

LEASE_ID_A = "lease-a-test"
LEASE_ID_B = "lease-b-test"


class ScriptedStateStorage(SurrealDBConsolidationStateMixin):
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return "test-brain"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, params))
        if not self.responses:
            return []
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_create_progress_uses_brain_scoped_deterministic_record_id() -> None:
    storage = ScriptedStateStorage([{"id": "consolidation_progress:row"}])
    state = {
        "brain_id": "test-brain",
        "run_id": "run-001",
        "schema_version": 11,
        "format_version": 1,
        "engine_version": "3.11.0:checkpoint-v1",
        "requested_strategies": ["prune"],
        "completed_strategies": [],
        "options_fingerprint": "fingerprint",
        "reference_time": "2026-09-23T00:00:00Z",
        "status": "queued",
        "current_strategy": "prune",
        "phase": "synapses",
        "cursor": None,
        "strategy_states": {"prune": {"status": "queued"}},
        "counters": {},
        "owner_token": "lease-token",
        "started_at": "2026-09-23T00:00:00Z",
        "updated_at": "2026-09-23T00:00:00Z",
        "last_error": None,
    }

    result = await storage.create_consolidation_progress(state)

    sql, params = storage.calls[0]
    assert "CREATE type::record('consolidation_progress', $record_id)" in sql
    assert params["record_id"] == storage._consolidation_record_id("test-brain")
    assert params["state"]["run_id"] == "run-001"
    assert result["id"] == "consolidation_progress:row"


@pytest.mark.asyncio
async def test_lease_create_succeeds_and_rejects_invalid_ttl() -> None:
    storage = ScriptedStateStorage([{"id": "lease"}])

    assert await storage.acquire_consolidation_lease("test-brain", LEASE_ID_A)
    assert "CREATE type::record('consolidation_lease'" in storage.calls[0][0]
    assert storage.calls[0][1]["lease"]["owner_token"] == LEASE_ID_A

    with pytest.raises(ValueError, match="between 30 and 3600"):
        await storage.acquire_consolidation_lease("test-brain", LEASE_ID_B, lease_seconds=10)
    assert len(storage.calls) == 1


@pytest.mark.asyncio
async def test_live_lease_is_not_stolen() -> None:
    storage = ScriptedStateStorage(
        RuntimeError("record already exists"),
        [],
    )

    assert not await storage.acquire_consolidation_lease("test-brain", LEASE_ID_B)
    assert len(storage.calls) == 2
    assert "expires_at <= $now" in storage.calls[1][0]


@pytest.mark.asyncio
async def test_expired_lease_can_be_stolen() -> None:
    storage = ScriptedStateStorage(
        RuntimeError("record already exists"),
        [{"id": "lease", "owner_token": LEASE_ID_B}],
    )

    assert await storage.acquire_consolidation_lease("test-brain", LEASE_ID_B)
    assert storage.calls[1][1]["lease"]["owner_token"] == LEASE_ID_B


@pytest.mark.asyncio
async def test_unexpected_lease_create_error_is_not_hidden() -> None:
    storage = ScriptedStateStorage(RuntimeError("connection lost"))

    with pytest.raises(RuntimeError, match="connection lost"):
        await storage.acquire_consolidation_lease("test-brain", LEASE_ID_A)
    assert len(storage.calls) == 1


@pytest.mark.asyncio
async def test_renew_and_release_are_fenced_by_owner_token() -> None:
    storage = ScriptedStateStorage(
        [{"id": "lease"}],
        [{"id": "lease"}],
    )

    assert await storage.renew_consolidation_lease("test-brain", LEASE_ID_A)
    assert await storage.release_consolidation_lease("test-brain", LEASE_ID_A)
    assert all("owner_token = $owner_token" in sql for sql, _ in storage.calls)
    assert all(params["owner_token"] == LEASE_ID_A for _, params in storage.calls)


@pytest.mark.asyncio
async def test_progress_claim_and_save_require_current_owner() -> None:
    storage = ScriptedStateStorage(
        [{"id": "progress", "owner_token": LEASE_ID_B}],
        [{"id": "progress", "owner_token": LEASE_ID_B, "phase": "neurons"}],
    )

    claimed = await storage.claim_consolidation_progress("test-brain", LEASE_ID_B)
    assert claimed is not None
    assert "status IN ['queued', 'running', 'paused', 'failed']" in storage.calls[0][0]

    saved = await storage.save_consolidation_progress(
        "test-brain",
        LEASE_ID_B,
        {"run_id": "run-001", "status": "paused", "phase": "neurons"},
    )
    assert saved is not None
    sql, params = storage.calls[1]
    assert "owner_token = $owner_token" in sql
    assert params["state"]["owner_token"] == LEASE_ID_B
    assert params["state"]["phase"] == "neurons"


@pytest.mark.asyncio
async def test_checkpoint_update_returns_none_after_lease_fence_changes() -> None:
    storage = ScriptedStateStorage([])

    result = await storage.save_consolidation_progress(
        "test-brain",
        "stale-owner",
        {"run_id": "run-001", "status": "running"},
    )

    assert result is None
    assert "owner_token = $owner_token" in storage.calls[0][0]


@pytest.mark.asyncio
async def test_semantic_recovery_cas_transitions_failed_to_paused_atomically() -> None:
    storage = ScriptedStateStorage([{"id": "progress", "status": "paused"}])
    old_owner = "failed-owner-test"
    recovery_owner = "recovery-owner-test"

    saved = await storage.compare_and_swap_semantic_link_recovery(
        brain_id="test-brain",
        expected_run_id="run-001",
        expected_owner_token=old_owner,
        expected_options_fingerprint="fingerprint",
        expected_status="failed",
        expected_phase="semantic_link_discovery_neurons",
        expected_updated_at="before-update",
        expected_reference_time="original-reference",
        lease_owner_token=recovery_owner,
        new_owner_token=recovery_owner,
        strategy_states={"semantic_link": {"phase": "semantic_link_restart_pending"}},
        counters={"semantic_link": {}},
        updated_at="after-update",
        new_status="paused",
    )

    assert saved is not None
    sql, params = storage.calls[0]
    assert "SET status = $new_status" in sql
    assert "status = $expected_status" in sql
    assert "expires_at > time::now()" in sql
    assert params["expected_status"] == "failed"
    assert params["new_status"] == "paused"
    assert params["expected_reference_time"] == "original-reference"


@pytest.mark.asyncio
async def test_semantic_recovery_cas_rejects_unapproved_status_transition() -> None:
    storage = ScriptedStateStorage([])
    old_owner = "owner-test"
    recovery_owner = "recovery-owner-test"

    with pytest.raises(ValueError, match="only transition failed status to paused"):
        await storage.compare_and_swap_semantic_link_recovery(
            brain_id="test-brain",
            expected_run_id="run-001",
            expected_owner_token=old_owner,
            expected_options_fingerprint="fingerprint",
            expected_status="running",
            expected_phase="semantic_link_discovery_neurons",
            expected_updated_at="before-update",
            expected_reference_time="original-reference",
            lease_owner_token=recovery_owner,
            new_owner_token=recovery_owner,
            strategy_states={"semantic_link": {"phase": "semantic_link_restart_pending"}},
            counters={"semantic_link": {}},
            updated_at="after-update",
            new_status="paused",
        )

    assert storage.calls == []

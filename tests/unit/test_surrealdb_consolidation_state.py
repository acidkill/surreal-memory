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
@pytest.mark.parametrize(
    "resolved_error", [None, "semantic_link (progress coordination: ConsolidationProgressError)"]
)
async def test_semantic_recovery_cas_transitions_failed_to_paused_atomically(
    resolved_error: str | None,
) -> None:
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
        resolved_source_error=resolved_error,
    )

    assert saved is not None
    sql, params = storage.calls[0]
    assert "SET status = $new_status" in sql
    assert "status = $expected_status" in sql
    assert "expires_at > time::now()" in sql
    assert params["expected_status"] == "failed"
    assert params["new_status"] == "paused"
    assert params["expected_reference_time"] == "original-reference"
    assert ("last_error = NONE" in sql) is (resolved_error is not None)
    assert ("last_error = $resolved_source_error" in sql) is (resolved_error is not None)
    assert params["resolved_source_error"] == resolved_error


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


def _native_transport_checkpoint(brain_id: str) -> dict[str, Any]:
    from datetime import UTC, datetime

    from surreal_memory.engine import consolidation_receipts as receipts

    state = {
        "brain_id": brain_id,
        "run_id": "synthetic-transport-run",
        "options_fingerprint": "a" * 64,
        "reference_time": datetime(2030, 1, 1, tzinfo=UTC),
        "schema_version": 12,
        "engine_version": "3.11.0:checkpoint-v1",
        "format_version": 1,
        "status": "paused",
        "current_strategy": "semantic_link",
        "phase": "semantic_link_apply",
        "requested_strategies": ["semantic_link"],
        "completed_strategies": [],
        "strategy_states": {},
        "counters": {},
        "work_progress_seq": 1,
        "owner_token": "synthetic-transport-owner",
    }
    state["native_handoff"] = receipts.prepare_native_handoff(None, state, fresh=False)
    return state


def _record_id_type() -> Any:
    sdk = pytest.importorskip(
        "surrealdb.data.types.record_id",
        reason="optional SurrealDB SDK required for typed transport",
    )
    return sdk.RecordID


@pytest.mark.asyncio
@pytest.mark.parametrize("brain_id", ["default", None])
async def test_progress_read_projects_sdk_id_for_native_and_hourly_digest(
    brain_id: str | None,
) -> None:
    import json

    from surreal_memory.engine import consolidation_receipts as receipts

    resolved = brain_id or "test-brain"
    raw = _native_transport_checkpoint(resolved)
    record_id = _record_id_type()(
        "consolidation_progress", ScriptedStateStorage._consolidation_record_id(resolved)
    )
    raw["id"] = record_id
    storage = ScriptedStateStorage([raw])
    assert receipts.native_handoff_valid(raw)
    with pytest.raises(TypeError, match="RecordID"):
        receipts._native_digest(raw)

    result = await storage.get_consolidation_progress(brain_id)

    assert result is not None and result is not raw
    digest = receipts._native_digest(result)
    assert result["id"] == str(record_id) and isinstance(result["id"], str)
    assert raw["id"] is record_id
    assert {key: value for key, value in result.items() if key != "id"} == {
        key: value for key, value in raw.items() if key != "id"
    }
    assert receipts.native_handoff_valid(result)
    assert receipts.native_payload_hash(result) == receipts.native_payload_hash(raw)
    assert receipts.checkpoint_hash(result) == receipts.checkpoint_hash(raw)
    json_result = json.loads(json.dumps(receipts._native_normalize(result)))
    assert receipts._native_digest(json_result) == digest
    assert receipts._native_digest({**result, "work_progress_seq": 2}) != digest
    assert len(storage.calls) == 1
    assert storage.calls[0][0].startswith("SELECT * FROM consolidation_progress")
    assert storage.calls[0][1] == {"brain_id": resolved}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["table", "key", "nonstring_key"])
async def test_progress_read_rejects_wrong_actual_sdk_record_identity(case: str) -> None:
    record_id = _record_id_type()
    table = "consolidation_lease" if case == "table" else "consolidation_progress"
    key: Any = ScriptedStateStorage._consolidation_record_id("default")
    if case == "key":
        key = "synthetic-wrong-key"
    elif case == "nonstring_key":
        key = 123
    storage = ScriptedStateStorage([{"id": record_id(table, key), "brain_id": "default"}])

    with pytest.raises(ValueError, match="checkpoint transport record identity"):
        await storage.get_consolidation_progress("default")

    assert len(storage.calls) == 1 and storage.calls[0][0].startswith("SELECT")


@pytest.mark.asyncio
@pytest.mark.parametrize("with_id", [True, False])
async def test_progress_read_preserves_existing_json_or_absent_id_without_sdk(
    with_id: bool,
) -> None:
    row: dict[str, Any] = {"brain_id": "test-brain", "run_id": "synthetic-json-run"}
    if with_id:
        row["id"] = "consolidation_progress:json-fixture"
    storage = ScriptedStateStorage([row])

    assert await storage.get_consolidation_progress() == row
    assert len(storage.calls) == 1


@pytest.mark.asyncio
async def test_progress_read_preserves_absent_checkpoint() -> None:
    storage = ScriptedStateStorage([])
    assert await storage.get_consolidation_progress("default") is None
    assert len(storage.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", ["unknown", "sdk_record"])
async def test_progress_projection_does_not_coerce_nested_payload_values(nested: str) -> None:
    from surreal_memory.engine import consolidation_receipts as receipts

    record_id = _record_id_type()
    row = _native_transport_checkpoint("default")
    row["id"] = record_id(
        "consolidation_progress", ScriptedStateStorage._consolidation_record_id("default")
    )
    value = object() if nested == "unknown" else record_id("neuron", "synthetic-nested")
    row["strategy_states"] = {"semantic_link": {"unexpected": value}}
    result = await ScriptedStateStorage([row]).get_consolidation_progress("default")
    assert result is not None
    assert result["strategy_states"]["semantic_link"]["unexpected"] is value
    assert not receipts.native_handoff_valid(result)
    with pytest.raises(TypeError):
        receipts._native_digest(result)

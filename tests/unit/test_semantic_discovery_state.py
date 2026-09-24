from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb.semantic_discovery_state import (
    SemanticDiscoveryStateConflictError,
    SurrealDBSemanticDiscoveryStateMixin,
)


class _SnapshotStorage(SurrealDBSemanticDiscoveryStateMixin):
    def __init__(self) -> None:
        self.current_brain_id = "brain-a"
        self.rows: dict[str, dict[str, Any]] = {}
        self.progress: dict[str, Any] = {}
        self.lease: dict[str, Any] = {}
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if sql.startswith("CREATE type::record('semantic_discovery_state'"):
            record_id = str(params["record_id"])
            if record_id in self.rows:
                raise RuntimeError("record already exists")
            record = dict(params["record"])
            record["id"] = f"semantic_discovery_state:{record_id}"
            self.rows[record_id] = record
            return [record]
        if sql.startswith("SELECT * FROM semantic_discovery_state WHERE id"):
            row = self.rows.get(str(params["record_id"]))
            return [row] if row else []
        if sql.startswith("SELECT * FROM semantic_discovery_state WITH INDEX"):
            return [
                row
                for row in self.rows.values()
                if row["brain_id"] == params["brain_id"]
                and row["state_id"] == params["state_id"]
                and row["revision"] == params["revision"]
            ][:1]
        if sql.startswith("SELECT * FROM consolidation_progress"):
            return [self.progress] if self.progress else []
        if sql.startswith("SELECT * FROM consolidation_lease"):
            return [self.lease] if self.lease else []
        return []


def _payload(**updates: Any) -> dict[str, Any]:
    return {
        "brain_id": "brain-a",
        "run_id": "run-1",
        "owner_token": "owner-old",
        "source_token": "source-token-1",
        "candidates": [["n1", 0.9, 3]],
        **updates,
    }


@pytest.mark.asyncio
async def test_snapshot_save_is_idempotent_and_rejects_conflicting_revision() -> None:
    storage = _SnapshotStorage()
    payload = _payload()

    await storage.save_semantic_discovery_state("state-owner-old", 1, payload)
    await storage.save_semantic_discovery_state("state-owner-old", 1, payload)

    assert len(storage.rows) == 1
    create_queries = [sql for sql, _ in storage.queries if sql.startswith("CREATE")]
    assert len(create_queries) == 2
    with pytest.raises(SemanticDiscoveryStateConflictError, match="conflicting content"):
        await storage.save_semantic_discovery_state(
            "state-owner-old", 1, _payload(candidates=[["n2", 0.8, 4]])
        )


@pytest.mark.asyncio
async def test_snapshot_payload_is_bounded_and_brain_bound() -> None:
    storage = _SnapshotStorage()

    with pytest.raises(ValueError, match="exceeds 1000000 bytes"):
        await storage.save_semantic_discovery_state("state-1", 1, _payload(large="x" * 1_000_001))
    with pytest.raises(ValueError, match="current brain"):
        await storage.save_semantic_discovery_state("state-1", 1, _payload(brain_id="brain-b"))
    with pytest.raises(ValueError, match="must not contain embedding vectors"):
        await storage.save_semantic_discovery_state(
            "state-1", 1, _payload(metadata={"embedding": [0.1, 0.2]})
        )
    assert not storage.rows


@pytest.mark.asyncio
async def test_snapshot_load_requires_active_lease_and_exact_manifest_reference() -> None:
    storage = _SnapshotStorage()
    payload = _payload()
    await storage.save_semantic_discovery_state("state-owner-old", 1, payload)
    storage.lease = {"owner_token": "owner-current"}
    storage.progress = {
        "brain_id": "brain-a",
        "run_id": "run-1",
        "owner_token": "owner-current",
        "strategy_states": {
            "semantic_discovery": {
                "state_id": "state-owner-old",
                "state_revision": 1,
                "source_token": "source-token-1",
            }
        },
    }

    loaded = await storage.load_semantic_discovery_state("state-owner-old", 1)
    assert loaded == payload

    storage.progress["strategy_states"]["semantic_discovery"]["state_revision"] = 2
    assert await storage.load_semantic_discovery_state("state-owner-old", 1) is None
    storage.progress["strategy_states"]["semantic_discovery"]["state_revision"] = 1
    storage.lease["owner_token"] = "rotated"  # noqa: S105 - synthetic lease owner
    assert await storage.load_semantic_discovery_state("state-owner-old", 1) is None

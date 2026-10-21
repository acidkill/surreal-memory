from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.semantic_discovery_state import (
    SemanticDiscoveryStateConflictError,
    SurrealDBSemanticDiscoveryStateMixin,
)
from surreal_memory.utils.timeutils import utcnow


class _SnapshotStorage(SurrealDBSemanticDiscoveryStateMixin):
    def __init__(self) -> None:
        self.current_brain_id = "brain-a"
        self.rows: dict[str, dict[str, Any]] = {}
        self.progress: dict[str, Any] = {}
        self.lease: dict[str, Any] = {}
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.deleted_batches: list[int] = []

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
        if sql.startswith("DELETE (SELECT VALUE id FROM semantic_discovery_state"):
            doomed = [
                key
                for key, row in self.rows.items()
                if row["created_at"] < params["cutoff"]
                and row["brain_id"] == params["brain_id"]
                and row["run_id"] not in params["keep"]
            ][: params["limit"]]
            self.deleted_batches.append(len(doomed))
            return [self.rows.pop(key)["id"] for key in doomed]
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


@pytest.mark.asyncio
async def test_snapshot_load_accepts_nested_json_string_manifest_only_for_exact_reference() -> None:
    storage = _SnapshotStorage()
    payload = _payload()
    await storage.save_semantic_discovery_state("state-owner-old", 1, payload)
    storage.lease = {"owner_token": "owner-current"}
    source_ref = {
        "state_id": "state-owner-old",
        "revision": 1,
        "state_revision": 1,
        "source_token": "source-token-1",
    }
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": source_ref,
    }
    # This is the persisted shape: pending contains the JSON encoding of a
    # list whose item is itself a JSON-encoded manifest.
    pending_manifest = json.dumps(
        [json.dumps(manifest, sort_keys=True, separators=(",", ":"))],
        separators=(",", ":"),
    )
    storage.progress = {
        "brain_id": "brain-a",
        "run_id": "run-1",
        "owner_token": "owner-current",
        "strategy_states": {"semantic_link": {"pending": [pending_manifest]}},
    }

    assert await storage.load_semantic_discovery_state("state-owner-old", 1) == payload

    for field, forged_value in (
        ("state_id", "other-state"),
        ("state_revision", 2),
        ("source_token", "other-source"),
    ):
        forged_ref = {**source_ref, field: forged_value}
        storage.progress["strategy_states"]["semantic_link"]["pending"] = [
            json.dumps(
                [json.dumps({**manifest, "source_state_ref": forged_ref})],
                separators=(",", ":"),
            )
        ]
        assert await storage.load_semantic_discovery_state("state-owner-old", 1) is None

    storage.progress["strategy_states"]["semantic_link"]["pending"] = [
        "not-json containing state-owner-old and source-token-1"
    ]
    assert await storage.load_semantic_discovery_state("state-owner-old", 1) is None


def _seed_snapshot(
    storage: _SnapshotStorage,
    key: str,
    *,
    run_id: str,
    age: timedelta,
    brain_id: str = "brain-a",
) -> None:
    storage.rows[key] = {
        "id": f"semantic_discovery_state:{key}",
        "brain_id": brain_id,
        "run_id": run_id,
        "created_at": utcnow() - age,
    }


@pytest.mark.asyncio
async def test_prune_deletes_expired_snapshots_of_finished_runs_only() -> None:
    storage = _SnapshotStorage()
    _seed_snapshot(storage, "finished-old-1", run_id="run-finished", age=timedelta(days=30))
    _seed_snapshot(storage, "finished-old-2", run_id="run-finished", age=timedelta(days=8))
    _seed_snapshot(storage, "finished-young", run_id="run-finished", age=timedelta(days=6))
    _seed_snapshot(
        storage, "other-brain", run_id="run-finished", age=timedelta(days=30), brain_id="brain-b"
    )

    deleted = await storage.prune_semantic_discovery_state(
        older_than=timedelta(days=7), keep_run_ids=[]
    )

    assert deleted == 2
    assert set(storage.rows) == {"finished-young", "other-brain"}


@pytest.mark.asyncio
async def test_prune_keeps_snapshots_of_resumable_runs_however_old() -> None:
    storage = _SnapshotStorage()
    _seed_snapshot(storage, "paused-old", run_id="run-paused", age=timedelta(days=90))
    _seed_snapshot(storage, "finished-old", run_id="run-finished", age=timedelta(days=90))

    deleted = await storage.prune_semantic_discovery_state(
        older_than=timedelta(days=7), keep_run_ids={"run-paused"}
    )

    assert deleted == 1
    assert set(storage.rows) == {"paused-old"}
    sql, params = storage.queries[-1]
    assert "WITH INDEX idx_sds_created_at" in sql
    assert params["keep"] == ["run-paused"]


@pytest.mark.asyncio
async def test_prune_deletes_in_batches_of_at_most_500_rows() -> None:
    storage = _SnapshotStorage()
    for index in range(1203):
        _seed_snapshot(storage, f"old-{index:04d}", run_id="run-finished", age=timedelta(days=30))

    deleted = await storage.prune_semantic_discovery_state(
        older_than=timedelta(days=7), keep_run_ids=()
    )

    assert deleted == 1203
    assert not storage.rows
    # Two full batches and one short one: the short batch ends the loop.
    assert storage.deleted_batches == [500, 500, 203]
    batches = [params for sql, params in storage.queries if sql.startswith("DELETE")]
    assert [params["limit"] for params in batches] == [500, 500, 500]


@pytest.mark.asyncio
async def test_prune_without_matches_returns_zero() -> None:
    storage = _SnapshotStorage()

    assert (
        await storage.prune_semantic_discovery_state(older_than=timedelta(days=7), keep_run_ids=[])
        == 0
    )
    _seed_snapshot(storage, "young", run_id="run-finished", age=timedelta(hours=1))
    assert (
        await storage.prune_semantic_discovery_state(older_than=timedelta(days=7), keep_run_ids=[])
        == 0
    )
    assert set(storage.rows) == {"young"}
    assert len([sql for sql, _ in storage.queries if sql.startswith("DELETE")]) == 2


@pytest.mark.asyncio
async def test_prune_cutoff_is_bound_relative_to_now() -> None:
    storage = _SnapshotStorage()

    before = utcnow()
    await storage.prune_semantic_discovery_state(older_than=timedelta(days=7), keep_run_ids=[])
    after = utcnow()

    _, params = storage.queries[-1]
    assert before - timedelta(days=7) <= params["cutoff"] <= after - timedelta(days=7)
    assert params["brain_id"] == "brain-a"


@pytest.mark.asyncio
@pytest.mark.parametrize("older_than", [timedelta(0), timedelta(days=-1)])
async def test_prune_rejects_non_positive_age(older_than: timedelta) -> None:
    storage = _SnapshotStorage()

    with pytest.raises(ValueError, match="positive timedelta"):
        await storage.prune_semantic_discovery_state(older_than=older_than, keep_run_ids=[])
    assert not storage.queries


@pytest.mark.asyncio
async def test_prune_rejects_a_bare_run_id_string() -> None:
    storage = _SnapshotStorage()

    # A str is a Collection[str]: iterating it would keep single characters.
    with pytest.raises(TypeError, match="not a string"):
        await storage.prune_semantic_discovery_state(
            older_than=timedelta(days=7), keep_run_ids="run-paused"
        )
    assert not storage.queries

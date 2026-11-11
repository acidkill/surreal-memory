"""Real deletion on a guard-registered, disposable SurrealDB instance only."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.storage.base import BrainDeletionConflictError
from surreal_memory.storage.surrealdb.store import _BRAIN_DELETE_TABLES, SurrealDBStorage
from tests._disposable_surrealdb import load_target
from tests.unit._surrealdb_live import ensure_real_surrealdb_sdk


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SurrealDBStorage]:
    target = load_target()
    if target is None:
        pytest.skip("requires a registered disposable SurrealDB container")
    ensure_real_surrealdb_sdk()
    target = target.child()
    storage = SurrealDBStorage(
        **target.storage_kwargs(), database="delete_" + uuid4().hex, embedding_dim=8
    )
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage._query(f"REMOVE NAMESPACE {target.namespace}")
        await storage.close()


async def seed(store: SurrealDBStorage, bid: str, *, name: str | None = None) -> None:
    await store.save_brain(Brain.create(name or bid, brain_id=bid))
    await store._query(
        "FOR $i IN $ids { CREATE type::record('neuron', $bid + '-' + <string>$i) "
        "SET brain_id = $bid, type = 'concept', content = 'synthetic'; };",
        ids=list(range(501)),
        bid=bid,
    )
    await store._query(
        "FOR $i IN $ids { CREATE type::record('semantic_discovery_state', $bid + '-' + <string>$i) "
        "CONTENT {brain_id: $bid, state_id: $bid + '-' + <string>$i, revision: 1, "
        "run_id: $bid, owner_token: 'synthetic', source_token: 'synthetic', "
        "payload: {synthetic: $payload}}; };",
        ids=list(range(9)),
        bid=bid,
        payload="x" * (1024 * 1024),
    )
    for table in (
        "consolidation_fiber_census",
        "consolidation_dedup_census",
        "consolidation_group_plan",
    ):
        await store._query(
            "CREATE type::record($table, $bid) CONTENT $row",
            table=table,
            bid=bid,
            row={
                "brain_id": bid,
                "run_id": bid,
                "strategy": "synthetic",
                "filter_fingerprint": "synthetic",
                "page_index": 0,
                "plan_id": bid,
                "kind": "synthetic",
                "item_key": "synthetic",
            },
        )


async def count(store: SurrealDBStorage, table: str, bid: str) -> int:
    rows = await store._query(
        "SELECT count() AS n FROM type::table($table) WHERE brain_id = $bid GROUP ALL",
        table=table,
        bid=bid,
    )
    return rows[0]["n"] if rows else 0


async def test_exact_delete_protection_isolation_and_large_batches(store: SurrealDBStorage) -> None:
    await seed(store, "target_full_id")
    await seed(store, "other")
    await store.save_brain(Brain.create("default", brain_id="default"))
    await store.save_brain(Brain.create("default", brain_id="protected-uuid"))
    for bid in ("default", "protected-uuid"):
        with pytest.raises(BrainDeletionConflictError):
            await store.delete_brain(bid)
    assert not await store.delete_brain("target_full")
    assert not await store.delete_brain("full_id")
    assert await store.acquire_consolidation_lease("target_full_id", "worker")
    with pytest.raises(BrainDeletionConflictError):
        await store.delete_brain("target_full_id")
    assert await count(store, "neuron", "target_full_id") == 501
    assert await store.release_consolidation_lease("target_full_id", "worker")
    # A literal scope must use the existing brain-first index on both supported DB versions.
    plan = await store._query_response(
        'SELECT VALUE id FROM semantic_discovery_state WHERE brain_id = "target_full_id" LIMIT 8 EXPLAIN'
    )
    print("snapshot-selector-plan", json.dumps(plan, default=str))
    assert "idx_sds_brain_run" in json.dumps(plan, default=str)
    brain = await store.get_brain("target_full_id")
    assert brain is not None and brain.id == "target_full_id"
    assert await store.delete_brain("target_full_id")
    assert not await store.delete_brain("target_full_id")
    for table in _BRAIN_DELETE_TABLES:
        assert await count(store, table, "target_full_id") == 0, table
    assert await count(store, "neuron", "other") == 501
    assert await count(store, "semantic_discovery_state", "other") == 9
    for table in (
        "consolidation_fiber_census",
        "consolidation_dedup_census",
        "consolidation_group_plan",
    ):
        assert await count(store, table, "other") == 1
        plan = await store._query_response(
            f'SELECT id FROM {table} WHERE brain_id = "other" LIMIT 8 EXPLAIN'  # noqa: S608 -- fixed table tuple
        )
        print(table + "-selector-plan", json.dumps(plan, default=str))
        assert "IndexScan" in json.dumps(plan, default=str)
    assert await store.get_brain("other") is not None
    assert await store.get_brain("default") is not None
    # clear remains a reset, retaining registry even when all scoped rows are removed.
    await store.clear("other")
    assert await store.get_brain("other") is not None


@pytest.mark.parametrize("failure", ["partial", "lease", "lost-ack"])
async def test_real_partial_failure_lease_race_retry_and_lost_ack(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from surrealdb.errors import ThrownError

    await seed(store, "target_full_id")
    query = store._query
    injected = False

    async def interrupt(sql: str, **params: Any) -> list[dict[str, Any]]:
        nonlocal injected
        if not injected and failure == "lease" and "LET $ids" in sql:
            injected = True
            assert await store.release_consolidation_lease("target_full_id", params["owner"])
            assert await store.acquire_consolidation_lease("target_full_id", "replacement-worker")
        result = await query(sql, **params)
        if not injected and failure == "partial" and "SELECT VALUE id FROM neuron " in sql:
            injected = True
            raise TimeoutError("synthetic lost batch response")
        if (
            not injected
            and failure == "lost-ack"
            and "DELETE type::record('brain', $brain_id)" in sql
        ):
            injected = True
            raise ConnectionError("synthetic lost registry response")
        return result

    monkeypatch.setattr(store, "_query", interrupt)
    if failure == "lost-ack":
        assert await store.delete_brain("target_full_id")
    else:
        with pytest.raises(
            (TimeoutError, ThrownError),
            match=r"synthetic lost batch response|brain deletion lease lost",
        ):
            await store.delete_brain("target_full_id")
        assert await store.get_brain("target_full_id") is not None
        remaining = await count(store, "neuron", "target_full_id")
        assert remaining == (1 if failure == "partial" else 501)
        monkeypatch.setattr(store, "_query", query)
        if failure == "lease":
            assert await store.release_consolidation_lease("target_full_id", "replacement-worker")
        assert await store.delete_brain("target_full_id")
    assert injected
    assert not await store.delete_brain("target_full_id")


async def test_real_exact_legacy_id_and_ambiguous_name(store: SurrealDBStorage) -> None:
    await seed(store, "legacy-id", name="legacy-name")
    await store._query(
        "CREATE neuron:legacy_name_row SET brain_id = 'legacy-name', type = 'concept', content = 'synthetic'"
    )
    assert not await store.delete_brain("legacy-name")
    await store.save_brain(Brain.create("legacy-name", brain_id="other-id"))
    with pytest.raises(BrainDeletionConflictError):
        await store.delete_brain("legacy-id")
    await store._query("DELETE type::record('brain', $bid)", bid="other-id")
    assert await store.delete_brain("legacy-id")
    assert await count(store, "neuron", "legacy-name") == 0

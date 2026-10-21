"""Disposable local-DB integration tests for semantic-discovery snapshot pruning.

These exercise real SurrealQL on SurrealDB 3.2+. Like the other changefeed and
snapshot suites, this one refuses non-loopback URLs: it deletes rows, so it must
never run against a shared or production endpoint.
"""

from __future__ import annotations

import ipaddress
import os
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.engine import semantic_discovery as semantic_discovery_module
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    CONSOLIDATION_ENGINE_VERSION,
    INCOMPLETE_STATUSES,
    PROGRESS_FORMAT_VERSION,
)
from surreal_memory.engine.semantic_discovery import SemanticDiscoveryResult
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_sds_prune_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        if hostname == "localhost":
            return True
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires explicit loopback SURREALDB_URL for a disposable SurrealDB >= 3.2",
    ),
]


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SurrealDBStorage]:
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="sds_prune_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-discovery-prune-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _save_snapshot(
    store: SurrealDBStorage, state_id: str, *, run_id: str, age: timedelta
) -> None:
    await store.save_semantic_discovery_state(
        state_id,
        1,
        {
            "brain_id": store._get_brain_id(),
            "run_id": run_id,
            "owner_token": "owner-" + run_id,
            "source_token": "source-" + run_id,
        },
    )
    # created_at defaults to time::now(); backdate it to simulate an old row.
    await store._query(
        "UPDATE semantic_discovery_state SET created_at = $created_at "
        "WHERE state_id = $state_id AND revision = 1",
        created_at=utcnow() - age,
        state_id=state_id,
    )


async def _state_ids(store: SurrealDBStorage) -> set[str]:
    rows = await store._query("SELECT VALUE state_id FROM semantic_discovery_state")
    return {str(state_id) for state_id in rows}


async def test_prune_removes_expired_snapshots_of_finished_runs_and_keeps_the_rest(
    store: SurrealDBStorage,
) -> None:
    await _save_snapshot(store, "finished-old-1", run_id="run-finished", age=timedelta(days=30))
    await _save_snapshot(store, "finished-old-2", run_id="run-finished", age=timedelta(days=8))
    await _save_snapshot(store, "finished-young", run_id="run-finished", age=timedelta(days=6))
    await _save_snapshot(store, "paused-old", run_id="run-paused", age=timedelta(days=90))
    own_brain = store._get_brain_id()
    other = Brain.create(name="semantic-discovery-prune-other")
    await store.save_brain(other)
    store.set_brain(other.id)
    await _save_snapshot(store, "other-brain-old", run_id="run-finished", age=timedelta(days=30))
    store.set_brain(own_brain)

    deleted = await store.prune_semantic_discovery_state(
        older_than=timedelta(days=7), keep_run_ids={"run-paused"}
    )

    assert deleted == 2
    assert await _state_ids(store) == {"finished-young", "paused-old", "other-brain-old"}


async def test_prune_deletes_in_batches_of_at_most_500_rows(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_id = store._get_brain_id()
    created_at = utcnow() - timedelta(days=30)
    rows = [
        {
            "state_id": f"bulk-{index:04d}",
            "revision": 1,
            "brain_id": brain_id,
            "run_id": "run-finished",
            "owner_token": "owner-bulk",
            "source_token": "source-bulk",
            "payload": {"index": index},
            "created_at": created_at,
        }
        for index in range(1203)
    ]
    await store._query("INSERT INTO semantic_discovery_state $rows RETURN NONE", rows=rows)
    assert len(await _state_ids(store)) == 1203

    batches: list[int] = []
    query = store._query

    async def recording_query(sql: str, **params: Any) -> list[dict[str, Any]]:
        result = await query(sql, **params)
        if sql.startswith("DELETE (SELECT VALUE id FROM semantic_discovery_state"):
            batches.append(len(result))
        return result

    monkeypatch.setattr(store, "_query", recording_query)

    deleted = await store.prune_semantic_discovery_state(
        older_than=timedelta(days=7), keep_run_ids=[]
    )

    assert deleted == 1203
    assert batches == [500, 500, 203]
    assert await _state_ids(store) == set()


async def test_prune_without_matches_returns_zero(store: SurrealDBStorage) -> None:
    assert (
        await store.prune_semantic_discovery_state(older_than=timedelta(days=7), keep_run_ids=[])
        == 0
    )
    await _save_snapshot(store, "young", run_id="run-finished", age=timedelta(hours=1))
    await _save_snapshot(store, "kept", run_id="run-paused", age=timedelta(days=30))

    assert (
        await store.prune_semantic_discovery_state(
            older_than=timedelta(days=7), keep_run_ids=["run-paused"]
        )
        == 0
    )
    assert await _state_ids(store) == {"young", "kept"}


async def test_unfinished_run_ids_come_from_progress_records(store: SurrealDBStorage) -> None:
    brain_id = store._get_brain_id()
    state: dict[str, Any] = {
        "brain_id": brain_id,
        "run_id": "run-paused",
        "schema_version": SCHEMA_VERSION,
        "format_version": PROGRESS_FORMAT_VERSION,
        "engine_version": CONSOLIDATION_ENGINE_VERSION,
        "requested_strategies": ["semantic_link"],
        "completed_strategies": [],
        "options_fingerprint": "fingerprint",
        "reference_time": utcnow(),
        "status": "paused",
        "current_strategy": "semantic_link",
        "phase": "semantic_link_discovery_neurons",
        "cursor": None,
        "strategy_states": {},
        "counters": {},
        "owner_token": "owner-paused",
        "started_at": utcnow(),
        "updated_at": utcnow(),
        "last_error": None,
    }
    await store.start_consolidation_progress(state)

    assert await store.get_consolidation_run_ids(statuses=INCOMPLETE_STATUSES) == ["run-paused"]

    await store.start_consolidation_progress({**state, "status": "completed"})
    assert await store.get_consolidation_run_ids(statuses=INCOMPLETE_STATUSES) == []


async def test_completed_semantic_link_run_prunes_snapshots_of_finished_runs(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _save_snapshot(store, "previous-run-1", run_id="run-previous", age=timedelta(days=30))
    await _save_snapshot(store, "previous-run-2", run_id="run-previous", age=timedelta(days=9))
    await _save_snapshot(store, "previous-young", run_id="run-previous", age=timedelta(days=1))
    current_runs: list[str] = []

    async def discover(
        storage: SurrealDBStorage, config: Any, *, run_id: str | None = None, **_: Any
    ) -> SemanticDiscoveryResult:
        # Stage a snapshot for the running run and age it past the cutoff: the
        # prune must keep it because the run that wrote it is still in progress.
        assert run_id is not None
        current_runs.append(run_id)
        await _save_snapshot(storage, "current-run", run_id=run_id, age=timedelta(days=30))
        return SemanticDiscoveryResult()

    monkeypatch.setattr(semantic_discovery_module, "discover_semantic_synapses", discover)

    report = await ConsolidationEngine(store).run([ConsolidationStrategy.SEMANTIC_LINK])

    assert report.extra["consolidation_status"] == "completed"
    assert len(current_runs) == 1
    assert report.semantic_discovery_states_pruned == 2
    assert await _state_ids(store) == {"previous-young", "current-run"}
    progress = await store.get_consolidation_progress()
    assert progress is not None
    assert progress["status"] == "completed"
    assert progress["run_id"] == current_runs[0]

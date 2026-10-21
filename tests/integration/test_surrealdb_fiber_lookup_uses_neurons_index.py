"""Live-DB pinning test: the neuron -> fiber lookup must ride ``idx_fiber_neurons``.

History. An earlier fix of ours for the same symptom hinted the query
away from every index with ``WITH NOINDEX``, because ``idx_fiber_brain`` — the only
usable index at the time — is ``brain_id`` alone: on a single-brain database it selects
100% of the rows and forces each one to be decoded, and once fibers carry vectors every fiber row
carries a 1024-float vector (measured: 279 ms with the index, 4.9 ms without). Upstream
#239 solved it the other way round, by adding an index the predicate can actually use —
``DEFINE INDEX idx_fiber_neurons ON fiber FIELDS neuron_ids.*, brain_id`` — plus an
explicit ``WITH INDEX`` hint, measured at 0.5 ms against our 4.4 ms. Their index wins, so
our hint is dropped and this test pins their plan.

What is pinned is the PLAN, never the clock: a timing assertion measures the machine.
Three directions, because one alone proves little:
  1. with ``contains_neuron`` the plan is an IndexScan on ``idx_fiber_neurons``;
  2. the negative control ``WITH NOINDEX`` on the same query is a TableScan (so the
     IndexScan above is a property of the query, not of everything the planner does);
  3. WITHOUT ``contains_neuron`` the hint must NOT be attached — the index cannot serve a
     predicate that does not mention ``neuron_ids``.

Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from base64 import b64encode
from dataclasses import replace
from datetime import datetime, timedelta
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.port is not None
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires an explicit loopback SurrealDB >= 3.2.0 URL",
    ),
]


@pytest_asyncio.fixture
async def store():
    database = "it_" + uuid.uuid4().hex[:12]
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database=database,
    )
    await storage.initialize()
    brain = Brain.create(name="fiber-index-plan-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        try:
            await SurrealDBStorage._query_response(storage, f"REMOVE DATABASE `{database}`")
        finally:
            await storage.close()


async def _seed(store: SurrealDBStorage, n: int = 6) -> str:
    """A handful of fibers; returns a neuron id that one of them contains."""
    anchor_id = ""
    for i in range(n):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"fiber anchor {i}")
        await store.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            tags={f"t{i}"},
        )
        await store.add_fiber(fiber)
        if i == 0:
            anchor_id = neuron.id
    return anchor_id


def _capture_sql(store: SurrealDBStorage) -> list[str]:
    zebrane: list[str] = []
    oryg = store._query

    async def obs(sql: str, **params):
        zebrane.append(sql)
        return await oryg(sql, **params)

    store._query = obs  # type: ignore[method-assign]
    return zebrane


async def _plan(store: SurrealDBStorage, sql: str, **params) -> str:
    plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN", **params)
    return json.dumps(plan, ensure_ascii=False, default=str)


async def test_the_index_exists_and_covers_neuron_ids_and_brain(store: SurrealDBStorage) -> None:
    """INFO FOR TABLE is the only proof an index exists — SELECT would say nothing."""
    info = await SurrealDBStorage._query_response(store, "INFO FOR TABLE fiber")
    ddl = json.dumps(info, ensure_ascii=False, default=str)
    assert "idx_fiber_neurons" in ddl, f"index missing from the live schema: {ddl[:400]}"
    assert "neuron_ids" in ddl and "brain_id" in ddl, (
        f"idx_fiber_neurons does not cover (neuron_ids[*], brain_id): {ddl[:400]}"
    )


async def test_contains_neuron_lookup_rides_the_neurons_index(store: SurrealDBStorage) -> None:
    nid = await _seed(store)
    zebrane = _capture_sql(store)

    fibers = await store.find_fibers(contains_neuron=nid)
    assert fibers, "the lookup must actually find the fiber it is planning for"

    (sql,) = [s for s in zebrane if "FROM fiber" in s and "contains_neuron" in s]
    assert "WITH INDEX idx_fiber_neurons" in sql, f"the hint is not on the query: {sql}"

    plan = await _plan(store, sql, brain_id=store._get_brain_id(), contains_neuron=nid)
    assert "IndexScan" in plan and "idx_fiber_neurons" in plan, (
        f"contains_neuron does not ride idx_fiber_neurons. plan={plan[:400]}"
    )


async def test_negative_control_without_the_index_is_a_table_scan(
    store: SurrealDBStorage,
) -> None:
    """Same query, hint swapped for WITH NOINDEX: the plan must degrade to a TableScan.

    Without this the assertion above proves nothing — a planner that reports IndexScan
    for everything would satisfy it just as well.
    """
    nid = await _seed(store)
    sql_noindex = (
        "SELECT * FROM fiber WITH NOINDEX "
        "WHERE brain_id = $brain_id AND neuron_ids CONTAINS $contains_neuron LIMIT 10"
    )
    plan = await _plan(store, sql_noindex, brain_id=store._get_brain_id(), contains_neuron=nid)
    assert "TableScan" in plan, f"WITH NOINDEX did not fall back to a table scan: {plan[:400]}"
    assert "idx_fiber_neurons" not in plan, f"NOINDEX plan still names the index: {plan[:400]}"


async def test_a_query_without_contains_neuron_does_not_carry_the_hint(
    store: SurrealDBStorage,
) -> None:
    """The index cannot serve a predicate that never mentions neuron_ids."""
    await _seed(store)
    zebrane = _capture_sql(store)

    await store.find_fibers(limit=5)

    fiber_queries = [s for s in zebrane if "FROM fiber" in s]
    assert fiber_queries, f"no fiber query was issued: {zebrane}"
    assert not [s for s in fiber_queries if "idx_fiber_neurons" in s], (
        f"the hint was attached to a query with no neuron predicate: {fiber_queries}"
    )


@pytest.mark.asyncio
async def test_prune_fiber_membership_matches_noindex_reference_and_uses_index(
    store: SurrealDBStorage,
) -> None:
    """The prune protection lookup is complete, brain-scoped, and index-backed."""
    active_fibers = [
        ({"above", "union"}, 0.9),
        ({"above", "equal"}, 0.8),
        ({"below"}, 0.79),
    ]
    for ids, salience in active_fibers:
        await store.add_fiber(
            replace(
                Fiber.create(
                    neuron_ids=ids,
                    synapse_ids=set(),
                    anchor_neuron_id=next(iter(ids)),
                    tags=set(),
                ),
                salience=salience,
            )
        )

    active_brain_id = store._get_brain_id()
    foreign_brain = Brain.create(name="fiber-index-foreign-it")
    await store.save_brain(foreign_brain)
    store.set_brain(foreign_brain.id)
    await store.add_fiber(
        replace(
            Fiber.create(
                neuron_ids={"above", "foreign"},
                synapse_ids=set(),
                anchor_neuron_id="above",
                tags=set(),
            ),
            salience=1.0,
        )
    )
    store.set_brain(active_brain_id)

    captured = _capture_sql(store)
    candidates = ["above", "union", "equal", "below", "foreign", "missing"]
    threshold_result = await store.get_fiber_neuron_ids_for(candidates, min_salience=0.8)
    (sql_threshold,) = [
        query for query in captured if "FROM fiber" in query and "salience >" in query
    ]
    assert "WITH INDEX idx_fiber_neurons" in sql_threshold
    threshold_params = {
        "brain_id": active_brain_id,
        "neuron_ids": candidates,
        "min_salience": 0.8,
    }
    noindex_sql = sql_threshold.replace(" WITH INDEX idx_fiber_neurons", " WITH NOINDEX")
    rows = await SurrealDBStorage._query_response(store, noindex_sql, **threshold_params)
    reference = {
        neuron_id
        for row in rows
        for neuron_id in row.get("neuron_ids", [])
        if neuron_id in candidates
    }
    assert threshold_result == reference == {"above", "union"}

    plan = await _plan(store, sql_threshold, **threshold_params)
    assert "idx_fiber_neurons" in plan and "IndexScan" in plan, plan[:1200]
    assert "idx_fiber_brain" not in plan and "TableScan" not in plan, plan[:1200]
    noindex_plan = await _plan(store, noindex_sql, **threshold_params)
    assert "TableScan" in noindex_plan, noindex_plan[:1200]
    assert "idx_fiber_neurons" not in noindex_plan, noindex_plan[:1200]

    without_threshold = await store.get_fiber_neuron_ids_for(candidates)
    without_sql = captured[-1]
    without_params = {"brain_id": active_brain_id, "neuron_ids": candidates}
    without_reference = await SurrealDBStorage._query_response(
        store,
        without_sql.replace(" WITH INDEX idx_fiber_neurons", " WITH NOINDEX"),
        **without_params,
    )
    assert (
        without_threshold
        == {
            neuron_id
            for row in without_reference
            for neuron_id in row.get("neuron_ids", [])
            if neuron_id in candidates
        }
        == {"above", "union", "equal", "below"}
    )
    without_plan = await _plan(store, without_sql, **without_params)
    assert "idx_fiber_neurons" in without_plan and "idx_fiber_brain" not in without_plan
    assert await store.get_fiber_neuron_ids_for(["missing"], min_salience=0.8) == set()


async def _seed_prune_scenario(
    store: SurrealDBStorage, prefix: str, reference_time: datetime
) -> dict[str, set[str]]:
    names = {
        "high_source",
        "high_target",
        "pinned_source",
        "pinned_target",
        "bridge_source",
        "bridge_target",
        "weak_source",
        "weak_target",
        "fiber_orphan",
        "plain_orphan",
    }
    ids = {name: f"{prefix}-{name.replace('_', '-')}" for name in names}
    for name, neuron_id in ids.items():
        neuron = replace(
            Neuron.create(type=NeuronType.CONCEPT, content=f"synthetic {prefix} {name}"),
            id=neuron_id,
            created_at=reference_time - timedelta(days=60),
        )
        await store.add_neuron(neuron)

    for neuron_id, salience, pinned in (
        (ids["high_source"], 0.9, False),
        (ids["pinned_source"], 0.1, True),
        (ids["fiber_orphan"], 0.1, False),
    ):
        fiber = replace(
            Fiber.create(
                neuron_ids={neuron_id},
                synapse_ids=set(),
                anchor_neuron_id=neuron_id,
                tags=set(),
            ),
            salience=salience,
            pinned=pinned,
        )
        await store.add_fiber(fiber)

    edges = {
        "high_edge": (ids["high_source"], ids["high_target"], 0.001),
        "pinned_edge": (ids["pinned_source"], ids["pinned_target"], 0.001),
        "bridge_edge": (ids["bridge_source"], ids["bridge_target"], 0.03),
        "weak_edge": (ids["weak_source"], ids["weak_target"], 0.001),
    }
    edge_ids: dict[str, set[str]] = {}
    for name, (source, target, weight) in edges.items():
        edge_id = f"{prefix}-{name.replace('_', '-')}"
        synapse = replace(
            Synapse.create(
                source, target, SynapseType.RELATED_TO, weight=weight, synapse_id=edge_id
            ),
            created_at=reference_time - timedelta(days=60),
        )
        await store.add_synapse(synapse)
        edge_ids[name] = {edge_id}
    return {
        "neurons": set(ids.values()),
        "edges": set().union(*edge_ids.values()),
        **{name: {ids[name]} for name in names},
        **edge_ids,
    }


async def _run_prune(
    store: SurrealDBStorage,
    reference_time: datetime,
    *,
    resume_run_id: str | None = None,
) -> ConsolidationReport:
    return await ConsolidationEngine(
        store,
        ConsolidationConfig(strategy_timeout_seconds=30, total_timeout_seconds=120),
    ).run(
        strategies=[ConsolidationStrategy.PRUNE],
        reference_time=reference_time,
        resume_run_id=resume_run_id,
    )


@pytest.mark.asyncio
async def test_real_prune_protection_and_persisted_resume_match_noindex_reference(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_time = datetime(2026, 10, 1, 12)
    indexed_brain_id = store._get_brain_id()
    indexed_ids = await _seed_prune_scenario(store, "indexed", reference_time)

    engine = ConsolidationEngine(
        store,
        ConsolidationConfig(strategy_timeout_seconds=30, total_timeout_seconds=120),
    )
    check_budget = engine._check_progress_budget
    pause_armed = True

    async def pause_at_committed_prune_checkpoint() -> None:
        nonlocal pause_armed
        if pause_armed and engine._strategy_progress_state().get("phase") == "retention_start":
            pause_armed = False
            raise ConsolidationPausedError("synthetic deterministic pause")
        await check_budget()

    monkeypatch.setattr(engine, "_check_progress_budget", pause_at_committed_prune_checkpoint)
    paused_report = await engine.run(
        strategies=[ConsolidationStrategy.PRUNE], reference_time=reference_time
    )
    assert paused_report.extra["consolidation_status"] == "paused"
    paused_state = await store.get_consolidation_progress(indexed_brain_id)
    assert paused_state is not None and paused_state["status"] == "paused"
    run_id = str(paused_state["run_id"])

    indexed_report = await _run_prune(store, reference_time, resume_run_id=run_id)
    indexed_state = await store.get_consolidation_progress(indexed_brain_id)
    assert indexed_state is not None and indexed_state["status"] == "completed"

    reference_brain = Brain.create(name="fiber-index-prune-noindex-it")
    await store.save_brain(reference_brain)
    store.set_brain(reference_brain.id)
    reference_ids = await _seed_prune_scenario(store, "noindex", reference_time)
    indexed_lookup = store.get_fiber_neuron_ids_for

    async def noindex_fiber_lookup(
        neuron_ids: list[str] | set[str], *, min_salience: float | None = None
    ) -> set[str]:
        candidates = set(neuron_ids)
        if not candidates:
            return set()
        params: dict[str, object] = {
            "brain_id": store._get_brain_id(),
            "neuron_ids": list(candidates),
        }
        if min_salience is None:
            sql = (
                "SELECT neuron_ids FROM fiber WITH NOINDEX WHERE "
                "brain_id = $brain_id AND neuron_ids CONTAINSANY $neuron_ids"
            )
        else:
            sql = (
                "SELECT neuron_ids FROM fiber WITH NOINDEX WHERE "
                "brain_id = $brain_id AND neuron_ids CONTAINSANY $neuron_ids "
                "AND salience > $min_salience"
            )
            params["min_salience"] = min_salience
        rows = await store._query(sql, **params)
        return {
            neuron_id
            for row in rows
            for neuron_id in row.get("neuron_ids", [])
            if neuron_id in candidates
        }

    monkeypatch.setattr(store, "get_fiber_neuron_ids_for", noindex_fiber_lookup)
    reference_report = await _run_prune(store, reference_time)
    reference_state = await store.get_consolidation_progress(reference_brain.id)
    assert reference_state is not None and reference_state["status"] == "completed"
    monkeypatch.setattr(store, "get_fiber_neuron_ids_for", indexed_lookup)

    assert indexed_report.extra["consolidation_status"] == "completed"
    assert indexed_report.synapses_pruned == reference_report.synapses_pruned == 1
    assert indexed_report.neurons_pruned == reference_report.neurons_pruned == 3
    assert (
        indexed_state["strategy_states"]["prune"]["counters"]
        == reference_state["strategy_states"]["prune"]["counters"]
    )

    async def remaining(brain_id: str, fixture_ids: dict[str, set[str]]):
        store.set_brain(brain_id)
        edges = await store.get_synapses_by_ids(fixture_ids["edges"])
        neurons = await store.find_neurons_by_ids(
            sorted(fixture_ids["neurons"]), include_embedding=False
        )
        return {edge.id for edge in edges}, {neuron.id for neuron in neurons}

    indexed_edges, indexed_neurons = await remaining(indexed_brain_id, indexed_ids)
    reference_edges, reference_neurons = await remaining(reference_brain.id, reference_ids)
    expected_indexed_edges = {
        next(iter(indexed_ids[name])) for name in ("high_edge", "pinned_edge", "bridge_edge")
    }
    expected_reference_edges = {
        next(iter(reference_ids[name])) for name in ("high_edge", "pinned_edge", "bridge_edge")
    }
    assert indexed_edges == expected_indexed_edges
    assert reference_edges == expected_reference_edges
    expected_indexed_neurons = {
        *indexed_ids["high_source"],
        *indexed_ids["high_target"],
        *indexed_ids["pinned_source"],
        *indexed_ids["pinned_target"],
        *indexed_ids["bridge_source"],
        *indexed_ids["bridge_target"],
        *indexed_ids["fiber_orphan"],
    }
    assert indexed_neurons == expected_indexed_neurons
    expected_reference_neurons = {
        value.replace("indexed-", "noindex-") for value in indexed_neurons
    }
    assert reference_neurons == expected_reference_neurons


async def _http_sql(store: SurrealDBStorage, sql: str) -> list[dict[str, object]]:
    parsed = urlparse(SURREALDB_URL or "")
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    credentials = b64encode(f"{SURREALDB_USER}:{SURREALDB_PASS}".encode()).decode()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/surrealql",
        "Authorization": f"Basic {credentials}",
        "surreal-ns": store._namespace,
        "surreal-db": store._database,
    }

    def send() -> list[dict[str, object]]:
        connection_type = HTTPSConnection if scheme == "https" else HTTPConnection
        connection = connection_type(parsed.hostname, parsed.port, timeout=10)
        try:
            connection.request("POST", "/sql", body=sql.encode(), headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                raise AssertionError(f"SurrealDB /sql returned HTTP {response.status}")
            return json.loads(response.read())
        finally:
            connection.close()

    return await asyncio.to_thread(send)


@pytest.mark.asyncio
async def test_readonly_transaction_keeps_snapshot_across_concurrent_writer(
    store: SurrealDBStorage,
) -> None:
    await store._query("CREATE fiber_index_snapshot_probe:probe SET marker = 'before' RETURN NONE")
    reader_sql = (
        "BEGIN TRANSACTION; "
        "SELECT VALUE marker FROM fiber_index_snapshot_probe "
        "WHERE id = type::record('fiber_index_snapshot_probe', 'probe'); "
        "SLEEP 1s; "
        "SELECT VALUE marker FROM fiber_index_snapshot_probe "
        "WHERE id = type::record('fiber_index_snapshot_probe', 'probe'); "
        "COMMIT TRANSACTION;"
    )
    reader = asyncio.create_task(_http_sql(store, reader_sql))
    await asyncio.sleep(0.1)
    await store._query("UPDATE fiber_index_snapshot_probe:probe SET marker = 'after' RETURN NONE")
    assert not reader.done(), "writer must commit while the read transaction is active"
    result = await reader
    assert all(item["status"] == "OK" for item in result), result
    reads = [item["result"] for item in result if isinstance(item.get("result"), list)]
    assert reads == [["before"], ["before"]]
    after = await store._query(
        "SELECT VALUE marker FROM fiber_index_snapshot_probe "
        "WHERE id = type::record('fiber_index_snapshot_probe', 'probe')"
    )
    assert after == ["after"]

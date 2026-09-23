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

import json
import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="fiber-index-plan-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
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

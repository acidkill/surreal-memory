"""Live integration tests for SurrealDB query-result shapes.

These run against a REAL SurrealDB >= 3.2.0 (skipped unless SURREALDB_URL is set).

``_query`` carried a shape heuristic -- ``result[0] if isinstance(result[0],
list) else result`` -- left over from the SDK 1.x era when ``query()`` returned
the RPC envelope. The pinned SDK (>=2.0.0) unwraps that envelope itself, so the
heuristic's only remaining effect was corrupting ``SELECT VALUE`` results on
array-typed fields: a list of per-row arrays collapsed to the first row's array,
which the caller then iterated as if it were the rows (#143; the residual was
#154's finding 3). A unit mock can pin our unwrapping code, but only a live
server proves which shapes the SDK actually hands us -- which is the half the
heuristic got wrong.

Run (real-db-test-runner drives this):
    SURREALDB_URL=http://localhost:<port> SURREALDB_USER=root SURREALDB_PASS=... \
        uv run --extra dev --extra server --extra surrealdb \
        pytest tests/integration/test_surrealdb_query_shapes.py -m integration -q
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
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
    """A fresh, initialised store scoped to its own throwaway database."""
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="query-shapes-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed_probe_table(store: SurrealDBStorage) -> None:
    """Three rows: two with tag arrays, one without (NULL projects as None).

    The fixture's database is throwaway, so no WHERE scoping is needed — every
    row in ``query_shape_probe`` is ours. (String-vs-RecordID ``IN`` filters
    silently match nothing; see test_surrealdb_pinning's header.)
    """
    await store._query("REMOVE TABLE IF EXISTS query_shape_probe")
    await store._query("CREATE query_shape_probe:a, query_shape_probe:b, query_shape_probe:c")
    await store._query(
        "UPDATE query_shape_probe SET tags = ['t1', 't2'] WHERE id = query_shape_probe:a"
    )
    await store._query("UPDATE query_shape_probe SET tags = ['t3'] WHERE id = query_shape_probe:b")


async def test_row_queries_return_row_dicts(store: SurrealDBStorage):
    await _seed_probe_table(store)
    rows = await store._query("SELECT * FROM query_shape_probe ORDER BY id")
    # Schemaless rows omit absent fields (row c has no tags key at all).
    assert [r.get("tags") for r in rows] == [["t1", "t2"], ["t3"], None]


async def test_value_query_on_scalar_field_is_flat(store: SurrealDBStorage):
    await _seed_probe_table(store)
    values = await store._query_values("SELECT VALUE id FROM query_shape_probe ORDER BY id")
    assert [str(v) for v in values] == [
        "query_shape_probe:a",
        "query_shape_probe:b",
        "query_shape_probe:c",
    ]


async def test_value_query_on_array_field_keeps_every_row(store: SurrealDBStorage):
    """The #143/#154.3 regression pin: the old heuristic returned only the
    first row's array here. All rows must survive, nesting intact."""
    await _seed_probe_table(store)
    values = await store._query_values("SELECT VALUE tags FROM query_shape_probe ORDER BY id")
    assert values == [["t1", "t2"], ["t3"], None]


async def test_connected_neuron_ids_through_real_synapses(store: SurrealDBStorage):
    """The one production caller of ``_query_values`` must keep working: two
    neurons joined by one synapse surface both endpoints."""
    a = await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="a"))
    b = await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="b"))
    await store.add_synapse(Synapse.create(source_id=a, target_id=b, type=SynapseType.RELATED_TO))

    connected = await store.get_connected_neuron_ids()
    assert connected == {a, b}

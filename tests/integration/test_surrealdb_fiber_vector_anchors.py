"""Live integration test for the fiber-level vector retriever (skipped unless SURREALDB_URL is
set — see tests/integration/test_surrealdb_query_shapes.py's header for how to run this against
a real SurrealDB >= 3.2.0).

The gap this fixes: `fiber` carries no vector field or index at all, and the only way retrieval
reaches a fiber is `find_fibers_batch` on neuron_ids membership — a fiber is invisible unless one
of ITS OWN neurons is already an activation anchor. A fiber whose neurons never win any of the
time/entity/keyword/embedding anchor races (but whose `summary` plainly answers the query) never
surfaces, no matter how well it matches. `find_fibers_by_embedding` (this fix) is the one retriever
that reaches a fiber directly, by its own precomputed `fiber_vec`, bypassing that requirement.

A unit test with a mocked storage cannot tell the difference between "the fiber HNSW index found
this" and "the mock said so" — only a real index, searched over more rows than a small `limit`,
proves it.

Results are identified by `anchor_neuron_id`, not `fiber.id`: `find_fibers`/`get_fiber` fold a
fiber's dashed uuid to underscores and never fold it back (a separate, documented, NOT-fixed-here
quirk — see `storage/surrealdb/store.py::_row_to_fiber`), so comparing returned fiber ids against
the dashed ids this test created them with would be comparing against the wrong quirk. The
production retriever step (`engine/retrieval.py`, "FIBER VECTOR ANCHORS") only ever reads
`fiber.anchor_neuron_id` for exactly this reason, so testing through that same field is also the
more faithful pin.
"""

from __future__ import annotations

import math
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

_DIM = 4
_FILLER_COUNT = 30
_QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]
_FILLER_VECTOR = [0.0, 1.0, 0.0, 0.0]  # orthogonal: cos == 0.0
_TARGET_VECTOR = [10.0, 1.0, 0.0, 0.0]  # cos(query, target) ~= 0.995
_EXPECTED_TARGET_COSINE = 10.0 / math.sqrt(101.0)


@pytest_asyncio.fixture
async def store():
    """A fresh store, scoped to its own throwaway database, with a small-dim HNSW index."""
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
        embedding_dim=_DIM,
    )
    await storage.initialize()
    brain = Brain.create(name="fiber-vec-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed(store: SurrealDBStorage) -> str:
    """30 filler fibers whose OWN vector points away from the query, plus one target fiber whose
    vector points at it — none of the fillers' anchor neurons are otherwise related, so a
    non-vector retriever finding a filler anchor would never surface the target. Returns the
    target's anchor neuron id."""
    for _ in range(_FILLER_COUNT):
        anchor = Neuron.create(type=NeuronType.CONCEPT, content="filler anchor")
        await store.add_neuron(anchor)
        fiber = Fiber.create(neuron_ids={anchor.id}, synapse_ids=set(), anchor_neuron_id=anchor.id)
        await store.add_fiber(fiber)
        await store.update_fiber_embeddings([(fiber.id, list(_FILLER_VECTOR))])

    target_anchor = Neuron.create(type=NeuronType.CONCEPT, content="the target anchor")
    await store.add_neuron(target_anchor)
    target_fiber = Fiber.create(
        neuron_ids={target_anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=target_anchor.id,
        summary="the fiber a query about the target concept should actually find",
    )
    await store.add_fiber(target_fiber)
    await store.update_fiber_embeddings([(target_fiber.id, list(_TARGET_VECTOR))])
    return target_anchor.id


@pytest.mark.timeout(120)
async def test_find_fibers_by_embedding_reaches_the_target_by_its_own_vector(
    store: SurrealDBStorage,
) -> None:
    target_anchor_id = await _seed(store)

    rows = await store.find_fibers_by_embedding(_QUERY_VECTOR, limit=5)
    scores_by_anchor = {fiber.anchor_neuron_id: score for fiber, score in rows}
    assert target_anchor_id in scores_by_anchor, (
        "the fiber vector index must reach the target at all"
    )
    assert abs(scores_by_anchor[target_anchor_id] - _EXPECTED_TARGET_COSINE) < 1e-6

    # Every filler scores strictly lower (orthogonal vs 0.995 cosine) — the target is not just
    # present, it wins.
    filler_scores = [s for anchor, s in scores_by_anchor.items() if anchor != target_anchor_id]
    assert all(s < scores_by_anchor[target_anchor_id] for s in filler_scores)


@pytest.mark.timeout(120)
async def test_find_fibers_by_embedding_scopes_to_the_calling_brain(
    store: SurrealDBStorage,
) -> None:
    """Same brain-scoping guarantee as `find_neurons_by_embedding` — a near-perfect match in a
    DIFFERENT brain must never surface."""
    first_brain_id = store.brain_id
    own_anchor = Neuron.create(type=NeuronType.CONCEPT, content="own anchor")
    await store.add_neuron(own_anchor)
    own_fiber = Fiber.create(
        neuron_ids={own_anchor.id}, synapse_ids=set(), anchor_neuron_id=own_anchor.id
    )
    await store.add_fiber(own_fiber)
    await store.update_fiber_embeddings([(own_fiber.id, [0.9, 0.1, 0.0, 0.0])])

    other_brain = Brain.create(name="fiber-vec-other-brain")
    await store.save_brain(other_brain)
    store.set_brain(other_brain.id)
    decoy_anchor = Neuron.create(type=NeuronType.CONCEPT, content="decoy anchor")
    await store.add_neuron(decoy_anchor)
    decoy_fiber = Fiber.create(
        neuron_ids={decoy_anchor.id}, synapse_ids=set(), anchor_neuron_id=decoy_anchor.id
    )
    await store.add_fiber(decoy_fiber)
    await store.update_fiber_embeddings([(decoy_fiber.id, list(_QUERY_VECTOR))])

    store.set_brain(first_brain_id)
    rows = await store.find_fibers_by_embedding(_QUERY_VECTOR, limit=10)
    anchors = {fiber.anchor_neuron_id for fiber, _score in rows}
    assert own_anchor.id in anchors
    assert decoy_anchor.id not in anchors

"""Live integration test for the KNN embedding-anchor path (skipped unless
SURREALDB_URL is set — see tests/integration/test_surrealdb_query_shapes.py's
header for how to run this against a real SurrealDB >= 3.2.0).

The defect this proves the fix for: the historical "scan" anchor path reads one
``find_neurons(limit=1000)`` page ordered by id. On a brain past that cap, a
memory whose id happens to sort late is structurally invisible to the semantic
retriever, no matter how well it matches the query — a unit test with a mocked
storage cannot tell the difference between "the index found it" and "the mock
said so". Only a real SurrealDB HNSW index, searched over more than 1000 real
rows, proves the vector index actually reaches what the scan cannot.

``tests/unit/test_embedding_anchors_knn.py`` covers every branch of the outcome
(threshold, tombstones, the retry, fallbacks) against mocks; this file is the
one live counterpart that proves the real index and the real cosine scale.
"""

from __future__ import annotations

import math
import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
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

# Small HNSW dimension: this test only needs geometric separation, not a real
# embedding model, and a tiny dimension keeps a 1200+ row seed fast.
_DIM = 4
_FILLER_COUNT = 1200
_QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]
# Orthogonal to the query: cos == 0.0, comfortably under the 0.3 ceiling every
# filler must stay below.
_FILLER_VECTOR = [0.0, 1.0, 0.0, 0.0]
# cos(query, target) == 10 / sqrt(101) ~= 0.99504, comfortably over the 0.95
# floor the target must clear.
_TARGET_VECTOR = [10.0, 1.0, 0.0, 0.0]
_EXPECTED_TARGET_COSINE = 10.0 / math.sqrt(101.0)
_TARGET_ID = "zzzzzzzz-target-neuron"


class _FixedVectorProvider:
    """Stub embedding provider: the query always embeds to one fixed vector.

    Only the storage layer needs to be real here — the point of this test is
    the SurrealDB HNSW index and the cosine scale it returns, not embedding
    quality, so pulling in a real model would add cost without adding proof.
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, _query: str) -> list[float]:
        return self._vector

    async def similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


@pytest_asyncio.fixture
async def store():
    """A fresh store, scoped to its own throwaway database, with a small-dim
    HNSW index so the 1200+ row seed below stays fast."""
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
        embedding_dim=_DIM,
    )
    await storage.initialize()
    brain = Brain.create(name="knn-anchors-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed_filler_and_target(store: SurrealDBStorage) -> None:
    """1200 filler neurons whose ids sort before the target's, plus the target.

    ``find_neurons`` orders by id and caps at 1000 (the scan path's page), so
    every filler id sorts lexically before ``_TARGET_ID`` — the target is
    guaranteed to fall outside that page while remaining fully reachable by a
    real vector-index search, which has no notion of id order at all.
    """
    neurons = [
        Neuron.create(
            type=NeuronType.CONCEPT,
            content=f"filler memory {i}",
            neuron_id=f"filler-{i:06d}",
            metadata={"_embedding": list(_FILLER_VECTOR)},
        )
        for i in range(_FILLER_COUNT)
    ]
    neurons.append(
        Neuron.create(
            type=NeuronType.CONCEPT,
            content="the target memory",
            neuron_id=_TARGET_ID,
            metadata={"_embedding": list(_TARGET_VECTOR)},
        )
    )
    await store.add_neurons_batch(neurons, record_change=False)


@pytest.mark.timeout(180)
async def test_knn_mode_reaches_the_target_beyond_the_scan_cap_and_scan_mode_does_not(
    store: SurrealDBStorage,
) -> None:
    """One seed, three related proofs against the same live index:

    1. the raw KNN call finds the target and reports its similarity on the
       cosine scale (not the old ``1/(1+distance)`` scale);
    2. the "knn" anchor mode surfaces that target as an anchor;
    3. the "scan" anchor mode — reading only the first id-ordered page — does
       not, because the target sorts past the 1000-row cap.
    """
    await _seed_filler_and_target(store)

    # 1. Storage layer: the target is found, and its score is cosine, not the
    # historical 1/(1+distance) scale.
    rows = await store.find_neurons_by_embedding(_QUERY_VECTOR, limit=30)
    scores_by_id = {neuron.id: score for neuron, score in rows}
    assert _TARGET_ID in scores_by_id, "the vector index must reach the target at all"
    assert abs(scores_by_id[_TARGET_ID] - _EXPECTED_TARGET_COSINE) < 1e-6
    # Distinguishability: at this cosine the two scales differ by only ~2.4e-5, so a
    # loose tolerance would pass on the old 1/(1+distance) scale too. Pin the gap
    # explicitly so the assertion can actually fail on the historical scale.
    _old_scale = 1.0 / (1.0 + (1.0 - _EXPECTED_TARGET_COSINE))
    assert abs(scores_by_id[_TARGET_ID] - _old_scale) > 1e-6, (
        "the reported score must be cosine, not the historical 1/(1+distance) scale"
    )

    # 2. Pipeline layer, "knn" mode: the target is a genuine anchor.
    knn_config = BrainConfig(embedding_anchor_mode="knn", embedding_similarity_threshold=0.5)
    knn_pipeline = ReflexPipeline(
        storage=store, config=knn_config, embedding_provider=_FixedVectorProvider(_QUERY_VECTOR)
    )
    knn_outcome = await knn_pipeline._find_embedding_anchors_outcome("find the target", top_k=10)
    assert _TARGET_ID in knn_outcome.anchor_ids

    # 3. Pipeline layer, "scan" mode: the same target, same threshold, same
    # brain — absent, because the scan never looks past the id-ordered page.
    scan_config = BrainConfig(embedding_anchor_mode="scan", embedding_similarity_threshold=0.5)
    scan_pipeline = ReflexPipeline(
        storage=store, config=scan_config, embedding_provider=_FixedVectorProvider(_QUERY_VECTOR)
    )
    scan_outcome = await scan_pipeline._find_embedding_anchors_outcome("find the target", top_k=10)
    assert _TARGET_ID not in scan_outcome.anchor_ids


async def test_two_brains_in_the_same_database_never_mix_knn_results(
    store: SurrealDBStorage,
) -> None:
    """The brain_id literal inlined into the KNN query must scope results
    exactly like ``find_neurons`` does — a neighbour from another brain must
    never surface, even when it is a far better match than anything of ours."""
    first_brain_id = store.brain_id
    own_neighbour = Neuron.create(
        type=NeuronType.CONCEPT,
        content="own brain neighbour",
        neuron_id="own-neighbour",
        metadata={"_embedding": [0.9, 0.1, 0.0, 0.0]},
    )
    await store.add_neuron(own_neighbour)

    other_brain = Brain.create(name="knn-anchors-other-brain")
    await store.save_brain(other_brain)
    store.set_brain(other_brain.id)
    # A near-perfect match for the query, planted in the OTHER brain — if
    # brain scoping ever leaked, this is the row that would surface.
    decoy = Neuron.create(
        type=NeuronType.CONCEPT,
        content="other brain decoy",
        neuron_id="other-brain-decoy",
        metadata={"_embedding": list(_QUERY_VECTOR)},
    )
    await store.add_neuron(decoy)

    store.set_brain(first_brain_id)
    rows = await store.find_neurons_by_embedding(_QUERY_VECTOR, limit=10)

    ids = {neuron.id for neuron, _ in rows}
    assert "own-neighbour" in ids
    assert "other-brain-decoy" not in ids


@pytest.mark.timeout(120)
async def test_a_zero_magnitude_vector_is_dropped_instead_of_scoring_as_a_perfect_match(
    store: SurrealDBStorage,
) -> None:
    """A zero-magnitude embedding makes the index report an unusable distance.

    Read naively that becomes similarity 1.0 — a perfect match for a row that
    cannot be ranked at all, which would then outrank every genuine neighbour
    and fill the anchor slots. The row must be dropped, and a query whose own
    vector is degenerate must fail loudly rather than rank on fabrications.
    """
    good = Neuron.create(
        type=NeuronType.CONCEPT,
        content="a genuine neighbour",
        neuron_id="zeroprobe-good",
        metadata={"_embedding": [0.9, 0.1, 0.0, 0.0]},
    )
    zero = Neuron.create(
        type=NeuronType.CONCEPT,
        content="a memory whose vector has no magnitude",
        neuron_id="zeroprobe-zero",
        metadata={"_embedding": [0.0, 0.0, 0.0, 0.0]},
    )
    await store.add_neurons_batch([good, zero], record_change=False)

    rows = await store.find_neurons_by_embedding(_QUERY_VECTOR, limit=10)
    by_id = {neuron.id: score for neuron, score in rows}
    assert "zeroprobe-good" in by_id, "the genuine neighbour must still be found"
    assert "zeroprobe-zero" not in by_id, (
        "an unrankable row must not be returned at all, let alone as a perfect match"
    )

    # The same row must not reach the anchor set through the pipeline either.
    config = BrainConfig(embedding_anchor_mode="knn", embedding_similarity_threshold=0.5)
    pipeline = ReflexPipeline(
        storage=store, config=config, embedding_provider=_FixedVectorProvider(_QUERY_VECTOR)
    )
    outcome = await pipeline._find_embedding_anchors_outcome("find something", top_k=10)
    assert "zeroprobe-zero" not in outcome.anchor_ids
    assert outcome.source == "knn"

    # A degenerate query vector leaves nothing rankable: loud failure, and the
    # pipeline reports the degradation instead of answering from fabrications.
    with pytest.raises(ValueError):
        await store.find_neurons_by_embedding([0.0, 0.0, 0.0, 0.0], limit=10)

    zero_query_pipeline = ReflexPipeline(
        storage=store,
        config=config,
        embedding_provider=_FixedVectorProvider([0.0, 0.0, 0.0, 0.0]),
    )
    zero_outcome = await zero_query_pipeline._find_embedding_anchors_outcome("q", top_k=10)
    assert zero_outcome.anchor_ids == []
    assert zero_outcome.source == "none:knn-bad-distance"

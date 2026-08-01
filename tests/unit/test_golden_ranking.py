"""Golden-ranking regression snapshot (U2).

Locks the DEFAULT-config recall ranking so the trust/recency calibration (opt-in,
neutral defaults trust_weight=0.0 / recency_weight=1.0) provably does NOT change
the ordering for existing users. Recorded on the pre-trust HEAD; must stay green
after the feature lands (the trust/recency code paths are branch-guarded off at
defaults). See RUNBOOK U2 + PLAN §PR2.

Determinism: fixed fiber ids, uniform salience/conductivity, embeddings off, no
context fingerprints — so the only differentiator is keyword/activation match,
which is float-deterministic for a given query.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage

# Ordered (fiber_id) ranking recorded on the pre-trust baseline for QUERY below.
# Regenerate ONLY intentionally (a real ranking change): run this test, read the
# assertion diff, and paste the observed order here.
#
# Re-recorded for the SQLite -> InMemoryStorage backend swap (PR6): SQLite's FTS5
# index stems "synapses" (query) to match "synapse" (g-synapse's neuron content),
# so g-synapse entered the SQLite-backed ranking. InMemoryStorage.find_neurons()
# does plain case-insensitive substring matching (no stemming), so it never
# anchors on g-synapse for this query and the fiber is correctly absent here —
# this is a known, accepted limitation of the in-memory test double relative to
# both real backends' full-text search, not a ranking regression.
GOLDEN_ORDER: list[str] = [
    "g-activation",
    "g-decay",
]

QUERY = "spreading activation through weighted synapses in the neural graph"

# (label, anchor content, fiber summary) — labels double as stable fiber ids.
_FIBERS: list[tuple[str, str, str]] = [
    (
        "g-activation",
        "spreading activation engine",
        "Spreading activation propagates signal strength",
    ),
    (
        "g-synapse",
        "weighted synapse conduction",
        "Weighted synapses carry the spreading activation signal",
    ),
    ("g-graph", "neural graph traversal", "The neural graph is traversed by spreading activation"),
    ("g-neural", "neural network topology", "Neural topology shapes graph connectivity"),
    ("g-consolidation", "memory consolidation", "Consolidation prunes weak connections over time"),
    ("g-embedding", "embedding configuration", "Configure embedding providers for semantic search"),
    ("g-decay", "activation decay schedule", "Activation decays on a logistic recency schedule"),
    ("g-reinforce", "hebbian reinforcement", "Reinforcement strengthens co-activated pathways"),
]


async def _build() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="golden_brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    for label, content, summary in _FIBERS:
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
        await store.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary=summary,
            fiber_id=label,
        )
        await store.add_fiber(fiber)
    return store


async def _ranking(config: BrainConfig) -> list[str]:
    store = await _build()
    try:
        pipeline = ReflexPipeline(store, config)
        result = await pipeline.query(QUERY)
        return list(result.fibers_matched)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_default_ranking_matches_golden_snapshot() -> None:
    ranking = await _ranking(BrainConfig(sufficiency_threshold=0.1))
    assert ranking == GOLDEN_ORDER


@pytest.mark.asyncio
async def test_ranking_is_deterministic_across_runs() -> None:
    a = await _ranking(BrainConfig(sufficiency_threshold=0.1))
    b = await _ranking(BrainConfig(sufficiency_threshold=0.1))
    assert a == b


@pytest.mark.asyncio
async def test_explicit_neutral_weights_match_golden_snapshot() -> None:
    # Explicit neutral trust/recency weights must reproduce the default ranking.
    ranking = await _ranking(
        BrainConfig(sufficiency_threshold=0.1, trust_weight=0.0, recency_weight=1.0)
    )
    assert ranking == GOLDEN_ORDER


@pytest.mark.asyncio
async def test_zero_trust_storage_reads_at_default() -> None:
    # Spy: with trust_weight=0.0 (default) the trust map is never built, so the
    # batch typed-memory read the trust path uses is never called.
    store = await _build()
    calls: list[int] = []
    orig = store.get_typed_memories_batch

    async def _spy(fiber_ids):  # type: ignore[no-untyped-def]
        calls.append(1)
        return await orig(fiber_ids)

    store.get_typed_memories_batch = _spy  # type: ignore[method-assign]
    try:
        pipeline = ReflexPipeline(store, BrainConfig(sufficiency_threshold=0.1))
        await pipeline.query(QUERY)
        assert calls == []
        assert pipeline._last_trust_map == {}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_trust_weight_demotes_low_trust_fiber() -> None:
    # Directional guard: two equally-matching fibers; with trust_weight=1.0 the
    # higher-trust fiber MUST outrank the lower-trust one. A sign/term swap in the
    # trust-multiply (score *= (1-tw) + tw*trust) would fail this.
    from surreal_memory.core.memory_types import MemoryType, TypedMemory

    store = InMemoryStorage()
    brain = Brain.create(name="trust_dir")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    for label, trust, tail in [("hi", 0.9, "alpha"), ("lo", 0.1, "beta")]:
        neuron = Neuron.create(
            type=NeuronType.CONCEPT, content=f"reciprocal rank fusion scoring {tail}"
        )
        await store.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary=f"reciprocal rank fusion scoring blends retrievers {tail}",
            fiber_id=label,
        )
        await store.add_fiber(fiber)
        await store.add_typed_memory(
            TypedMemory.create(fiber_id=label, memory_type=MemoryType.FACT, trust_score=trust)
        )
    try:
        pipeline = ReflexPipeline(store, BrainConfig(sufficiency_threshold=0.1, trust_weight=1.0))
        ranking = list((await pipeline.query("reciprocal rank fusion scoring")).fibers_matched)
        assert "hi" in ranking and "lo" in ranking
        assert ranking.index("hi") < ranking.index("lo")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_trust_storage_read_when_weight_positive() -> None:
    # With trust_weight>0 the trust map IS built (batch typed-memory read happens).
    store = await _build()
    calls: list[int] = []
    orig = store.get_typed_memories_batch

    async def _spy(fiber_ids):  # type: ignore[no-untyped-def]
        calls.append(1)
        return await orig(fiber_ids)

    store.get_typed_memories_batch = _spy  # type: ignore[method-assign]
    try:
        pipeline = ReflexPipeline(store, BrainConfig(sufficiency_threshold=0.1, trust_weight=0.5))
        await pipeline.query(QUERY)
        assert calls != []
        assert pipeline._last_trust_map != {}
    finally:
        await store.close()

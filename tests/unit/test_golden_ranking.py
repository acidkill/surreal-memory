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

import tempfile
from pathlib import Path

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.sqlite_store import SQLiteStorage

# Ordered (fiber_id) ranking recorded on the pre-trust baseline for QUERY below.
# Regenerate ONLY intentionally (a real ranking change): run this test, read the
# assertion diff, and paste the observed order here.
GOLDEN_ORDER: list[str] = [
    "g-activation",
    "g-decay",
    "g-synapse",
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


async def _build() -> SQLiteStorage:
    tmp = tempfile.mkdtemp()
    store = SQLiteStorage(Path(tmp) / "golden.db")
    await store.initialize()
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

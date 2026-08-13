"""K2 (run 013) — connectivity scored on the organic subgraph.

Code-index neurons (``metadata.indexed = true``) are structural leaves emitted by
``CodebaseIndexer`` — one tree per file, never the endpoint of a memory the user
recalls. Counting them in the connectivity denominator deflated the score on
code-heavy brains: a brain with 20k indexed neurons + 20k organic neurons at a
healthy 6 synapses/neuron reported ~3, because half the denominator was inert.

This pins two things:
1. ``get_stats`` returns ``structural_neuron_count`` (live, backend-computed).
2. ``connectivity`` is scored against the ORGANIC count
   (``connectivity_neuron_count = neuron_count - structural_neuron_count``), so
   adding code-index neurons does not move the score.

See DECISIONS.md (run 013, K1) — variant (c): both endpoints organic.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.diagnostics import DiagnosticsEngine
from surreal_memory.storage.memory_store import InMemoryStorage


async def _seed_brain(
    *, indexed: int, organic: int, edges_per_organic: int
) -> tuple[InMemoryStorage, str]:
    """Build a brain with `indexed` code-index neurons + `organic` organic ones.

    Only organic neurons carry semantic synapses (mirrors the real encoder: all
    code-index edges are indexed-indexed within one file, see codebase_encoder.py).
    """
    store = InMemoryStorage()
    brain = Brain.create(name="k2-structural", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for i in range(indexed):
        await store.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT,
                content=f"file:{i}",
                neuron_id=f"idx-{i}",
                metadata={"indexed": True},
            )
        )

    for i in range(organic):
        await store.add_neuron(
            Neuron.create(type=NeuronType.ENTITY, content=f"mem-{i}", neuron_id=f"org-{i}")
        )

    sids: set[str] = set()
    if organic > 0:
        for i in range(edges_per_organic * organic):
            src = f"org-{i % organic}"
            tgt = f"org-{(i + 1) % organic}"
            sid = f"e-{i}"
            await store.add_synapse(
                Synapse.create(
                    source_id=src,
                    target_id=tgt,
                    type=SynapseType.RELATED_TO,
                    weight=0.6,
                    synapse_id=sid,
                )
            )
            sids.add(sid)

    await store.add_fiber(
        Fiber.create(
            neuron_ids={f"org-{i}" for i in range(organic)} | {f"idx-{i}" for i in range(indexed)},
            synapse_ids=sids,
            anchor_neuron_id="org-0" if organic else "idx-0",
            fiber_id="f-0",
        )
    )
    return store, brain.id


@pytest_asyncio.fixture
async def mixed_brain() -> tuple[InMemoryStorage, str]:
    # 20 indexed + 20 organic, organic graph at 3 edges/neuron.
    return await _seed_brain(indexed=20, organic=20, edges_per_organic=3)


class TestStructuralNeuronCountReported:
    @pytest.mark.asyncio
    async def test_stats_exposes_structural_count(
        self, mixed_brain: tuple[InMemoryStorage, str]
    ) -> None:
        store, brain_id = mixed_brain
        stats = await store.get_stats(brain_id)

        assert stats["neuron_count"] == 40
        assert stats["structural_neuron_count"] == 20

    @pytest.mark.asyncio
    async def test_enhanced_stats_inherits_structural_count(
        self, mixed_brain: tuple[InMemoryStorage, str]
    ) -> None:
        store, brain_id = mixed_brain
        enhanced = await store.get_enhanced_stats(brain_id)

        assert enhanced["neuron_count"] == 40
        assert enhanced["structural_neuron_count"] == 20

    @pytest.mark.asyncio
    async def test_report_carries_connectivity_neuron_count(
        self, mixed_brain: tuple[InMemoryStorage, str]
    ) -> None:
        store, brain_id = mixed_brain
        report = await DiagnosticsEngine(store).analyze(brain_id)

        assert report.neuron_count == 40
        assert report.structural_neuron_count == 20
        assert report.connectivity_neuron_count == 20


class TestConnectivityOnOrganicSubgraph:
    @pytest.mark.asyncio
    async def test_indexed_neurons_do_not_deflate_connectivity(
        self, mixed_brain: tuple[InMemoryStorage, str]
    ) -> None:
        """Adding 20 structural leaves must not move the score vs an all-organic brain."""
        store, mixed_id = mixed_brain
        organic_store, organic_id = await _seed_brain(indexed=0, organic=20, edges_per_organic=3)

        mixed_report = await DiagnosticsEngine(store).analyze(mixed_id)
        organic_report = await DiagnosticsEngine(organic_store).analyze(organic_id)

        assert mixed_report.connectivity == pytest.approx(organic_report.connectivity, rel=1e-6)

    @pytest.mark.asyncio
    async def test_pure_indexed_brain_is_not_divide_by_zero(
        self,
    ) -> None:
        """connectivity_neuron_count == 0 must not raise; connectivity is 0."""
        store, brain_id = await _seed_brain(indexed=10, organic=0, edges_per_organic=0)
        report = await DiagnosticsEngine(store).analyze(brain_id)

        assert report.connectivity_neuron_count == 0
        assert report.connectivity == 0.0

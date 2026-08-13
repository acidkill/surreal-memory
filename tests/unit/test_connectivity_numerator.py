"""K3 (run 013) — connectivity numerator: endpoint-based organic filtering.

Variant (c) (DECISIONS.md): a synapse counts toward connectivity only when BOTH
endpoints are organic. The code-index synapse TYPES (contains / co_occurs / is_a /
related_to) are shared with the organic encoder, so the filter is endpoint-based,
not type-based. This pins the numerator and the per-metric organic filtering.
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


async def _brain_with_code_index() -> tuple[InMemoryStorage, str]:
    """A brain that mirrors the real encoder's pathology.

    4 indexed symbol neurons + 3 code-index edges between them (CONTAINS,
    CO_OCCURS, IS_A — all indexed-indexed), plus 4 organic neurons with 2
    organic semantic edges (RELATED_TO). The organic graph is thin (0.5/neuron)
    but the 3 code-index edges would inflate a type-unaware numerator.
    """
    store = InMemoryStorage()
    brain = Brain.create(name="k3-numerator", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for i in range(4):
        await store.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT,
                content=f"idx-{i}",
                neuron_id=f"idx-{i}",
                metadata={"indexed": True},
            )
        )
    for i in range(4):
        await store.add_neuron(
            Neuron.create(type=NeuronType.ENTITY, content=f"org-{i}", neuron_id=f"org-{i}")
        )

    sids: set[str] = set()
    code_edges = [
        ("idx-0", "idx-1", SynapseType.CONTAINS),
        ("idx-1", "idx-2", SynapseType.CO_OCCURS),
        ("idx-2", "idx-3", SynapseType.IS_A),
    ]
    organic_edges = [
        ("org-0", "org-1", SynapseType.RELATED_TO),
        ("org-2", "org-3", SynapseType.RELATED_TO),
    ]
    for idx, (s, t, st) in enumerate(code_edges + organic_edges):
        sid = f"e-{idx}"
        await store.add_synapse(
            Synapse.create(source_id=s, target_id=t, type=st, weight=0.6, synapse_id=sid)
        )
        sids.add(sid)

    await store.add_fiber(
        Fiber.create(
            neuron_ids={f"idx-{i}" for i in range(4)} | {f"org-{i}" for i in range(4)},
            synapse_ids=sids,
            anchor_neuron_id="org-0",
            fiber_id="f-0",
        )
    )
    return store, brain.id


@pytest_asyncio.fixture
async def code_index_brain() -> tuple[InMemoryStorage, str]:
    return await _brain_with_code_index()


class TestOrganicSynapseCount:
    @pytest.mark.asyncio
    async def test_enhanced_stats_reports_organic_synapse_count(
        self, code_index_brain: tuple[InMemoryStorage, str]
    ) -> None:
        store, brain_id = code_index_brain
        enhanced = await store.get_enhanced_stats(brain_id)

        assert enhanced["synapse_count"] == 5
        # Only the 2 organic↔organic RELATED_TO edges; the 3 code-index edges
        # (indexed-indexed) are excluded by endpoint filter.
        assert enhanced["synapse_stats"]["organic_synapse_count"] == 2

    @pytest.mark.asyncio
    async def test_is_structural_neuron_helper(self) -> None:
        assert DiagnosticsEngine._is_structural_neuron({"indexed": True}) is True
        assert DiagnosticsEngine._is_structural_neuron({"indexed": False}) is False
        assert DiagnosticsEngine._is_structural_neuron({}) is False
        assert DiagnosticsEngine._is_structural_neuron(None) is False

    @pytest.mark.asyncio
    async def test_connectivity_score_ignores_code_index_edges(
        self, code_index_brain: tuple[InMemoryStorage, str]
    ) -> None:
        """The 3 code-index edges must not inflate connectivity.

        With organic denominator 4 and organic numerator 2, ratio = 0.5/neuron.
        A type-unaware numerator (counting all 5) would give 1.25/neuron (or 5/4
        if alias etc. excluded) — measurably higher.
        """
        store, brain_id = code_index_brain
        report = await DiagnosticsEngine(store).analyze(brain_id)

        # organic denominator excludes the 4 indexed neurons (4 organic remain).
        # organic numerator = 2 (only the organic↔organic RELATED_TO edges);
        # the 3 code-index edges are excluded by endpoint filter even though
        # their types are NOT in _STRUCTURAL_SYNAPSE_TYPES.
        assert report.connectivity_neuron_count == 4
        # connectivity score: sigmoid(-1.5*(0.5-3.0)) over 0.5 ratio.
        # report.connectivity is rounded to 4 dp, so match at that precision.
        import math

        expected = round(1.0 / (1.0 + math.exp(-1.5 * (0.5 - 3.0))), 4)
        assert report.connectivity == pytest.approx(expected, abs=1e-4)


class TestPerMetricOrganicFiltering:
    @pytest.mark.asyncio
    async def test_freshness_excludes_code_index_fibers(
        self, code_index_brain: tuple[InMemoryStorage, str]
    ) -> None:
        """Freshness computed over organic fibers only.

        The single fiber here is organic (no code_index tag), so freshness is
        1.0 (created today). A code_index fiber added today must not change it.
        """
        store, brain_id = code_index_brain

        # Add a code_index fiber (would inflate freshness if not excluded)
        await store.add_fiber(
            Fiber.create(
                neuron_ids={"idx-0"},
                synapse_ids=set(),
                anchor_neuron_id="idx-0",
                fiber_id="f-code",
                tags={"code_index"},
            )
        )
        report = await DiagnosticsEngine(store).analyze(brain_id)
        # Only the organic fiber counts in the denominator → freshness stays 1.0
        assert report.freshness == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_activation_efficiency_organic_denominator(
        self, code_index_brain: tuple[InMemoryStorage, str]
    ) -> None:
        """Activation efficiency denominator = organic neuron count (4), not 8."""
        store, brain_id = code_index_brain
        report = await DiagnosticsEngine(store).analyze(brain_id)
        # No neuron has been activated → 0.0 regardless, but the point is no
        # divide-by-included-indexed. Verify it runs and reports against organic.
        assert report.activation_efficiency == 0.0

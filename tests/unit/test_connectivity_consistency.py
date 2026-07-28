"""One brain, one connectivity number — every surface must agree.

`smem health` scores connectivity on the semantic graph (alias dedup rows and
audit/provenance edges excluded, see ``DiagnosticsEngine._STRUCTURAL_SYNAPSE_TYPES``).
Four other surfaces used to divide the RAW synapse total by the neuron count, so the
same brain was simultaneously advertised as 13.5 synapses/neuron and 1.4 — and the
loudest number was the one that hid the problem.

This pins the agreement itself, not each formula separately: whatever the exclusion
set becomes, diagnostics / smem_stats hints / the maintenance pulse / topology
knowledge_density must keep returning the SAME ratio for the SAME brain.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.diagnostics import DiagnosticsEngine
from surreal_memory.engine.topology_analysis import compute_topology
from surreal_memory.mcp.maintenance_handler import (
    HealthPulse,
    MaintenanceHandler,
    _evaluate_thresholds,
)
from surreal_memory.mcp.stats_handler import StatsHandler
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import MaintenanceConfig, UnifiedConfig

# 20 neurons; 180 alias rows (dedup plumbing) + 20 real relations.
# Raw ratio 10.0/neuron, semantic ratio 1.0/neuron — a 10x gap, so any surface
# still counting the raw total is impossible to mistake for a rounding artefact.
NEURONS = 20
ALIAS_EDGES = 180
SEMANTIC_EDGES = 20
EXPECTED_RATIO = SEMANTIC_EDGES / NEURONS  # 1.0
RAW_RATIO = (ALIAS_EDGES + SEMANTIC_EDGES) / NEURONS  # 10.0


# ── Fixtures ─────────────────────────────────────────────────────


async def _build_brain() -> tuple[InMemoryStorage, str]:
    """A brain whose synapse table is 90% dedup plumbing."""
    store = InMemoryStorage()
    brain = Brain.create(name="connectivity-consistency", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for i in range(NEURONS):
        await store.add_neuron(
            Neuron.create(type=NeuronType.ENTITY, content=f"node-{i}", neuron_id=f"n-{i}")
        )

    synapse_ids: set[str] = set()
    edges: list[tuple[str, str, SynapseType]] = [
        (f"n-{i % NEURONS}", f"n-{(i + 1) % NEURONS}", SynapseType.ALIAS)
        for i in range(ALIAS_EDGES)
    ] + [
        (f"n-{i % NEURONS}", f"n-{(i + 3) % NEURONS}", SynapseType.RELATED_TO)
        for i in range(SEMANTIC_EDGES)
    ]
    for idx, (src, tgt, stype) in enumerate(edges):
        sid = f"s-{idx}"
        await store.add_synapse(
            Synapse.create(
                source_id=src,
                target_id=tgt,
                type=stype,
                weight=0.0 if stype is SynapseType.ALIAS else 0.6,
                synapse_id=sid,
            )
        )
        synapse_ids.add(sid)

    await store.add_fiber(
        Fiber.create(
            neuron_ids={f"n-{i}" for i in range(NEURONS)},
            synapse_ids=synapse_ids,
            anchor_neuron_id="n-0",
            fiber_id="f-0",
        )
    )
    return store, brain.id


@pytest_asyncio.fixture
async def brain() -> tuple[InMemoryStorage, str]:
    return await _build_brain()


class _FakeServer(MaintenanceHandler):
    """Minimal MCP server stub exposing the storage the pulse reads."""

    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage
        self.config = UnifiedConfig(maintenance=MaintenanceConfig())

    async def get_storage(self) -> InMemoryStorage:
        return self._storage

    async def _maybe_run_expiry_cleanup(self) -> int:
        return 0


# ── Per-surface ratios ───────────────────────────────────────────


async def _diagnostics_ratio(store: InMemoryStorage, brain_id: str) -> float:
    report = await DiagnosticsEngine(store).analyze(brain_id)
    return report.semantic_synapse_count / report.neuron_count


async def _stats_hint(store: InMemoryStorage, brain_id: str) -> str:
    """The smem_stats connectivity hint for this brain."""
    handler = StatsHandler.__new__(StatsHandler)
    stats = await store.get_enhanced_stats(brain_id)
    hints = await handler._generate_stats_hints(store, brain_id, stats)
    return next(h for h in hints if "connectivity" in h.lower())


async def _pulse(store: InMemoryStorage) -> HealthPulse:
    pulse = await _FakeServer(store)._health_pulse()
    assert pulse is not None
    return pulse


# ── The regression that matters ──────────────────────────────────


class TestAllSurfacesAgree:
    """Same brain, same ratio — whatever the exclusion set says."""

    @pytest.mark.asyncio
    async def test_every_surface_reports_the_semantic_ratio(
        self, brain: tuple[InMemoryStorage, str]
    ) -> None:
        store, brain_id = brain

        diagnostics = await _diagnostics_ratio(store, brain_id)
        pulse = await _pulse(store)
        topo = await compute_topology(store, brain_id)
        hint = await _stats_hint(store, brain_id)

        assert diagnostics == pytest.approx(EXPECTED_RATIO)
        assert pulse.connectivity == pytest.approx(EXPECTED_RATIO)
        assert topo.knowledge_density == pytest.approx(EXPECTED_RATIO)
        # The hint prints the same ratio it was computed from.
        assert f"{EXPECTED_RATIO:.1f} semantic synapses/neuron" in hint

    @pytest.mark.asyncio
    async def test_none_of_them_quote_the_raw_ratio(
        self, brain: tuple[InMemoryStorage, str]
    ) -> None:
        """10.0/neuron is the number that hid the thin graph."""
        store, brain_id = brain

        pulse = await _pulse(store)
        topo = await compute_topology(store, brain_id)
        hint = await _stats_hint(store, brain_id)

        assert pulse.connectivity != pytest.approx(RAW_RATIO)
        assert topo.knowledge_density != pytest.approx(RAW_RATIO)
        assert f"{RAW_RATIO:.1f}" not in hint

    @pytest.mark.asyncio
    async def test_raw_totals_are_still_reported_untouched(
        self, brain: tuple[InMemoryStorage, str]
    ) -> None:
        """Excluding structural edges from a RATIO must not hide them from counts."""
        store, brain_id = brain
        report = await DiagnosticsEngine(store).analyze(brain_id)
        pulse = await _pulse(store)

        assert report.synapse_count == ALIAS_EDGES + SEMANTIC_EDGES
        assert pulse.synapse_count == ALIAS_EDGES + SEMANTIC_EDGES
        assert pulse.semantic_synapse_count == SEMANTIC_EDGES

    @pytest.mark.asyncio
    async def test_a_semantic_brain_agrees_too(self) -> None:
        """The agreement is not an artefact of the alias-heavy shape."""
        store = InMemoryStorage()
        b = Brain.create(name="all-semantic", config=BrainConfig(), owner_id="test")
        await store.save_brain(b)
        store.set_brain(b.id)
        for i in range(NEURONS):
            await store.add_neuron(
                Neuron.create(type=NeuronType.ENTITY, content=f"node-{i}", neuron_id=f"n-{i}")
            )
        sids: set[str] = set()
        for i in range(SEMANTIC_EDGES):
            sid = f"s-{i}"
            await store.add_synapse(
                Synapse.create(
                    source_id=f"n-{i % NEURONS}",
                    target_id=f"n-{(i + 1) % NEURONS}",
                    type=SynapseType.CAUSED_BY,
                    weight=0.6,
                    synapse_id=sid,
                )
            )
            sids.add(sid)
        await store.add_fiber(
            Fiber.create(
                neuron_ids={f"n-{i}" for i in range(NEURONS)},
                synapse_ids=sids,
                anchor_neuron_id="n-0",
                fiber_id="f-0",
            )
        )

        diagnostics = await _diagnostics_ratio(store, b.id)
        pulse = await _pulse(store)
        topo = await compute_topology(store, b.id)

        assert diagnostics == pytest.approx(EXPECTED_RATIO)
        assert pulse.connectivity == pytest.approx(EXPECTED_RATIO)
        assert topo.knowledge_density == pytest.approx(EXPECTED_RATIO)


# ── Unit-consistent messaging ────────────────────────────────────


class TestHintsStateTheUnit:
    """A ratio and a 0-1 score must never be printed against the same target."""

    @pytest.mark.asyncio
    async def test_stats_hint_names_the_unit_and_the_exclusion(
        self, brain: tuple[InMemoryStorage, str]
    ) -> None:
        store, brain_id = brain
        hint = await _stats_hint(store, brain_id)

        assert "semantic synapses/neuron" in hint
        assert "3-8 per neuron" in hint
        assert "alias and audit edges excluded" in hint
        # "target: 3+" sat next to a number that could be a 0-1 score.
        assert "target: 3+" not in hint

    def test_maintenance_hint_names_the_unit_and_the_exclusion(self) -> None:
        hints = _evaluate_thresholds(
            fiber_count=1,
            neuron_count=NEURONS,
            synapse_count=ALIAS_EDGES + SEMANTIC_EDGES,
            connectivity=EXPECTED_RATIO,
            orphan_ratio=0.0,
            cfg=MaintenanceConfig(),
        )
        message = next(h.message for h in hints if "connectivity" in h.message.lower())

        assert "semantic synapses/neuron" in message
        assert "3-8 per neuron" in message
        assert "alias and audit edges excluded" in message
        # Alert routing keys off this prefix — keep it recognisable.
        assert message.startswith("Low connectivity (")


# ── Behaviour the exclusion is supposed to restore ───────────────


class TestSuppressedSignalsFireAgain:
    """Alias inflation used to silence the very hints this brain needs."""

    @pytest.mark.asyncio
    async def test_low_connectivity_hint_fires_on_an_alias_flooded_brain(
        self, brain: tuple[InMemoryStorage, str]
    ) -> None:
        """At the raw 10.0/neuron both thresholds (<2.0, <1.5) stayed silent."""
        store, brain_id = brain

        assert "connectivity" in (await _stats_hint(store, brain_id)).lower()
        pulse = await _pulse(store)
        assert any("connectivity" in h.message.lower() for h in pulse.hints)

    @pytest.mark.asyncio
    async def test_pulse_falls_back_to_the_raw_total_without_a_breakdown(
        self, brain: tuple[InMemoryStorage, str]
    ) -> None:
        """A backend that cannot break down by type must still produce a pulse.

        Over-reporting connectivity is the safe failure: it can only withhold a
        hint, never fabricate a low-connectivity alarm.
        """
        store, _ = brain

        async def _no_breakdown(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("backend has no per-type synapse stats")

        store.get_enhanced_stats = _no_breakdown  # type: ignore[method-assign]
        pulse = await _pulse(store)

        assert pulse.connectivity == pytest.approx(RAW_RATIO)
        assert pulse.semantic_synapse_count == ALIAS_EDGES + SEMANTIC_EDGES

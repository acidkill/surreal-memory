"""Connectivity must score the semantic graph, and say so in matching units.

Two regressions are pinned here:

1. `alias` rows are dedup plumbing (weight 0.0, one per repeated mention). On a real
   brain they were 89% of all synapses and pinned connectivity at its 1.0 ceiling
   while the semantic graph held 1.4 edges/neuron — the metric hid the weakness it
   exists to expose.
2. The 0-1 connectivity score was reported next to the raw "3-8 synapses/neuron"
   target, so a saturated 1.0 read as "under-connected".
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.diagnostics import DiagnosticsEngine, _build_dynamic_action
from surreal_memory.storage.memory_store import InMemoryStorage

# ── Helpers ──────────────────────────────────────────────────────


def _stats(**counts: int) -> dict[str, Any]:
    """Build a synapse_stats payload shaped like the storage backends emit."""
    return {"by_type": {stype: {"count": n} for stype, n in counts.items()}}


async def _brain_with(
    name: str,
    *,
    neuron_count: int,
    edges: list[tuple[str, str, SynapseType]],
) -> tuple[InMemoryStorage, str]:
    """Storage holding `neuron_count` neurons wired by `edges`, all in one fiber."""
    store = InMemoryStorage()
    brain = Brain.create(name=name, config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for i in range(neuron_count):
        await store.add_neuron(
            Neuron.create(type=NeuronType.ENTITY, content=f"node-{i}", neuron_id=f"n-{i}")
        )
    synapse_ids: set[str] = set()
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
            neuron_ids={f"n-{i}" for i in range(neuron_count)},
            synapse_ids=synapse_ids,
            anchor_neuron_id="n-0",
            fiber_id=f"f-{name}",
        )
    )
    return store, brain.id


@pytest_asyncio.fixture
async def alias_only_brain() -> tuple[InMemoryStorage, str]:
    """20 neurons joined exclusively by dedup `alias` edges — 10 edges/neuron of nothing."""
    edges = [(f"n-{i % 20}", f"n-{(i + 1) % 20}", SynapseType.ALIAS) for i in range(200)]
    return await _brain_with("alias-only", neuron_count=20, edges=edges)


# ── Structural exclusion ─────────────────────────────────────────


class TestSemanticSynapseCount:
    """`alias` and audit/provenance edges are not part of the semantic graph."""

    def test_alias_is_excluded(self) -> None:
        stats = _stats(alias=137871, co_occurs=4711, related_to=2069)
        total = 137871 + 4711 + 2069
        assert DiagnosticsEngine._count_semantic_synapses(total, stats) == 4711 + 2069

    def test_audit_and_provenance_edges_are_excluded(self) -> None:
        """They point at an agent/user/source record, not at another memory."""
        stats = _stats(
            related_to=100,
            stored_by=151,
            verified_at=7,
            approved_by=3,
            source_of=11,
        )
        assert DiagnosticsEngine._count_semantic_synapses(272, stats) == 100

    def test_similar_to_is_kept(self) -> None:
        """Embedding-derived, but it joins two content neurons and is traversed on recall."""
        stats = _stats(similar_to=4177, alias=100)
        assert DiagnosticsEngine._count_semantic_synapses(4277, stats) == 4177

    def test_table_structure_edges_are_kept(self) -> None:
        """in_row/in_column link content to content — real structure, not plumbing."""
        stats = _stats(in_row=40, in_column=83, has_value=166)
        assert DiagnosticsEngine._count_semantic_synapses(289, stats) == 289

    def test_member_name_keys_also_match(self) -> None:
        """Some fixtures key by enum member name rather than value."""
        stats = _stats(ALIAS=50, RELATED_TO=10)
        assert DiagnosticsEngine._count_semantic_synapses(60, stats) == 10

    def test_missing_breakdown_falls_back_to_total(self) -> None:
        """A possibly-inflated number beats silently claiming zero connectivity."""
        assert DiagnosticsEngine._count_semantic_synapses(500, {}) == 500

    def test_plain_int_counts_supported(self) -> None:
        """by_type entries are sometimes bare counts rather than dicts."""
        stats: dict[str, Any] = {"by_type": {"alias": 90, "related_to": 10}}
        assert DiagnosticsEngine._count_semantic_synapses(100, stats) == 10

    def test_never_negative(self) -> None:
        """Inconsistent totals must not produce a negative semantic count."""
        stats = _stats(alias=500)
        assert DiagnosticsEngine._count_semantic_synapses(100, stats) == 0


class TestConnectivityIgnoresPlumbing:
    """The score follows the semantic graph, not write volume."""

    def test_alias_flood_does_not_saturate_score(self) -> None:
        """The live brain's shape: 13.5 raw edges/neuron, 1.4 of them semantic."""
        stats = _stats(alias=137871, co_occurs=4711, similar_to=4177, related_to=2069)
        total, neurons = 148828, 11457

        inflated = DiagnosticsEngine._compute_connectivity(total, neurons)
        semantic = DiagnosticsEngine._count_semantic_synapses(total, stats)
        honest = DiagnosticsEngine._compute_connectivity(semantic, neurons)

        assert inflated > 0.99, "raw counting saturates the sigmoid"
        assert honest < 0.2, "the semantic graph is genuinely thin"

    @pytest.mark.asyncio
    async def test_alias_only_brain_scores_near_zero(
        self, alias_only_brain: tuple[InMemoryStorage, str]
    ) -> None:
        """10 alias edges/neuron is still an empty semantic graph."""
        store, brain_id = alias_only_brain
        report = await DiagnosticsEngine(store).analyze(brain_id)

        assert report.synapse_count == 200
        assert report.semantic_synapse_count == 0
        assert report.connectivity < 0.02
        assert "LOW_CONNECTIVITY" in {w.code for w in report.warnings}

    @pytest.mark.asyncio
    async def test_semantic_edges_still_count(self) -> None:
        """Same brain, real relations instead of alias rows — score recovers."""
        edges = [(f"n-{i % 20}", f"n-{(i + 1) % 20}", SynapseType.RELATED_TO) for i in range(200)]
        store, brain_id = await _brain_with("semantic", neuron_count=20, edges=edges)
        report = await DiagnosticsEngine(store).analyze(brain_id)

        assert report.semantic_synapse_count == 200
        assert report.connectivity > 0.99
        assert "LOW_CONNECTIVITY" not in {w.code for w in report.warnings}


# ── Unit-consistent messaging ────────────────────────────────────


class TestWarningUnitsMatch:
    """A 0-1 score must never be printed against the raw 3-8/neuron target."""

    def _warn(self, **overrides: Any) -> tuple[list[Any], list[str]]:
        engine = DiagnosticsEngine.__new__(DiagnosticsEngine)
        kwargs: dict[str, Any] = {
            "neuron_count": 100,
            "synapse_count": 1000,
            "fiber_count": 10,
            "raw_connectivity": 1.0,
            "semantic_synapse_count": 100,
            "connectivity_score": DiagnosticsEngine._compute_connectivity(100, 100),
            "synapse_stats": _stats(alias=900, related_to=60, co_occurs=40),
            "orphan_rate": 0.0,
            "consolidation_ratio": 1.0,
            "freshness": 1.0,
            "fibers": [],
        }
        kwargs.update(overrides)
        return engine._generate_diagnostics(**kwargs)

    def test_message_quotes_the_semantic_ratio_not_the_score(self) -> None:
        warnings, _ = self._warn()
        warning = next(w for w in warnings if w.code == "LOW_CONNECTIVITY")
        assert "1.0 semantic synapses/neuron" in warning.message
        # The normalised score (~0.05) must not be the number sitting next to "3-8".
        assert "0.05 synapses/neuron" not in warning.message
        assert "score" not in warning.message.lower()

    def test_target_is_stated_in_per_neuron_units(self) -> None:
        warnings, recommendations = self._warn()
        warning = next(w for w in warnings if w.code == "LOW_CONNECTIVITY")
        assert "per neuron" in warning.message
        assert any("per neuron" in r for r in recommendations)

    def test_score_travels_in_details_not_in_the_ratio_text(self) -> None:
        warnings, _ = self._warn()
        details = next(w for w in warnings if w.code == "LOW_CONNECTIVITY").details
        assert details["semantic_synapses_per_neuron"] == pytest.approx(1.0)
        assert details["semantic_synapse_count"] == 100
        assert details["structural_synapse_count"] == 900
        assert 0.0 <= details["connectivity_score"] <= 1.0
        assert details["connectivity_score"] < 0.5

    def test_recommended_gap_is_measured_against_semantic_edges(self) -> None:
        """Advice must not count the 900 alias rows as progress toward 3/neuron."""
        _, recommendations = self._warn()
        rec = next(r for r in recommendations if r.startswith("Low connectivity"))
        assert "~200 more connections" in rec

    def test_action_string_labels_both_units(self) -> None:
        action = _build_dynamic_action(
            "connectivity",
            "fallback",
            {"neuron_count": 100, "synapse_count": 1000, "semantic_synapse_count": 100},
        )
        assert "1.0 semantic synapses/neuron" in action
        assert "0-1 connectivity score" in action
        # The old text claimed a bare "target: 3.0+" beside a 0-1 rendered score.
        assert "target: 3.0+" not in action

    def test_action_falls_back_to_total_without_semantic_metric(self) -> None:
        """Older callers pass no semantic count; the string must still be coherent."""
        action = _build_dynamic_action(
            "connectivity", "fallback", {"neuron_count": 100, "synapse_count": 300}
        )
        assert "3.0 semantic synapses/neuron" in action


class TestLowDiversityThreshold:
    """LOW_DIVERSITY is about type coverage; a 17-type brain is not under-covered."""

    def _warn(self, synapse_stats: dict[str, Any], synapse_count: int) -> list[Any]:
        engine = DiagnosticsEngine.__new__(DiagnosticsEngine)
        warnings, _ = engine._generate_diagnostics(
            neuron_count=100,
            synapse_count=synapse_count,
            fiber_count=10,
            raw_connectivity=5.0,
            synapse_stats=synapse_stats,
            orphan_rate=0.0,
            consolidation_ratio=1.0,
            freshness=1.0,
            fibers=[],
        )
        return warnings

    def test_seventeen_types_does_not_fire(self) -> None:
        """Real brain distribution: skewed, but 17 distinct types are in use."""
        stats = _stats(
            alias=137871,
            co_occurs=4711,
            similar_to=4177,
            related_to=2069,
            happened_at=1976,
            after=1868,
            involves=1014,
            has_value=166,
            stored_by=151,
            in_column=83,
            effective_for=70,
            felt=41,
            used_with=25,
            before=20,
            caused_by=5,
            evolves_from=3,
            leads_to=2,
        )
        codes = {w.code for w in self._warn(stats, 154252)}
        assert "LOW_DIVERSITY" not in codes

    def test_three_types_does_not_fire(self) -> None:
        codes = {w.code for w in self._warn(_stats(alias=10, related_to=10, felt=10), 30)}
        assert "LOW_DIVERSITY" not in codes

    def test_two_types_fires(self) -> None:
        warnings = self._warn(_stats(related_to=50, co_occurs=50), 100)
        warning = next(w for w in warnings if w.code == "LOW_DIVERSITY")
        assert warning.details["types_used"] == 2
        # Counts on both sides of the comparison — never a 0-1 entropy score.
        assert "2 of 8" in warning.message

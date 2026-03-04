"""Invariant and property tests — catch definition drift and environment leaks.

These tests verify properties that should hold across ALL layers, not just
within a single module. They catch bugs that unit tests miss because unit
tests mirror implementation rather than challenging assumptions.

Three categories:
1. Environment invariants: test runs leave no trace in production
2. Cross-layer properties: definitions consistent across modules
3. Sanity smoke: health metrics match raw storage counts
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from neural_memory.engine.diagnostics import DiagnosticsEngine
from neural_memory.storage.memory_store import InMemoryStorage

# ── Shared fixtures ──────────────────────────────────────────────


@pytest_asyncio.fixture
async def mixed_storage() -> InMemoryStorage:
    """Brain with neurons in different connectivity states.

    Layout:
    - n-syn-a, n-syn-b: connected via synapse only (no fiber)
    - n-fib-a, n-fib-b, n-fib-c: in fiber only (no synapse)
    - n-both: in both synapse and fiber
    - n-anchor: fiber anchor, has synapse
    - n-orphan-1, n-orphan-2: truly orphaned (no synapse, no fiber)
    """
    store = InMemoryStorage()
    brain = Brain.create(name="mixed", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    neurons = [
        Neuron.create(type=NeuronType.ENTITY, content="syn-a", neuron_id="n-syn-a"),
        Neuron.create(type=NeuronType.ENTITY, content="syn-b", neuron_id="n-syn-b"),
        Neuron.create(type=NeuronType.SPATIAL, content="fib-a", neuron_id="n-fib-a"),
        Neuron.create(type=NeuronType.SPATIAL, content="fib-b", neuron_id="n-fib-b"),
        Neuron.create(type=NeuronType.SPATIAL, content="fib-c", neuron_id="n-fib-c"),
        Neuron.create(type=NeuronType.CONCEPT, content="both", neuron_id="n-both"),
        Neuron.create(type=NeuronType.ACTION, content="anchor", neuron_id="n-anchor"),
        Neuron.create(type=NeuronType.ENTITY, content="orphan-1", neuron_id="n-orphan-1"),
        Neuron.create(type=NeuronType.ENTITY, content="orphan-2", neuron_id="n-orphan-2"),
    ]
    for n in neurons:
        await store.add_neuron(n)

    synapses = [
        Synapse.create(
            source_id="n-syn-a",
            target_id="n-syn-b",
            type=SynapseType.RELATED_TO,
            weight=0.5,
            synapse_id="s-1",
        ),
        Synapse.create(
            source_id="n-both",
            target_id="n-anchor",
            type=SynapseType.INVOLVES,
            weight=0.7,
            synapse_id="s-2",
        ),
    ]
    for s in synapses:
        await store.add_synapse(s)

    fiber = Fiber.create(
        neuron_ids={"n-fib-a", "n-fib-b", "n-fib-c", "n-both", "n-anchor"},
        synapse_ids={"s-2"},
        anchor_neuron_id="n-anchor",
        fiber_id="f-1",
    )
    await store.add_fiber(fiber)

    return store


# ── 1. Cross-layer property: orphan definition consistency ───────


class TestOrphanDefinitionConsistency:
    """Verify that diagnostics and consolidation agree on what is an orphan.

    INVARIANT: If diagnostics says a neuron is NOT orphaned, consolidation
    must NOT prune it. If diagnostics says it IS orphaned, consolidation
    should prune it (given eligible conditions).
    """

    @pytest.mark.asyncio
    async def test_diagnostics_and_consolidation_agree_on_orphans(
        self, mixed_storage: InMemoryStorage
    ) -> None:
        """Both modules should identify the same set of orphan neurons."""
        brain_id = mixed_storage._current_brain_id

        # Get orphan rate from diagnostics
        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)
        diag_orphan_rate = report.orphan_rate

        # Get orphan count from consolidation (dry run)
        config = ConsolidationConfig(
            prune_min_inactive_days=0.0,
            prune_weight_threshold=1.0,  # Won't prune synapses
            prune_isolated_neurons=True,
        )
        engine = ConsolidationEngine(mixed_storage, config)
        consol_report = await engine.run(
            strategies=[ConsolidationStrategy.PRUNE],
            dry_run=True,
        )

        # Derive rate from consolidation for comparison
        consol_orphan_rate = consol_report.neurons_pruned / report.neuron_count

        # Both modules must agree on orphan rate
        assert diag_orphan_rate == pytest.approx(consol_orphan_rate, abs=0.01), (
            f"Diagnostics orphan_rate={diag_orphan_rate:.3f} != "
            f"consolidation rate={consol_orphan_rate:.3f}"
        )
        # And both should find exactly 2 orphans (n-orphan-1, n-orphan-2)
        assert consol_report.neurons_pruned == 2, (
            f"Consolidation found {consol_report.neurons_pruned} orphans, expected 2"
        )

    @pytest.mark.asyncio
    async def test_fiber_members_never_pruned(self, mixed_storage: InMemoryStorage) -> None:
        """Neurons in fibers must survive consolidation prune."""
        config = ConsolidationConfig(
            prune_min_inactive_days=0.0,
            prune_weight_threshold=1.0,
            prune_isolated_neurons=True,
        )
        engine = ConsolidationEngine(mixed_storage, config)
        await engine.run(
            strategies=[ConsolidationStrategy.PRUNE],
            dry_run=False,
        )

        # Fiber members should survive
        for nid in ["n-fib-a", "n-fib-b", "n-fib-c", "n-both", "n-anchor"]:
            neuron = await mixed_storage.get_neuron(nid)
            assert neuron is not None, f"Fiber member {nid} was incorrectly pruned"

        # Synapse-only neurons survive too
        for nid in ["n-syn-a", "n-syn-b"]:
            neuron = await mixed_storage.get_neuron(nid)
            assert neuron is not None, f"Synapse-linked {nid} was incorrectly pruned"

        # True orphans should be gone
        for nid in ["n-orphan-1", "n-orphan-2"]:
            neuron = await mixed_storage.get_neuron(nid)
            assert neuron is None, f"Orphan {nid} should have been pruned"

    @pytest.mark.asyncio
    async def test_synapse_only_neurons_not_orphaned(self, mixed_storage: InMemoryStorage) -> None:
        """Neurons connected only via synapses (not in fibers) are not orphans."""
        diag = DiagnosticsEngine(mixed_storage)
        brain_id = mixed_storage._current_brain_id
        report = await diag.analyze(brain_id)

        # 9 neurons, 7 connected (2 via synapse, 5 via fiber), 2 orphans
        expected_rate = 2 / 9
        assert report.orphan_rate == pytest.approx(expected_rate, abs=0.01)


# ── 2. Sanity smoke: health metrics match raw counts ─────────────


class TestHealthMetricsSanity:
    """Verify health report metrics match direct storage queries."""

    @pytest.mark.asyncio
    async def test_neuron_count_matches(self, mixed_storage: InMemoryStorage) -> None:
        """Health neuron_count must equal actual neuron count in storage."""
        brain_id = mixed_storage._current_brain_id
        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)

        actual = await mixed_storage.get_stats(brain_id)
        assert report.neuron_count == actual["neuron_count"]

    @pytest.mark.asyncio
    async def test_synapse_count_matches(self, mixed_storage: InMemoryStorage) -> None:
        """Health synapse_count must equal actual synapse count."""
        brain_id = mixed_storage._current_brain_id
        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)

        actual = await mixed_storage.get_stats(brain_id)
        assert report.synapse_count == actual["synapse_count"]

    @pytest.mark.asyncio
    async def test_fiber_count_matches(self, mixed_storage: InMemoryStorage) -> None:
        """Health fiber_count must equal actual fiber count."""
        brain_id = mixed_storage._current_brain_id
        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)

        actual = await mixed_storage.get_stats(brain_id)
        assert report.fiber_count == actual["fiber_count"]

    @pytest.mark.asyncio
    async def test_orphan_count_matches_raw_query(self, mixed_storage: InMemoryStorage) -> None:
        """Orphan count from health must match manual graph traversal."""
        brain_id = mixed_storage._current_brain_id
        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)

        # Manual count: walk synapses + fibers
        all_synapses = await mixed_storage.get_all_synapses()
        connected: set[str] = set()
        for s in all_synapses:
            connected.add(s.source_id)
            connected.add(s.target_id)

        fibers = await mixed_storage.get_fibers(limit=100000)
        for f in fibers:
            connected.update(f.neuron_ids)

        all_neurons = await mixed_storage.find_neurons(limit=100000)
        manual_orphan_count = sum(1 for n in all_neurons if n.id not in connected)
        manual_rate = manual_orphan_count / len(all_neurons) if all_neurons else 0.0

        assert report.orphan_rate == pytest.approx(manual_rate, abs=0.001)

    @pytest.mark.asyncio
    async def test_activation_efficiency_matches_raw(self, mixed_storage: InMemoryStorage) -> None:
        """Activation efficiency must match actual neuron state query."""
        brain_id = mixed_storage._current_brain_id

        # Activate some neurons to have non-zero efficiency
        for nid in ["n-syn-a", "n-both", "n-anchor"]:
            state = await mixed_storage.get_neuron_state(nid)
            if state:
                await mixed_storage.update_neuron_state(state.activate(0.5))

        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)

        # Manual: count neurons with access_frequency > 0
        all_neurons = await mixed_storage.find_neurons(limit=100000)
        activated = 0
        for n in all_neurons:
            state = await mixed_storage.get_neuron_state(n.id)
            if state and state.access_frequency > 0:
                activated += 1

        expected = activated / len(all_neurons) if all_neurons else 0.0
        assert report.activation_efficiency == pytest.approx(expected, abs=0.01)

    @pytest.mark.asyncio
    async def test_connectivity_matches_raw(self, mixed_storage: InMemoryStorage) -> None:
        """Connectivity should reflect actual synapse/neuron ratio."""
        brain_id = mixed_storage._current_brain_id
        diag = DiagnosticsEngine(mixed_storage)
        report = await diag.analyze(brain_id)

        stats = await mixed_storage.get_stats(brain_id)
        ratio = stats["synapse_count"] / max(stats["neuron_count"], 1)
        # Connectivity uses sigmoid: 1 - e^(-ratio/target)
        # Just verify it's > 0 when ratio > 0
        if ratio > 0:
            assert report.connectivity > 0.0


# ── 3. Prune safety: fiber integrity after consolidation ─────────


class TestPruneSafety:
    """Verify consolidation never breaks fiber integrity."""

    @pytest.mark.asyncio
    async def test_fiber_neuron_ids_intact_after_prune(
        self, mixed_storage: InMemoryStorage
    ) -> None:
        """All neuron_ids in a fiber must still exist after prune."""
        config = ConsolidationConfig(
            prune_min_inactive_days=0.0,
            prune_weight_threshold=1.0,
            prune_isolated_neurons=True,
        )
        engine = ConsolidationEngine(mixed_storage, config)
        await engine.run(
            strategies=[ConsolidationStrategy.PRUNE],
            dry_run=False,
        )

        fibers = await mixed_storage.get_fibers(limit=100000)
        for fiber in fibers:
            for nid in fiber.neuron_ids:
                neuron = await mixed_storage.get_neuron(nid)
                assert neuron is not None, (
                    f"Fiber {fiber.id} references neuron {nid} which was pruned"
                )

    @pytest.mark.asyncio
    async def test_health_improves_after_prune(self, mixed_storage: InMemoryStorage) -> None:
        """Orphan rate should decrease (or stay 0) after pruning orphans."""
        brain_id = mixed_storage._current_brain_id
        diag = DiagnosticsEngine(mixed_storage)

        before = await diag.analyze(brain_id)

        config = ConsolidationConfig(
            prune_min_inactive_days=0.0,
            prune_weight_threshold=1.0,
            prune_isolated_neurons=True,
        )
        engine = ConsolidationEngine(mixed_storage, config)
        await engine.run(
            strategies=[ConsolidationStrategy.PRUNE],
            dry_run=False,
        )

        after = await diag.analyze(brain_id)
        assert after.orphan_rate <= before.orphan_rate


# ── 4. Environment invariant: E2E test isolation ─────────────────


class TestE2EIsolation:
    """Verify E2E tests use isolated storage via NEURALMEMORY_DIR."""

    def test_e2e_client_fixture_uses_tmp_path(self) -> None:
        """The E2E client fixture must accept tmp_path and monkeypatch params."""
        import inspect

        from tests.e2e.test_api import client

        sig = inspect.signature(client)
        params = list(sig.parameters.keys())
        assert "tmp_path" in params, "E2E client fixture must use tmp_path for isolated storage"
        assert "monkeypatch" in params, (
            "E2E client fixture must use monkeypatch to override NEURALMEMORY_DIR"
        )

    def test_neuralmemory_dir_env_respected(self, tmp_path: object) -> None:
        """get_neuralmemory_dir() must respect NEURALMEMORY_DIR env var."""
        import os
        from pathlib import Path

        from neural_memory.unified_config import get_neuralmemory_dir

        original = os.environ.get("NEURALMEMORY_DIR")
        try:
            test_dir = str(tmp_path)
            os.environ["NEURALMEMORY_DIR"] = test_dir
            result = get_neuralmemory_dir()
            assert result == Path(test_dir).resolve()
        finally:
            if original is None:
                os.environ.pop("NEURALMEMORY_DIR", None)
            else:
                os.environ["NEURALMEMORY_DIR"] = original

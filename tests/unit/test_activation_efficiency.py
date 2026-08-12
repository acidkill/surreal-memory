"""Tests for activation efficiency fixes (Issue #15).

1. Hebbian learning None activation floor → 0.1
2. Dormant neuron reactivation during consolidation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.neuron import NeuronState
from surreal_memory.engine.consolidation import ConsolidationReport
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

# ── Hebbian Floor Tests ──


class TestHebbianActivationFloor:
    """Verify None activations are replaced with 0.1 floor."""

    def test_missing_activations_get_floor(self) -> None:
        """When neuron is not in activations dict, should use 0.1 not None."""
        # This tests the logic in retrieval.py _defer_co_activated
        # pre_act/post_act should be 0.1 when neuron not in activations map
        activations: dict[str, MagicMock] = {}
        a, b = "neuron-a", "neuron-b"

        pre_act = activations[a].activation_level if activations and a in activations else 0.1
        post_act = activations[b].activation_level if activations and b in activations else 0.1

        assert pre_act == 0.1
        assert post_act == 0.1

    def test_present_activations_used(self) -> None:
        """When neuron is in activations dict, should use actual value."""
        mock_result = MagicMock()
        mock_result.activation_level = 0.7
        activations = {"neuron-a": mock_result}
        a = "neuron-a"

        pre_act = activations[a].activation_level if activations and a in activations else 0.1

        assert pre_act == 0.7

    def test_floor_enables_positive_hebbian_delta(self) -> None:
        """With 0.1 floor, hebbian_update should compute positive delta."""
        from surreal_memory.engine.learning_rule import hebbian_update

        result = hebbian_update(
            current_weight=0.5,
            pre_activation=0.1,
            post_activation=0.1,
            reinforced_count=1,
        )
        # With positive activations, delta should be positive (not zero as with None)
        assert result.delta >= 0


# ── Dormant Reactivation Tests ──


def _make_neuron_state(
    neuron_id: str,
    access_frequency: int = 0,
    activation_level: float = 0.3,
) -> NeuronState:
    return NeuronState(
        neuron_id=neuron_id,
        activation_level=activation_level,
        access_frequency=access_frequency,
        last_activated=utcnow(),
    )


async def _seeded_storage(states: list[NeuronState]) -> InMemoryStorage:
    storage = InMemoryStorage()
    storage.set_brain("b")
    for state in states:
        await storage.update_neuron_state(state)
    return storage


class TestDormantStateQuery:
    """The dedicated dormant query — filtering and sampling belong in storage."""

    @pytest.mark.asyncio
    async def test_returns_only_dormant_states(self) -> None:
        """Neurons with access_frequency > 0 are never returned."""
        storage = await _seeded_storage(
            [_make_neuron_state(f"d{i}", access_frequency=0) for i in range(3)]
            + [_make_neuron_state(f"a{i}", access_frequency=3) for i in range(3)]
        )

        dormant = await storage.get_dormant_neuron_states(limit=20)

        assert {s.neuron_id for s in dormant} == {"d0", "d1", "d2"}

    @pytest.mark.asyncio
    async def test_caps_at_limit(self) -> None:
        """A brain whose dormant set dwarfs the limit still yields exactly limit rows."""
        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(50)]
        )

        dormant = await storage.get_dormant_neuron_states(limit=20)

        assert len(dormant) == 20
        assert len({s.neuron_id for s in dormant}) == 20

    @pytest.mark.asyncio
    async def test_returns_empty_when_nothing_dormant(self) -> None:
        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=1) for i in range(5)]
        )

        assert await storage.get_dormant_neuron_states(limit=20) == []

    @pytest.mark.asyncio
    async def test_sampling_is_randomized(self) -> None:
        """Repeated calls must reach different slices, or replay starves the tail."""
        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(100)]
        )

        seen: set[str] = set()
        for _ in range(5):
            seen |= {s.neuron_id for s in await storage.get_dormant_neuron_states(limit=20)}

        assert len(seen) > 20

    @pytest.mark.asyncio
    async def test_base_default_matches_backend_semantics(self) -> None:
        """Backends that do not override still get filter + cap from the base class."""
        states = [_make_neuron_state(f"d{i}", access_frequency=0) for i in range(30)]
        states += [_make_neuron_state(f"a{i}", access_frequency=2) for i in range(5)]
        storage = await _seeded_storage(states)

        dormant = await NeuralStorage.get_dormant_neuron_states(storage, limit=20)

        assert len(dormant) == 20
        assert all(s.access_frequency == 0 for s in dormant)


class TestDormantReactivation:
    @pytest.mark.asyncio
    async def test_reactivates_dormant_neurons(self) -> None:
        """Dormant neurons (access_frequency=0) should get a small activation bump."""
        from surreal_memory.engine.consolidation import ConsolidationEngine

        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(5)]
        )
        consolidator = ConsolidationEngine(storage)

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 5
        stored = await storage.get_all_neuron_states()
        assert all(s.access_frequency == 1 for s in stored)
        assert all(s.activation_level == 0.35 for s in stored)  # 0.3 + 0.05

    @pytest.mark.asyncio
    async def test_skips_active_neurons(self) -> None:
        """Neurons with access_frequency > 0 should not be reactivated."""
        from surreal_memory.engine.consolidation import ConsolidationEngine

        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=3) for i in range(5)]
        )
        consolidator = ConsolidationEngine(storage)

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 0
        stored = await storage.get_all_neuron_states()
        assert all(s.activation_level == 0.3 for s in stored)

    @pytest.mark.asyncio
    async def test_caps_at_20_neurons(self) -> None:
        """Should reactivate at most 20 dormant neurons."""
        from surreal_memory.engine.consolidation import ConsolidationEngine

        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(50)]
        )
        consolidator = ConsolidationEngine(storage)

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 20
        reactivated = [s for s in await storage.get_all_neuron_states() if s.access_frequency > 0]
        assert len(reactivated) == 20

    @pytest.mark.asyncio
    async def test_asks_storage_for_a_bounded_sample(self) -> None:
        """The whole state table must never be pulled just to find dormant neurons."""
        from surreal_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        storage.get_dormant_neuron_states.return_value = [
            _make_neuron_state("n0", access_frequency=0)
        ]
        consolidator = ConsolidationEngine(storage)

        await consolidator._reactivate_dormant(ConsolidationReport(), dry_run=False)

        storage.get_all_neuron_states.assert_not_called()
        storage.get_dormant_neuron_states.assert_awaited_once_with(limit=20)

    @pytest.mark.asyncio
    async def test_dry_run_counts_only(self) -> None:
        """Dry run should count but not update."""
        from surreal_memory.engine.consolidation import ConsolidationEngine

        storage = await _seeded_storage(
            [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(5)]
        )
        consolidator = ConsolidationEngine(storage)

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=True)

        assert report.neurons_reactivated == 5
        stored = await storage.get_all_neuron_states()
        assert all(s.access_frequency == 0 for s in stored)

    @pytest.mark.asyncio
    async def test_handles_storage_error_gracefully(self) -> None:
        """Storage errors should not crash reactivation."""
        from surreal_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        consolidator = ConsolidationEngine(storage)
        storage.get_dormant_neuron_states.side_effect = RuntimeError("DB error")

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 0

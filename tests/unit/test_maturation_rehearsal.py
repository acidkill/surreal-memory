"""Tests for maturation rehearsal during reinforcement (Issue #11)."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import NeuronState
from neural_memory.engine.lifecycle import ReinforcementManager
from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.utils.timeutils import utcnow


def _make_neuron_state(neuron_id: str) -> NeuronState:
    return NeuronState(
        neuron_id=neuron_id,
        activation_level=0.5,
        access_frequency=1,
        last_activated=utcnow(),
    )


def _make_fiber_stub(fiber_id: str) -> SimpleNamespace:
    """Minimal fiber-like object with just an id."""
    return SimpleNamespace(id=fiber_id)


def _make_maturation(fiber_id: str, brain_id: str = "test-brain") -> MaturationRecord:
    return MaturationRecord(
        fiber_id=fiber_id,
        brain_id=brain_id,
        stage=MemoryStage.EPISODIC,
        stage_entered_at=utcnow() - timedelta(days=10),
        rehearsal_count=0,
        reinforcement_timestamps=(),
    )


class TestMaturationRehearsalOnReinforce:
    @pytest.mark.asyncio
    async def test_reinforce_triggers_rehearsal(self) -> None:
        """Reinforcing neurons should rehearse maturation records of connected fibers."""
        storage = AsyncMock()
        storage.get_neuron_states_batch.return_value = {
            "n1": _make_neuron_state("n1"),
        }
        storage.find_fibers_batch.return_value = [_make_fiber_stub("f1")]
        record = _make_maturation("f1")
        storage.get_maturation.return_value = record

        mgr = ReinforcementManager()
        await mgr.reinforce(storage, ["n1"])

        storage.find_fibers_batch.assert_called_once()
        storage.get_maturation.assert_called_once_with("f1")
        storage.save_maturation.assert_called_once()
        saved = storage.save_maturation.call_args[0][0]
        assert saved.rehearsal_count == 1
        assert len(saved.reinforcement_timestamps) == 1

    @pytest.mark.asyncio
    async def test_rehearsal_accumulates_timestamps(self) -> None:
        """Multiple reinforcements should accumulate distinct timestamps."""
        storage = AsyncMock()
        storage.get_neuron_states_batch.return_value = {
            "n1": _make_neuron_state("n1"),
        }
        storage.find_fibers_batch.return_value = [_make_fiber_stub("f1")]

        # Start with 2 existing timestamps
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="test-brain",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=utcnow() - timedelta(days=10),
            rehearsal_count=2,
            reinforcement_timestamps=(
                (utcnow() - timedelta(days=3)).isoformat(),
                (utcnow() - timedelta(days=1)).isoformat(),
            ),
        )
        storage.get_maturation.return_value = record

        mgr = ReinforcementManager()
        await mgr.reinforce(storage, ["n1"])

        saved = storage.save_maturation.call_args[0][0]
        assert saved.rehearsal_count == 3
        assert len(saved.reinforcement_timestamps) == 3

    @pytest.mark.asyncio
    async def test_no_maturation_record_skips_gracefully(self) -> None:
        """If no maturation record exists, rehearsal is skipped without error."""
        storage = AsyncMock()
        storage.get_neuron_states_batch.return_value = {
            "n1": _make_neuron_state("n1"),
        }
        storage.find_fibers_batch.return_value = [_make_fiber_stub("f1")]
        storage.get_maturation.return_value = None

        mgr = ReinforcementManager()
        result = await mgr.reinforce(storage, ["n1"])

        assert result == 1  # neuron reinforced
        storage.save_maturation.assert_not_called()

    @pytest.mark.asyncio
    async def test_rehearsal_caps_fibers_at_10(self) -> None:
        """At most 10 fibers should be rehearsed per reinforce call."""
        storage = AsyncMock()
        states = {f"n{i}": _make_neuron_state(f"n{i}") for i in range(15)}
        storage.get_neuron_states_batch.return_value = states
        # Return 15 fibers
        fibers = [_make_fiber_stub(f"f{i}") for i in range(15)]
        storage.find_fibers_batch.return_value = fibers
        storage.get_maturation.return_value = _make_maturation("any")

        mgr = ReinforcementManager()
        await mgr.reinforce(storage, [f"n{i}" for i in range(15)])

        # Should cap at 10 fibers
        assert storage.get_maturation.call_count == 10
        assert storage.save_maturation.call_count == 10

    @pytest.mark.asyncio
    async def test_rehearsal_error_does_not_break_reinforce(self) -> None:
        """If maturation rehearsal fails, neuron reinforcement still succeeds."""
        storage = AsyncMock()
        storage.get_neuron_states_batch.return_value = {
            "n1": _make_neuron_state("n1"),
        }
        storage.find_fibers_batch.side_effect = RuntimeError("DB error")

        mgr = ReinforcementManager()
        result = await mgr.reinforce(storage, ["n1"])

        # Neuron was still reinforced despite maturation failure
        assert result == 1
        storage.update_neuron_states_batch.assert_called_once()


class TestMatureOrphanedRecords:
    """Tests for orphaned maturation record handling in consolidation."""

    @pytest.mark.asyncio
    async def test_mature_skips_orphaned_fk_error(self) -> None:
        """_mature should skip records that trigger FK constraint errors."""
        from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport

        storage = AsyncMock()
        storage.current_brain_id = "test-brain"
        storage.cleanup_orphaned_maturations.return_value = 0

        record = MaturationRecord(
            fiber_id="orphan-fiber",
            brain_id="test-brain",
            stage=MemoryStage.SHORT_TERM,
            stage_entered_at=utcnow() - timedelta(hours=1),
        )
        storage.find_maturations.return_value = [record]
        storage.save_maturation.side_effect = sqlite3.IntegrityError(
            "FOREIGN KEY constraint failed"
        )
        storage.get_fibers.return_value = []

        config = AsyncMock()
        config.summarize_min_cluster_size = 5
        config.summarize_tag_overlap_threshold = 0.5

        engine = ConsolidationEngine(storage, config)
        report = ConsolidationReport(started_at=utcnow())

        # Should not raise — orphaned FK errors are caught
        await engine._mature(report, utcnow(), dry_run=False)

    @pytest.mark.asyncio
    async def test_mature_calls_cleanup_first(self) -> None:
        """_mature should clean up orphaned records before processing."""
        from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport

        storage = AsyncMock()
        storage.current_brain_id = "test-brain"
        storage.cleanup_orphaned_maturations.return_value = 3
        storage.find_maturations.return_value = []
        storage.get_fibers.return_value = []

        config = AsyncMock()
        config.summarize_min_cluster_size = 5
        config.summarize_tag_overlap_threshold = 0.5

        engine = ConsolidationEngine(storage, config)
        report = ConsolidationReport(started_at=utcnow())

        await engine._mature(report, utcnow(), dry_run=False)

        storage.cleanup_orphaned_maturations.assert_called_once()

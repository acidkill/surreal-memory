"""Tests for maturation rehearsal during reinforcement (Issue #11)."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.neuron import NeuronState
from surreal_memory.engine.lifecycle import ReinforcementManager
from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.utils.timeutils import utcnow


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
    async def test_missing_maturation_record_is_created_on_reinforce(self) -> None:
        """A fiber with no maturation row gets one, seeded at SHORT_TERM.

        This used to be a silent skip. A fiber without a maturation row can
        never advance a stage, so skipping meant it stayed invisible to
        maturation forever -- measured at 874 of 2819 fibers on the live brain.
        Reinforcement is the one moment we know the fiber is alive, so that is
        where the row gets created.
        """
        from surreal_memory.engine.memory_stages import MemoryStage

        storage = AsyncMock()
        storage.get_neuron_states_batch.return_value = {
            "n1": _make_neuron_state("n1"),
        }
        storage.find_fibers_batch.return_value = [_make_fiber_stub("f1")]
        storage.get_maturation.return_value = None

        mgr = ReinforcementManager()
        result = await mgr.reinforce(storage, ["n1"])

        assert result == 1  # neuron reinforced
        storage.save_maturation.assert_called_once()
        saved = storage.save_maturation.call_args.args[0]
        assert saved.fiber_id == "f1"
        assert saved.stage == MemoryStage.SHORT_TERM
        # The recall that created the row also counts as its first rehearsal.
        assert saved.rehearsal_count == 1

    @pytest.mark.asyncio
    async def test_rehearsal_neuron_limit_still_caps_the_lookup(self) -> None:
        """The neuron fan-out into find_fibers_batch is still capped, just raised and configurable.

        Uncapping fiber rehearsal entirely could scale unbounded with brain
        size, so the guard now lives on the neuron side
        (``rehearsal_neuron_limit``, default 15 -- see
        ``BrainConfig.reinforcement_neuron_limit`` for why 15 and not a larger
        jump) instead of a second, redundant fixed-10 fiber cap.
        """
        storage = AsyncMock()
        states = {f"n{i}": _make_neuron_state(f"n{i}") for i in range(30)}
        storage.get_neuron_states_batch.return_value = states
        storage.find_fibers_batch.return_value = [_make_fiber_stub("f0")]
        storage.get_maturation.return_value = _make_maturation("f0")

        mgr = ReinforcementManager()  # default rehearsal_neuron_limit=15
        await mgr.reinforce(storage, [f"n{i}" for i in range(30)])

        called_neuron_ids = storage.find_fibers_batch.call_args.args[0]
        assert len(called_neuron_ids) == 15

    @pytest.mark.asyncio
    async def test_rehearsal_reaches_all_fibers_actually_hit_by_recall(self) -> None:
        """Every fiber recall actually surfaced should be rehearsed, not an arbitrary first 10.

        RUN-010 VALIDATE-FIRST (section B). `for fiber in fibers[:10]` truncates
        the dedup'd result of `find_fibers_batch` to a fixed first 10 regardless
        of how many distinct fibers recall actually surfaced, capping the
        EPISODIC -> SEMANTIC spacing gate's only rehearsal source at a size that
        does not scale with the brain or the recall. Must FAIL on origin/main.
        """
        storage = AsyncMock()
        states = {f"n{i}": _make_neuron_state(f"n{i}") for i in range(15)}
        storage.get_neuron_states_batch.return_value = states
        fibers = [_make_fiber_stub(f"f{i}") for i in range(15)]
        storage.find_fibers_batch.return_value = fibers
        storage.get_maturation.return_value = _make_maturation("any")

        mgr = ReinforcementManager()
        await mgr.reinforce(storage, [f"n{i}" for i in range(15)])

        assert storage.save_maturation.call_count == 15, (
            "all 15 fibers recall surfaced should be rehearsed, not an arbitrary first 10"
        )

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
        from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport

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
        from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport

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


class TestStagesAdvancedPerHop:
    """RUN-010 VALIDATE-FIRST (section D1): stages_advanced flattens 3 hops."""

    @pytest.mark.asyncio
    async def test_stage_advances_broken_out_by_hop(self) -> None:
        """The report must say which hop advanced, not just how many advanced in total.

        `report.stages_advanced` sums stm->working, working->episodic, and
        episodic->semantic into one counter, so "15 advanced" and "zero new
        semantic" are indistinguishable without reading raw maturation rows.
        Must FAIL on origin/main (no per-hop breakdown exists yet).
        """
        from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport

        now = utcnow()
        stm_record = MaturationRecord(
            fiber_id="f-stm",
            brain_id="test-brain",
            stage=MemoryStage.SHORT_TERM,
            stage_entered_at=now - timedelta(minutes=40),
        )
        working_record = MaturationRecord(
            fiber_id="f-working",
            brain_id="test-brain",
            stage=MemoryStage.WORKING,
            stage_entered_at=now - timedelta(hours=5),
        )
        episodic_record = MaturationRecord(
            fiber_id="f-episodic",
            brain_id="test-brain",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=now - timedelta(days=8),
            rehearsal_count=3,
            reinforcement_timestamps=(
                (now - timedelta(days=3)).isoformat(),
                (now - timedelta(days=2)).isoformat(),
                (now - timedelta(days=1)).isoformat(),
            ),
        )

        storage = AsyncMock()
        storage.current_brain_id = "test-brain"
        storage.cleanup_orphaned_maturations.return_value = 0
        storage.find_maturations.return_value = [stm_record, working_record, episodic_record]
        storage.get_fibers.return_value = []

        config = AsyncMock()
        config.summarize_min_cluster_size = 5
        config.summarize_tag_overlap_threshold = 0.5

        engine = ConsolidationEngine(storage, config)
        report = ConsolidationReport(started_at=now)

        await engine._mature(report, now, dry_run=False)

        assert report.stages_advanced == 3
        breakdown = report.extra["stage_transitions"]
        assert breakdown == {
            "stm_to_working": 1,
            "working_to_episodic": 1,
            "episodic_to_semantic": 1,
        }
        assert sum(breakdown.values()) == report.stages_advanced

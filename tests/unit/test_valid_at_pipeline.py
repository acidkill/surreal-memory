"""Regression: point-in-time recall (valid_at) through the REAL pipeline.

U3's valid_at recall tests mocked ReflexPipeline, so they never exercised the
pipeline's own event-time filter `_fiber_valid_at`. The Spectron demo surfaced the
bug: BuildFiberStep stamps every fiber with a ZERO-WIDTH time window
(time_start == time_end == write time), and `_fiber_valid_at` required
`time_start <= dt <= time_end`, so any valid_at other than that exact microsecond
excluded the fiber — making logical point-in-time recall impossible for the common
"no explicit event time" case. Fix: a zero-width window is treated as unbounded;
only real intervals (time_start < time_end) are event-time filtered.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline, _fiber_valid_at
from surreal_memory.storage.sqlite_store import SQLiteStorage

_T = datetime(2026, 1, 15, 12, 0, 0)


class TestFiberValidAtUnit:
    def _fiber(self, start: datetime | None, end: datetime | None) -> Fiber:
        return Fiber.create(
            neuron_ids={"n"},
            synapse_ids=set(),
            anchor_neuron_id="n",
            summary="x",
            time_start=start,
            time_end=end,
        )

    def test_zero_width_window_is_unbounded(self) -> None:
        f = self._fiber(_T, _T)  # zero-width (the BuildFiberStep default shape)
        assert _fiber_valid_at(f, _T - timedelta(days=30)) is True
        assert _fiber_valid_at(f, _T) is True
        assert _fiber_valid_at(f, _T + timedelta(days=30)) is True

    def test_real_interval_is_bounded(self) -> None:
        f = self._fiber(_T, _T + timedelta(days=100))
        assert _fiber_valid_at(f, _T + timedelta(days=50)) is True
        assert _fiber_valid_at(f, _T - timedelta(days=1)) is False  # before start
        assert _fiber_valid_at(f, _T + timedelta(days=200)) is False  # after end

    def test_open_bounds_are_unbounded(self) -> None:
        assert _fiber_valid_at(self._fiber(None, None), _T) is True
        assert _fiber_valid_at(self._fiber(_T, None), _T + timedelta(days=5)) is True
        assert _fiber_valid_at(self._fiber(None, _T), _T - timedelta(days=5)) is True


@pytest.fixture
async def storage() -> SQLiteStorage:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = SQLiteStorage(Path(tmpdir) / "test.db")
        await s.initialize()
        brain = Brain.create(name="valid_at_test")
        await s.save_brain(brain)
        s.set_brain(brain.id)
        yield s
        await s.close()


class TestValidAtThroughPipeline:
    async def test_zero_width_fiber_survives_valid_at_filter(self, storage: SQLiteStorage) -> None:
        # A fiber with a zero-width event window (as BuildFiberStep produces) must be
        # returned by a valid_at recall for a DIFFERENT time — the pre-fix bug dropped it.
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="Emma lives in Oslo Norway")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="Emma lives in Oslo the capital of Norway",
            time_start=_T,
            time_end=_T,  # zero-width, like every fact without an extracted event time
        )
        await storage.add_fiber(fiber)

        pipeline = ReflexPipeline(storage, BrainConfig())
        # valid_at a month before the write time — pre-fix this returned [] because the
        # zero-width window only matched the exact write microsecond.
        result = await pipeline.query("where does Emma live Oslo", valid_at=_T - timedelta(days=30))
        assert fiber.id in result.fibers_matched

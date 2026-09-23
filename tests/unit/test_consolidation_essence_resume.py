from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError


class _Progress:
    def __init__(
        self,
        *,
        pause_phase: str | None = None,
        pause_cursor: str | None = None,
    ) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"essence_backfill": {}}}
        self.pause_phase = pause_phase
        self.pause_cursor = pause_cursor
        self.writes: list[dict[str, Any]] = []

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return self.state["strategy_states"].setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        current = self.strategy_state(strategy)
        current.update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
            updated_at=datetime.now(UTC),
        )
        self.writes.append(dict(current))
        if phase == self.pause_phase and (self.pause_cursor is None or cursor == self.pause_cursor):
            self.pause_phase = None
            raise ConsolidationPausedError("simulated interruption")


class _Storage:
    def __init__(
        self,
        fibers: list[Fiber],
        *,
        fail_after_write: bool = False,
    ) -> None:
        self.fibers = {fiber.id: fiber for fiber in fibers}
        self.writes: list[str] = []
        self.fail_after_write = fail_after_write

    async def get_fibers(self, *, limit: int) -> list[Fiber]:
        # Match the real API's bounded result shape; engine applies stable ID order.
        return list(self.fibers.values())[:limit]

    async def get_neuron(self, neuron_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=neuron_id, content=f"content for {neuron_id}")

    async def update_fiber(self, fiber: Fiber) -> None:
        self.writes.append(fiber.id)
        self.fibers[fiber.id] = fiber
        if self.fail_after_write:
            self.fail_after_write = False
            raise RuntimeError("simulated crash after database application")


class _Generator:
    def __init__(self, result: str = "saved essence") -> None:
        self.result = result
        self.calls = 0

    async def generate(self, _content: str, *, priority: int) -> str:
        assert priority == 5
        self.calls += 1
        return self.result


def _fiber(fiber_id: str) -> Fiber:
    return Fiber.create(
        neuron_ids={f"neuron-{fiber_id}"},
        synapse_ids=set(),
        anchor_neuron_id=f"neuron-{fiber_id}",
        fiber_id=fiber_id,
    )


def _engine(storage: _Storage, progress: _Progress | None = None) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.ESSENCE_BACKFILL
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest.mark.asyncio
async def test_resume_applies_checkpointed_essence_without_another_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import fidelity

    storage = _Storage([_fiber("fiber-a")])
    progress = _Progress(pause_phase="essence_pending")
    generator = _Generator()
    monkeypatch.setattr(fidelity, "get_essence_generator", lambda _strategy: generator)
    engine = _engine(storage, progress)

    # If the process dies before the result checkpoint, the completed external
    # LLM call can repeat. Once pending is saved, resume must not call it again.
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._essence_backfill(ConsolidationReport(), dry_run=False)

    assert generator.calls == 1
    assert storage.writes == []
    pending_state = progress.strategy_state("essence_backfill")
    assert pending_state["phase"] == "essence_pending"
    assert pending_state["cursor"] is None
    assert pending_state["pending"] == ['{"fiber_id":"fiber-a","essence":"saved essence"}']

    report = ConsolidationReport()
    await engine._essence_backfill(report, dry_run=False)

    assert generator.calls == 1
    assert storage.writes == ["fiber-a"]
    assert storage.fibers["fiber-a"].essence == "saved essence"
    assert report.essences_generated == 1
    assert progress.strategy_state("essence_backfill")["pending"] == []


@pytest.mark.asyncio
async def test_resume_does_not_repeat_db_write_after_applied_unit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import fidelity

    storage = _Storage([_fiber("fiber-a")])
    progress = _Progress(pause_phase="essence_scan", pause_cursor="fiber-a")
    generator = _Generator()
    monkeypatch.setattr(fidelity, "get_essence_generator", lambda _strategy: generator)
    engine = _engine(storage, progress)

    # Interruption after the idempotent write and its cursor checkpoint models a
    # process that must resume without either another LLM call or another write.
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._essence_backfill(ConsolidationReport(), dry_run=False)

    assert storage.writes == ["fiber-a"]
    report = ConsolidationReport()
    await engine._essence_backfill(report, dry_run=False)

    assert generator.calls == 1
    assert storage.writes == ["fiber-a"]
    assert report.essences_generated == 1


@pytest.mark.asyncio
async def test_dry_run_never_writes_storage_or_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import fidelity

    storage = _Storage([_fiber("fiber-a")])
    progress = _Progress()
    generator = _Generator()
    monkeypatch.setattr(fidelity, "get_essence_generator", lambda _strategy: generator)
    engine = _engine(storage, progress)

    report = ConsolidationReport()
    await engine._essence_backfill(report, dry_run=True)

    assert generator.calls == 1
    assert storage.writes == []
    assert progress.writes == []
    assert report.essences_generated == 1


@pytest.mark.asyncio
async def test_pending_result_replay_skips_write_when_db_applied_before_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import fidelity

    storage = _Storage([_fiber("fiber-a")], fail_after_write=True)
    progress = _Progress()
    generator = _Generator()
    monkeypatch.setattr(fidelity, "get_essence_generator", lambda _strategy: generator)
    engine = _engine(storage, progress)

    with pytest.raises(RuntimeError, match="after database application"):
        await engine._essence_backfill(ConsolidationReport(), dry_run=False)

    assert generator.calls == 1
    assert storage.writes == ["fiber-a"]
    assert storage.fibers["fiber-a"].essence == "saved essence"
    assert progress.strategy_state("essence_backfill")["phase"] == "essence_pending"

    report = ConsolidationReport()
    await engine._essence_backfill(report, dry_run=False)

    assert generator.calls == 1
    assert storage.writes == ["fiber-a"]
    assert report.essences_generated == 1
    assert progress.strategy_state("essence_backfill")["pending"] == []

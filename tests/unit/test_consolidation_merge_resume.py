from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage

REFERENCE_TIME = datetime(2026, 9, 20, 12, 30)


class _SimulatedCrash(BaseException):
    """Model process death after a storage write but before its checkpoint."""


class _Progress:
    def __init__(self, pause_phase: str | None = None, pause_occurrence: int = 1) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"merge": {}}}
        self.pause_phase = pause_phase
        self.pause_occurrence = pause_occurrence
        self.phase_counts: dict[str, int] = {}
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
        state = self.strategy_state(strategy)
        state.update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
            updated_at=REFERENCE_TIME,
        )
        self.writes.append(dict(state))
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        if phase == self.pause_phase and self.phase_counts[phase] == self.pause_occurrence:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated merge budget exhaustion")


class _FiberStorage:
    current_brain_id = "merge-brain"

    def __init__(self, *, typed: bool = True, matured: bool = True) -> None:
        self.fibers = {fiber.id: fiber for fiber in _sources()}
        self.typed_memories: dict[str, TypedMemory] = {}
        self.maturations: dict[str, MaturationRecord] = {}
        self.crash_after: str | None = None
        if typed:
            self.typed_memories = {
                fiber_id: TypedMemory(
                    fiber_id=fiber_id,
                    memory_type=MemoryType.FACT,
                    priority=Priority.HIGH if fiber_id.endswith("0") else Priority.NORMAL,
                )
                for fiber_id in self.fibers
            }
        if matured:
            self.maturations = {
                fiber_id: MaturationRecord(
                    fiber_id=fiber_id,
                    brain_id=self.current_brain_id,
                    stage=MemoryStage.EPISODIC,
                    stage_entered_at=REFERENCE_TIME - timedelta(days=4),
                    rehearsal_count=1,
                    reinforcement_timestamps=((REFERENCE_TIME - timedelta(days=3)).isoformat(),),
                )
                for fiber_id in self.fibers
            }

    def _crash_if(self, operation: str) -> None:
        if self.crash_after == operation:
            self.crash_after = None
            raise _SimulatedCrash(f"process stopped after {operation}")

    async def get_fibers(self, limit: int = 10, **_kwargs: Any) -> list[Fiber]:
        return sorted(self.fibers.values(), key=lambda fiber: fiber.id)[:limit]

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        return self.fibers.get(fiber_id)

    async def add_fiber(self, fiber: Fiber) -> str:
        self.fibers[fiber.id] = fiber
        self._crash_if("add_fiber")
        return fiber.id

    async def delete_fiber(self, fiber_id: str) -> bool:
        deleted = self.fibers.pop(fiber_id, None) is not None
        self._crash_if("delete_fiber")
        return deleted

    async def get_typed_memories_batch(self, fiber_ids: list[str]) -> dict[str, TypedMemory]:
        return {
            fiber_id: self.typed_memories[fiber_id]
            for fiber_id in fiber_ids
            if fiber_id in self.typed_memories
        }

    async def add_typed_memory(self, record: TypedMemory) -> None:
        self.typed_memories[record.fiber_id] = record
        self._crash_if("add_typed_memory")

    async def delete_typed_memory(self, fiber_id: str) -> None:
        self.typed_memories.pop(fiber_id, None)

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        return self.maturations.get(fiber_id)

    async def save_maturation(self, record: MaturationRecord) -> None:
        self.maturations[record.fiber_id] = record
        self._crash_if("save_maturation")


def _sources() -> list[Fiber]:
    shared = {"n-1", "n-2", "n-3"}
    return [
        Fiber(
            id=f"src-{index}",
            neuron_ids=set(shared),
            synapse_ids={f"edge-{index}"},
            anchor_neuron_id=f"n-{index + 1}",
            pathway=[f"n-{index + 1}"],
            summary=f"summary {index}",
            salience=0.6 + index * 0.1,
            frequency=index + 1,
            auto_tags={f"auto:{index}"},
            agent_tags={f"agent:{index}"},
            metadata={"source": index},
            created_at=REFERENCE_TIME - timedelta(minutes=index),
        )
        for index in range(2)
    ]


def _engine(storage: _FiberStorage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.MERGE
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


async def _resume_and_assert_complete(engine: ConsolidationEngine, storage: _FiberStorage) -> None:
    report = ConsolidationReport()
    await engine._merge(report, dry_run=False)

    assert {fiber.id for fiber in await storage.get_fibers(limit=100)} == {
        next(iter(storage.fibers))
    }
    successor = next(iter(storage.fibers.values()))
    assert successor.metadata["merged_from"] == ["src-0", "src-1"]
    assert report.fibers_merged == 2
    assert report.fibers_created == 1
    assert report.fibers_removed == 2
    assert await storage.get_typed_memories_batch(["src-0", "src-1"]) == {}
    assert (await storage.get_typed_memories_batch([successor.id])).get(successor.id) is not None
    assert await storage.get_maturation(successor.id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "occurrence"),
    [
        ("merge_pending", 1),
        ("merge_fiber_created", 1),
        ("merge_typed_reassigned", 1),
        ("merge_maturation_transferred", 1),
        ("merge_deleting_sources", 1),
        ("merge_deleting_sources", 2),
        ("merge_deleting_sources", 3),
        ("merge_scan", 1),
    ],
)
async def test_merge_resumes_after_each_committed_phase(phase: str, occurrence: int) -> None:
    storage = _FiberStorage()
    progress = _Progress(phase, occurrence)
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated merge budget"):
        await engine._merge(ConsolidationReport(), dry_run=False)

    assert progress.strategy_state("merge")["phase"] == phase
    progress.pause_phase = None
    await _resume_and_assert_complete(engine, storage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_phase"),
    [
        ("add_fiber", "merge_pending"),
        ("add_typed_memory", "merge_fiber_created"),
        ("save_maturation", "merge_typed_reassigned"),
        ("delete_fiber", "merge_deleting_sources"),
    ],
)
async def test_merge_retries_storage_write_completed_before_checkpoint(
    operation: str, expected_phase: str
) -> None:
    storage = _FiberStorage()
    storage.crash_after = operation
    progress = _Progress()
    engine = _engine(storage, progress)

    with pytest.raises(_SimulatedCrash, match=f"after {operation}"):
        await engine._merge(ConsolidationReport(), dry_run=False)

    assert progress.strategy_state("merge")["phase"] == expected_phase
    await _resume_and_assert_complete(engine, storage)


@pytest.mark.asyncio
async def test_merge_refuses_to_use_source_changed_after_pending_checkpoint() -> None:
    storage = _FiberStorage()
    progress = _Progress("merge_pending")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError):
        await engine._merge(ConsolidationReport(), dry_run=False)

    storage.fibers["src-0"] = replace(storage.fibers["src-0"], summary="changed while paused")
    with pytest.raises(RuntimeError, match="changed after its checkpoint"):
        await engine._merge(ConsolidationReport(), dry_run=False)

    assert "src-0" in storage.fibers and "src-1" in storage.fibers
    assert not any(f.metadata.get("merged_from") for f in storage.fibers.values())


@pytest.mark.asyncio
async def test_merge_dry_run_never_writes_graph_or_checkpoint() -> None:
    storage = _FiberStorage()
    progress = _Progress("merge_pending")
    engine = _engine(storage, progress)
    report = ConsolidationReport()

    await engine._merge(report, dry_run=True)

    assert set(storage.fibers) == {"src-0", "src-1"}
    assert set(storage.typed_memories) == {"src-0", "src-1"}
    assert set(storage.maturations) == {"src-0", "src-1"}
    assert progress.writes == []
    assert report.fibers_merged == 2
    assert report.fibers_created == 1
    assert report.fibers_removed == 0


def _large_disjoint_merge_groups() -> list[Fiber]:
    """Twelve independent postings exceed the former 50,000-pair ceiling."""
    return [
        Fiber(
            id=f"candidate-{group:02d}-{member:03d}",
            neuron_ids={f"neuron-{group:02d}"},
            synapse_ids=set(),
            anchor_neuron_id=f"neuron-{group:02d}",
            pathway=[f"neuron-{group:02d}"],
            created_at=REFERENCE_TIME,
        )
        for group in range(12)
        for member in range(100)
    ]


@pytest.mark.asyncio
async def test_merge_scans_every_posting_after_fifty_thousand_candidate_pairs() -> None:
    storage = _FiberStorage(typed=False, matured=False)
    engine = _engine(storage, _Progress())
    fibers = _large_disjoint_merge_groups()
    storage.fibers = {fiber.id: fiber for fiber in fibers}
    report = ConsolidationReport()
    await engine._merge(report, dry_run=True)

    assert report.fibers_merged == 1200
    assert report.fibers_created == 12
    assert {detail.original_fiber_ids[0][:12] for detail in report.merge_details} == {
        f"candidate-{group:02d}" for group in range(12)
    }


@pytest.mark.asyncio
async def test_merge_candidate_scan_deadline_fails_before_any_graph_write() -> None:
    storage = _FiberStorage(typed=False, matured=False)
    progress = _Progress()
    engine = _engine(storage, progress)
    engine._strategy_deadline = 0.0
    fibers = _large_disjoint_merge_groups()

    async def all_fibers(*, created_before: datetime | None = None) -> list[Fiber]:
        return fibers

    engine._all_fibers_paged = all_fibers  # type: ignore[method-assign]
    report = ConsolidationReport()
    with pytest.raises(ConsolidationPausedError, match="budget is down"):
        await engine._merge(report, dry_run=False)

    assert report.fibers_created == 0
    assert progress.strategy_state("merge")["phase"] == "merge_candidate_scan"
    assert not any(fiber.metadata.get("merged_from") for fiber in storage.fibers.values())


@pytest.mark.asyncio
async def test_merge_resumes_candidate_scan_after_fifty_thousand_pairs() -> None:
    import json

    storage = _FiberStorage(typed=False, matured=False)
    storage.fibers = {fiber.id: fiber for fiber in _large_disjoint_merge_groups()}
    progress = _Progress("merge_candidate_scan", 52)
    engine = _engine(storage, progress)

    async def all_fibers(*, created_before: datetime | None = None) -> list[Fiber]:
        return list(reversed(storage.fibers.values()))

    engine._all_fibers_paged = all_fibers  # type: ignore[method-assign]
    with pytest.raises(ConsolidationPausedError, match="simulated merge"):
        await engine._merge(ConsolidationReport(), dry_run=False)

    saved = progress.strategy_state("merge")
    scan = json.loads(saved["cursor"])
    assert saved["phase"] == "merge_candidate_scan"
    assert scan["pair_checks"] >= 51_000
    assert len(scan["parents"]) == 1200
    assert len(storage.fibers) == 1200

    progress.pause_phase = None
    report = ConsolidationReport()
    await engine._merge(report, dry_run=False)

    assert report.fibers_merged == 1200
    assert report.fibers_created == 12
    assert report.fibers_removed == 1200
    assert len(storage.fibers) == 12
    assert progress.strategy_state("merge")["phase"] == "merge_complete"
    assert len(progress.phase_counts) >= 3


@pytest.mark.asyncio
async def test_merge_retries_crash_in_first_planned_group_then_finishes_remaining() -> None:
    storage = _FiberStorage(typed=False, matured=False)
    storage.fibers = {fiber.id: fiber for fiber in _large_disjoint_merge_groups()}
    storage.crash_after = "add_fiber"
    progress = _Progress()
    engine = _engine(storage, progress)

    async def all_fibers(*, created_before: datetime | None = None) -> list[Fiber]:
        return list(storage.fibers.values())

    engine._all_fibers_paged = all_fibers  # type: ignore[method-assign]
    with pytest.raises(_SimulatedCrash, match="after add_fiber"):
        await engine._merge(ConsolidationReport(), dry_run=False)

    assert progress.strategy_state("merge")["phase"] == "merge_pending"
    report = ConsolidationReport()
    await engine._merge(report, dry_run=False)

    assert report.fibers_merged == 1200
    assert report.fibers_created == 12
    assert report.fibers_removed == 1200
    assert len(storage.fibers) == 12
    assert all(fiber.metadata.get("merged_from") for fiber in storage.fibers.values())


@pytest.mark.asyncio
async def test_merge_refuses_changed_fiber_during_checkpointed_candidate_scan() -> None:
    storage = _FiberStorage(typed=False, matured=False)
    storage.fibers = {fiber.id: fiber for fiber in _large_disjoint_merge_groups()}
    progress = _Progress("merge_candidate_scan", 2)
    engine = _engine(storage, progress)

    async def all_fibers(*, created_before: datetime | None = None) -> list[Fiber]:
        return list(storage.fibers.values())

    engine._all_fibers_paged = all_fibers  # type: ignore[method-assign]
    with pytest.raises(ConsolidationPausedError):
        await engine._merge(ConsolidationReport(), dry_run=False)

    original = storage.fibers["candidate-00-000"]
    storage.fibers[original.id] = replace(original, summary="changed after checkpoint")
    with pytest.raises(RuntimeError, match="frozen fiber snapshot changed"):
        await engine._merge(ConsolidationReport(), dry_run=False)

    assert progress.strategy_state("merge")["phase"] == "merge_candidate_scan"
    assert not any(fiber.metadata.get("merged_from") for fiber in storage.fibers.values())


@pytest.mark.asyncio
async def test_merge_resumes_next_planned_group_without_repeating_completed_one() -> None:
    import json

    storage = _FiberStorage(typed=False, matured=False)
    storage.fibers = {fiber.id: fiber for fiber in _large_disjoint_merge_groups()}
    progress = _Progress("merge_scan", 2)
    engine = _engine(storage, progress)

    async def all_fibers(*, created_before: datetime | None = None) -> list[Fiber]:
        return list(storage.fibers.values())

    engine._all_fibers_paged = all_fibers  # type: ignore[method-assign]
    with pytest.raises(ConsolidationPausedError):
        await engine._merge(ConsolidationReport(), dry_run=False)

    plan = json.loads(progress.strategy_state("merge")["cursor"])
    assert plan["next_index"] == 1
    assert len(storage.fibers) == 1101
    first_successor = next(
        fiber for fiber in storage.fibers.values() if fiber.metadata.get("merged_from")
    )

    report = ConsolidationReport()
    await engine._merge(report, dry_run=False)
    assert first_successor.id in storage.fibers
    assert len(storage.fibers) == 12
    assert report.fibers_merged == 1200
    assert report.fibers_created == 12
    assert report.fibers_removed == 1200

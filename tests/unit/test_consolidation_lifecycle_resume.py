from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

REFERENCE_TIME = datetime(2026, 9, 20, 12, 30)


class _Progress:
    def __init__(self, pause_after: int | None = None) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"lifecycle": {}}}
        self.pause_after = pause_after
        self.checkpoints = 0

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
        self.strategy_state(strategy).update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )
        self.checkpoints += 1
        if self.pause_after == self.checkpoints:
            raise ConsolidationPausedError("test budget exhausted")


def _neuron(neuron_id: str, *, age_days: int = 200) -> Neuron:
    return Neuron(
        id=neuron_id,
        content=f"memory {neuron_id}",
        type=NeuronType.CONCEPT,
        created_at=REFERENCE_TIME - timedelta(days=age_days),
        metadata={"lifecycle_state": "active", "priority": 0},
    )


class _Storage:
    current_brain_id = "default"

    def __init__(self, neurons: list[Neuron]) -> None:
        self.neurons = {neuron.id: neuron for neuron in neurons}
        self.updates: list[tuple[str, str]] = []
        self.change_on_refetch: str | None = None
        self.changed = False

    async def find_neurons_after_id(
        self,
        cursor_id: str | None,
        *,
        limit: int,
        created_before: datetime,
        ephemeral: bool | None,
        include_embedding: bool,
    ) -> list[Neuron]:
        assert include_embedding is False
        rows = sorted(self.neurons.values(), key=lambda item: item.id)
        if cursor_id is not None:
            rows = [row for row in rows if row.id > cursor_id]
        rows = [row for row in rows if row.created_at <= created_before]
        return rows[:limit]

    async def find_neurons_by_ids(
        self, neuron_ids: list[str], *, include_embedding: bool = False
    ) -> list[Neuron]:
        if self.change_on_refetch and not self.changed and self.change_on_refetch in neuron_ids:
            target = self.change_on_refetch
            self.neurons[target] = _neuron(target, age_days=1)
            self.changed = True
        return [self.neurons[nid] for nid in neuron_ids if nid in self.neurons]

    async def get_all_neuron_states(self) -> list[NeuronState]:
        return []

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        return {}

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        self.updates.append((neuron_id, lifecycle_state))
        neuron = self.neurons[neuron_id]
        self.neurons[neuron_id] = Neuron(
            id=neuron.id,
            content=neuron.content,
            type=neuron.type,
            created_at=neuron.created_at,
            metadata={**neuron.metadata, "lifecycle_state": lifecycle_state},
        )


def _engine(storage: _Storage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage)
    engine._active_strategy = ConsolidationStrategy.LIFECYCLE
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest.mark.asyncio
async def test_lifecycle_resumes_after_last_committed_neuron() -> None:
    storage = _Storage([_neuron("n-1"), _neuron("n-2")])
    progress = _Progress(pause_after=1)
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="test budget"):
        await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)

    assert storage.updates == [("n-1", "archived")]
    assert progress.strategy_state("lifecycle")["cursor"] == "n-1"

    progress.pause_after = None
    report = ConsolidationReport()
    await engine._lifecycle(report, REFERENCE_TIME, dry_run=False)

    assert storage.updates == [("n-1", "archived"), ("n-2", "archived")]
    assert progress.strategy_state("lifecycle")["cursor"] == "n-2"
    assert report.extra["lifecycle_states_updated"] == 2


@pytest.mark.asyncio
async def test_lifecycle_revalidates_changed_neuron_before_updating() -> None:
    storage = _Storage([_neuron("n-1")])
    storage.change_on_refetch = "n-1"
    progress = _Progress()
    engine = _engine(storage, progress)

    await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)

    assert storage.changed
    assert storage.updates == []
    assert progress.strategy_state("lifecycle")["cursor"] == "n-1"


@pytest.mark.asyncio
async def test_lifecycle_dry_run_does_not_write_or_checkpoint() -> None:
    storage = _Storage([_neuron("n-1")])
    progress = _Progress()
    engine = _engine(storage, progress)
    report = ConsolidationReport()

    await engine._lifecycle(report, REFERENCE_TIME, dry_run=True)

    assert report.extra["lifecycle_states_updated"] == 1
    assert storage.updates == []
    assert progress.checkpoints == 0


@pytest.mark.asyncio
async def test_lifecycle_scans_beyond_ten_thousand_without_singleton_reads_for_noops() -> None:
    class CountingStorage(_Storage):
        def __init__(self, neurons: list[Neuron]) -> None:
            super().__init__(neurons)
            self.singleton_reads = 0
            self.pages = 0

        async def find_neurons_after_id(
            self,
            cursor_id: str | None,
            *,
            limit: int,
            created_before: datetime,
            ephemeral: bool | None,
            include_embedding: bool,
        ) -> list[Neuron]:
            self.pages += 1
            return await super().find_neurons_after_id(
                cursor_id,
                limit=limit,
                created_before=created_before,
                ephemeral=ephemeral,
                include_embedding=include_embedding,
            )

        async def find_neurons_by_ids(
            self, neuron_ids: list[str], *, include_embedding: bool = False
        ) -> list[Neuron]:
            self.singleton_reads += 1
            return await super().find_neurons_by_ids(
                neuron_ids, include_embedding=include_embedding
            )

    storage = CountingStorage(
        [_neuron(f"n-{i:05d}").with_metadata(lifecycle_state="archived") for i in range(10_001)]
    )
    progress = _Progress()
    await _engine(storage, progress)._lifecycle(
        ConsolidationReport(), REFERENCE_TIME, dry_run=False
    )

    assert progress.strategy_state("lifecycle")["cursor"] == "n-10000"
    assert progress.checkpoints == 21
    assert storage.singleton_reads == 0
    assert storage.pages == 21
    assert storage.updates == []


@pytest.mark.asyncio
async def test_lifecycle_noop_pages_survive_two_restarts() -> None:
    storage = _Storage(
        [_neuron(f"n-{i:04d}").with_metadata(lifecycle_state="archived") for i in range(1_201)]
    )
    progress = _Progress(pause_after=1)
    engine = _engine(storage, progress)

    for checkpoint, expected_cursor in ((1, "n-0499"), (2, "n-0999")):
        with pytest.raises(ConsolidationPausedError, match="test budget"):
            await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)
        assert progress.checkpoints == checkpoint
        assert progress.strategy_state("lifecycle")["cursor"] == expected_cursor
        progress.pause_after = checkpoint + 1

    progress.pause_after = None
    await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)
    assert progress.strategy_state("lifecycle")["cursor"] == "n-1200"
    assert progress.checkpoints == 3
    assert storage.updates == []


@pytest.mark.asyncio
async def test_lifecycle_failed_update_remains_pending_and_retries_once() -> None:
    class FailingStorage(_Storage):
        def __init__(self, neurons: list[Neuron]) -> None:
            super().__init__(neurons)
            self.fail_once = True

        async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
            if neuron_id == "n-1" and self.fail_once:
                self.fail_once = False
                raise OSError("temporary write failure")
            await super().update_neuron_lifecycle(neuron_id, lifecycle_state)

    storage = FailingStorage([_neuron("n-1"), _neuron("n-2")])
    progress = _Progress()
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="1 neuron update"):
        await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)

    assert progress.strategy_state("lifecycle")["pending"] == ["n-1"]
    assert progress.strategy_state("lifecycle")["cursor"] == "n-2"
    assert storage.updates == [("n-2", "archived")]

    report = ConsolidationReport()
    await engine._lifecycle(report, REFERENCE_TIME, dry_run=False)

    assert storage.updates == [("n-2", "archived"), ("n-1", "archived")]
    assert progress.strategy_state("lifecycle")["pending"] == []
    assert report.extra["lifecycle_states_updated"] == 2


@pytest.mark.asyncio
async def test_lifecycle_write_before_failed_checkpoint_is_not_duplicated() -> None:
    class FailingProgress(_Progress):
        fail_before_save = True

        async def checkpoint(
            self,
            strategy: str,
            phase: str,
            *,
            cursor: str | None = None,
            pending: list[str] | None = None,
            counters: dict[str, int | float] | None = None,
        ) -> None:
            if self.fail_before_save:
                self.fail_before_save = False
                raise OSError("temporary checkpoint failure")
            await super().checkpoint(
                strategy, phase, cursor=cursor, pending=pending, counters=counters
            )

    storage = _Storage([_neuron("n-1")])
    progress = FailingProgress()
    engine = _engine(storage, progress)

    with pytest.raises(OSError, match="temporary checkpoint"):
        await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)
    assert storage.updates == [("n-1", "archived")]
    assert progress.checkpoints == 0

    await engine._lifecycle(ConsolidationReport(), REFERENCE_TIME, dry_run=False)
    assert storage.updates == [("n-1", "archived")]
    assert progress.strategy_state("lifecycle")["cursor"] == "n-1"

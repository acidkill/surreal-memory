from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from tests.unit.test_consolidation_group_plan import _PlanStorage


class _Progress:
    def __init__(self, pause_phase: str | None = None) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"infer": {}}}
        self.pause_phase = pause_phase
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
        )
        self.writes.append(dict(current))
        if phase == self.pause_phase:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated interruption")


class _Storage:
    def __init__(self, synapses: list[Synapse] | None = None) -> None:
        self.synapses = {synapse.id: synapse for synapse in synapses or []}
        self.add_calls: list[Synapse] = []
        self.update_calls: list[Synapse] = []
        self.counts = [("neuron-a", "neuron-b", 4, 0.8)]
        self.prune_calls = 0

    async def get_co_activation_counts(self, *, since: datetime, min_count: int):
        return [item for item in self.counts if item[2] >= min_count]

    async def get_synapses(self, *, limit: int, offset: int):
        return list(self.synapses.values())[offset : offset + limit]

    async def add_synapse(self, synapse: Synapse) -> str:
        key = {synapse.source_id, synapse.target_id}
        if any({item.source_id, item.target_id} == key for item in self.synapses.values()):
            raise ValueError("pair already exists")
        self.add_calls.append(synapse)
        self.synapses[synapse.id] = synapse
        return synapse.id

    async def update_synapse(self, synapse: Synapse) -> None:
        self.update_calls.append(synapse)
        self.synapses[synapse.id] = synapse

    async def get_neurons_batch(self, _neuron_ids: list[str]):
        return {}

    async def get_fibers(self, *, limit: int):
        return []

    async def prune_co_activations(self, *, older_than: datetime) -> int:
        self.prune_calls += 1
        return 0


class _DurableProgress(_Progress):
    def __init__(self, pause_phase: str | None = None) -> None:
        super().__init__(pause_phase)
        self.state["run_id"] = "infer-test-run"


class _DurableStorage(_Storage):
    def __init__(self, counts: list[tuple[str, str, int, float]]) -> None:
        super().__init__()
        self.counts = counts
        self.current_brain_id = "infer-test-brain"
        self.plan_storage = _PlanStorage()
        self.synapse_pair_probes: list[tuple[str, str]] = []
        self.prune_ids = ["co_activations:old-a", "co_activations:old-b"]
        self.pruned_ids: list[str] = []

    def _get_brain_id(self) -> str:
        return str(self.current_brain_id)

    async def _query(self, sql: str, **params: Any):
        return await self.plan_storage._query(sql, **params)

    async def iter_co_activation_counts(
        self,
        *,
        since: datetime,
        until: datetime,
        min_count: int = 1,
        after_pair: tuple[str, str] | None = None,
        page_size: int = 500,
    ):
        del since, until
        rows = sorted(self.counts, key=lambda row: (row[0], row[1]))
        for row in rows:
            if row[2] >= min_count and (after_pair is None or row[:2] > after_pair):
                yield row

    async def find_existing_synapse_pairs(self, pairs: list[tuple[str, str]]):
        self.synapse_pair_probes.extend(pairs)
        return {
            pair
            for pair in pairs
            if any(
                {synapse.source_id, synapse.target_id} == set(pair)
                for synapse in self.synapses.values()
            )
        }

    async def get_co_activation_prune_page(
        self, older_than: datetime, after_id: str | None = None, *, limit: int = 500
    ):
        del older_than
        remaining = [
            event_id
            for event_id in self.prune_ids
            if event_id not in self.pruned_ids and (after_id is None or event_id > after_id)
        ]
        return remaining[:limit]

    async def prune_co_activation_ids(self, event_ids: list[str]) -> int:
        fresh = [event_id for event_id in event_ids if event_id not in self.pruned_ids]
        self.pruned_ids.extend(fresh)
        return len(fresh)

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        *,
        limit: int,
        offset: int = 0,
    ):
        rows = [
            synapse
            for synapse in self.synapses.values()
            if (source_id is None or synapse.source_id == source_id)
            and (target_id is None or synapse.target_id == target_id)
        ]
        return rows[offset : offset + limit]

    async def get_synapse(self, synapse_id: str):
        return self.synapses.get(synapse_id)

    async def get_fiber(self, _fiber_id: str):
        return None


def _engine(storage: _Storage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(
        storage,
        ConsolidationConfig(infer_co_activation_threshold=3),
    )
    engine._active_strategy = ConsolidationStrategy.INFER
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


def _durable_engine(
    storage: _DurableStorage, progress: _DurableProgress, *, max_per_run: int = 50
) -> ConsolidationEngine:
    engine = ConsolidationEngine(
        storage,
        ConsolidationConfig(infer_co_activation_threshold=3, infer_max_per_run=max_per_run),
    )
    engine._active_strategy = ConsolidationStrategy.INFER
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


def _pending_operation(progress: _Progress) -> dict[str, Any]:
    pending = progress.strategy_state("infer")["pending"]
    for serialized in pending:
        item = json.loads(serialized)
        if item.get("kind") in {
            "infer_add_pending",
            "infer_reinforce_pending",
            "infer_prune_pending",
        }:
            return item
    raise AssertionError("pending operation snapshot was not saved")


@pytest.mark.asyncio
async def test_add_resumes_from_exact_pending_model_without_duplicate() -> None:
    storage = _Storage()
    progress = _Progress(pause_phase="infer_add_pending")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), datetime.now(UTC), dry_run=False)

    operation = _pending_operation(progress)
    assert operation["kind"] == "infer_add_pending"
    assert operation["pair_key"] == '["neuron-a","neuron-b"]'
    assert operation["synapse"]["source_id"] == "neuron-a"
    assert operation["synapse"]["target_id"] == "neuron-b"
    assert storage.add_calls == []

    report = ConsolidationReport()
    await engine._infer(report, datetime.now(UTC), dry_run=False)

    assert len(storage.add_calls) == 1
    assert report.synapses_inferred == 1
    inferred = next(iter(storage.synapses.values()))
    assert inferred.id == operation["synapse"]["id"]
    assert inferred.reinforced_count == 0
    assert storage.update_calls == []


@pytest.mark.asyncio
async def test_add_after_checkpoint_pause_does_not_add_or_reinforce_twice() -> None:
    storage = _Storage()
    progress = _Progress(pause_phase="infer_add")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), datetime.now(UTC), dry_run=False)

    assert len(storage.add_calls) == 1
    assert progress.strategy_state("infer")["pending"]  # manifest remains durable
    report = ConsolidationReport()
    await engine._infer(report, datetime.now(UTC), dry_run=False)

    assert len(storage.add_calls) == 1
    assert storage.update_calls == []
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_reinforcement_pending_snapshot_replays_without_second_delta() -> None:
    existing = Synapse.create(
        "neuron-a",
        "neuron-b",
        SynapseType.CO_OCCURS,
        weight=0.4,
    )
    storage = _Storage([existing])
    progress = _Progress(pause_phase="infer_reinforce_pending")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), datetime.now(UTC), dry_run=False)

    operation = _pending_operation(progress)
    snapshot = operation["synapse"]
    assert operation["kind"] == "infer_reinforce_pending"
    assert snapshot["reinforced_count"] == existing.reinforced_count + 1
    assert snapshot["weight"] > existing.weight
    assert storage.update_calls == []

    report = ConsolidationReport()
    await engine._infer(report, datetime.now(UTC), dry_run=False)

    assert len(storage.update_calls) == 1
    persisted = storage.synapses[existing.id]
    assert persisted.reinforced_count == existing.reinforced_count + 1
    assert persisted.weight == snapshot["weight"]
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_reinforcement_applied_before_pause_is_not_reapplied_on_resume() -> None:
    existing = Synapse.create(
        "neuron-a",
        "neuron-b",
        SynapseType.CO_OCCURS,
        weight=0.4,
    )
    storage = _Storage([existing])
    progress = _Progress(pause_phase="infer_reinforce")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), datetime.now(UTC), dry_run=False)

    assert len(storage.update_calls) == 1
    report = ConsolidationReport()
    await engine._infer(report, datetime.now(UTC), dry_run=False)

    assert len(storage.update_calls) == 1
    assert storage.synapses[existing.id].reinforced_count == 1
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_add_applied_before_timeout_replays_without_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage()
    progress = _Progress()
    engine = _engine(storage, progress)
    add = storage.add_synapse
    calls = 0

    async def add_then_timeout(synapse: Synapse) -> str:
        nonlocal calls
        calls += 1
        result = await add(synapse)
        if calls == 1:
            raise TimeoutError("write committed but response was lost")
        return result

    monkeypatch.setattr(storage, "add_synapse", add_then_timeout)

    with pytest.raises(TimeoutError, match="response was lost"):
        await engine._infer(ConsolidationReport(), datetime.now(UTC), dry_run=False)

    assert len(storage.synapses) == 1
    assert _pending_operation(progress)["kind"] == "infer_add_pending"

    report = ConsolidationReport()
    await engine._infer(report, datetime.now(UTC), dry_run=False)

    assert calls == 1
    assert len(storage.synapses) == 1
    assert storage.update_calls == []
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_reinforcement_applied_before_timeout_replays_exact_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = Synapse.create(
        "neuron-a",
        "neuron-b",
        SynapseType.CO_OCCURS,
        weight=0.4,
    )
    storage = _Storage([existing])
    progress = _Progress()
    engine = _engine(storage, progress)
    update = storage.update_synapse
    calls = 0

    async def update_then_timeout(synapse: Synapse) -> None:
        nonlocal calls
        calls += 1
        await update(synapse)
        if calls == 1:
            raise TimeoutError("write committed but response was lost")

    monkeypatch.setattr(storage, "update_synapse", update_then_timeout)

    with pytest.raises(TimeoutError, match="response was lost"):
        await engine._infer(ConsolidationReport(), datetime.now(UTC), dry_run=False)

    pending_snapshot = _pending_operation(progress)["synapse"]
    assert storage.synapses[existing.id].reinforced_count == existing.reinforced_count + 1

    report = ConsolidationReport()
    await engine._infer(report, datetime.now(UTC), dry_run=False)

    assert calls == 2
    assert storage.synapses[existing.id].reinforced_count == existing.reinforced_count + 1
    assert storage.synapses[existing.id].weight == pending_snapshot["weight"]
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_durable_infer_stages_and_resumes_without_duplicate_synapse_write() -> None:
    storage = _DurableStorage(
        [
            ("neuron-a", "neuron-b", 4, 0.7),
            ("neuron-a", "neuron-c", 9, 0.9),
        ]
    )
    progress = _DurableProgress(pause_phase="infer_add_pending")
    engine = _durable_engine(storage, progress, max_per_run=1)
    reference_time = datetime.now(UTC)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), reference_time, dry_run=False)

    staged = list(storage.plan_storage.candidates_by_id.values())
    assert len(staged) == 2
    assert storage.synapse_pair_probes == [("neuron-a", "neuron-b"), ("neuron-a", "neuron-c")]
    pending = _pending_operation(progress)
    assert pending["synapse"]["source_id"] == "neuron-a"
    assert pending["synapse"]["target_id"] == "neuron-c"
    assert storage.add_calls == []

    report = ConsolidationReport()
    await engine._infer(report, reference_time, dry_run=False)

    assert len(storage.add_calls) == 1
    assert storage.add_calls[0].target_id == "neuron-c"
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_durable_reinforcement_pending_replays_exactly_once() -> None:
    original = Synapse.create("neuron-a", "neuron-b", SynapseType.CO_OCCURS, weight=0.4)
    storage = _DurableStorage([("neuron-a", "neuron-b", 4, 0.8)])
    storage.synapses[original.id] = original
    progress = _DurableProgress(pause_phase="infer_reinforce_pending")
    engine = _durable_engine(storage, progress)
    reference_time = datetime.now(UTC)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), reference_time, dry_run=False)

    saved = _pending_operation(progress)["synapse"]
    assert storage.update_calls == []
    report = ConsolidationReport()
    await engine._infer(report, reference_time, dry_run=False)

    assert len(storage.update_calls) == 1
    assert storage.synapses[original.id].reinforced_count == original.reinforced_count + 1
    assert storage.synapses[original.id].weight == saved["weight"]
    assert report.synapses_inferred == 1


@pytest.mark.asyncio
async def test_durable_prune_batch_resumes_after_delete_without_second_write() -> None:
    storage = _DurableStorage([])
    progress = _DurableProgress(pause_phase="infer_prune")
    engine = _durable_engine(storage, progress)
    reference_time = datetime.now(UTC)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._infer(ConsolidationReport(), reference_time, dry_run=False)

    assert storage.pruned_ids == ["co_activations:old-a", "co_activations:old-b"]
    report = ConsolidationReport()
    await engine._infer(report, reference_time, dry_run=False)

    assert storage.pruned_ids == ["co_activations:old-a", "co_activations:old-b"]
    assert report.co_activations_pruned == 2

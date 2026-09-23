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


def _engine(storage: _Storage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(
        storage,
        ConsolidationConfig(infer_co_activation_threshold=3),
    )
    engine._active_strategy = ConsolidationStrategy.INFER
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


def _pending_operation(progress: _Progress) -> dict[str, Any]:
    pending = progress.strategy_state("infer")["pending"]
    for serialized in pending:
        item = json.loads(serialized)
        if item.get("kind") != "infer_manifest":
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

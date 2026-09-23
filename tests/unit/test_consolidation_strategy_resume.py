from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.dream import DreamResult
from surreal_memory.engine.hippocampal_replay import ReplayResult


class _Progress:
    def __init__(self, pause_phase: str | None = None) -> None:
        self.state: dict[str, Any] = {"strategy_states": {}}
        self.pause_phase = pause_phase
        self.writes: list[dict[str, Any]] = []
        self.checkpoint_hook: Any | None = None

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return cast("dict[str, Any]", self.state["strategy_states"].setdefault(strategy, {}))

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
        current.update(phase=phase, cursor=cursor)
        if pending is not None:
            current["pending"] = list(pending)
        else:
            current.pop("pending", None)
        if counters is not None:
            current["counters"] = dict(counters)
        self.writes.append(dict(current))
        if self.checkpoint_hook is not None:
            await self.checkpoint_hook(strategy, phase, cursor)
        if phase == self.pause_phase:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated process interruption")


class _Storage:
    def __init__(self, *, synapses: list[Synapse] | None = None) -> None:
        self.current_brain_id = "brain-test"
        self.brain = SimpleNamespace(config=SimpleNamespace())
        self.synapses = {synapse.id: synapse for synapse in synapses or []}
        self.neurons: dict[str, Neuron] = {}
        self.add_synapse_calls: list[Synapse] = []
        self.update_synapse_calls: list[Synapse] = []
        self.add_neuron_calls: list[Neuron] = []
        self.cancel_after_add = False
        self.cancel_after_neuron = False
        self.cancel_after_update = False

    async def get_brain(self, _brain_id: str) -> Any:
        return self.brain

    async def add_synapse(self, synapse: Synapse) -> str:
        self.add_synapse_calls.append(synapse)
        if any(
            current.source_id == synapse.source_id and current.target_id == synapse.target_id
            for current in self.synapses.values()
        ):
            raise ValueError("synapse pair already exists")
        self.synapses[synapse.id] = synapse
        if self.cancel_after_add:
            self.cancel_after_add = False
            raise asyncio.CancelledError()
        return synapse.id

    async def get_synapses(self, *, source_id: str | None = None, **_kwargs: Any) -> list[Synapse]:
        return [
            synapse
            for synapse in self.synapses.values()
            if source_id is None or synapse.source_id == source_id
        ]

    async def update_synapse(self, synapse: Synapse) -> None:
        self.update_synapse_calls.append(synapse)
        self.synapses[synapse.id] = synapse
        if self.cancel_after_update:
            self.cancel_after_update = False
            raise asyncio.CancelledError()

    async def find_neurons(self, *, type: NeuronType | None = None, **_kwargs: Any) -> list[Neuron]:
        return [neuron for neuron in self.neurons.values() if type is None or neuron.type == type]

    async def add_neuron(self, neuron: Neuron) -> str:
        self.add_neuron_calls.append(neuron)
        self.neurons[neuron.id] = neuron
        if self.cancel_after_neuron:
            self.cancel_after_neuron = False
            raise asyncio.CancelledError()
        return neuron.id


def _engine(
    strategy: ConsolidationStrategy,
    storage: _Storage,
    progress: _Progress,
) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, ConsolidationConfig())  # type: ignore[arg-type]
    engine._active_strategy = strategy
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest.mark.asyncio
async def test_dream_replays_exact_pending_synapse_after_write_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage()
    progress = _Progress()
    engine = _engine(ConsolidationStrategy.DREAM, storage, progress)
    synapse = Synapse.create("neuron-a", "neuron-b", SynapseType.RELATED_TO, weight=0.1)
    calls = 0

    async def fake_dream(*_args: Any, **_kwargs: Any) -> DreamResult:
        nonlocal calls
        calls += 1
        return DreamResult([synapse])

    monkeypatch.setattr("surreal_memory.engine.dream.dream", fake_dream)
    storage.cancel_after_add = True
    with pytest.raises(asyncio.CancelledError):
        await engine._dream(ConsolidationReport(), dry_run=False)

    report = ConsolidationReport()
    await engine._dream(report, dry_run=False)

    assert calls == 1
    assert list(storage.synapses) == [synapse.id]
    assert report.dream_synapses_created == 1
    assert progress.strategy_state("dream")["phase"] == "completed"


async def test_replay_replays_exact_pending_update_without_applying_weight_delta_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Synapse.create("source-a", "target-a", SynapseType.RELATED_TO, weight=0.5)
    storage = _Storage(synapses=[original])
    progress = _Progress()
    engine = _engine(ConsolidationStrategy.REPLAY, storage, progress)
    calls = 0

    async def fake_replay(target: Any, _config: Any, **_kwargs: Any) -> ReplayResult:
        nonlocal calls
        calls += 1
        observed = (await target.get_synapses(source_id=original.source_id))[0]
        await target.update_synapse(replace(observed, weight=0.6))
        replayed = (await target.get_synapses(source_id=original.source_id))[0]
        await target.update_synapse(replace(replayed, weight=0.7))
        return ReplayResult(episodes_replayed=1, synapses_strengthened=2)

    monkeypatch.setattr(
        "surreal_memory.engine.hippocampal_replay.hippocampal_replay",
        fake_replay,
    )
    storage.cancel_after_update = True
    with pytest.raises(asyncio.CancelledError):
        await engine._replay(ConsolidationReport(), dry_run=False)

    report = ConsolidationReport()
    await engine._replay(report, dry_run=False)

    assert calls == 1
    assert storage.synapses[original.id].weight == pytest.approx(0.7)
    assert len(storage.update_synapse_calls) == 2
    assert report.extra["replay_ltp"] == 2
    assert report.extra["replay_episodes"] == 1
    assert progress.strategy_state("replay")["phase"] == "completed"


@pytest.mark.asyncio
async def test_replay_revalidates_synapse_changed_after_plan_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Synapse.create("source-b", "target-b", SynapseType.RELATED_TO, weight=0.5)
    storage = _Storage(synapses=[original])
    progress = _Progress()
    engine = _engine(ConsolidationStrategy.REPLAY, storage, progress)

    async def fake_replay(target: Any, _config: Any, **_kwargs: Any) -> ReplayResult:
        observed = (await target.get_synapses(source_id=original.source_id))[0]
        await target.update_synapse(replace(observed, weight=0.6))
        return ReplayResult(episodes_replayed=1, synapses_strengthened=1)

    async def external_change(_strategy: str, phase: str, _cursor: str | None) -> None:
        if phase == "replay_pending":
            storage.synapses[original.id] = replace(original, weight=0.7)

    monkeypatch.setattr(
        "surreal_memory.engine.hippocampal_replay.hippocampal_replay",
        fake_replay,
    )
    progress.checkpoint_hook = external_change
    report = ConsolidationReport()
    await engine._replay(report, dry_run=False)

    assert storage.synapses[original.id].weight == pytest.approx(0.7)
    assert report.extra["replay_ltp"] == 0
    assert report.extra["replay_ltd"] == 0


@pytest.mark.asyncio
async def test_schema_replays_checkpointed_create_plan_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage()
    progress = _Progress()
    engine = _engine(ConsolidationStrategy.SCHEMA, storage, progress)
    calls = 0
    schema = Neuron.create(
        NeuronType.SCHEMA,
        "Schema: test — entities: common patterns (12 memories)",
        metadata={"tags": ["test"], "schema_version": 1, "cluster_size": 12},
    )

    async def fake_schema(target: Any, _config: Any, *, dry_run: bool = False) -> int:
        nonlocal calls
        calls += 1
        if dry_run:
            return 1
        await target.add_neuron(schema)
        return 1

    monkeypatch.setattr(
        "surreal_memory.engine.schema_assimilation.batch_schema_assimilation",
        fake_schema,
    )
    storage.cancel_after_neuron = True
    with pytest.raises(asyncio.CancelledError):
        await engine._schema(ConsolidationReport(), dry_run=False)

    report = ConsolidationReport()
    await engine._schema(report, dry_run=False)

    assert calls == 1
    assert list(storage.neurons) == [schema.id]
    assert report.extra["schemas_created"] == 1
    assert progress.strategy_state("schema")["phase"] == "completed"


@pytest.mark.asyncio
async def test_dry_run_does_not_write_strategy_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = _Storage()
    functions = {
        "surreal_memory.engine.dream.dream": lambda *_a, **_kw: DreamResult(),
        "surreal_memory.engine.hippocampal_replay.hippocampal_replay": (
            lambda *_a, **_kw: ReplayResult()
        ),
        "surreal_memory.engine.schema_assimilation.batch_schema_assimilation": (
            lambda *_a, **_kw: 0
        ),
        "surreal_memory.engine.interference.batch_interference_scan": (
            lambda *_a, **_kw: SimpleNamespace(fan_effects_flagged=0)
        ),
    }

    for target, function in functions.items():

        async def async_function(*args: Any, _function: Any = function, **kwargs: Any) -> Any:
            return _function(*args, **kwargs)

        monkeypatch.setattr(target, async_function)

    for strategy, method in (
        (ConsolidationStrategy.DREAM, "_dream"),
        (ConsolidationStrategy.REPLAY, "_replay"),
        (ConsolidationStrategy.SCHEMA, "_schema"),
        (ConsolidationStrategy.INTERFERENCE, "_interference"),
        (ConsolidationStrategy.LEARN_HABITS, "_learn_habits"),
    ):
        progress = _Progress()
        engine = _engine(strategy, storage, progress)
        report = ConsolidationReport()
        handler = getattr(engine, method)
        if strategy is ConsolidationStrategy.LEARN_HABITS:
            await handler(
                report, reference_time=__import__("datetime").datetime.now(), dry_run=True
            )
        else:
            await handler(report, dry_run=True)
        assert progress.writes == []


@pytest.mark.asyncio
async def test_interference_restores_completed_read_only_result_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage()
    progress = _Progress()
    progress.strategy_state("interference").update(
        phase="completed",
        counters={"interference_fan_effects": 6},
    )
    engine = _engine(ConsolidationStrategy.INTERFERENCE, storage, progress)

    async def unexpected_scan(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("completed interference strategy should not rescan")

    monkeypatch.setattr(
        "surreal_memory.engine.interference.batch_interference_scan",
        unexpected_scan,
    )
    report = ConsolidationReport()
    await engine._interference(report, dry_run=False)

    assert report.extra["interference_fan_effects"] == 6


@pytest.mark.asyncio
async def test_habit_learning_resumes_from_first_unfinished_mining_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage()
    progress = _Progress()
    progress.strategy_state("learn_habits").update(
        phase="habits_action_completed",
        counters={
            "habits_learned": 2,
            "action_events_pruned": 3,
            "query_patterns_learned": 4,
        },
    )
    engine = _engine(ConsolidationStrategy.LEARN_HABITS, storage, progress)
    calls: list[str] = []

    async def unexpected_action(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("action")
        raise AssertionError("completed action-log stage should not rerun")

    async def query_patterns(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("query")
        return SimpleNamespace(patterns_learned=5)

    async def tool_habits(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("tool")
        return [], SimpleNamespace(habits_learned=1)

    monkeypatch.setattr("surreal_memory.engine.sequence_mining.learn_habits", unexpected_action)
    monkeypatch.setattr(
        "surreal_memory.engine.query_pattern_mining.learn_query_patterns",
        query_patterns,
    )
    monkeypatch.setattr("surreal_memory.engine.sequence_mining.learn_tool_habits", tool_habits)

    report = ConsolidationReport()
    await engine._learn_habits(
        report,
        reference_time=__import__("datetime").datetime.now(),
        dry_run=False,
    )

    assert calls == ["query", "tool"]
    assert report.habits_learned == 3
    assert report.action_events_pruned == 3
    assert report.query_patterns_learned == 5
    assert progress.strategy_state("learn_habits")["phase"] == "completed"

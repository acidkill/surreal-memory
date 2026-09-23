from __future__ import annotations

from typing import Any, cast

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.query_pattern_mining import (
    QueryPatternCandidate,
    _query_candidate_manifest,
    learn_query_patterns,
)
from surreal_memory.engine.sequence_mining import (
    HabitCandidate,
    HabitReport,
    _materialize_habits,
    strengthen_sequential_pair,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


class _Progress:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {"run_id": "run-frozen", "strategy_states": {}}
        self.pause_candidate_checkpoint = True

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        states = cast("dict[str, dict[str, Any]]", self.state["strategy_states"])
        return states.setdefault(strategy, {})

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
        if (
            self.pause_candidate_checkpoint
            and phase == "habits_query_pending"
            and cursor == "candidate:1"
        ):
            self.pause_candidate_checkpoint = False
            raise ConsolidationPausedError("simulated crash before cursor persistence")
        current.update(phase=phase, cursor=cursor)
        if pending is not None:
            current["pending"] = list(pending)
        if counters is not None:
            current["counters"] = dict(counters)


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(
        name="habit-resume-test",
        config=BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0),
        owner_id="test",
    )
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


async def _seed_query_events(storage: InMemoryStorage, rounds: int = 5) -> None:
    for index in range(rounds):
        session = f"query-session-{index}"
        await storage.record_action(
            action_type="recall",
            action_context="authentication jwt tokens security",
            session_id=session,
        )
        await storage.record_action(
            action_type="recall",
            action_context="middleware express routing handlers",
            session_id=session,
        )


@pytest.mark.asyncio
async def test_materialize_habits_retry_after_fiber_write_is_idempotent(
    store: InMemoryStorage,
) -> None:
    candidate = HabitCandidate(
        steps=("open_file", "edit_file", "open_file", "edit_file"),
        frequency=3,
        avg_duration_seconds=4.0,
        confidence=1.0,
    )
    config = BrainConfig(reinforcement_delta=0.1)
    saved_manifest: list[dict[str, object]] | None = None

    async def lose_cursor(manifest: list[dict[str, object]], _cursor: int) -> None:
        nonlocal saved_manifest
        saved_manifest = manifest
        raise RuntimeError("simulated interruption after fiber write")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        await _materialize_habits(
            store,
            [candidate],
            config,
            HabitReport(),
            source="action_log",
            run_id="run-1",
            on_progress=lose_cursor,
        )

    assert saved_manifest is not None
    existing_fibers = await store.find_fibers(metadata_key="_habit_pattern", limit=20)
    assert len(existing_fibers) == 1
    synapses = await store.get_synapses(type=SynapseType.BEFORE)
    assert len(synapses) == 2
    before_by_edge = {(synapse.source_id, synapse.target_id): synapse for synapse in synapses}
    assert sorted(len(synapse.metadata["_habit_effect_ids"]) for synapse in synapses) == [1, 2]

    replayed = await _materialize_habits(
        store,
        [candidate],
        config,
        HabitReport(),
        source="action_log",
        run_id="run-1",
        start_cursor=0,
    )

    after_synapses = await store.get_synapses(type=SynapseType.BEFORE)
    after_by_edge = {(synapse.source_id, synapse.target_id): synapse for synapse in after_synapses}
    fibers_after = await store.find_fibers(metadata_key="_habit_pattern", limit=20)
    assert len(replayed) == 1
    assert len(fibers_after) == 1
    for edge, before in before_by_edge.items():
        after = after_by_edge[edge]
        assert after.weight == before.weight
        assert after.metadata["sequential_count"] == before.metadata["sequential_count"]
        assert after.reinforced_count == before.reinforced_count


@pytest.mark.asyncio
async def test_sequential_pair_effect_replays_once_but_new_run_reinforces(
    store: InMemoryStorage,
) -> None:
    await store.add_neuron(
        Neuron.create(type=NeuronType.ACTION, content="plan", metadata={"_habit_action": True})
    )
    await store.add_neuron(
        Neuron.create(type=NeuronType.ACTION, content="execute", metadata={"_habit_action": True})
    )
    config = BrainConfig(reinforcement_delta=0.1, default_synapse_weight=0.3)

    first = await strengthen_sequential_pair(
        store, "plan", "execute", config, effect_id="run-1:pair"
    )
    assert first is not None
    retry = await strengthen_sequential_pair(
        store, "plan", "execute", config, effect_id="run-1:pair"
    )
    assert retry is not None
    assert retry.weight == first.weight
    assert retry.metadata["sequential_count"] == first.metadata["sequential_count"]
    assert retry.reinforced_count == first.reinforced_count

    next_run = await strengthen_sequential_pair(
        store, "plan", "execute", config, effect_id="run-2:pair"
    )
    assert next_run is not None
    assert next_run.weight == pytest.approx(first.weight + 0.1)
    assert next_run.metadata["sequential_count"] == first.metadata["sequential_count"] + 1
    assert next_run.reinforced_count == first.reinforced_count + 1


@pytest.mark.asyncio
async def test_query_candidate_manifest_survives_interruption_and_changed_events(
    store: InMemoryStorage,
) -> None:
    brain_id = store.current_brain_id
    assert brain_id is not None
    brain = await store.get_brain(brain_id)
    assert brain is not None
    reference_time = utcnow()
    await _seed_query_events(store)

    captured: list[dict[str, object]] = []

    async def persist_manifest(manifest: list[dict[str, object]], cursor: int) -> None:
        if cursor == 0:
            captured[:] = manifest
        else:
            raise RuntimeError("simulated interruption after query synapse write")

    with pytest.raises(RuntimeError, match="query synapse write"):
        await learn_query_patterns(
            store,
            brain.config,
            reference_time,
            run_id="run-query",
            on_progress=persist_manifest,
        )
    assert captured
    first_candidate = captured[0]
    topics = first_candidate["topics"]
    assert isinstance(topics, list)
    source_id, target_id = f"concept-{topics[0]}", f"concept-{topics[1]}"
    before = await store.get_synapses(
        source_id=source_id, target_id=target_id, type=SynapseType.BEFORE
    )
    assert before
    before_synapse = before[0]

    # These new events would change a fresh mining pass; the saved manifest is
    # the contract for this frozen run and must remain authoritative.
    for index in range(4):
        await store.record_action(
            action_type="recall",
            action_context="newly introduced topic resilience checkpoint",
            session_id=f"changed-session-{index}",
        )

    report = await learn_query_patterns(
        store,
        brain.config,
        reference_time,
        run_id="run-query",
        resume_manifest=captured,
        resume_cursor=0,
    )
    assert report.patterns_learned == len(captured)
    after_resume = await store.get_synapses(
        source_id=source_id, target_id=target_id, type=SynapseType.BEFORE
    )
    assert after_resume
    assert after_resume[0].weight == before_synapse.weight
    assert (
        after_resume[0].metadata["sequential_count"] == before_synapse.metadata["sequential_count"]
    )

    # A distinct consolidation run is allowed to apply the intended increase,
    # but retries of that new run remain idempotent too.
    await learn_query_patterns(
        store,
        brain.config,
        reference_time,
        run_id="run-query-next",
        resume_manifest=captured,
    )
    once_next_run = await store.get_synapses(
        source_id=source_id, target_id=target_id, type=SynapseType.BEFORE
    )
    assert once_next_run
    assert once_next_run[0].weight == pytest.approx(before_synapse.weight + 0.05)
    await learn_query_patterns(
        store,
        brain.config,
        reference_time,
        run_id="run-query-next",
        resume_manifest=captured,
    )
    twice_next_run = await store.get_synapses(
        source_id=source_id, target_id=target_id, type=SynapseType.BEFORE
    )
    assert twice_next_run
    assert twice_next_run[0].weight == once_next_run[0].weight
    assert (
        _query_candidate_manifest(
            [QueryPatternCandidate(tuple(topics), int(str(first_candidate["frequency"])), 1.0)]
        )[0]["topics"]
        == topics
    )


@pytest.mark.asyncio
async def test_consolidation_persists_and_reuses_query_plan_after_write_crash(
    store: InMemoryStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain_id = store.current_brain_id
    assert brain_id is not None
    brain = await store.get_brain(brain_id)
    assert brain is not None
    await _seed_query_events(store)
    reference_time = utcnow()
    progress = _Progress()
    engine = ConsolidationEngine(store)
    engine._progress_session = cast("Any", progress)
    engine._active_strategy = ConsolidationStrategy.LEARN_HABITS

    import surreal_memory.engine.sequence_mining as sequence_mining

    async def no_action_habits(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[Any], HabitReport]:
        return [], HabitReport()

    async def no_tool_habits(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[Any], HabitReport]:
        return [], HabitReport()

    monkeypatch.setattr(sequence_mining, "learn_habits", no_action_habits)
    monkeypatch.setattr(sequence_mining, "learn_tool_habits", no_tool_habits)

    with pytest.raises(ConsolidationPausedError):
        await engine._learn_habits(ConsolidationReport(), reference_time, dry_run=False)

    state = progress.strategy_state(ConsolidationStrategy.LEARN_HABITS.value)
    assert state["phase"] == "habits_query_pending"
    pending = state["pending"]
    assert len(pending) == 1
    import json

    saved = json.loads(pending[0])
    assert saved["version"] == 1
    assert saved["run_id"] == "run-frozen"
    assert saved["stage"] == "query"
    assert saved["candidates"]

    candidate = saved["candidates"][0]
    topics = candidate["topics"]
    synapses = await store.get_synapses(
        source_id=f"concept-{topics[0]}",
        target_id=f"concept-{topics[1]}",
        type=SynapseType.BEFORE,
    )
    assert synapses
    before = synapses[0]
    assert state["cursor"] == "candidate:0"

    # A retry must use the durable manifest, not the changed event history.
    for index in range(4):
        await store.record_action(
            action_type="recall",
            action_context="a different topic introduced after the frozen plan",
            session_id=f"new-session-{index}",
        )
    import surreal_memory.engine.query_pattern_mining as query_pattern_mining

    def unexpected_remining(_events: Any) -> list[Any]:
        pytest.fail("resumed query stage re-mined changed source data")
        return []

    monkeypatch.setattr(query_pattern_mining, "mine_query_topic_pairs", unexpected_remining)
    await engine._learn_habits(ConsolidationReport(), reference_time, dry_run=False)

    state = progress.strategy_state(ConsolidationStrategy.LEARN_HABITS.value)
    assert state["phase"] == "completed"
    after = await store.get_synapses(
        source_id=f"concept-{topics[0]}",
        target_id=f"concept-{topics[1]}",
        type=SynapseType.BEFORE,
    )
    assert after
    assert after[0].weight == before.weight
    assert after[0].metadata["sequential_count"] == before.metadata["sequential_count"]

"""Unit tests for sequence mining — habit detection from action events."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from surreal_memory.core.action_event import ActionEvent
from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import NeuronType
from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.sequence_mining import (
    SequencePair,
    _existing_habit_steps,
    _guard_habit_candidate_expansion,
    _SequencePairAccumulator,
    extract_habit_candidates,
    heuristic_habit_name,
    learn_habits,
    learn_tool_habits,
    mine_sequential_pairs,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

# ── Fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """InMemoryStorage with a brain context configured for habit testing."""
    s = InMemoryStorage()
    brain = Brain.create(
        name="habit-test",
        config=BrainConfig(
            habit_min_frequency=2,
            sequential_window_seconds=60.0,
        ),
        owner_id="test",
    )
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


# ── mine_sequential_pairs ────────────────────────────────────────


async def test_habit_dedup_scans_beyond_first_thousand(store: InMemoryStorage) -> None:
    """An older habit cannot disappear from dedup after 1000 newer fibers."""
    for i in range(1001):
        fiber = Fiber.create(
            neuron_ids={"anchor"},
            synapse_ids=set(),
            anchor_neuron_id="anchor",
            fiber_id=f"habit-{i:05d}",
            metadata={"_habit_pattern": True, "_workflow_actions": [f"action-{i}"]},
        )
        await store.add_fiber(fiber)

    steps = await _existing_habit_steps(store)
    assert len(steps) == 1001
    assert ("action-1000",) in steps


class TestMineSequentialPairs:
    """Tests for mine_sequential_pairs pure function."""

    def test_three_events_one_session(self) -> None:
        """Three consecutive events in one session produce two pairs."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(
                id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base
            ),
            ActionEvent(
                id="2",
                brain_id="b",
                session_id="s1",
                action_type="remember",
                created_at=base + timedelta(seconds=5),
            ),
            ActionEvent(
                id="3",
                brain_id="b",
                session_id="s1",
                action_type="context",
                created_at=base + timedelta(seconds=10),
            ),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        assert len(pairs) == 2
        pair_keys = {(p.action_a, p.action_b) for p in pairs}
        assert ("recall", "remember") in pair_keys
        assert ("remember", "context") in pair_keys
        for p in pairs:
            assert p.count == 1

    def test_window_filters_large_gaps(self) -> None:
        """Events separated by more than window_seconds are not counted."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(
                id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base
            ),
            ActionEvent(
                id="2",
                brain_id="b",
                session_id="s1",
                action_type="remember",
                created_at=base + timedelta(seconds=120),
            ),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        assert len(pairs) == 0

    def test_groups_by_session_id(self) -> None:
        """Events in different sessions are mined independently."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(
                id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base
            ),
            ActionEvent(
                id="2",
                brain_id="b",
                session_id="s1",
                action_type="remember",
                created_at=base + timedelta(seconds=5),
            ),
            ActionEvent(
                id="3", brain_id="b", session_id="s2", action_type="recall", created_at=base
            ),
            ActionEvent(
                id="4",
                brain_id="b",
                session_id="s2",
                action_type="remember",
                created_at=base + timedelta(seconds=5),
            ),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        # Same pair from two sessions should aggregate count
        assert len(pairs) == 1
        assert pairs[0].action_a == "recall"
        assert pairs[0].action_b == "remember"
        assert pairs[0].count == 2

    def test_empty_list_returns_empty(self) -> None:
        """Empty event list produces no pairs."""
        pairs = mine_sequential_pairs([], window_seconds=30.0)
        assert pairs == []

    def test_avg_gap_seconds(self) -> None:
        """Average gap is computed correctly across sessions."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            ActionEvent(id="1", brain_id="b", session_id="s1", action_type="A", created_at=base),
            ActionEvent(
                id="2",
                brain_id="b",
                session_id="s1",
                action_type="B",
                created_at=base + timedelta(seconds=10),
            ),
            ActionEvent(id="3", brain_id="b", session_id="s2", action_type="A", created_at=base),
            ActionEvent(
                id="4",
                brain_id="b",
                session_id="s2",
                action_type="B",
                created_at=base + timedelta(seconds=20),
            ),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=60.0)

        assert len(pairs) == 1
        assert pairs[0].avg_gap_seconds == pytest.approx(15.0)

    def test_sorted_by_count_descending(self) -> None:
        """Results are sorted by count descending."""
        base = datetime(2026, 1, 1, 10, 0)
        events = [
            # recall->remember appears twice
            ActionEvent(
                id="1", brain_id="b", session_id="s1", action_type="recall", created_at=base
            ),
            ActionEvent(
                id="2",
                brain_id="b",
                session_id="s1",
                action_type="remember",
                created_at=base + timedelta(seconds=5),
            ),
            ActionEvent(
                id="3", brain_id="b", session_id="s2", action_type="recall", created_at=base
            ),
            ActionEvent(
                id="4",
                brain_id="b",
                session_id="s2",
                action_type="remember",
                created_at=base + timedelta(seconds=5),
            ),
            # context->recall appears once
            ActionEvent(
                id="5", brain_id="b", session_id="s3", action_type="context", created_at=base
            ),
            ActionEvent(
                id="6",
                brain_id="b",
                session_id="s3",
                action_type="recall",
                created_at=base + timedelta(seconds=5),
            ),
        ]
        pairs = mine_sequential_pairs(events, window_seconds=30.0)

        assert pairs[0].count >= pairs[-1].count


# ── extract_habit_candidates ─────────────────────────────────────


class TestExtractHabitCandidates:
    """Tests for extract_habit_candidates pure function."""

    def test_bigrams_from_frequent_pairs(self) -> None:
        """Frequent pairs produce bigram candidates."""
        pairs = [
            SequencePair(action_a="recall", action_b="remember", count=5, avg_gap_seconds=3.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3, total_sessions=10)

        assert len(candidates) >= 1
        bigrams = [c for c in candidates if len(c.steps) == 2]
        assert len(bigrams) == 1
        assert bigrams[0].steps == ("recall", "remember")
        assert bigrams[0].frequency == 5
        assert bigrams[0].confidence == pytest.approx(0.5)

    def test_trigrams_from_chained_pairs(self) -> None:
        """A->B and B->C produce trigram A->B->C."""
        pairs = [
            SequencePair(action_a="recall", action_b="edit", count=4, avg_gap_seconds=2.0),
            SequencePair(action_a="edit", action_b="test", count=3, avg_gap_seconds=5.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3, total_sessions=5)

        trigrams = [c for c in candidates if len(c.steps) == 3]
        assert len(trigrams) == 1
        assert trigrams[0].steps == ("recall", "edit", "test")
        assert trigrams[0].frequency == 3  # min(4, 3)
        assert trigrams[0].avg_duration_seconds == pytest.approx(7.0)

    def test_filters_by_min_frequency(self) -> None:
        """Pairs below min_frequency are excluded."""
        pairs = [
            SequencePair(action_a="recall", action_b="remember", count=1, avg_gap_seconds=3.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3)

        assert candidates == []

    def test_skips_cycles(self) -> None:
        """A->B + B->A should not produce trigram A->B->A."""
        pairs = [
            SequencePair(action_a="recall", action_b="edit", count=5, avg_gap_seconds=2.0),
            SequencePair(action_a="edit", action_b="recall", count=5, avg_gap_seconds=2.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=3)

        trigrams = [c for c in candidates if len(c.steps) == 3]
        assert len(trigrams) == 0

    def test_empty_pairs_returns_empty(self) -> None:
        """Empty pairs list returns empty candidates."""
        candidates = extract_habit_candidates([], min_frequency=1)
        assert candidates == []

    def test_confidence_calculation(self) -> None:
        """Confidence is frequency / total_sessions."""
        pairs = [
            SequencePair(action_a="A", action_b="B", count=6, avg_gap_seconds=1.0),
        ]
        candidates = extract_habit_candidates(pairs, min_frequency=1, total_sessions=12)

        assert len(candidates) >= 1
        assert candidates[0].confidence == pytest.approx(0.5)


# ── heuristic_habit_name ─────────────────────────────────────────


class TestHeuristicHabitName:
    """Tests for heuristic_habit_name pure function."""

    def test_joins_steps_with_hyphen(self) -> None:
        """Multiple steps are joined with hyphens."""
        name = heuristic_habit_name(("recall", "edit", "test"))
        assert name == "recall-edit-test"

    def test_single_step(self) -> None:
        """Single step returns the step itself."""
        name = heuristic_habit_name(("recall",))
        assert name == "recall"

    def test_two_steps(self) -> None:
        """Two steps produce hyphen-joined name."""
        name = heuristic_habit_name(("remember", "context"))
        assert name == "remember-context"


# ── learn_habits (integration) ───────────────────────────────────


class TestLearnHabits:
    """Integration tests for learn_habits async pipeline."""

    async def test_full_pipeline(self, store: InMemoryStorage) -> None:
        """Record actions, mine habits, verify neurons + synapses + fiber."""
        # Record repeated action sequences across multiple sessions
        # Pattern: recall -> remember (repeated 3 times in 3 sessions)
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        learned, report = await learn_habits(store, config, now)

        assert report.sequences_analyzed == 6
        assert len(learned) >= 1

        habit = learned[0]
        assert habit.steps == ("recall", "remember")
        assert habit.name == "recall-remember"
        assert habit.frequency >= 2

        # Verify workflow fiber was created
        assert habit.workflow_fiber is not None
        assert habit.workflow_fiber.summary == "recall-remember"
        assert habit.workflow_fiber.metadata.get("_habit_pattern") is True
        assert habit.workflow_fiber.metadata.get("_workflow_actions") == ["recall", "remember"]

        # Verify BEFORE synapses were created
        assert len(habit.sequence_synapses) >= 1
        for syn in habit.sequence_synapses:
            assert syn.type == SynapseType.BEFORE

    async def test_insufficient_events_returns_empty(self, store: InMemoryStorage) -> None:
        """Fewer than 2 events produces no habits."""
        await store.record_action("recall", session_id="s0")

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        learned, report = await learn_habits(store, config, now)

        assert learned == []
        assert report.sequences_analyzed == 1
        assert report.habits_learned == 0

    async def test_action_neurons_created(self, store: InMemoryStorage) -> None:
        """Learned habits create ACTION neurons for each step."""
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("edit", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        await learn_habits(store, config, now)

        # Verify ACTION neurons exist
        recall_neurons = await store.find_neurons(content_exact="recall", type=NeuronType.ACTION)
        edit_neurons = await store.find_neurons(content_exact="edit", type=NeuronType.ACTION)
        assert len(recall_neurons) >= 1
        assert len(edit_neurons) >= 1

    async def test_workflow_fiber_metadata(self, store: InMemoryStorage) -> None:
        """Workflow fiber carries correct habit metadata."""
        for session_num in range(4):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)
            await store.record_action("context", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        learned, report = await learn_habits(store, config, now)

        assert report.habits_learned >= 1

        # Find the workflow fiber with the most steps (trigram if available)
        names = {h.name for h in learned}

        # At minimum, bigrams should be present
        assert any("recall" in n for n in names)

        # Check all learned habits have correct metadata
        for habit in learned:
            fiber = habit.workflow_fiber
            assert fiber.metadata.get("_habit_pattern") is True
            assert fiber.metadata.get("_habit_frequency") >= 2
            assert "_habit_confidence" in fiber.metadata
            assert fiber.metadata["_workflow_actions"] == list(habit.steps)

    async def test_below_frequency_threshold_no_habits(self, store: InMemoryStorage) -> None:
        """Events below habit_min_frequency produce no habits."""
        # Only one session -- frequency=1 which is below min_frequency=2
        await store.record_action("recall", session_id="s0")
        await store.record_action("remember", session_id="s0")

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        learned, report = await learn_habits(store, config, now)

        assert learned == []
        assert report.habits_learned == 0

    async def test_report_fields(self, store: InMemoryStorage) -> None:
        """HabitReport fields are populated correctly."""
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        _learned, report = await learn_habits(store, config, now)

        assert report.sequences_analyzed == 6
        assert report.habits_learned >= 1
        assert report.pairs_strengthened >= 1
        assert isinstance(report.action_events_pruned, int)


# ── learn_tool_habits (tool_events → habits) ─────────────────────


def _tool_event(tool: str, at: datetime) -> dict:
    """Synthetic tool-event row as buffered by insert_tool_events."""
    return {"tool_name": tool, "session_id": "", "created_at": at, "success": True}


class TestToolHabits:
    """Tool-usage sequences from the tool_events buffer become habit fibers."""

    async def test_tool_sequences_become_habit_fibers(self, store: InMemoryStorage) -> None:
        """Repeated Read → Edit → Bash runs materialize a _habit_pattern fiber."""
        now = utcnow()
        t = now - timedelta(minutes=30)
        events = []
        for _ in range(3):  # 3 repetitions ≥ habit_min_frequency=2
            for tool in ("Read", "Edit", "Bash"):
                events.append(_tool_event(tool, t))
                t += timedelta(seconds=10)  # within 60s window
            t += timedelta(minutes=10)  # gap breaks the sequence between runs
        await store.insert_tool_events(store.current_brain_id, events)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        learned, report = await learn_tool_habits(store, config, now)

        assert report.habits_learned >= 1
        habits = await store.find_fibers(metadata_key="_habit_pattern", limit=100)
        assert habits, "tool habits must be listable via the _habit_pattern marker"
        assert any(h.metadata.get("_habit_source") == "tool_events" for h in habits)
        names = {h.name for h in learned}
        assert any("Read" in n and "Edit" in n for n in names)

    async def test_reads_tail_past_legacy_limit_and_keeps_global_frequency_floor(
        self, store: InMemoryStorage
    ) -> None:
        brain_id = store.current_brain_id
        assert brain_id is not None
        now = utcnow()
        created_at = now - timedelta(minutes=1)
        rows = [
            {
                "id": f"{index:05d}",
                "brain_id": brain_id,
                "tool_name": "NoiseA" if index % 2 == 0 else "NoiseB",
                "session_id": "prefix",
                "created_at": created_at,
            }
            for index in range(4999)
        ]
        rows.append(
            {
                "id": "04999a",
                "brain_id": brain_id,
                "tool_name": "TailA",
                "session_id": "tail",
                "created_at": created_at,
            }
        )
        for index in range(51):
            rows.extend(
                (
                    {
                        "id": f"z{index:04d}b",
                        "brain_id": brain_id,
                        "tool_name": "TailB",
                        "session_id": "tail",
                        "created_at": created_at,
                    },
                    {
                        "id": f"z{index:04d}c",
                        "brain_id": brain_id,
                        "tool_name": "Break",
                        "session_id": "tail",
                        "created_at": created_at,
                    },
                )
            )
            if index < 50:
                rows.append(
                    {
                        "id": f"z{index + 1:04d}a",
                        "brain_id": brain_id,
                        "tool_name": "TailA",
                        "session_id": "tail",
                        "created_at": created_at,
                    }
                )
        store._action_events[brain_id] = rows

        learned, report = await learn_tool_habits(
            store,
            BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0),
            now,
        )

        assert report.sequences_analyzed == len(rows)
        assert any("TailA" in habit.name and "TailB" in habit.name for habit in learned)

    async def test_global_tool_frequency_floor_rejects_tail_only_noise(
        self, store: InMemoryStorage
    ) -> None:
        brain_id = store.current_brain_id
        assert brain_id is not None
        now = utcnow()
        created_at = now - timedelta(minutes=1)
        rows = [
            {
                "id": f"{index:05d}",
                "brain_id": brain_id,
                "tool_name": "NoiseA" if index % 2 == 0 else "NoiseB",
                "session_id": "prefix",
                "created_at": created_at,
            }
            for index in range(5000)
        ]
        for index in range(30):
            rows.extend(
                (
                    {
                        "id": f"z{index:04d}a",
                        "brain_id": brain_id,
                        "tool_name": "RareA",
                        "session_id": "tail",
                        "created_at": created_at,
                    },
                    {
                        "id": f"z{index:04d}b",
                        "brain_id": brain_id,
                        "tool_name": "RareB",
                        "session_id": "tail",
                        "created_at": created_at,
                    },
                    {
                        "id": f"z{index:04d}c",
                        "brain_id": brain_id,
                        "tool_name": "Break",
                        "session_id": "tail",
                        "created_at": created_at,
                    },
                )
            )
        store._action_events[brain_id] = rows

        learned, report = await learn_tool_habits(
            store,
            BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0),
            now,
        )

        assert report.sequences_analyzed == len(rows)
        assert not any("RareA" in habit.name for habit in learned)

    async def test_resumed_tool_manifest_does_not_read_source_events(
        self, store: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fail_if_read(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
            raise AssertionError("resumed tool mining must use its frozen manifest")

        monkeypatch.setattr(store, "get_tool_events_for_mining", fail_if_read)
        learned, report = await learn_tool_habits(
            store,
            BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0),
            utcnow(),
            run_id="resume-tool",
            resume_manifest=[
                {
                    "steps": ["Read", "Write"],
                    "frequency": 2,
                    "avg_duration_seconds": 1.0,
                    "confidence": 1.0,
                }
            ],
            resume_cursor=1,
        )

        assert learned == []
        assert report.sequences_analyzed == 0

    async def test_action_habit_adjacency_crosses_page_boundary(
        self, store: InMemoryStorage
    ) -> None:
        brain_id = store.current_brain_id
        assert brain_id is not None
        now = utcnow()
        created_at = now - timedelta(minutes=1)
        rows = [
            {
                "id": f"{index:04d}",
                "brain_id": brain_id,
                "session_id": f"prefix-{index}",
                "action_type": "noise",
                "action_context": "",
                "tags": (),
                "fiber_id": None,
                "created_at": created_at,
            }
            for index in range(999)
        ]
        rows.extend(
            (
                {
                    "id": "0999",
                    "brain_id": brain_id,
                    "session_id": "tail",
                    "action_type": "TailA",
                    "action_context": "",
                    "tags": (),
                    "fiber_id": None,
                    "created_at": created_at,
                },
                {
                    "id": "z0000b",
                    "brain_id": brain_id,
                    "session_id": "tail",
                    "action_type": "TailB",
                    "action_context": "",
                    "tags": (),
                    "fiber_id": None,
                    "created_at": created_at,
                },
                {
                    "id": "z0000c",
                    "brain_id": brain_id,
                    "session_id": "tail",
                    "action_type": "Break",
                    "action_context": "",
                    "tags": (),
                    "fiber_id": None,
                    "created_at": created_at,
                },
                {
                    "id": "z0001a",
                    "brain_id": brain_id,
                    "session_id": "tail",
                    "action_type": "TailA",
                    "action_context": "",
                    "tags": (),
                    "fiber_id": None,
                    "created_at": created_at,
                },
                {
                    "id": "z0001b",
                    "brain_id": brain_id,
                    "session_id": "tail",
                    "action_type": "TailB",
                    "action_context": "",
                    "tags": (),
                    "fiber_id": None,
                    "created_at": created_at,
                },
            )
        )
        store._action_events[brain_id] = rows

        learned, report = await learn_habits(
            store,
            BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0),
            now,
        )

        assert report.sequences_analyzed == 1004
        assert any(habit.steps == ("TailA", "TailB") for habit in learned)

    def test_streaming_pairs_match_reference_miner_ties(self) -> None:
        timestamp = utcnow()
        events = [
            ActionEvent(id="01", session_id="first", action_type="A", created_at=timestamp),
            ActionEvent(id="02", session_id="second", action_type="C", created_at=timestamp),
            ActionEvent(id="03", session_id="first", action_type="B", created_at=timestamp),
            ActionEvent(id="04", session_id="second", action_type="D", created_at=timestamp),
        ]
        accumulator = _SequencePairAccumulator(60)
        for event in events:
            accumulator.add(event.session_id, event.action_type, event.created_at)

        assert accumulator.pairs() == mine_sequential_pairs(events, 60)

        tool_events = [
            ActionEvent(id="t1", action_type="ToolA", created_at=timestamp),
            ActionEvent(id="t2", action_type="ToolB", created_at=timestamp),
            ActionEvent(id="t3", action_type="ToolB", created_at=timestamp),
            ActionEvent(id="t4", action_type="ToolC", created_at=timestamp),
        ]
        tool_accumulator = _SequencePairAccumulator(60, ignore_self=True)
        for event in tool_events:
            tool_accumulator.add(None, event.action_type, event.created_at)
        expected = [
            pair
            for pair in mine_sequential_pairs(tool_events, 60)
            if pair.action_a != pair.action_b
        ]
        assert tool_accumulator.pairs() == expected

    def test_dense_trigram_expansion_fails_closed(self) -> None:
        pairs = [
            *(SequencePair(f"left-{index}", "middle", 2, 1.0) for index in range(510)),
            *(SequencePair("middle", f"right-{index}", 2, 1.0) for index in range(510)),
        ]

        with pytest.raises(RuntimeError, match="candidate expansion limit"):
            _guard_habit_candidate_expansion(pairs, min_frequency=2)

    async def test_self_pairs_do_not_form_habits(self, store: InMemoryStorage) -> None:
        """Bash → Bash repetition is not a workflow and must not become a habit."""
        now = utcnow()
        t = now - timedelta(minutes=5)
        events = [_tool_event("Bash", t + timedelta(seconds=5 * i)) for i in range(10)]
        await store.insert_tool_events(store.current_brain_id, events)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        learned, report = await learn_tool_habits(store, config, now)

        assert learned == []
        assert report.habits_learned == 0

    async def test_idempotent_across_runs(self, store: InMemoryStorage) -> None:
        """A second mining run must not duplicate already-materialized habits."""
        now = utcnow()
        t = now - timedelta(minutes=30)
        events = []
        for _ in range(3):
            for tool in ("Grep", "Read"):
                events.append(_tool_event(tool, t))
                t += timedelta(seconds=10)
            t += timedelta(minutes=10)
        await store.insert_tool_events(store.current_brain_id, events)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        _, first = await learn_tool_habits(store, config, now)
        assert first.habits_learned >= 1
        count_after_first = len(await store.find_fibers(metadata_key="_habit_pattern", limit=100))

        _, second = await learn_tool_habits(store, config, now)
        count_after_second = len(await store.find_fibers(metadata_key="_habit_pattern", limit=100))

        assert second.habits_learned == 0
        assert count_after_second == count_after_first

    async def test_no_tool_events_is_noop(self, store: InMemoryStorage) -> None:
        """Backends/brains without tool events return an empty report."""
        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        learned, report = await learn_tool_habits(store, config, utcnow())

        assert learned == []
        assert report.habits_learned == 0

    async def test_volume_scaled_frequency_floor(self, store: InMemoryStorage) -> None:
        """On dense streams, low-frequency combos must not qualify as habits.

        400 events raise the effective floor to 400/100 = 4, so a pair seen only
        3 times (= config threshold) is rejected as noise.
        """
        now = utcnow()
        t = now - timedelta(hours=8)
        events = []
        # Dense filler: 394 alternating Bash/Read (self-pair-free noise floor
        # carrier; Bash→Read itself is frequent and WILL qualify).
        for i in range(394):
            events.append(_tool_event("Bash" if i % 2 == 0 else "Read", t))
            t += timedelta(seconds=20)
        # Rare pair Grep→Write occurs exactly 3 times — below the scaled floor.
        for _ in range(3):
            events.append(_tool_event("Grep", t))
            t += timedelta(seconds=10)
            events.append(_tool_event("Write", t))
            t += timedelta(minutes=10)
        await store.insert_tool_events(store.current_brain_id, events)

        config = BrainConfig(habit_min_frequency=3, sequential_window_seconds=60.0)
        learned, _ = await learn_tool_habits(store, config, now)

        names = {h.name for h in learned}
        assert not any("Grep" in n for n in names), (
            "freq-3 pair must be rejected when the volume-scaled floor is 4"
        )
        assert any("Bash" in n and "Read" in n for n in names), (
            "genuinely frequent pairs must still qualify"
        )

    async def test_action_habits_idempotent_across_runs(self, store: InMemoryStorage) -> None:
        """learn_habits must not re-create the same habit on every consolidation.

        Regression: each `consolidate` run created another `recall-remember`
        fiber because nothing consumed the mined action events.
        """
        for session_num in range(3):
            sid = f"s{session_num}"
            await store.record_action("recall", session_id=sid)
            await store.record_action("remember", session_id=sid)

        config = BrainConfig(habit_min_frequency=2, sequential_window_seconds=60.0)
        now = utcnow()

        _, first = await learn_habits(store, config, now)
        assert first.habits_learned >= 1
        count_first = len(await store.find_fibers(metadata_key="_habit_pattern", limit=100))

        _, second = await learn_habits(store, config, now)
        count_second = len(await store.find_fibers(metadata_key="_habit_pattern", limit=100))

        assert second.habits_learned == 0
        assert count_second == count_first

"""Sequence mining — detect habitual action patterns and create workflow fibers.

Mines action event sequences to discover repeated patterns:
1. Group events by session, find consecutive pairs within time window
2. Extract bigram/trigram candidates meeting frequency threshold
3. Create ACTION neurons + BEFORE synapses + WORKFLOW fibers

Zero LLM dependency — pure frequency-based pattern detection.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.core.action_event import ActionEvent
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class SequencePair:
    """A consecutive pair of actions observed in sessions.

    Attributes:
        action_a: First action type
        action_b: Second action type
        count: Number of times this pair was observed
        avg_gap_seconds: Average time gap between A and B
    """

    action_a: str
    action_b: str
    count: int
    avg_gap_seconds: float


@dataclass(frozen=True)
class HabitCandidate:
    """A candidate habit pattern extracted from sequential pairs.

    Attributes:
        steps: Ordered tuple of action types forming the habit
        frequency: Number of times this pattern was observed
        avg_duration_seconds: Average total duration of the pattern
        confidence: Frequency / total sessions (how consistently it appears)
    """

    steps: tuple[str, ...]
    frequency: int
    avg_duration_seconds: float
    confidence: float


HabitProgressCallback = Callable[[list[dict[str, object]], int], Awaitable[None]]


def _habit_candidate_manifest(candidates: Sequence[HabitCandidate]) -> list[dict[str, object]]:
    return [
        {
            "steps": list(candidate.steps),
            "frequency": candidate.frequency,
            "avg_duration_seconds": candidate.avg_duration_seconds,
            "confidence": candidate.confidence,
        }
        for candidate in candidates
    ]


def _habit_candidates_from_manifest(
    manifest: Sequence[Mapping[str, object]],
) -> list[HabitCandidate]:
    candidates: list[HabitCandidate] = []
    for item in manifest:
        steps = item.get("steps")
        if not isinstance(steps, (list, tuple)):
            raise ValueError("invalid habit candidate manifest")
        candidates.append(
            HabitCandidate(
                steps=tuple(str(step) for step in steps),
                frequency=int(str(item["frequency"])),
                avg_duration_seconds=float(str(item["avg_duration_seconds"])),
                confidence=float(str(item["confidence"])),
            )
        )
    return candidates


def _habit_effect_id(run_id: str, source: str, candidate: HabitCandidate, pair_index: int) -> str:
    payload = json.dumps(
        {"run_id": run_id, "source": source, "steps": list(candidate.steps), "pair": pair_index},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _habit_effect_metadata(
    metadata: Mapping[str, object],
    effect_id: str | None,
    effect_run_id: str | None,
) -> tuple[dict[str, object], bool]:
    updated = dict(metadata)
    if effect_id is None:
        return updated, False
    run_id = effect_run_id or effect_id
    stored_run_id = str(updated.get("_habit_effect_run_id") or "")
    stored_effects = updated.get("_habit_effect_ids")
    effect_ids = (
        {str(value) for value in stored_effects}
        if isinstance(stored_effects, (list, tuple, set)) and stored_run_id == run_id
        else set()
    )
    if stored_run_id == run_id and effect_id in effect_ids:
        return updated, True
    effect_ids.add(effect_id)
    updated.update(
        _habit_effect_id=effect_id,
        _habit_effect_run_id=run_id,
        _habit_effect_ids=sorted(effect_ids),
    )
    return updated, False


@dataclass(frozen=True)
class LearnedHabit:
    """A fully materialized habit in the neural graph.

    Attributes:
        name: Heuristic name (e.g., "recall-edit-test")
        steps: Ordered action types
        frequency: How often this pattern occurs
        workflow_fiber: The WORKFLOW fiber created for this habit
        sequence_synapses: BEFORE synapses connecting the action neurons
    """

    name: str
    steps: tuple[str, ...]
    frequency: int
    workflow_fiber: Fiber
    sequence_synapses: list[Synapse]


@dataclass
class HabitReport:
    """Report from habit learning operations.

    Attributes:
        sequences_analyzed: Total action events processed
        pairs_strengthened: Sequential pairs that had existing synapses reinforced
        habits_learned: New habits materialized in the graph
        action_events_pruned: Old action events cleaned up
    """

    sequences_analyzed: int = 0
    pairs_strengthened: int = 0
    habits_learned: int = 0
    action_events_pruned: int = 0


def mine_sequential_pairs(
    events: list[ActionEvent],
    window_seconds: float,
) -> list[SequencePair]:
    """Mine consecutive action pairs from event sequences.

    Groups events by session_id, sorts by created_at, and counts
    pairs of consecutive actions within the time window.

    Args:
        events: List of action events (any order)
        window_seconds: Maximum gap between A and B to count as sequential

    Returns:
        List of SequencePair sorted by count descending
    """
    # Group by session
    sessions: dict[str | None, list[ActionEvent]] = defaultdict(list)
    for event in events:
        sessions[event.session_id].append(event)

    # Count pairs
    pair_gaps: dict[tuple[str, str], list[float]] = defaultdict(list)

    for session_events in sessions.values():
        sorted_events = sorted(session_events, key=lambda e: e.created_at)
        for i in range(len(sorted_events) - 1):
            a = sorted_events[i]
            b = sorted_events[i + 1]
            gap = (b.created_at - a.created_at).total_seconds()
            if gap <= window_seconds:
                pair_gaps[(a.action_type, b.action_type)].append(gap)

    results: list[SequencePair] = []
    for (action_a, action_b), gaps in pair_gaps.items():
        results.append(
            SequencePair(
                action_a=action_a,
                action_b=action_b,
                count=len(gaps),
                avg_gap_seconds=sum(gaps) / len(gaps) if gaps else 0.0,
            )
        )

    results.sort(key=lambda p: p.count, reverse=True)
    return results


def extract_habit_candidates(
    pairs: list[SequencePair],
    min_frequency: int,
    total_sessions: int = 1,
) -> list[HabitCandidate]:
    """Extract habit candidates from sequential pairs.

    Builds bigrams and trigrams from pairs meeting the frequency threshold.

    Args:
        pairs: Sequential pairs from mine_sequential_pairs
        min_frequency: Minimum count for a pair to be considered
        total_sessions: Total number of sessions for confidence calculation

    Returns:
        List of HabitCandidate sorted by frequency descending
    """
    # Filter pairs by min_frequency
    frequent_pairs = [p for p in pairs if p.count >= min_frequency]
    if not frequent_pairs:
        return []

    candidates: list[HabitCandidate] = []
    seen_steps: set[tuple[str, ...]] = set()

    # Build bigrams
    for pair in frequent_pairs:
        steps = (pair.action_a, pair.action_b)
        if steps not in seen_steps:
            seen_steps.add(steps)
            candidates.append(
                HabitCandidate(
                    steps=steps,
                    frequency=pair.count,
                    avg_duration_seconds=pair.avg_gap_seconds,
                    confidence=pair.count / max(total_sessions, 1),
                )
            )

    # Build trigrams: A→B + B→C = A→B→C
    pair_map: dict[str, list[SequencePair]] = defaultdict(list)
    for pair in frequent_pairs:
        pair_map[pair.action_a].append(pair)

    for pair_ab in frequent_pairs:
        for pair_bc in pair_map.get(pair_ab.action_b, []):
            if pair_bc.action_b == pair_ab.action_a:
                continue  # Skip cycles
            tri_steps = (pair_ab.action_a, pair_ab.action_b, pair_bc.action_b)
            if tri_steps in seen_steps:
                continue
            seen_steps.add(tri_steps)
            freq = min(pair_ab.count, pair_bc.count)
            if freq >= min_frequency:
                candidates.append(
                    HabitCandidate(
                        steps=tri_steps,
                        frequency=freq,
                        avg_duration_seconds=pair_ab.avg_gap_seconds + pair_bc.avg_gap_seconds,
                        confidence=freq / max(total_sessions, 1),
                    )
                )

    candidates.sort(key=lambda c: c.frequency, reverse=True)
    return candidates


def heuristic_habit_name(steps: tuple[str, ...]) -> str:
    """Generate a human-readable name from action steps.

    Args:
        steps: Ordered action types

    Returns:
        Hyphen-joined name (e.g., "recall-edit-test")
    """
    return "-".join(steps)


async def strengthen_sequential_pair(
    storage: NeuralStorage,
    action_a: str,
    action_b: str,
    config: BrainConfig,
    *,
    effect_id: str | None = None,
    effect_run_id: str | None = None,
) -> Synapse | None:
    """Find or create a BEFORE synapse with durable per-effect receipts."""
    neurons_a = await storage.find_neurons(content_exact=action_a, type=NeuronType.ACTION)
    neurons_b = await storage.find_neurons(content_exact=action_b, type=NeuronType.ACTION)

    if not neurons_a or not neurons_b:
        return None

    neuron_a = neurons_a[0]
    neuron_b = neurons_b[0]
    existing = await storage.get_synapses(
        source_id=neuron_a.id,
        target_id=neuron_b.id,
        type=SynapseType.BEFORE,
    )

    if existing:
        synapse = existing[0]
        metadata, already_applied = _habit_effect_metadata(
            synapse.metadata, effect_id, effect_run_id
        )
        if already_applied:
            return synapse
        seq_count = synapse.metadata.get("sequential_count", 0) + 1
        metadata["sequential_count"] = seq_count
        reinforced = Synapse(
            id=synapse.id,
            source_id=synapse.source_id,
            target_id=synapse.target_id,
            type=synapse.type,
            weight=min(1.0, synapse.weight + config.reinforcement_delta),
            direction=synapse.direction,
            metadata=metadata,
            reinforced_count=synapse.reinforced_count + 1,
            last_activated=utcnow(),
            created_at=synapse.created_at,
        )
        await storage.update_synapse(reinforced)
        return reinforced

    create_metadata: dict[str, object] = {"sequential_count": 1, "_habit": True}
    create_metadata, _ = _habit_effect_metadata(create_metadata, effect_id, effect_run_id)
    synapse = Synapse.create(
        source_id=neuron_a.id,
        target_id=neuron_b.id,
        type=SynapseType.BEFORE,
        weight=config.default_synapse_weight,
        metadata=create_metadata,
    )
    await storage.add_synapse(synapse)
    return synapse


async def learn_habits(
    storage: NeuralStorage,
    config: BrainConfig,
    reference_time: datetime,
    *,
    run_id: str | None = None,
    resume_manifest: Sequence[Mapping[str, object]] | None = None,
    resume_cursor: int = 0,
    on_progress: HabitProgressCallback | None = None,
) -> tuple[list[LearnedHabit], HabitReport]:
    """Learn action-log habits, optionally resuming a frozen candidate plan."""
    report = HabitReport()
    if resume_manifest is not None:
        candidates = _habit_candidates_from_manifest(resume_manifest)
    else:
        since = reference_time - timedelta(days=30)
        events = await storage.get_action_sequences(since=since)
        report.sequences_analyzed = len(events)
        if len(events) < 2:
            return [], report

        pairs = mine_sequential_pairs(events, config.sequential_window_seconds)
        if not pairs:
            return [], report
        session_ids = {event.session_id for event in events if event.session_id}
        total_sessions = max(len(session_ids), 1)
        candidates = extract_habit_candidates(pairs, config.habit_min_frequency, total_sessions)
        if not candidates:
            return [], report
        existing_steps = await _existing_habit_steps(storage)
        candidates = [
            candidate for candidate in candidates if tuple(candidate.steps) not in existing_steps
        ]
        if not candidates:
            return [], report

    manifest = _habit_candidate_manifest(candidates)
    if on_progress is not None and resume_manifest is None:
        await on_progress(manifest, 0)
    learned = await _materialize_habits(
        storage,
        candidates,
        config,
        report,
        source="action_log",
        run_id=run_id,
        start_cursor=resume_cursor,
        on_progress=on_progress,
    )

    prune_cutoff = reference_time - timedelta(days=60)
    report.action_events_pruned = await storage.prune_action_events(prune_cutoff)
    return learned, report


async def _existing_habit_steps(storage: NeuralStorage) -> set[tuple[str, ...]]:
    """Step-sequences of all materialized habits, without a top-N cutoff."""
    steps: set[tuple[str, ...]] = set()
    cursor: str | None = None
    while True:
        page = await storage.get_fibers_after_id(cursor, limit=1000)
        for fiber in page:
            if "_habit_pattern" in fiber.metadata:
                actions = fiber.metadata.get("_workflow_actions")
                if actions:
                    steps.add(tuple(actions))
        if len(page) < 1000:
            break
        cursor = page[-1].id
    return steps


async def _materialize_habits(
    storage: NeuralStorage,
    candidates: list[HabitCandidate],
    config: BrainConfig,
    report: HabitReport,
    source: str,
    *,
    run_id: str | None = None,
    start_cursor: int = 0,
    on_progress: HabitProgressCallback | None = None,
) -> list[LearnedHabit]:
    """Materialize candidates with optional stable effect receipts and checkpoints."""
    learned: list[LearnedHabit] = []
    if not candidates:
        return learned

    manifest = _habit_candidate_manifest(candidates)
    all_steps = {step for candidate in candidates for step in candidate.steps}
    existing_actions = await storage.find_neurons_exact_batch(
        list(all_steps), type=NeuronType.ACTION
    )

    for candidate_index, candidate in enumerate(candidates):
        if candidate_index < start_cursor:
            continue

        neuron_ids: list[str] = []
        for step in candidate.steps:
            if step in existing_actions:
                neuron_ids.append(existing_actions[step].id)
            else:
                neuron = Neuron.create(
                    type=NeuronType.ACTION,
                    content=step,
                    metadata={"_habit_action": True},
                )
                await storage.add_neuron(neuron)
                existing_actions[step] = neuron
                neuron_ids.append(neuron.id)

        sequence_synapses: list[Synapse] = []
        for pair_index in range(len(neuron_ids) - 1):
            effect_id = _habit_effect_id(run_id, source, candidate, pair_index) if run_id else None
            synapse = await strengthen_sequential_pair(
                storage,
                candidate.steps[pair_index],
                candidate.steps[pair_index + 1],
                config,
                effect_id=effect_id,
                effect_run_id=run_id,
            )
            if synapse:
                sequence_synapses.append(synapse)
                report.pairs_strengthened += 1

        name = heuristic_habit_name(candidate.steps)
        candidate_key = hashlib.sha256(
            json.dumps(
                {"source": source, "steps": list(candidate.steps)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        workflow_effect_id = f"{run_id}:{source}:{candidate_key}" if run_id else None
        metadata: dict[str, object] = {
            "_workflow_actions": list(candidate.steps),
            "_habit_pattern": True,
            "_habit_frequency": candidate.frequency,
            "_habit_confidence": candidate.confidence,
            "_habit_source": source,
        }
        if workflow_effect_id:
            metadata["_habit_effect_id"] = workflow_effect_id
        fiber_id = (
            "habit-" + hashlib.sha256(workflow_effect_id.encode("utf-8")).hexdigest()[:32]
            if workflow_effect_id
            else None
        )
        workflow_fiber = await storage.get_fiber(fiber_id) if fiber_id is not None else None
        if workflow_fiber is None:
            workflow_fiber = Fiber.create(
                neuron_ids=set(neuron_ids),
                synapse_ids={synapse.id for synapse in sequence_synapses},
                anchor_neuron_id=neuron_ids[0],
                pathway=neuron_ids,
                summary=name,
                tags=set(),
                metadata=metadata,
                fiber_id=fiber_id,
            )
            await storage.add_fiber(workflow_fiber)

        learned.append(
            LearnedHabit(
                name=name,
                steps=candidate.steps,
                frequency=candidate.frequency,
                workflow_fiber=workflow_fiber,
                sequence_synapses=sequence_synapses,
            )
        )
        report.habits_learned += 1
        if on_progress is not None:
            await on_progress(manifest, candidate_index + 1)

    return learned


# Tool-usage habit mining tunables (module-level — no BrainConfig migration).
_TOOL_HABIT_LOOKBACK_DAYS = 30
_TOOL_HABIT_MAX_EVENTS = 5000
_TOOL_HABIT_MAX = 25  # cap materialized tool habits per run to avoid flooding
# Tool events are orders of magnitude denser than action events (hundreds per
# session vs a handful), so the configured habit_min_frequency (default 3) lets
# nearly every tool combination qualify — live-tested at ~1.7k events it drained
# a 134-habit backlog of mostly freq-3 noise. Scale the frequency floor with
# volume: 1 per this many events (e.g. 1700 events → floor 17).
_TOOL_HABIT_FREQ_DIVISOR = 100


async def learn_tool_habits(
    storage: NeuralStorage,
    config: BrainConfig,
    reference_time: datetime,
    *,
    run_id: str | None = None,
    resume_manifest: Sequence[Mapping[str, object]] | None = None,
    resume_cursor: int = 0,
    on_progress: HabitProgressCallback | None = None,
) -> tuple[list[LearnedHabit], HabitReport]:
    """Learn tool habits, optionally resuming a frozen candidate plan."""
    from surreal_memory.core.action_event import ActionEvent

    report = HabitReport()
    if resume_manifest is not None:
        candidates = _habit_candidates_from_manifest(resume_manifest)
    else:
        brain_id = storage.current_brain_id
        if not brain_id:
            return [], report

        since = reference_time - timedelta(days=_TOOL_HABIT_LOOKBACK_DAYS)
        rows = await storage.get_tool_events_for_mining(
            brain_id, since=since, limit=_TOOL_HABIT_MAX_EVENTS
        )
        if len(rows) < 2:
            return [], report

        events: list[ActionEvent] = []
        for row in rows:
            created = row.get("created_at")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    continue
            if not isinstance(created, datetime):
                continue
            if created.tzinfo is not None:
                created = created.replace(tzinfo=None)
            tool = str(row.get("tool_name") or "").strip()
            if not tool:
                continue
            events.append(
                ActionEvent(
                    brain_id=brain_id,
                    session_id=None,
                    action_type=tool,
                    created_at=created,
                )
            )
        report.sequences_analyzed = len(events)
        if len(events) < 2:
            return [], report

        pairs = mine_sequential_pairs(events, config.sequential_window_seconds)
        pairs = [pair for pair in pairs if pair.action_a != pair.action_b]
        if not pairs:
            return [], report

        effective_min_freq = max(
            config.habit_min_frequency, len(events) // _TOOL_HABIT_FREQ_DIVISOR
        )
        candidates = extract_habit_candidates(pairs, effective_min_freq, total_sessions=1)
        if not candidates:
            return [], report
        candidates = [
            dc_replace(candidate, confidence=min(1.0, candidate.confidence))
            if candidate.confidence > 1.0
            else candidate
            for candidate in candidates
        ]

        existing_steps = await _existing_habit_steps(storage)
        candidates = [
            candidate for candidate in candidates if tuple(candidate.steps) not in existing_steps
        ]
        candidates = candidates[:_TOOL_HABIT_MAX]
        if not candidates:
            return [], report

    manifest = _habit_candidate_manifest(candidates)
    if on_progress is not None and resume_manifest is None:
        await on_progress(manifest, 0)
    learned = await _materialize_habits(
        storage,
        candidates,
        config,
        report,
        source="tool_events",
        run_id=run_id,
        start_cursor=resume_cursor,
        on_progress=on_progress,
    )
    return learned, report

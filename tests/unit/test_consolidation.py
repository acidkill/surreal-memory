"""Tests for memory consolidation — high-frequency fibers boost synapses."""

from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.lifecycle import DecayManager
from surreal_memory.engine.memory_stages import (
    MaturationRecord,
    MemoryStage,
    compute_stage_transition,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


@pytest_asyncio.fixture
async def consolidation_storage() -> InMemoryStorage:
    """Storage with fibers at different frequencies."""
    store = InMemoryStorage()
    brain = Brain.create(name="consolidation_test", brain_id="cons-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Create neurons
    n1 = Neuron.create(type=NeuronType.ENTITY, content="alpha", neuron_id="n-1")
    n2 = Neuron.create(type=NeuronType.ENTITY, content="beta", neuron_id="n-2")
    n3 = Neuron.create(type=NeuronType.ENTITY, content="gamma", neuron_id="n-3")
    for n in [n1, n2, n3]:
        await store.add_neuron(n)

    # Synapses for high-frequency fiber
    s1 = Synapse.create(
        source_id="n-1",
        target_id="n-2",
        type=SynapseType.RELATED_TO,
        weight=0.5,
        synapse_id="syn-hi-1",
    )
    # Synapse for low-frequency fiber
    s2 = Synapse.create(
        source_id="n-2",
        target_id="n-3",
        type=SynapseType.RELATED_TO,
        weight=0.5,
        synapse_id="syn-lo-1",
    )
    await store.add_synapse(s1)
    await store.add_synapse(s2)

    # High-frequency fiber (frequency=10)
    hi_fiber = Fiber(
        id="fiber-hi",
        neuron_ids={"n-1", "n-2"},
        synapse_ids={"syn-hi-1"},
        anchor_neuron_id="n-1",
        pathway=["n-1", "n-2"],
        frequency=10,
    )
    # Low-frequency fiber (frequency=2)
    lo_fiber = Fiber(
        id="fiber-lo",
        neuron_ids={"n-2", "n-3"},
        synapse_ids={"syn-lo-1"},
        anchor_neuron_id="n-2",
        pathway=["n-2", "n-3"],
        frequency=2,
    )
    await store.add_fiber(hi_fiber)
    await store.add_fiber(lo_fiber)

    return store


@pytest.mark.asyncio
async def test_high_frequency_fiber_consolidated(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Synapses in high-freq fiber boosted by boost_delta."""
    manager = DecayManager()
    await manager.consolidate(
        consolidation_storage,
        frequency_threshold=5,
        boost_delta=0.03,
    )

    synapse = await consolidation_storage.get_synapse("syn-hi-1")
    assert synapse is not None
    assert synapse.weight == pytest.approx(0.53, abs=1e-9)


@pytest.mark.asyncio
async def test_low_frequency_fiber_unchanged(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Synapses in low-freq fiber untouched."""
    manager = DecayManager()
    await manager.consolidate(
        consolidation_storage,
        frequency_threshold=5,
        boost_delta=0.03,
    )

    synapse = await consolidation_storage.get_synapse("syn-lo-1")
    assert synapse is not None
    assert synapse.weight == pytest.approx(0.5, abs=1e-9)


@pytest.mark.asyncio
async def test_returns_consolidated_count(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Return value matches number of synapses updated."""
    manager = DecayManager()
    count = await manager.consolidate(
        consolidation_storage,
        frequency_threshold=5,
        boost_delta=0.03,
    )

    # Only the high-frequency fiber's 1 synapse should be consolidated
    assert count == 1


# ── INFER strategy tests ────────────────────────────────────────


@pytest_asyncio.fixture
async def infer_storage() -> InMemoryStorage:
    """Storage with neurons and co-activation events for inference."""
    store = InMemoryStorage()
    brain = Brain.create(name="infer_test", brain_id="infer-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Create neurons
    n1 = Neuron.create(type=NeuronType.ENTITY, content="python programming", neuron_id="in-1")
    n2 = Neuron.create(type=NeuronType.ENTITY, content="python testing", neuron_id="in-2")
    n3 = Neuron.create(type=NeuronType.ENTITY, content="python debugging", neuron_id="in-3")
    for n in [n1, n2, n3]:
        await store.add_neuron(n)

    # Existing synapse between n1 and n2
    s1 = Synapse.create(
        source_id="in-1",
        target_id="in-2",
        type=SynapseType.RELATED_TO,
        weight=0.4,
        synapse_id="syn-existing",
    )
    await store.add_synapse(s1)

    # Create a fiber containing all neurons
    fiber = Fiber(
        id="fiber-infer",
        neuron_ids={"in-1", "in-2", "in-3"},
        synapse_ids={"syn-existing"},
        anchor_neuron_id="in-1",
        pathway=["in-1", "in-2", "in-3"],
        frequency=5,
    )
    await store.add_fiber(fiber)

    # Record co-activation events (n1,n3 have no existing synapse — should be inferred)
    for _ in range(5):
        await store.record_co_activation("in-1", "in-3", 0.8)
    # n1,n2 already have a synapse — should be reinforced
    for _ in range(4):
        await store.record_co_activation("in-1", "in-2", 0.7)

    return store


@pytest.mark.asyncio
async def test_infer_creates_new_synapse(infer_storage: InMemoryStorage) -> None:
    """INFER creates CO_OCCURS synapse for pairs without existing connections."""
    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)
    report = await engine.run(strategies=[ConsolidationStrategy.INFER])

    assert report.synapses_inferred >= 1

    # Check that a synapse was created between in-1 and in-3
    synapses = await infer_storage.get_synapses(source_id="in-1", target_id="in-3")
    reverse = await infer_storage.get_synapses(source_id="in-3", target_id="in-1")
    all_found = synapses + reverse
    assert len(all_found) >= 1
    inferred = all_found[0]
    assert inferred.type == SynapseType.CO_OCCURS
    assert inferred.metadata.get("_inferred") is True


@pytest.mark.asyncio
async def test_infer_reinforces_existing_synapse(infer_storage: InMemoryStorage) -> None:
    """INFER reinforces existing synapses for pairs that already have connections."""
    original = await infer_storage.get_synapse("syn-existing")
    assert original is not None
    original_weight = original.weight

    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)
    await engine.run(strategies=[ConsolidationStrategy.INFER])

    updated = await infer_storage.get_synapse("syn-existing")
    assert updated is not None
    assert updated.weight > original_weight


@pytest.mark.asyncio
async def test_infer_prunes_old_co_activations(infer_storage: InMemoryStorage) -> None:
    """INFER prunes co-activation events outside the window."""
    config = ConsolidationConfig(infer_co_activation_threshold=3, infer_window_days=7)
    engine = ConsolidationEngine(infer_storage, config)
    report = await engine.run(strategies=[ConsolidationStrategy.INFER])

    assert report.co_activations_pruned >= 0


@pytest.mark.asyncio
async def test_infer_dry_run(infer_storage: InMemoryStorage) -> None:
    """Dry run reports counts but doesn't modify storage."""
    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)

    # Count synapses before
    synapses_before = await infer_storage.get_synapses()
    count_before = len(synapses_before)

    report = await engine.run(strategies=[ConsolidationStrategy.INFER], dry_run=True)

    assert report.synapses_inferred >= 1
    assert report.dry_run is True

    # Synapses unchanged
    synapses_after = await infer_storage.get_synapses()
    assert len(synapses_after) == count_before


@pytest.mark.asyncio
async def test_infer_report_in_summary(infer_storage: InMemoryStorage) -> None:
    """INFER results appear in report summary."""
    config = ConsolidationConfig(infer_co_activation_threshold=3)
    engine = ConsolidationEngine(infer_storage, config)
    report = await engine.run(strategies=[ConsolidationStrategy.INFER])

    summary = report.summary()
    assert "Synapses inferred" in summary
    assert "Co-activations pruned" in summary


# ── Parallel tier execution tests ──────────────────────────────


@pytest.mark.asyncio
async def test_strategy_tiers_cover_all_strategies() -> None:
    """All non-ALL strategies appear in exactly one tier."""
    all_in_tiers: set[ConsolidationStrategy] = set()
    for tier in ConsolidationEngine.STRATEGY_TIERS:
        # No overlap between tiers
        assert not (all_in_tiers & tier), f"Overlap detected: {all_in_tiers & tier}"
        all_in_tiers |= tier

    expected = {s for s in ConsolidationStrategy if s != ConsolidationStrategy.ALL}
    assert all_in_tiers == expected


@pytest.mark.asyncio
async def test_run_all_strategies_parallel(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Running ALL strategies via tiered parallel produces a valid report."""
    engine = ConsolidationEngine(consolidation_storage)
    report = await engine.run(strategies=[ConsolidationStrategy.ALL])

    assert report.duration_ms >= 0
    assert not report.dry_run


@pytest.mark.asyncio
async def test_run_single_strategy_still_works(
    consolidation_storage: InMemoryStorage,
) -> None:
    """A single strategy request still works through the tier system."""
    engine = ConsolidationEngine(consolidation_storage)
    report = await engine.run(strategies=[ConsolidationStrategy.PRUNE])

    # Should complete without error
    assert report.duration_ms >= 0


@pytest.mark.asyncio
async def test_run_multiple_same_tier_strategies(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Multiple strategies from the same tier run in parallel."""
    engine = ConsolidationEngine(consolidation_storage)
    # LEARN_HABITS and DEDUP are in the same tier as PRUNE
    report = await engine.run(
        strategies=[
            ConsolidationStrategy.PRUNE,
            ConsolidationStrategy.DEDUP,
        ]
    )

    assert report.duration_ms >= 0


@pytest.mark.asyncio
async def test_run_strategies_across_tiers(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Strategies from different tiers execute in correct tier order."""
    execution_order: list[str] = []

    original_prune = engine_cls._prune if (engine_cls := ConsolidationEngine) else None  # noqa: F841

    # Patch strategies to record execution order
    engine = ConsolidationEngine(consolidation_storage)

    async def tracking_prune(report, ref_time, dry_run):
        execution_order.append("prune")

    async def tracking_merge(report, dry_run):
        execution_order.append("merge")

    async def tracking_enrich(report, dry_run):
        execution_order.append("enrich")

    engine._prune = tracking_prune  # type: ignore[assignment]
    engine._merge = tracking_merge  # type: ignore[assignment]
    engine._enrich = tracking_enrich  # type: ignore[assignment]

    await engine.run(
        strategies=[
            ConsolidationStrategy.ENRICH,  # tier 4
            ConsolidationStrategy.PRUNE,  # tier 1
            ConsolidationStrategy.MERGE,  # tier 2
        ]
    )

    # Tier order: prune(1) -> merge(2) -> enrich(4)
    assert execution_order == ["prune", "merge", "enrich"]


@pytest.mark.asyncio
async def test_run_default_none_strategies(
    consolidation_storage: InMemoryStorage,
) -> None:
    """Passing None defaults to ALL strategies."""
    engine = ConsolidationEngine(consolidation_storage)
    report = await engine.run(strategies=None)

    assert report.duration_ms >= 0


@pytest.mark.asyncio
async def test_run_strategy_dispatcher(
    consolidation_storage: InMemoryStorage,
) -> None:
    """_run_strategy dispatches to the correct method."""
    engine = ConsolidationEngine(consolidation_storage)
    report = ConsolidationReport()

    called = False

    async def mock_dream(report, dry_run):
        nonlocal called
        called = True

    engine._dream = mock_dream  # type: ignore[assignment]

    from surreal_memory.utils.timeutils import utcnow

    await engine._run_strategy(ConsolidationStrategy.DREAM, report, utcnow(), dry_run=True)

    assert called


@pytest.mark.asyncio
async def test_merge_never_removes_habit_fiber() -> None:
    """A `_habit_pattern` fiber must survive _merge even at 100% neuron overlap.

    Merging deletes the member fibers and the merged fiber drops the marker, so
    without the guard a learned habit would silently vanish from `smem habits
    list` and habits could never accumulate over time.
    """
    store = InMemoryStorage()
    brain = Brain.create(name="merge_habit_test", brain_id="mh-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for nid in ("n-a", "n-b", "n-c"):
        await store.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    shared = {"n-a", "n-b", "n-c"}  # 100% overlap → Jaccard 1.0, well above 0.5
    plain1 = Fiber(
        id="plain-1",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=5,
    )
    plain2 = Fiber(
        id="plain-2",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-b",
        pathway=["n-b"],
        frequency=5,
    )
    habit = Fiber(
        id="habit-1",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=5,
        metadata={"_habit_pattern": True, "_workflow_actions": ["a", "b"]},
    )
    for f in (plain1, plain2, habit):
        await store.add_fiber(f)

    engine = ConsolidationEngine(store, ConsolidationConfig())
    await engine._merge(ConsolidationReport(), dry_run=False)

    remaining = {f.id for f in await store.get_fibers(limit=100)}
    assert "habit-1" in remaining, "merge must never delete a _habit_pattern fiber"


class _MaturationCapableStorage(InMemoryStorage):
    """InMemoryStorage plus a real maturation table, for merge-inheritance tests.

    InMemoryStorage inherits NeuralStorage's no-op maturation defaults (save is
    a no-op, get/find always return empty), so a test built on it can exercise
    the real Union-Find/Jaccard merge path but can never observe whether
    maturation was read or written. This adds a trivial dict-backed table so
    run 010 section C (merge losing maturation) can be reproduced end-to-end
    against real reads/writes instead of mocked ones.
    """

    def __init__(self) -> None:
        super().__init__()
        self._maturations: dict[str, MaturationRecord] = {}

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        return self._maturations.get(fiber_id)

    async def save_maturation(self, record: MaturationRecord) -> None:
        self._maturations[record.fiber_id] = record

    async def find_maturations(
        self,
        stage: MemoryStage | None = None,
        min_rehearsal_count: int = 0,
    ) -> list[MaturationRecord]:
        return [
            m
            for m in self._maturations.values()
            if (stage is None or m.stage == stage) and m.rehearsal_count >= min_rehearsal_count
        ]


@pytest.mark.asyncio
async def test_merge_inherits_maturation_from_sources() -> None:
    """A fiber created by _merge must inherit source maturation, not restart at STM.

    RUN-010 VALIDATE-FIRST (section C). `_merge` deletes the source fibers and
    adds a brand-new one; on origin/main it never reads or writes the
    maturation table for that new fiber, so consolidation destroys the exact
    progress (stage, rehearsal_count, reinforcement_timestamps) it exists to
    measure. Two sources reinforced on two distinct calendar days must combine
    into a record that, after one more reinforcement on a third day, clears
    the EPISODIC -> SEMANTIC spacing gate -- exactly as an unmerged fiber
    would. This must FAIL on origin/main.
    """
    store = _MaturationCapableStorage()
    brain = Brain.create(name="merge_maturation_test", brain_id="mm-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for nid in ("n-a", "n-b"):
        await store.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    shared = {"n-a", "n-b"}
    day1 = utcnow() - timedelta(days=2)
    day2 = utcnow() - timedelta(days=1)

    source_a = Fiber(
        id="src-a",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=3,
        created_at=day1,
    )
    source_b = Fiber(
        id="src-b",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-b",
        pathway=["n-b"],
        frequency=1,
        created_at=day1,
    )
    for f in (source_a, source_b):
        await store.add_fiber(f)

    await store.save_maturation(
        MaturationRecord(
            fiber_id="src-a",
            brain_id="mm-brain",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=day1 - timedelta(days=8),
            rehearsal_count=1,
            reinforcement_timestamps=(day1.isoformat(),),
        )
    )
    await store.save_maturation(
        MaturationRecord(
            fiber_id="src-b",
            brain_id="mm-brain",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=day1 - timedelta(days=8),
            rehearsal_count=1,
            reinforcement_timestamps=(day2.isoformat(),),
        )
    )

    engine = ConsolidationEngine(store, ConsolidationConfig())
    await engine._merge(ConsolidationReport(), dry_run=False)

    remaining = await store.get_fibers(limit=100)
    assert len(remaining) == 1, "the two sources should have merged into one fiber"
    merged = remaining[0]
    assert merged.id not in ("src-a", "src-b")

    inherited = await store.get_maturation(merged.id)
    assert inherited is not None, (
        "merge must carry the sources' MaturationRecord forward, not silently drop it"
    )
    assert inherited.stage == MemoryStage.EPISODIC
    assert inherited.distinct_reinforcement_days == 2

    third_day = day2 + timedelta(days=1)
    rehearsed = inherited.rehearse(third_day)
    promoted = compute_stage_transition(rehearsed, now=third_day)
    assert promoted.stage == MemoryStage.SEMANTIC, (
        "with reinforcements spread across 3 distinct days the merged fiber must "
        "clear the spacing gate exactly like an unmerged fiber would"
    )


@pytest.mark.asyncio
async def test_merge_maturation_combination_semantics() -> None:
    """Locks in the exact inheritance rule, not just the SEMANTIC end-to-end outcome.

    Two sources at different stages: the WORKING source has been in its stage
    far longer than the EPISODIC source. The merged record must take the
    *highest* stage (EPISODIC) but the *oldest* stage_entered_at across ALL
    sources regardless of which source that came from -- otherwise a fiber
    that already had a 20-day head start in a lower stage would have that
    dwell time silently discarded just because another source outranked it.
    rehearsal_count is summed (each source's count is real, independent
    rehearsal history), and reinforcement_timestamps is the union of both.
    """
    store = _MaturationCapableStorage()
    brain = Brain.create(name="merge_semantics_test", brain_id="ms-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for nid in ("n-a", "n-b"):
        await store.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    shared = {"n-a", "n-b"}
    old_entered = utcnow() - timedelta(days=20)
    recent_entered = utcnow() - timedelta(hours=1)
    ts_a1 = (utcnow() - timedelta(days=5)).isoformat()
    ts_a2 = (utcnow() - timedelta(days=3)).isoformat()
    ts_b1 = (utcnow() - timedelta(hours=1)).isoformat()

    source_a = Fiber(
        id="src-a",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=1,
        created_at=old_entered,
    )
    source_b = Fiber(
        id="src-b",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-b",
        pathway=["n-b"],
        frequency=1,
        created_at=old_entered,
    )
    for f in (source_a, source_b):
        await store.add_fiber(f)

    await store.save_maturation(
        MaturationRecord(
            fiber_id="src-a",
            brain_id="ms-brain",
            stage=MemoryStage.WORKING,
            stage_entered_at=old_entered,
            rehearsal_count=2,
            reinforcement_timestamps=(ts_a1, ts_a2),
        )
    )
    await store.save_maturation(
        MaturationRecord(
            fiber_id="src-b",
            brain_id="ms-brain",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=recent_entered,
            rehearsal_count=1,
            reinforcement_timestamps=(ts_b1,),
        )
    )

    engine = ConsolidationEngine(store, ConsolidationConfig())
    await engine._merge(ConsolidationReport(), dry_run=False)

    remaining = await store.get_fibers(limit=100)
    assert len(remaining) == 1
    merged = await store.get_maturation(remaining[0].id)
    assert merged is not None

    assert merged.stage == MemoryStage.EPISODIC, "must take the highest stage of any source"
    assert merged.stage_entered_at == old_entered, (
        "must keep the oldest stage_entered_at across ALL sources, not just the "
        "winning-stage source, or a source's dwell time is silently discarded"
    )
    assert merged.rehearsal_count == 3, "rehearsal_count is summed across sources"
    assert set(merged.reinforcement_timestamps) == {ts_a1, ts_a2, ts_b1}, (
        "reinforcement_timestamps is the union of every source's timestamps"
    )


@pytest.mark.asyncio
async def test_merge_rehearsal_count_matches_the_deduplicated_timestamp_union() -> None:
    """rehearsal_count must stay derived from the timestamp union, not an
    independent sum, or a timestamp collision across sources could break the
    rehearsal_count == len(reinforcement_timestamps) invariant every
    organically-created MaturationRecord maintains (each rehearse() call
    increments both together). Two sources sharing one identical timestamp
    prove it: summing counts would give 2, but only 1 distinct rehearsal
    event is actually evidenced by the timestamps.
    """
    store = _MaturationCapableStorage()
    brain = Brain.create(name="merge_rehearsal_count_test", brain_id="mrc-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for nid in ("n-a", "n-b"):
        await store.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    shared = {"n-a", "n-b"}
    entered = utcnow() - timedelta(days=10)
    shared_ts = (utcnow() - timedelta(days=1)).isoformat()

    source_a = Fiber(
        id="src-a",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=1,
        created_at=entered,
    )
    source_b = Fiber(
        id="src-b",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-b",
        pathway=["n-b"],
        frequency=1,
        created_at=entered,
    )
    for f in (source_a, source_b):
        await store.add_fiber(f)

    for fid in ("src-a", "src-b"):
        await store.save_maturation(
            MaturationRecord(
                fiber_id=fid,
                brain_id="mrc-brain",
                stage=MemoryStage.EPISODIC,
                stage_entered_at=entered,
                rehearsal_count=1,
                reinforcement_timestamps=(shared_ts,),  # identical on both sources
            )
        )

    engine = ConsolidationEngine(store, ConsolidationConfig())
    await engine._merge(ConsolidationReport(), dry_run=False)

    remaining = await store.get_fibers(limit=100)
    merged = await store.get_maturation(remaining[0].id)
    assert merged is not None

    assert merged.reinforcement_timestamps == (shared_ts,), "the timestamp collapses to one"
    assert merged.rehearsal_count == len(merged.reinforcement_timestamps) == 1, (
        "rehearsal_count must track the deduplicated union (1), not a naive sum (2)"
    )


@pytest.mark.asyncio
async def test_merge_never_removes_reasoning_pattern_fiber() -> None:
    """A `_reasoning_pattern` fiber must survive _merge even at 100% neuron overlap.

    Same failure as the habit guard above, one marker along: merging deletes the
    member fibers and the merged fiber replaces their metadata wholesale, so the
    `_source_model` / `_reasoning_category` / `_reasoning_confidence` keys that
    category coverage is computed from disappear. Learned patterns are especially
    exposed because every pattern in a category shares that category's concept
    neuron, which is exactly what the overlap check keys on — so a mining run's
    whole output can be merged into one metadata-less fiber and coverage falls
    back to zero with the traces already marked processed.
    """
    store = InMemoryStorage()
    brain = Brain.create(name="merge_reasoning_test", brain_id="mr-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for nid in ("n-a", "n-b", "n-c"):
        await store.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    shared = {"n-a", "n-b", "n-c"}  # 100% overlap → Jaccard 1.0
    plain = Fiber(
        id="plain-1",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=5,
    )
    patterns = [
        Fiber(
            id=f"pattern-{i}",
            neuron_ids=shared,
            synapse_ids=set(),
            anchor_neuron_id="n-a",
            pathway=["n-a"],
            frequency=5,
            metadata={
                "_reasoning_pattern": True,
                "_source_model": "claude-fable-5",
                "_reasoning_category": "debugging",
                "_reasoning_confidence": 0.6,
            },
        )
        for i in (1, 2)
    ]
    for f in (plain, *patterns):
        await store.add_fiber(f)

    engine = ConsolidationEngine(store, ConsolidationConfig())
    await engine._merge(ConsolidationReport(), dry_run=False)

    survivors = await store.get_fibers(limit=100)
    remaining = {f.id for f in survivors}
    assert {"pattern-1", "pattern-2"} <= remaining, (
        "merge must never delete a _reasoning_pattern fiber"
    )
    for f in survivors:
        if f.id in {"pattern-1", "pattern-2"}:
            assert f.metadata.get("_source_model") == "claude-fable-5"
            assert f.metadata.get("_reasoning_category") == "debugging"


@pytest.mark.asyncio
async def test_merge_never_removes_pinned_fiber_without_pattern_markers() -> None:
    """A fiber pinned for any reason must survive _merge at 100% neuron overlap.

    _merge already skips fibers carrying `_habit_pattern` / `_reasoning_pattern`
    markers, so the pinned fiber here deliberately carries NEITHER: with such a
    marker the existing guard rescues it and the test passes with or without the
    `pinned` check, proving nothing. Unprotected today is every other reason a
    fiber is pinned — a trained knowledge base, or an explicit user pin. Merge
    deletes the member fibers and the merged fiber keeps only `merged_from`, so
    the pin goes with them.
    """
    store = InMemoryStorage()
    brain = Brain.create(name="merge_pinned_test", brain_id="mp-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for nid in ("n-a", "n-b", "n-c"):
        await store.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    shared = {"n-a", "n-b", "n-c"}  # 100% overlap → Jaccard 1.0, well above 0.5
    plain1 = Fiber(
        id="plain-1",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=5,
    )
    plain2 = Fiber(
        id="plain-2",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-b",
        pathway=["n-b"],
        frequency=5,
    )
    pinned_fiber = Fiber(
        id="pinned-1",
        neuron_ids=shared,
        synapse_ids=set(),
        anchor_neuron_id="n-a",
        pathway=["n-a"],
        frequency=5,
        pinned=True,
    )
    assert not any(
        marker in pinned_fiber.metadata for marker in ("_habit_pattern", "_reasoning_pattern")
    ), "the fixture must not qualify for the pre-existing marker guard"

    for f in (plain1, plain2, pinned_fiber):
        await store.add_fiber(f)

    engine = ConsolidationEngine(store, ConsolidationConfig())
    await engine._merge(ConsolidationReport(), dry_run=False)

    remaining = {f.id for f in await store.get_fibers(limit=100)}
    assert "pinned-1" in remaining, "merge must never delete a pinned fiber"

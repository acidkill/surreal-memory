from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)
from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.engine.semantic_discovery import SemanticDiscoveryResult

_REFERENCE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


class _Progress:
    def __init__(self, strategy: str, pause_phase: str | None = None) -> None:
        self.state: dict[str, Any] = {"strategy_states": {strategy: {}}}
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
        state = self.strategy_state(strategy)
        state.update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )
        self.writes.append(dict(state))
        if phase == self.pause_phase:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated interruption")


class _Storage:
    def __init__(
        self,
        *,
        fibers: list[Fiber] | None = None,
        maturations: list[MaturationRecord] | None = None,
        neurons: list[Neuron] | None = None,
        synapses: list[Synapse] | None = None,
    ) -> None:
        self.current_brain_id = "brain-1"
        self.brain = SimpleNamespace(config=SimpleNamespace(essence_generator="extractive"))
        self.fibers = {fiber.id: fiber for fiber in fibers or []}
        self.maturations = {item.fiber_id: item for item in maturations or []}
        self.neurons = {neuron.id: neuron for neuron in neurons or []}
        self.synapses = {synapse.id: synapse for synapse in synapses or []}
        self.added_neurons: list[str] = []
        self.added_synapses: list[str] = []
        self.saved_maturations: list[str] = []
        self.cleanup_calls = 0
        self.backfill_calls = 0

    async def get_brain(self, _brain_id: str) -> Any:
        return self.brain

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def get_fibers(self, *, limit: int) -> list[Fiber]:
        return list(self.fibers.values())[:limit]

    async def get_total_fiber_count(self) -> int:
        return len(self.fibers)

    async def find_maturations(self) -> list[MaturationRecord]:
        return list(self.maturations.values())

    async def save_maturation(self, record: MaturationRecord) -> None:
        self.saved_maturations.append(record.fiber_id)
        self.maturations[record.fiber_id] = record

    async def get_promotion_candidates(
        self, *, min_frequency: int, source_type: str
    ) -> list[dict[str, Any]]:
        return []

    async def promote_memory_type(self, **_kwargs: Any) -> bool:
        return False

    async def cleanup_orphaned_maturations(self) -> int:
        self.cleanup_calls += 1
        return 0

    async def backfill_maturations(self) -> dict[str, int]:
        self.backfill_calls += 1
        return {}

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        return self.neurons.get(neuron_id)

    async def add_neuron(self, neuron: Neuron) -> str:
        if neuron.id in self.neurons:
            raise ValueError("neuron already exists")
        self.added_neurons.append(neuron.id)
        self.neurons[neuron.id] = neuron
        return neuron.id

    async def find_neurons(self, *, type: NeuronType, limit: int, offset: int) -> list[Neuron]:
        found = sorted(
            (neuron for neuron in self.neurons.values() if neuron.type == type),
            key=lambda neuron: neuron.id,
        )
        return found[offset : offset + limit]

    async def get_synapses(
        self,
        *,
        type: SynapseType | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Synapse]:
        found = sorted(
            (edge for edge in self.synapses.values() if type is None or edge.type == type),
            key=lambda edge: edge.id,
        )
        return found[offset : offset + limit] if limit is not None else found

    async def get_synapses_paged(self, *, type: SynapseType) -> list[Synapse]:
        return await self.get_synapses(type=type)

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        return self.synapses.get(synapse_id)

    async def add_synapse(self, synapse: Synapse) -> str:
        if synapse.id in self.synapses:
            raise ValueError("synapse already exists")
        self.added_synapses.append(synapse.id)
        self.synapses[synapse.id] = synapse
        return synapse.id


def _engine(
    storage: _Storage,
    strategy: ConsolidationStrategy,
    progress: _Progress,
) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._active_strategy = strategy
    engine._progress_session = progress  # type: ignore[assignment]

    async def no_reactivation(_report: ConsolidationReport, _dry_run: bool) -> None:
        return None

    engine._reactivate_dormant = no_reactivation  # type: ignore[method-assign]
    return engine


def _causal_chain() -> list[Synapse]:
    return [
        Synapse.create(
            "cause-a",
            "cause-b",
            SynapseType.CAUSED_BY,
            weight=0.8,
            synapse_id="edge-ab",
        ),
        Synapse.create(
            "cause-b",
            "cause-c",
            SynapseType.CAUSED_BY,
            weight=0.6,
            synapse_id="edge-bc",
        ),
    ]


def _pattern_fixtures() -> tuple[list[Fiber], list[MaturationRecord]]:
    fibers = [
        Fiber.create(
            neuron_ids={"entity-shared", f"entity-{index}", f"anchor-{index}"},
            synapse_ids=set(),
            anchor_neuron_id=f"anchor-{index}",
            summary=f"synthetic event {index}",
            tags={"alpha", "shared"},
            fiber_id=f"fiber-{index}",
        )
        for index in range(3)
    ]
    maturations = [
        MaturationRecord(
            fiber_id=fiber.id,
            brain_id="brain-1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=_REFERENCE_TIME,
            rehearsal_count=3,
        )
        for fiber in fibers
    ]
    return fibers, maturations


@pytest.mark.asyncio
async def test_mature_replays_completed_stage_unit_without_saving_twice() -> None:
    record = MaturationRecord(
        fiber_id="fiber-stage",
        brain_id="brain-1",
        stage=MemoryStage.SHORT_TERM,
        stage_entered_at=_REFERENCE_TIME - timedelta(days=2),
    )
    storage = _Storage(maturations=[record])
    progress = _Progress("mature", pause_phase="mature_stage")
    engine = _engine(storage, ConsolidationStrategy.MATURE, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._mature(ConsolidationReport(), _REFERENCE_TIME, dry_run=False)

    assert storage.saved_maturations == ["fiber-stage"]
    assert storage.maturations["fiber-stage"].stage == MemoryStage.WORKING

    await _engine(storage, ConsolidationStrategy.MATURE, progress)._mature(
        ConsolidationReport(), _REFERENCE_TIME, dry_run=False
    )

    assert storage.saved_maturations == ["fiber-stage"]
    assert progress.strategy_state("mature")["phase"] == "mature_complete"


@pytest.mark.asyncio
async def test_mature_pattern_retry_reuses_saved_ids_without_duplicate_writes() -> None:
    fibers, maturations = _pattern_fixtures()
    storage = _Storage(fibers=fibers, maturations=maturations)
    progress = _Progress("mature", pause_phase="mature_pattern")
    engine = _engine(storage, ConsolidationStrategy.MATURE, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._mature(ConsolidationReport(), _REFERENCE_TIME, dry_run=False)

    assert len(storage.added_neurons) == 1
    assert len(storage.added_synapses) == 1

    await _engine(storage, ConsolidationStrategy.MATURE, progress)._mature(
        ConsolidationReport(), _REFERENCE_TIME, dry_run=False
    )

    assert len(storage.added_neurons) == 1
    assert len(storage.added_synapses) == 1


@pytest.mark.asyncio
async def test_mature_rejects_pattern_snapshot_when_fiber_data_changes() -> None:
    fibers, maturations = _pattern_fixtures()
    storage = _Storage(fibers=fibers, maturations=maturations)
    progress = _Progress("mature", pause_phase="mature_pattern_pending")

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.MATURE, progress)._mature(
            ConsolidationReport(), _REFERENCE_TIME, dry_run=False
        )

    changed = Fiber.create(
        neuron_ids={"entity-shared", "entity-changed", "anchor-0"},
        synapse_ids=set(),
        anchor_neuron_id="anchor-0",
        summary="changed synthetic event",
        tags={"different"},
        fiber_id="fiber-0",
    )
    storage.fibers[changed.id] = changed

    with pytest.raises(ConsolidationProgressError, match="source data changed"):
        await _engine(storage, ConsolidationStrategy.MATURE, progress)._mature(
            ConsolidationReport(), _REFERENCE_TIME, dry_run=False
        )

    assert storage.added_neurons == []
    assert storage.added_synapses == []


@pytest.mark.asyncio
async def test_mature_dry_run_does_not_change_storage_or_progress() -> None:
    record = MaturationRecord(
        fiber_id="fiber-stage",
        brain_id="brain-1",
        stage=MemoryStage.SHORT_TERM,
        stage_entered_at=_REFERENCE_TIME - timedelta(days=2),
    )
    storage = _Storage(maturations=[record])
    progress = _Progress("mature")
    before = json.loads(json.dumps(progress.state))

    await _engine(storage, ConsolidationStrategy.MATURE, progress)._mature(
        ConsolidationReport(), _REFERENCE_TIME, dry_run=True
    )

    assert storage.saved_maturations == []
    assert storage.cleanup_calls == 0
    assert storage.backfill_calls == 0
    assert progress.writes == []
    assert progress.state == before


@pytest.mark.asyncio
async def test_enrich_replays_saved_synapse_after_interruption_without_duplicate() -> None:
    storage = _Storage(synapses=_causal_chain())
    progress = _Progress("enrich", pause_phase="enrich_apply")
    engine = _engine(storage, ConsolidationStrategy.ENRICH, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._enrich(ConsolidationReport(), dry_run=False)

    assert len(storage.added_synapses) == 1
    pending = json.loads(progress.strategy_state("enrich")["pending"][0])
    saved_id = pending["synapses"][0]["id"]

    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.ENRICH, progress)._enrich(report, dry_run=False)

    assert storage.added_synapses == [saved_id]
    assert report.synapses_enriched == 1
    assert progress.strategy_state("enrich")["pending"] == []


@pytest.mark.asyncio
async def test_enrich_rejects_snapshot_when_causal_input_changes() -> None:
    storage = _Storage(synapses=_causal_chain())
    progress = _Progress("enrich", pause_phase="enrich_pending")

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.ENRICH, progress)._enrich(
            ConsolidationReport(), dry_run=False
        )

    storage.synapses["edge-bc"] = Synapse(
        **{
            **storage.synapses["edge-bc"].__dict__,
            "weight": 0.2,
        }
    )
    with pytest.raises(ConsolidationProgressError, match="source data changed"):
        await _engine(storage, ConsolidationStrategy.ENRICH, progress)._enrich(
            ConsolidationReport(), dry_run=False
        )

    assert storage.added_synapses == []


@pytest.mark.asyncio
async def test_enrich_dry_run_does_not_change_storage_or_progress() -> None:
    storage = _Storage(synapses=_causal_chain())
    progress = _Progress("enrich")
    before = json.loads(json.dumps(progress.state))
    before_synapses = dict(storage.synapses)

    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.ENRICH, progress)._enrich(report, dry_run=True)

    assert report.synapses_enriched == 1
    assert storage.synapses == before_synapses
    assert storage.added_synapses == []
    assert progress.writes == []
    assert progress.state == before


def _semantic_result() -> SemanticDiscoveryResult:
    edge = Synapse.create(
        "semantic-a",
        "semantic-b",
        SynapseType.SIMILAR_TO,
        weight=0.54,
        metadata={"_semantic_discovery": True},
        synapse_id="semantic-edge-a-b",
    )
    return SemanticDiscoveryResult(
        neurons_embedded=2,
        pairs_evaluated=1,
        synapses_created=1,
        eligible_total=2,
        synapses=[edge],
    )


async def _semantic_discovery(*_args: Any) -> SemanticDiscoveryResult:
    return _semantic_result()


@pytest.mark.asyncio
async def test_semantic_link_replays_saved_synapse_after_interruption_without_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT,
            f"concept {suffix}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"semantic-{suffix}",
        )
        for suffix in ("a", "b")
    ]
    storage = _Storage(neurons=neurons)
    progress = _Progress("semantic_link", pause_phase="semantic_link_apply")
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery.discover_semantic_synapses",
        _semantic_discovery,
    )

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert storage.added_synapses == ["semantic-edge-a-b"]
    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        report, dry_run=False
    )

    assert storage.added_synapses == ["semantic-edge-a-b"]
    assert report.semantic_synapses_created == 1
    assert progress.strategy_state("semantic_link")["pending"] == []


@pytest.mark.asyncio
async def test_semantic_link_rejects_snapshot_when_embedding_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT,
            f"concept {suffix}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"semantic-{suffix}",
        )
        for suffix in ("a", "b")
    ]
    storage = _Storage(neurons=neurons)
    progress = _Progress("semantic_link", pause_phase="semantic_link_pending")
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery.discover_semantic_synapses",
        _semantic_discovery,
    )

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    storage.neurons["semantic-a"] = storage.neurons["semantic-a"].with_metadata(
        _embedding=[0.0, 1.0]
    )
    with pytest.raises(ConsolidationProgressError, match="source data changed"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert storage.added_synapses == []


@pytest.mark.asyncio
async def test_semantic_link_dry_run_does_not_change_storage_or_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT,
            f"concept {suffix}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"semantic-{suffix}",
        )
        for suffix in ("a", "b")
    ]
    storage = _Storage(neurons=neurons)
    progress = _Progress("semantic_link")
    before = json.loads(json.dumps(progress.state))
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery.discover_semantic_synapses",
        _semantic_discovery,
    )

    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        report, dry_run=True
    )

    assert report.semantic_synapses_created == 1
    assert storage.added_synapses == []
    assert progress.writes == []
    assert progress.state == before

from __future__ import annotations

import json
import sys
from dataclasses import replace as dc_replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine import semantic_discovery as semantic_discovery_module
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
from surreal_memory.engine.semantic_discovery import (
    SemanticDiscoveryResult,
    discover_semantic_synapses,
)

_REFERENCE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


class _Progress:
    def __init__(
        self,
        strategy: str,
        pause_phase: str | None = None,
        *,
        pause_after_phase_calls: int = 1,
    ) -> None:
        self.state: dict[str, Any] = {"strategy_states": {strategy: {}}}
        self.pause_phase = pause_phase
        self.pause_after_phase_calls = pause_after_phase_calls
        self.phase_calls: dict[str, int] = {}
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
        self.phase_calls[phase] = self.phase_calls.get(phase, 0) + 1
        if phase == self.pause_phase and self.phase_calls[phase] == self.pause_after_phase_calls:
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

    async def find_neurons_after_id(
        self,
        cursor_id: str | None,
        *,
        limit: int,
        ephemeral: bool | None = None,
        include_embedding: bool = False,
    ) -> list[Neuron]:
        found = sorted(self.neurons.values(), key=lambda neuron: neuron.id)
        if cursor_id is not None:
            found = [neuron for neuron in found if neuron.id > cursor_id]
        return found[:limit]

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

    async def get_synapses_after_id(
        self,
        cursor_id: str | None,
        *,
        limit: int,
    ) -> list[Synapse]:
        found = sorted(self.synapses.values(), key=lambda edge: edge.id)
        if cursor_id is not None:
            found = [edge for edge in found if edge.id > cursor_id]
        return found[:limit]

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


async def _semantic_discovery(*_args: Any, **_kwargs: Any) -> SemanticDiscoveryResult:
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
async def test_semantic_link_resume_accepts_store_public_id_normalization_without_duplicate(
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
    planned = Synapse.create(
        "semantic-a",
        "semantic-b",
        SynapseType.SIMILAR_TO,
        weight=0.54,
        metadata={"_semantic_discovery": True},
        synapse_id="similar_to-" + "a" * 32,
    )
    result = SemanticDiscoveryResult(
        neurons_embedded=2,
        pairs_evaluated=1,
        synapses_created=1,
        eligible_total=2,
        synapses=[planned],
    )

    async def discover(*_args: Any, **_kwargs: Any) -> SemanticDiscoveryResult:
        return result

    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery.discover_semantic_synapses", discover
    )

    async def add_with_surreal_public_id(synapse: Synapse) -> str:
        persisted = dc_replace(synapse, id=synapse.id.replace("_", "-"))
        storage.added_synapses.append(synapse.id)
        storage.synapses[persisted.id] = persisted
        return persisted.id

    async def get_with_surreal_id(synapse_id: str) -> Synapse | None:
        return storage.synapses.get(synapse_id) or storage.synapses.get(
            synapse_id.replace("_", "-")
        )

    monkeypatch.setattr(storage, "add_synapse", add_with_surreal_public_id)
    monkeypatch.setattr(storage, "get_synapse", get_with_surreal_id)
    progress = _Progress("semantic_link")
    durable_checkpoint = progress.checkpoint

    async def fail_before_apply_checkpoint(
        strategy: str,
        phase: str,
        **kwargs: Any,
    ) -> None:
        if phase == "semantic_link_apply":
            raise ConsolidationPausedError("simulated post-write interruption")
        await durable_checkpoint(strategy, phase, **kwargs)

    monkeypatch.setattr(progress, "checkpoint", fail_before_apply_checkpoint)
    with pytest.raises(ConsolidationPausedError, match="post-write interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert storage.added_synapses == [planned.id]
    assert len(storage.synapses) == 1
    monkeypatch.setattr(progress, "checkpoint", durable_checkpoint)
    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        report, dry_run=False
    )

    assert storage.added_synapses == [planned.id]
    assert len(storage.synapses) == 1
    assert report.semantic_synapses_created == 1
    assert progress.strategy_state("semantic_link")["pending"] == []


@pytest.mark.asyncio
async def test_semantic_link_resumes_after_first_durable_discovery_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
            f"semantic candidate {index}",
            metadata={"_embedding": [1.0 - index * 0.01, index * 0.01]},
            neuron_id=f"semantic-node-{index}",
        )
        for index in range(4)
    ]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._EMBEDDING_PAGE_SIZE", 2)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._SYNAPSE_PAGE_SIZE", 2)

    expected_storage = _Storage(neurons=neurons)
    expected_storage.brain.config = config
    legacy_result = await discover_semantic_synapses(expected_storage, config)
    expected_synapse_ids = sorted(synapse.id for synapse in legacy_result.synapses)

    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_neurons",
        pause_after_phase_calls=2,
    )
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    checkpoint = progress.strategy_state("semantic_link")
    assert checkpoint["phase"] == "semantic_link_discovery_neurons"
    discovery_state = json.loads(checkpoint["pending"][0])
    assert discovery_state["stage"] == "neurons"
    assert discovery_state["cursor"] == "semantic-node-1"
    assert storage.added_synapses == []

    progress.pause_phase = "semantic_link_apply"
    progress.pause_after_phase_calls = 1
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert len(storage.added_synapses) == 1

    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert sorted(storage.added_synapses) == expected_synapse_ids
    assert len(storage.added_synapses) == len(set(storage.added_synapses))

    changed_storage = _Storage(neurons=neurons)
    changed_storage.brain.config = config
    changed_progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_neurons",
        pause_after_phase_calls=2,
    )
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(
            changed_storage, ConsolidationStrategy.SEMANTIC_LINK, changed_progress
        )._semantic_link(ConsolidationReport(), dry_run=False)
    changed_storage.neurons["semantic-node-0"] = changed_storage.neurons[
        "semantic-node-0"
    ].with_metadata(_embedding=[0.0, 1.0])
    with pytest.raises(ConsolidationProgressError, match="source changed before its cursor"):
        await _engine(
            changed_storage, ConsolidationStrategy.SEMANTIC_LINK, changed_progress
        )._semantic_link(ConsolidationReport(), dry_run=False)
    assert changed_storage.added_synapses == []


@pytest.mark.asyncio
async def test_semantic_link_resumes_after_synapse_keyset_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
            f"semantic candidate {index}",
            metadata={"_embedding": [1.0 - index * 0.01, index * 0.01]},
            neuron_id=f"semantic-node-{index}",
        )
        for index in range(4)
    ]
    existing = [
        Synapse.create(
            neurons[index].id,
            neurons[index + 1].id,
            SynapseType.CAUSED_BY,
            weight=0.8,
            synapse_id=f"existing-edge-{index}",
        )
        for index in range(3)
    ]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._EMBEDDING_PAGE_SIZE", 2)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._SYNAPSE_PAGE_SIZE", 2)

    expected_storage = _Storage(neurons=neurons, synapses=existing)
    legacy_result = await discover_semantic_synapses(expected_storage, config)
    expected_synapse_ids = sorted(synapse.id for synapse in legacy_result.synapses)

    storage = _Storage(neurons=neurons, synapses=existing)
    storage.brain.config = config
    progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_synapses",
        pause_after_phase_calls=2,
    )
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    checkpoint = progress.strategy_state("semantic_link")
    discovery_state = json.loads(checkpoint["pending"][0])
    assert checkpoint["phase"] == "semantic_link_discovery_synapses"
    assert discovery_state["stage"] == "synapses"
    assert discovery_state["cursor"] == "existing-edge-1"

    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert sorted(storage.added_synapses) == expected_synapse_ids
    assert len(storage.added_synapses) == len(set(storage.added_synapses))


def _semantic_similarity_fixture(
    neuron_count: int = 6,
) -> tuple[list[Neuron], SimpleNamespace]:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
            f"semantic candidate {index}",
            metadata={"_embedding": [1.0 - index * 0.01, index * 0.01]},
            neuron_id=f"semantic-node-{index:02d}",
        )
        for index in range(neuron_count)
    ]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )
    return neurons, config


def _force_semantic_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._EMBEDDING_PAGE_SIZE", 2)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._SYNAPSE_PAGE_SIZE", 2)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._YIELD_EVERY_ROWS", 2)


@pytest.mark.asyncio
async def test_semantic_link_resumes_similarity_cursor_without_recomputing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "_MAX_SIMILARITY_CHECKPOINTS", 4)
    original_cosine = semantic_discovery_module._cosine_similarity
    similarity_calls = 0

    def counted_cosine(left: list[float], right: list[float]) -> float:
        nonlocal similarity_calls
        similarity_calls += 1
        return original_cosine(left, right)

    monkeypatch.setattr(semantic_discovery_module, "_cosine_similarity", counted_cosine)
    expected_storage = _Storage(neurons=neurons)
    legacy_result = await discover_semantic_synapses(expected_storage, config)
    expected_synapse_ids = sorted(synapse.id for synapse in legacy_result.synapses)

    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_similarity",
        pause_after_phase_calls=3,
    )
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    checkpoint = progress.strategy_state("semantic_link")
    discovery_state = json.loads(checkpoint["pending"][0])
    assert checkpoint["phase"] == "semantic_link_discovery_similarity"
    assert discovery_state["stage"] == "similarity"
    assert discovery_state["version"] == 2
    assert discovery_state["cursor"] == "3"
    assert discovery_state["similarity_backend"] == "python_float64"
    assert len(discovery_state["results"]) == discovery_state["synapses_created"]
    assert all(len(result) == 3 for result in discovery_state["results"])
    assert storage.added_synapses == []

    similarity_calls = 0
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert similarity_calls == 2 * (len(neurons) - 1)
    assert sorted(storage.added_synapses) == expected_synapse_ids
    assert len(storage.added_synapses) == len(set(storage.added_synapses))
    assert progress.phase_calls["semantic_link_discovery_similarity"] == 4
    for write in progress.writes:
        if write["phase"] == "semantic_link_discovery_similarity" and write["cursor"] is not None:
            pending_bytes = len(
                json.dumps(write["pending"], ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            assert pending_bytes <= semantic_discovery_module._MAX_SIMILARITY_CHECKPOINT_BYTES


@pytest.mark.asyncio
async def test_semantic_link_budget_pause_checkpoints_last_completed_similarity_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "_MAX_SIMILARITY_CHECKPOINTS", 4)
    original_cosine = semantic_discovery_module._cosine_similarity
    similarity_calls = 0

    def counted_cosine(left: list[float], right: list[float]) -> float:
        nonlocal similarity_calls
        similarity_calls += 1
        return original_cosine(left, right)

    monkeypatch.setattr(semantic_discovery_module, "_cosine_similarity", counted_cosine)
    expected = await discover_semantic_synapses(_Storage(neurons=neurons), config)
    similarity_calls = 0
    snapshots: list[dict[str, Any]] = []

    async def save_checkpoint(_phase: str, _cursor: str | None, details: dict[str, Any]) -> None:
        snapshots.append(details)

    budget_checks = 0

    async def pause_before_fourth_row() -> None:
        nonlocal budget_checks
        budget_checks += 1
        if budget_checks == 4:
            raise ConsolidationPausedError("simulated budget exhaustion")

    with pytest.raises(ConsolidationPausedError, match="budget exhaustion"):
        await discover_semantic_synapses(
            _Storage(neurons=neurons),
            config,
            checkpoint=save_checkpoint,
            budget_check=pause_before_fourth_row,
        )

    durable = snapshots[-1]
    assert durable["stage"] == "similarity"
    assert durable["version"] == 2
    assert durable["cursor"] == "2"
    assert len(durable["results"]) == durable["synapses_created"]
    assert similarity_calls == 3 * (len(neurons) - 1)

    similarity_calls = 0
    resumed = await discover_semantic_synapses(
        _Storage(neurons=neurons),
        config,
        checkpoint=save_checkpoint,
        resume_state=durable,
        budget_check=lambda: _completed_budget_check(),
    )

    assert similarity_calls == (len(neurons) - 3) * (len(neurons) - 1)
    assert sorted(edge.id for edge in resumed.synapses) == sorted(
        edge.id for edge in expected.synapses
    )
    assert len({edge.id for edge in resumed.synapses}) == len(resumed.synapses)


async def _completed_budget_check() -> None:
    return None


@pytest.mark.asyncio
async def test_semantic_link_resume_at_pair_cap_does_not_create_more_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    config.semantic_discovery_max_pairs = 1
    _force_semantic_python_fallback(monkeypatch)
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_similarity",
        pause_after_phase_calls=2,
    )

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    discovery_state = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    assert discovery_state["cursor"] == "0"
    assert discovery_state["synapses_created"] == 1
    assert len(discovery_state["results"]) == 1

    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert len(storage.added_synapses) == 1
    assert len(storage.added_synapses) == len(set(storage.added_synapses))


@pytest.mark.asyncio
async def test_semantic_link_rejects_changed_source_after_similarity_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_similarity",
        pause_after_phase_calls=3,
    )
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    storage.neurons["semantic-node-00"] = storage.neurons["semantic-node-00"].with_metadata(
        _embedding=[0.0, 1.0]
    )
    with pytest.raises(ConsolidationProgressError, match="source changed"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert storage.added_synapses == []


@pytest.mark.asyncio
async def test_semantic_link_restarts_from_last_durable_similarity_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    original_cosine = semantic_discovery_module._cosine_similarity
    similarity_calls = 0

    def counted_cosine(left: list[float], right: list[float]) -> float:
        nonlocal similarity_calls
        similarity_calls += 1
        return original_cosine(left, right)

    monkeypatch.setattr(semantic_discovery_module, "_cosine_similarity", counted_cosine)
    expected_storage = _Storage(neurons=neurons)
    legacy_result = await discover_semantic_synapses(expected_storage, config)
    expected_synapse_ids = sorted(synapse.id for synapse in legacy_result.synapses)

    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress("semantic_link")
    durable_checkpoint = progress.checkpoint
    similarity_checkpoint_calls = 0

    async def fail_before_durable_write(
        strategy: str,
        phase: str,
        **kwargs: Any,
    ) -> None:
        nonlocal similarity_checkpoint_calls
        if phase == "semantic_link_discovery_similarity":
            similarity_checkpoint_calls += 1
            if similarity_checkpoint_calls == 3:
                raise OSError("simulated checkpoint persistence failure")
        await durable_checkpoint(strategy, phase, **kwargs)

    monkeypatch.setattr(progress, "checkpoint", fail_before_durable_write)
    similarity_calls = 0
    with pytest.raises(OSError, match="checkpoint persistence failure"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    checkpoint = progress.strategy_state("semantic_link")
    discovery_state = json.loads(checkpoint["pending"][0])
    assert checkpoint["phase"] == "semantic_link_discovery_similarity"
    assert discovery_state["version"] == 2
    assert discovery_state["cursor"] == "1"
    assert storage.added_synapses == []

    monkeypatch.setattr(progress, "checkpoint", durable_checkpoint)
    similarity_calls = 0
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert similarity_calls == (len(neurons) - 2) * (len(neurons) - 1)
    assert sorted(storage.added_synapses) == expected_synapse_ids
    assert len(storage.added_synapses) == len(set(storage.added_synapses))


@pytest.mark.asyncio
async def test_semantic_link_fails_when_similarity_checkpoint_exceeds_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "_MAX_SIMILARITY_CHECKPOINT_BYTES", 1)
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress("semantic_link")

    with pytest.raises(ConsolidationProgressError, match="payload exceeds"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert storage.added_synapses == []


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

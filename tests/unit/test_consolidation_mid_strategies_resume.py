from __future__ import annotations

import bisect
import json
import sys
import weakref
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
        self.state: dict[str, Any] = {
            "strategy_states": {strategy: {}},
            "run_id": "test-run",
        }
        self.owner_token = "fixture-" + "owner"
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
        self.semantic_source_generation = 0
        self.semantic_discovery_states: dict[tuple[str, int], dict[str, Any]] = {}

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

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        return {
            neuron_id: self.neurons[neuron_id]
            for neuron_id in neuron_ids
            if neuron_id in self.neurons
        }

    async def capture_semantic_source_token(self) -> str:
        return str(self.semantic_source_generation)

    async def assert_semantic_source_unchanged(self, token: str) -> None:
        if token != str(self.semantic_source_generation):
            raise RuntimeError("semantic source mutation fence detected a write")

    async def save_semantic_discovery_state(
        self, state_id: str, revision: int, payload: dict[str, Any]
    ) -> None:
        key = (state_id, revision)
        if key in self.semantic_discovery_states:
            normalized = json.loads(json.dumps(payload))
            if self.semantic_discovery_states[key] != normalized:
                prior = self.semantic_discovery_states[key]
                changed = [
                    name
                    for name in set(prior) | set(normalized)
                    if prior.get(name) != normalized.get(name)
                ]
                raise RuntimeError(f"semantic discovery state revisions are immutable: {changed}")
            return
        self.semantic_discovery_states[key] = json.loads(json.dumps(payload))

    async def load_semantic_discovery_state(
        self, state_id: str, revision: int
    ) -> dict[str, Any] | None:
        payload = self.semantic_discovery_states.get((state_id, revision))
        return json.loads(json.dumps(payload)) if payload is not None else None

    async def find_existing_synapse_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> set[tuple[str, str]]:
        requested = {tuple(sorted(pair)) for pair in pairs}
        return {
            tuple(sorted((synapse.source_id, synapse.target_id)))
            for synapse in self.synapses.values()
            if tuple(sorted((synapse.source_id, synapse.target_id))) in requested
        }

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


def _staged_semantic_state(
    storage: _Storage, pending: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(pending[0])
    ref = manifest.get("source_state_ref")
    assert isinstance(ref, dict)
    state = storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]
    return manifest, state


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
    changed_storage.semantic_source_generation += 1
    with pytest.raises(ConsolidationProgressError, match="source mutation fence detected a write"):
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
    assert discovery_state["cursor"] == "existing-edge-2"

    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert sorted(storage.added_synapses) == expected_synapse_ids
    assert len(storage.added_synapses) == len(set(storage.added_synapses))


@pytest.mark.asyncio
async def test_semantic_link_bounds_candidate_memory_and_replays_large_neuron_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neuron_count = 12_000
    page_size = 100
    candidate_cap = 8

    class _SyntheticPagedStorage(_Storage):
        def __init__(self) -> None:
            super().__init__()
            self.live_neurons = 0
            self.max_live_neurons = 0
            self.degree_map = {
                f"{kind}-{index:05d}": 0
                for kind in ("concept", "entity")
                for index in range(neuron_count // 2)
            }
            self.ordered_ids = sorted(self.degree_map)

        def _release_neuron(self) -> None:
            self.live_neurons -= 1

        async def find_neurons_after_id(
            self,
            cursor_id: str | None,
            *,
            limit: int,
            ephemeral: bool | None = None,
            include_embedding: bool = False,
        ) -> list[Neuron]:
            del ephemeral, include_embedding
            start = bisect.bisect_right(self.ordered_ids, cursor_id) if cursor_id else 0
            page_ids = self.ordered_ids[start : start + limit]
            page = []
            for neuron_id in page_ids:
                neuron = Neuron.create(
                    NeuronType.CONCEPT if neuron_id.startswith("concept-") else NeuronType.ENTITY,
                    neuron_id,
                    metadata={"_embedding": [1.0, 0.0, 0.0, 0.0]},
                    neuron_id=neuron_id,
                )
                self.live_neurons += 1
                weakref.finalize(neuron, self._release_neuron)
                page.append(neuron)
            self.max_live_neurons = max(self.max_live_neurons, self.live_neurons)
            return page

        async def get_synapse_degrees(self) -> dict[str, int]:
            return self.degree_map

        async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
            return {
                neuron_id: Neuron.create(
                    NeuronType.CONCEPT if neuron_id.startswith("concept-") else NeuronType.ENTITY,
                    neuron_id,
                    metadata={"_embedding": [1.0, 0.0, 0.0, 0.0]},
                    neuron_id=neuron_id,
                )
                for neuron_id in neuron_ids
            }

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
    monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", page_size)
    monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", candidate_cap)

    expected_storage = _SyntheticPagedStorage()
    expected_storage.brain.config = config
    expected_progress = _Progress("semantic_link")
    await _engine(
        expected_storage, ConsolidationStrategy.SEMANTIC_LINK, expected_progress
    )._semantic_link(ConsolidationReport(), dry_run=False)
    expected_synapses = sorted(expected_storage.added_synapses)

    storage = _SyntheticPagedStorage()
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
    discovery_state = json.loads(checkpoint["pending"][0])
    assert checkpoint["phase"] == "semantic_link_discovery_neurons"
    assert discovery_state["stage"] == "neurons"
    assert discovery_state["cursor"] == "concept-00099"
    assert storage.added_synapses == []

    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    assert sorted(storage.added_synapses) == expected_synapses
    assert len(storage.added_synapses) == len(set(storage.added_synapses))
    assert progress.writes[-1]["phase"] == "semantic_link_complete"
    discovery_writes = [
        write
        for write in progress.writes
        if write["phase"].startswith("semantic_link_discovery_neurons")
    ]
    _manifest, source_state = _staged_semantic_state(storage, discovery_writes[-1]["pending"])
    assert source_state["eligible_total"] == neuron_count
    assert storage.max_live_neurons <= candidate_cap + (2 * page_size)
    expected_selected = {
        *(f"concept-{index:05d}" for index in range(candidate_cap // 2)),
        *(f"entity-{index:05d}" for index in range(candidate_cap // 2)),
    }
    assert all(
        endpoint in expected_selected
        for synapse_id in storage.added_synapses
        for endpoint in (
            storage.synapses[synapse_id].source_id,
            storage.synapses[synapse_id].target_id,
        )
    )


@pytest.mark.asyncio
async def test_semantic_link_no_degree_fallback_keeps_interleaved_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            neuron_type,
            f"{neuron_type.value} candidate {index}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"{prefix}-{index}",
        )
        for neuron_type, prefix in (
            (NeuronType.CONCEPT, "concept"),
            (NeuronType.ENTITY, "entity"),
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
    monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", 2)
    monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 4)

    legacy = await discover_semantic_synapses(_Storage(neurons=neurons), config)
    resumable_storage = _Storage(neurons=neurons)
    checkpoints: list[dict[str, Any]] = []

    async def checkpoint(_stage: str, _cursor: str | None, details: dict[str, Any]) -> None:
        checkpoints.append(details)

    resumable = await discover_semantic_synapses(resumable_storage, config, checkpoint=checkpoint)

    assert sorted(synapse.id for synapse in resumable.synapses) == sorted(
        synapse.id for synapse in legacy.synapses
    )
    assert resumable.eligible_total == 8
    neuron_checkpoints = [item for item in checkpoints if item["stage"] == "neurons"]
    _manifest, state = _staged_semantic_state(
        resumable_storage, [json.dumps(neuron_checkpoints[-1])]
    )
    assert state["last_scan_details"]["eligible_neurons"] == 8
    expected_candidates = {"concept-0", "concept-1", "entity-0", "entity-1"}
    assert all(
        endpoint in expected_candidates
        for synapse in resumable.synapses
        for endpoint in (synapse.source_id, synapse.target_id)
    )


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
    assert discovery_state["version"] == 3
    assert discovery_state["cursor"] == "3"
    _manifest, source_state = _staged_semantic_state(storage, checkpoint["pending"])
    assert source_state["run_id"] == "test-run"
    assert source_state["owner_token"] == "fixture-" + "owner"
    legacy = source_state["legacy_checkpoint"]
    assert legacy["similarity_backend"] == "python_float64"
    assert len(legacy["results"]) == legacy["synapses_created"]
    assert all(len(result) == 3 for result in legacy["results"])
    assert storage.added_synapses == []

    similarity_calls = 0
    progress.owner_token = "new-" + "owner"
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )

    latest_discovery = next(
        write
        for write in reversed(progress.writes)
        if write["phase"].startswith("semantic_link_discovery_")
    )
    _manifest, resumed_source_state = _staged_semantic_state(storage, latest_discovery["pending"])
    assert resumed_source_state["run_id"] == "test-run"
    # The immutable revision retains its original owner provenance on takeover,
    # keeping an orphaned same-revision retry byte-for-byte deterministic.
    assert resumed_source_state["owner_token"] == "fixture-" + "owner"
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
            assert pending_bytes <= semantic_discovery_module._MAX_NEURON_CHECKPOINT_BYTES


@pytest.mark.asyncio
async def test_semantic_link_preserves_similarity_checkpoint_during_source_replay(
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

    durable = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    assert durable["stage"] == "similarity"
    assert durable["version"] == 3
    assert durable["cursor"] == "3"
    replay_writes: list[tuple[str, str | None]] = []

    async def checkpoint(stage: str, cursor: str | None, _details: dict[str, Any]) -> None:
        replay_writes.append((stage, cursor))

    async def pause_before_next_similarity_row() -> None:
        raise ConsolidationPausedError("simulated replay budget exhaustion")

    with pytest.raises(ConsolidationPausedError, match="replay budget exhaustion"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=durable,
            budget_check=pause_before_next_similarity_row,
        )

    # The external source state resumes without regressing the committed row.
    assert replay_writes == []
    assert durable["cursor"] == "3"


def test_semantic_sha256_state_round_trips_across_block_boundaries() -> None:
    import hashlib

    for size in (0, 1, 55, 56, 63, 64, 65, 127, 1024):
        payload = bytes((index * 37) % 256 for index in range(size))
        expected = hashlib.sha256(payload).hexdigest()
        original = semantic_discovery_module._SerializableSHA256()
        split = size // 2
        original.update(payload[:split])
        restored = semantic_discovery_module._SerializableSHA256(original.export())
        restored.update(payload[split:])
        assert restored.hexdigest() == expected


@pytest.mark.asyncio
async def test_semantic_v1_similarity_rebuild_advances_without_replacing_legacy_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "_YIELD_EVERY_ROWS", 1)
    storage = _Storage(neurons=neurons)
    snapshots: list[dict[str, Any]] = []

    async def save_checkpoint(_stage: str, _cursor: str | None, details: dict[str, Any]) -> None:
        snapshots.append(details)

    checks = 0

    async def pause_after_two_rows() -> None:
        nonlocal checks
        if storage.semantic_discovery_states and list(storage.semantic_discovery_states.values())[
            -1
        ].get("source_complete"):
            checks += 1
            if checks == 3:
                raise ConsolidationPausedError("simulated legacy checkpoint")

    with pytest.raises(ConsolidationPausedError, match="legacy checkpoint"):
        await discover_semantic_synapses(
            storage, config, checkpoint=save_checkpoint, budget_check=pause_after_two_rows
        )
    manifest, state = _staged_semantic_state(storage, [json.dumps(snapshots[-1])])
    legacy_v2 = state["legacy_checkpoint"]
    assert legacy_v2["cursor"] == "1"

    # Model a deployed v1 checkpoint: digest/counters exist, but no serialized
    # result vector or per-row rebuild accumulator exists.
    legacy_v1 = {
        key: value
        for key, value in legacy_v2.items()
        if key not in {"results", "similarity_backend"}
    }
    legacy_v1["version"] = 1
    next_revision = manifest["source_state_ref"]["revision"] + 1
    migrated_state = {**state, "legacy_checkpoint": legacy_v1}
    migrated_state["state_revision"] = next_revision
    migrated_state["revision"] = next_revision
    await storage.save_semantic_discovery_state(
        manifest["source_state_ref"]["state_id"], next_revision, migrated_state
    )
    legacy_pointer = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "similarity",
        "cursor": "1",
        "source_state_ref": {
            "state_id": manifest["source_state_ref"]["state_id"],
            "revision": next_revision,
        },
    }

    budget_calls = 0

    async def pause_before_second_replayed_row() -> None:
        nonlocal budget_calls
        budget_calls += 1
        if budget_calls == 2:
            raise ConsolidationPausedError("simulated bounded replay deadline")

    with pytest.raises(ConsolidationPausedError, match="bounded replay deadline"):
        await discover_semantic_synapses(
            storage,
            config,
            checkpoint=save_checkpoint,
            resume_state=legacy_pointer,
            budget_check=pause_before_second_replayed_row,
        )
    after_first_replay = snapshots[-1]
    assert after_first_replay["version"] == 3
    assert after_first_replay["cursor"] == "1"
    _manifest, replay_state = _staged_semantic_state(storage, [json.dumps(after_first_replay)])
    assert replay_state["legacy_checkpoint"]["version"] == 1
    assert replay_state["similarity_rebuild"]["next_row"] == 1
    assert replay_state["similarity_rebuild"]["cursor_verified"] is False
    assert replay_state["similarity_rebuild"]["results"]

    budget_calls = 0

    async def pause_after_legacy_cursor_verification() -> None:
        nonlocal budget_calls
        budget_calls += 1
        if budget_calls == 2:
            raise ConsolidationPausedError("simulated post-migration deadline")

    with pytest.raises(ConsolidationPausedError, match="post-migration deadline"):
        await discover_semantic_synapses(
            storage,
            config,
            checkpoint=save_checkpoint,
            resume_state=after_first_replay,
            budget_check=pause_after_legacy_cursor_verification,
        )
    promoted = snapshots[-1]
    assert promoted["version"] == 3
    assert promoted["cursor"] == "1"
    _manifest, promoted_state = _staged_semantic_state(storage, [json.dumps(promoted)])
    assert promoted_state["legacy_checkpoint"]["version"] == 2
    assert promoted_state["legacy_checkpoint"]["cursor"] == "1"
    assert promoted_state["similarity_rebuild"] is None

    expected = await discover_semantic_synapses(_Storage(neurons=neurons), config)
    resumed = await discover_semantic_synapses(
        storage, config, checkpoint=save_checkpoint, resume_state=promoted
    )
    assert sorted(edge.id for edge in resumed.synapses) == sorted(
        edge.id for edge in expected.synapses
    )
    assert len(storage.added_synapses) == 0


@pytest.mark.asyncio
async def test_semantic_v1_synapse_prefix_rebuild_advances_across_budgeted_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    edges = [
        Synapse.create(
            neurons[index % len(neurons)].id,
            neurons[(index + 1) % len(neurons)].id,
            SynapseType.CAUSED_BY,
            weight=0.8,
            synapse_id=f"legacy-edge-{index:02d}",
        )
        for index in range(8)
    ]
    storage = _Storage(neurons=neurons, synapses=edges)
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(
        semantic_discovery_module,
        "_effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr(semantic_discovery_module, "_SYNAPSE_PAGE_SIZE", 2)

    expected_prefix = semantic_discovery_module._SemanticFingerprintBuilder(
        storage.current_brain_id, config
    )
    for neuron in neurons:
        expected_prefix.add_neuron(neuron)
    neuron_digest = expected_prefix.prefix_digest()
    expected_prefix.begin_edges()
    for edge in edges[:2]:
        expected_prefix.add_synapse(edge)
    durable: dict[str, Any] = {
        "kind": "semantic_link_discovery",
        "version": 1,
        "stage": "synapses",
        "cursor": edges[1].id,
        "prefix_digest": expected_prefix.prefix_digest(),
        "neuron_digest": neuron_digest,
    }
    phase = "neurons"
    edge_page_budget_checks = 0

    async def checkpoint(stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        nonlocal durable, phase
        durable = details | {"cursor": cursor}
        phase = stage

    async def pause_after_edge_page() -> None:
        nonlocal edge_page_budget_checks
        if phase == "synapses":
            edge_page_budget_checks += 1
            if edge_page_budget_checks == 1:
                raise ConsolidationPausedError("simulated first edge replay deadline")

    with pytest.raises(ConsolidationPausedError, match="first edge replay deadline"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=durable,
            budget_check=pause_after_edge_page,
        )
    assert durable["version"] == 3
    assert durable["stage"] == "synapses"
    assert durable["cursor"] == edges[1].id
    _manifest, first_rebuild = _staged_semantic_state(storage, [json.dumps(durable)])
    assert first_rebuild["legacy_edge_cursor"] == edges[1].id
    assert first_rebuild["legacy_edge_verified"] is True

    edge_page_budget_checks = 0

    async def pause_after_next_edge_page() -> None:
        nonlocal edge_page_budget_checks
        if phase == "synapses":
            edge_page_budget_checks += 1
            if edge_page_budget_checks == 2:
                raise ConsolidationPausedError("simulated second edge replay deadline")

    first_durable = durable
    with pytest.raises(ConsolidationPausedError, match="second edge replay deadline"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=first_durable,
            budget_check=pause_after_next_edge_page,
        )
    assert durable["cursor"] == edges[3].id
    _manifest, second_rebuild = _staged_semantic_state(storage, [json.dumps(durable)])
    assert second_rebuild["edge_cursor"] == edges[3].id
    assert second_rebuild["legacy_edge_verified"] is True

    result = await semantic_discovery_module._discover_semantic_synapses_resumable(
        storage,
        config,
        checkpoint=checkpoint,
        resume_state=durable,
        budget_check=None,
    )
    assert result.source_fingerprint is not None
    assert durable["stage"] == "similarity"

    storage.semantic_source_generation += 1
    with pytest.raises(RuntimeError, match="mutation fence detected a write"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=first_durable,
            budget_check=None,
        )


@pytest.mark.asyncio
async def test_semantic_discovery_manifest_stays_small_for_ten_thousand_long_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
            f"long candidate {index}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"candidate-{index:05d}-{'x' * 40}",
        )
        for index in range(10_001)
    ]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=20,
        essence_generator="extractive",
    )
    monkeypatch.setattr(
        semantic_discovery_module,
        "_effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", 10_000)
    storage = _Storage(neurons=neurons)
    durable: dict[str, Any] = {}

    async def checkpoint(_stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        nonlocal durable
        durable = details | {"cursor": cursor}

    calls = 0

    async def one_page_budget() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConsolidationPausedError("simulated candidate state budget")

    with pytest.raises(ConsolidationPausedError, match="candidate state budget"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=None,
            budget_check=one_page_budget,
        )

    manifest_bytes = len(
        json.dumps([json.dumps(durable, separators=(",", ":"))], separators=(",", ":")).encode()
    )
    assert durable["version"] == 3
    assert manifest_bytes < 512 * 1024
    _manifest, source_state = _staged_semantic_state(storage, [json.dumps(durable)])
    assert len(source_state["candidates"]) == 10_000
    assert all(len(candidate[0]) > 50 for candidate in source_state["candidates"])
    source_state_bytes = len(
        json.dumps(source_state, ensure_ascii=False, separators=(",", ":")).encode()
    )
    assert source_state_bytes < semantic_discovery_module._MAX_REBUILD_STATE_BYTES


@pytest.mark.asyncio
async def test_semantic_v1_neuron_checkpoint_at_36988_advances_in_bounded_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix_length = 36_988
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
            f"large checkpoint neuron {index}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"neuron-{index:05d}",
        )
        for index in range(prefix_length + 2)
    ]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=20,
        essence_generator="extractive",
    )
    monkeypatch.setattr(
        semantic_discovery_module,
        "_effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", 5000)
    monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 2)
    storage = _Storage(neurons=neurons)
    fingerprint = semantic_discovery_module._SemanticFingerprintBuilder(
        storage.current_brain_id, config
    )
    for neuron in neurons[:prefix_length]:
        fingerprint.add_neuron(neuron)
    durable: dict[str, Any] = {
        "kind": "semantic_link_discovery",
        "version": 1,
        "stage": "neurons",
        "cursor": neurons[prefix_length - 1].id,
        "prefix_digest": fingerprint.prefix_digest(),
    }

    async def checkpoint(stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        nonlocal durable
        durable = details | {"cursor": cursor}

    for _ in range(10):
        budget_checks = 0

        async def two_page_budget() -> None:
            nonlocal budget_checks
            budget_checks += 1
            if budget_checks == 3:
                raise ConsolidationPausedError("simulated 600-second page budget")

        try:
            await semantic_discovery_module._discover_semantic_synapses_resumable(
                storage,
                config,
                checkpoint=checkpoint,
                resume_state=durable,
                budget_check=two_page_budget,
            )
        except ConsolidationPausedError as exc:
            if "600-second page budget" not in str(exc):
                raise
        _manifest, rebuild = _staged_semantic_state(storage, [json.dumps(durable)])
        if rebuild["legacy_neuron_verified"]:
            break
    else:
        pytest.fail("v1 prefix rebuild did not reach its 36,988-row commit cursor")

    assert rebuild["legacy_neuron_cursor"] == neurons[prefix_length - 1].id
    assert rebuild["neuron_cursor"] >= neurons[prefix_length - 1].id
    assert rebuild["legacy_neuron_verified"] is True
    assert durable["version"] == 3
    manifest_bytes = len(
        json.dumps([json.dumps(durable, separators=(",", ":"))], separators=(",", ":")).encode()
    )
    assert manifest_bytes < 512 * 1024


@pytest.mark.asyncio
async def test_semantic_v1_neuron_prefix_rebuild_is_bounded_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 2)
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    expected = semantic_discovery_module._SemanticFingerprintBuilder(
        storage.current_brain_id, config
    )
    for neuron in neurons[:4]:
        expected.add_neuron(neuron)
    durable: dict[str, Any] = {
        "kind": "semantic_link_discovery",
        "version": 1,
        "stage": "neurons",
        "cursor": neurons[3].id,
        "prefix_digest": expected.prefix_digest(),
    }
    fetch_cursors: list[str | None] = []
    original_fetch = storage.find_neurons_after_id

    async def observed_fetch(cursor_id: str | None, **kwargs: Any) -> list[Neuron]:
        fetch_cursors.append(cursor_id)
        return await original_fetch(cursor_id, **kwargs)

    storage.find_neurons_after_id = observed_fetch  # type: ignore[method-assign]

    async def checkpoint(stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        nonlocal durable
        assert stage == details["stage"]
        durable = details | {"cursor": cursor}

    async def one_page_budget() -> None:
        nonlocal budget_calls
        budget_calls += 1
        if budget_calls == 2:
            raise ConsolidationPausedError("simulated rebuild deadline")

    budget_calls = 0
    with pytest.raises(ConsolidationPausedError, match="rebuild deadline"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=durable,
            budget_check=one_page_budget,
        )
    assert durable["version"] == 3
    assert durable["cursor"] == neurons[1].id
    _manifest, rebuild = _staged_semantic_state(storage, [json.dumps(durable)])
    assert rebuild["legacy_neuron_cursor"] == neurons[3].id
    assert rebuild["legacy_neuron_verified"] is False
    durable_bytes = len(
        json.dumps(
            [json.dumps(durable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert durable_bytes <= semantic_discovery_module._MAX_NEURON_CHECKPOINT_BYTES
    assert all(len(candidate) == 3 for candidate in rebuild["candidates"])

    first_durable = durable
    budget_calls = 0
    with pytest.raises(ConsolidationPausedError, match="rebuild deadline"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=first_durable,
            budget_check=one_page_budget,
        )
    assert durable["cursor"] == neurons[3].id
    _manifest, rebuild = _staged_semantic_state(storage, [json.dumps(durable)])
    assert rebuild["legacy_neuron_verified"] is True

    result = await semantic_discovery_module._discover_semantic_synapses_resumable(
        storage,
        config,
        checkpoint=checkpoint,
        resume_state=durable,
        budget_check=None,
    )
    assert result.eligible_total == len(neurons)
    assert result.synapses
    assert all(
        endpoint in {neurons[0].id, neurons[1].id}
        for synapse in result.synapses
        for endpoint in (synapse.source_id, synapse.target_id)
    )
    assert fetch_cursors == [None, neurons[1].id, neurons[3].id, neurons[5].id]

    storage.semantic_source_generation += 1
    with pytest.raises(RuntimeError, match="mutation fence detected a write"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=first_durable,
            budget_check=None,
        )


@pytest.mark.asyncio
async def test_semantic_v3_rebuild_rejects_malformed_serialized_digest() -> None:
    neurons, config = _semantic_similarity_fixture()
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    malformed = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "cursor": None,
        "source_rebuild": {
            "source_token": "0",
            "legacy_checkpoint": {
                "kind": "semantic_link_discovery",
                "version": 1,
                "stage": "neurons",
                "cursor": None,
            },
            "fingerprint": {
                "sha256": {"h": [], "n": 0, "b": ""},
                "neuron_count": 0,
                "edge_count": 0,
                "edges_started": False,
                "finished": False,
            },
            "eligible_total": 0,
            "type_ranks": {"concept": 0, "entity": 0},
            "candidates": [],
            "bounded": False,
            "degree_mode": False,
        },
    }

    async def checkpoint(_stage: str, _cursor: str | None, _details: dict[str, Any]) -> None:
        raise AssertionError("malformed checkpoint must fail before writing")

    with pytest.raises(RuntimeError, match="serialized SHA-256 state"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=malformed,
            budget_check=None,
        )


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
    storage = _Storage(neurons=neurons)

    async def save_checkpoint(_phase: str, _cursor: str | None, details: dict[str, Any]) -> None:
        snapshots.append(details)

    budget_checks = 0

    async def pause_before_fourth_row() -> None:
        nonlocal budget_checks
        latest_state = list(storage.semantic_discovery_states.values())[-1]
        if not latest_state.get("source_complete"):
            return
        budget_checks += 1
        if budget_checks == 4:
            raise ConsolidationPausedError("simulated budget exhaustion")

    with pytest.raises(ConsolidationPausedError, match="budget exhaustion"):
        await discover_semantic_synapses(
            storage,
            config,
            checkpoint=save_checkpoint,
            budget_check=pause_before_fourth_row,
        )

    durable = snapshots[-1]
    assert durable["stage"] == "similarity"
    assert durable["version"] == 3
    assert durable["cursor"] == "2"
    _manifest, staged = _staged_semantic_state(storage, [json.dumps(durable)])
    logical = staged["legacy_checkpoint"]
    assert len(logical["results"]) == logical["synapses_created"]
    assert similarity_calls == 3 * (len(neurons) - 1)

    similarity_calls = 0
    resumed = await discover_semantic_synapses(
        storage,
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
    _manifest, staged = _staged_semantic_state(
        storage, progress.strategy_state("semantic_link")["pending"]
    )
    assert staged["legacy_checkpoint"]["synapses_created"] == 1
    assert len(staged["legacy_checkpoint"]["results"]) == 1

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
    storage.semantic_source_generation += 1
    with pytest.raises(ConsolidationProgressError, match="mutation fence detected a write"):
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
    assert discovery_state["version"] == 3
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
    monkeypatch.setattr(semantic_discovery_module, "_MAX_REBUILD_STATE_BYTES", 1)
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress("semantic_link")

    with pytest.raises(ConsolidationProgressError, match="staged source state exceeds"):
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

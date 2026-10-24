from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)
from surreal_memory.utils.timeutils import utcnow
from tests.unit.test_consolidation_mid_strategies_resume import (
    _engine,
    _force_semantic_python_fallback,
    _Progress,
    _Storage,
)
from tests.unit.test_semantic_discovery_state import _SnapshotStorage


def _semantic_link_fixture(
    monkeypatch: pytest.MonkeyPatch, *, neuron_count: int = 5, pair_cap: int = 2
) -> tuple[_Storage, _Progress, datetime]:
    reference_time = utcnow()
    neurons = [
        replace(
            Neuron.create(
                NeuronType.CONCEPT,
                f"semantic candidate {index}",
                neuron_id=f"semantic-node-{index}",
                metadata={"_embedding": [1.0, 0.0]},
            ),
            created_at=reference_time - timedelta(days=1),
        )
        for index in range(neuron_count)
    ]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.0,
        semantic_discovery_max_pairs=pair_cap,
        essence_generator="extractive",
    )
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._EMBEDDING_PAGE_SIZE", 2)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._SYNAPSE_PAGE_SIZE", 2)

    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress("semantic_link")
    progress.reference_time = reference_time
    return storage, progress, reference_time


async def _continue_until_manifest_field(
    storage: _Storage, progress: _Progress, field: str
) -> dict[str, object]:
    for _ in range(20):
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            pass
        state = progress.strategy_state("semantic_link")
        if state.get("pending"):
            manifest = json.loads(state["pending"][0])
            if field in manifest:
                return manifest
        if state.get("phase") == "semantic_link_complete":
            break
    pytest.fail(f"semantic_link did not checkpoint continuation field {field}")


@pytest.mark.asyncio
async def test_semantic_link_continues_capped_batches_until_all_pairs_are_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    pair_cap = storage.brain.config.semantic_discovery_max_pairs

    paused_batches = 0
    for _ in range(10):
        before = len(storage.added_synapses)
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            paused_batches += 1
        assert len(storage.added_synapses) - before <= pair_cap
        if progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete":
            break
    else:
        pytest.fail("semantic_link did not drain all capped discovery batches")

    assert len(storage.added_synapses) == 10
    assert len(storage.added_synapses) == len(set(storage.added_synapses))
    assert paused_batches >= 4


@pytest.mark.asyncio
async def test_applied_batch_keeps_its_staged_source_loadable_under_the_current_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    with pytest.raises(ConsolidationPausedError):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    manifest = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    ref = manifest["source_state_ref"]
    payload = storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]
    snapshot_storage = _SnapshotStorage()
    snapshot_storage.current_brain_id = storage.current_brain_id
    await snapshot_storage.save_semantic_discovery_state(ref["state_id"], ref["revision"], payload)
    snapshot_storage.progress = {**progress.state, "owner_token": progress.owner_token}
    snapshot_storage.lease = {"owner_token": progress.owner_token}
    assert (
        await snapshot_storage.load_semantic_discovery_state(ref["state_id"], ref["revision"])
        == payload
    )


@pytest.mark.asyncio
async def test_semantic_link_resume_after_batch_checkpoint_and_later_discovery_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    pair_cap = storage.brain.config.semantic_discovery_max_pairs
    progress.pause_phase = "semantic_link_discovery_batch"

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert len(storage.added_synapses) == pair_cap

    progress.pause_phase = "semantic_link_discovery_similarity"
    progress.pause_after_phase_calls = (
        progress.phase_calls.get("semantic_link_discovery_similarity", 0) + 2
    )
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert len(storage.added_synapses) == pair_cap

    for _ in range(10):
        before = len(storage.added_synapses)
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            pass
        assert len(storage.added_synapses) - before <= pair_cap
        if progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete":
            break
    else:
        pytest.fail("semantic_link did not resume after both durable interruptions")

    assert len(storage.added_synapses) == 10
    assert len(storage.added_synapses) == len(set(storage.added_synapses))
    assert progress.strategy_state("semantic_link")["counters"]["semantic_synapses_created"] == 10


@pytest.mark.asyncio
async def test_semantic_link_rejects_source_mutation_between_capped_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    pair_cap = storage.brain.config.semantic_discovery_max_pairs
    with pytest.raises(ConsolidationPausedError, match="bounded pair batch"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert len(storage.added_synapses) == pair_cap

    continuation = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    assert continuation["version"] == 4  # Older engines must reject unconsumed batches.

    candidate = storage.neurons["semantic-node-0"]
    storage.neurons[candidate.id] = candidate.with_metadata(_embedding=[0.0, 1.0])
    storage.semantic_source_generation += 1
    with pytest.raises(ConsolidationProgressError, match="source mutation fence detected a write"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert len(storage.added_synapses) == pair_cap
    assert progress.strategy_state("semantic_link")["phase"] != "semantic_link_complete"


@pytest.mark.asyncio
async def test_semantic_link_windows_cover_nodes_outside_initial_candidate_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch, neuron_count=7)
    pair_cap = storage.brain.config.semantic_discovery_max_pairs
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", 4)

    for _ in range(20):
        before = len(storage.added_synapses)
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            pass
        assert len(storage.added_synapses) - before <= pair_cap
        if progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete":
            break
    else:
        pytest.fail("semantic_link did not finish candidate windows")

    covered_node_ids = {
        endpoint
        for synapse in storage.synapses.values()
        for endpoint in (synapse.source_id, synapse.target_id)
    }
    assert covered_node_ids == set(storage.neurons)
    assert 1 < len(storage.added_synapses) < 7 * 6 // 2
    assert len(storage.added_synapses) == len(set(storage.added_synapses))


@pytest.mark.asyncio
async def test_semantic_link_rejects_tampered_batch_and_window_continuations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    await _continue_until_manifest_field(storage, progress, "next_batch")
    batch_state = progress.strategy_state("semantic_link")
    batch_manifest = json.loads(batch_state["pending"][0])
    batch_manifest["next_batch"] += 1
    batch_state["pending"] = [json.dumps(batch_manifest)]
    before = len(storage.added_synapses)
    with pytest.raises(ConsolidationProgressError, match="invalid semantic_link applied-batch"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert len(storage.added_synapses) == before
    assert batch_state["phase"] != "semantic_link_complete"

    window_storage, window_progress, _ = _semantic_link_fixture(monkeypatch)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", 4)
    await _continue_until_manifest_field(window_storage, window_progress, "next_window")
    window_state = window_progress.strategy_state("semantic_link")
    window_manifest = json.loads(window_state["pending"][0])
    window_manifest["next_window"] += 1
    window_state["pending"] = [json.dumps(window_manifest)]
    before = len(window_storage.added_synapses)
    with pytest.raises(ConsolidationProgressError, match="invalid semantic_link applied-batch"):
        await _engine(
            window_storage, ConsolidationStrategy.SEMANTIC_LINK, window_progress
        )._semantic_link(ConsolidationReport(), dry_run=False)
    assert len(window_storage.added_synapses) == before
    assert window_state["phase"] != "semantic_link_complete"


@pytest.mark.asyncio
async def test_semantic_link_rejects_source_mutation_between_population_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", 4)
    await _continue_until_manifest_field(storage, progress, "next_window")
    before = len(storage.added_synapses)

    candidate = storage.neurons["semantic-node-4"]
    storage.neurons[candidate.id] = candidate.with_metadata(_embedding=[0.0, 1.0])
    storage.semantic_source_generation += 1
    with pytest.raises(ConsolidationProgressError, match="source mutation fence detected a write"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert len(storage.added_synapses) == before
    assert progress.strategy_state("semantic_link")["phase"] != "semantic_link_complete"


@pytest.mark.asyncio
async def test_semantic_link_ignores_neurons_created_after_frozen_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, reference_time = _semantic_link_fixture(monkeypatch)
    await _continue_until_manifest_field(storage, progress, "next_batch")
    original_ids = set(storage.neurons)
    late_neuron = replace(
        Neuron.create(
            NeuronType.CONCEPT,
            "post-reference candidate",
            neuron_id="semantic-node-late",
            metadata={"_embedding": [1.0, 0.0]},
        ),
        created_at=reference_time + timedelta(seconds=1),
    )
    storage.neurons[late_neuron.id] = late_neuron
    storage.semantic_source_generation += 1
    storage.post_reference_source_generations.add(storage.semantic_source_generation)

    for _ in range(10):
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            pass
        if progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete":
            break
    else:
        pytest.fail("semantic_link did not finish against its frozen reference time")

    linked_ids = {
        endpoint
        for synapse in storage.synapses.values()
        for endpoint in (synapse.source_id, synapse.target_id)
    }
    assert late_neuron.id not in linked_ids
    assert linked_ids <= original_ids
    assert len(storage.added_synapses) == 10
    assert progress.strategy_state("semantic_link")["counters"]["semantic_synapses_created"] == 10


@pytest.mark.asyncio
async def test_semantic_link_id_order_windows_skip_ineligible_rows_and_cover_singleton_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, reference_time = _semantic_link_fixture(monkeypatch)
    ineligible = [
        replace(
            Neuron.create(
                NeuronType.CONCEPT,
                "",
                neuron_id="semantic-node-1a",
                metadata={"_embedding": [1.0, 0.0]},
            ),
            created_at=reference_time - timedelta(days=1),
        ),
        replace(
            Neuron.create(
                NeuronType.TIME,
                "2026-10-24",
                neuron_id="semantic-node-2a",
                metadata={"_embedding": [1.0, 0.0]},
            ),
            created_at=reference_time - timedelta(days=1),
        ),
        replace(
            Neuron.create(
                NeuronType.CONCEPT,
                "missing vector",
                neuron_id="semantic-node-3a",
            ),
            created_at=reference_time - timedelta(days=1),
        ),
    ]
    storage.neurons.update({neuron.id: neuron for neuron in ineligible})
    eligible_ids = {f"semantic-node-{index}" for index in range(5)}
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", 4)

    reports: list[ConsolidationReport] = []
    per_call_creates: list[int] = []
    for _ in range(20):
        report = ConsolidationReport()
        before = len(storage.added_synapses)
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                report, dry_run=False
            )
        except ConsolidationPausedError:
            pass
        reports.append(report)
        per_call_creates.append(len(storage.added_synapses) - before)
        if progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete":
            break
    else:
        pytest.fail("semantic_link did not finish ID-order windows")

    linked_ids = {
        endpoint
        for synapse in storage.synapses.values()
        for endpoint in (synapse.source_id, synapse.target_id)
    }
    assert linked_ids == eligible_ids
    assert len(storage.added_synapses) == 7
    assert all(created <= 2 for created in per_call_creates)
    assert progress.strategy_state("semantic_link")["counters"]["semantic_synapses_created"] == 7
    assert reports[-1].extra["semantic_coverage_complete"] is True
    assert reports[-1].extra["semantic_coverage_offset"] == 3
    assert not any(report.extra.get("semantic_coverage_complete") for report in reports[:-1])


@pytest.mark.asyncio
async def test_semantic_link_requires_numpy_for_oversized_frozen_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _reference_time = _semantic_link_fixture(monkeypatch)
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery._FALLBACK_MAX_NEURONS", 3)

    with pytest.raises(ConsolidationProgressError, match="numpy is required to complete"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert storage.added_synapses == []
    assert progress.strategy_state("semantic_link")["phase"] != "semantic_link_complete"

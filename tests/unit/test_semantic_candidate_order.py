from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine import semantic_discovery as semantic_discovery_module
from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationLeaseLostError,
    ConsolidationPausedError,
    ConsolidationProgressError,
)
from surreal_memory.utils.timeutils import utcnow
from tests.unit.test_consolidation_mid_strategies_resume import (
    _completed_budget_check,
    _engine,
    _force_semantic_python_fallback,
    _Progress,
    _semantic_similarity_fixture,
    _staged_semantic_state,
    _Storage,
)


def _grouped_type_neurons() -> list[Neuron]:
    return [
        Neuron.create(
            kind,
            f"semantic candidate {kind.value} {index}",
            metadata={"_embedding": [1.0 - index * 0.01, index * 0.01]},
            neuron_id=f"{kind.value}-{index:02d}",
        )
        for kind in (NeuronType.CONCEPT, NeuronType.ENTITY)
        for index in range(3)
    ]


def _summary(result: Any) -> tuple[Any, ...]:
    return (
        result.neurons_embedded,
        result.pairs_evaluated,
        result.synapses_created,
        result.skipped_existing,
        result.skipped_created_this_run,
        result.eligible_total,
        sorted((synapse.id, synapse.weight) for synapse in result.synapses),
        result.coverage_offset,
        result.coverage_complete,
    )


async def _paused_candidate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Neuron], Any, _Storage, dict[str, Any], dict[str, Any]]:
    neurons = _grouped_type_neurons()
    _fixture_neurons, config = _semantic_similarity_fixture()
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "_MAX_SIMILARITY_CHECKPOINTS", 4)
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
    manifest = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    _manifest, source_state = _staged_semantic_state(
        storage, progress.strategy_state("semantic_link")["pending"]
    )
    assert manifest["stage"] == "similarity"
    assert source_state["bounded"] is False
    assert source_state["legacy_checkpoint"]["version"] == 2
    return neurons, config, storage, manifest, source_state


def _scan_priority(neuron_id: str) -> int:
    kind, suffix = neuron_id.split("-")
    rank = int(suffix)
    return rank * 2 + (kind == "entity")


async def _resume_snapshot(
    storage: _Storage, config: Any, manifest: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []

    async def checkpoint(_stage: str, _cursor: str | None, details: dict[str, Any]) -> None:
        checkpoints.append(details)

    result = await semantic_discovery_module._discover_semantic_synapses_resumable(
        storage,
        config,
        checkpoint=checkpoint,
        resume_state=manifest,
        budget_check=None,
    )
    latest = checkpoints[-1]
    ref = latest["source_state_ref"]
    return result, storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_numpy", [False, True])
async def test_semantic_candidate_priorities_survive_repeated_similarity_resume(
    monkeypatch: pytest.MonkeyPatch,
    use_numpy: bool,
) -> None:
    neurons = _grouped_type_neurons()
    _fixture_neurons, config = _semantic_similarity_fixture()
    if use_numpy:
        pytest.importorskip("numpy")
        monkeypatch.setattr(
            semantic_discovery_module,
            "_effective_embedding",
            lambda _config: (True, "fixture", "fixture"),
        )
        monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", 2)
        monkeypatch.setattr(semantic_discovery_module, "_SYNAPSE_PAGE_SIZE", 2)
        monkeypatch.setattr(semantic_discovery_module, "_YIELD_EVERY_ROWS", 2)
    else:
        _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "_MAX_SIMILARITY_CHECKPOINTS", 4)

    expected = await semantic_discovery_module._discover_semantic_synapses_resumable(
        _Storage(neurons=neurons),
        config,
        checkpoint=lambda *_args, **_kwargs: _completed_budget_check(),
        resume_state=None,
        budget_check=None,
    )

    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress(
        "semantic_link",
        pause_phase="semantic_link_discovery_similarity",
        pause_after_phase_calls=3,
    )
    for _ in range(2):
        with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        checkpoint = progress.strategy_state("semantic_link")
        manifest = json.loads(checkpoint["pending"][0])
        _manifest, staged = _staged_semantic_state(storage, checkpoint["pending"])
        assert staged["bounded"] is False
        assert manifest["stage"] == "similarity"
        progress.pause_phase = "semantic_link_discovery_similarity"
        progress.pause_after_phase_calls = progress.phase_calls[progress.pause_phase] + 1

    resumed = await semantic_discovery_module._discover_semantic_synapses_resumable(
        storage,
        config,
        checkpoint=lambda *_args, **_kwargs: _completed_budget_check(),
        resume_state=json.loads(progress.strategy_state("semantic_link")["pending"][0]),
        budget_check=None,
    )
    assert _summary(resumed) == _summary(expected)


@pytest.mark.asyncio
async def test_legacy_raw_order_similarity_checkpoint_repairs_stale_priorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config, storage, manifest, state = await _paused_candidate_snapshot(monkeypatch)
    candidates = state["candidates"]
    assert [item[0] for item in candidates] == [
        "concept-00",
        "entity-00",
        "concept-01",
        "entity-01",
        "concept-02",
        "entity-02",
    ]
    # Recreate the old persisted pairing: IDs are in interleaved snapshot order,
    # while each carries the priority it had in the grouped scan order.
    for candidate, stale_priority in zip(candidates, (0, 2, 4, 1, 3, 5), strict=True):
        candidate[1] = stale_priority

    result, resumed_state = await _resume_snapshot(storage, config, manifest)
    assert [item[0] for item in resumed_state["candidates"]] == [item[0] for item in candidates]
    assert [item[1] for item in resumed_state["candidates"]] == [0, 1, 2, 3, 4, 5]
    expected = await semantic_discovery_module._discover_semantic_synapses_resumable(
        _Storage(neurons=neurons),
        config,
        checkpoint=lambda *_args, **_kwargs: _completed_budget_check(),
        resume_state=None,
        budget_check=None,
    )
    assert _summary(result) == _summary(expected)


@pytest.mark.asyncio
async def test_legacy_priority_sorted_digest_accepts_raw_id_order_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neurons, config, storage, manifest, state = await _paused_candidate_snapshot(monkeypatch)
    candidates = state["candidates"]
    # Old snapshots can contain grouped scan IDs even when their similarity
    # result indices and digest use priority-sorted candidate coordinates.
    candidates.sort(key=lambda item: item[0])
    for candidate in candidates:
        candidate[1] = _scan_priority(candidate[0])

    result, resumed_state = await _resume_snapshot(storage, config, manifest)
    assert [item[0] for item in resumed_state["candidates"]] == [
        "concept-00",
        "entity-00",
        "concept-01",
        "entity-01",
        "concept-02",
        "entity-02",
    ]
    assert [item[1] for item in resumed_state["candidates"]] == [0, 1, 2, 3, 4, 5]
    expected = await semantic_discovery_module._discover_semantic_synapses_resumable(
        _Storage(neurons=neurons),
        config,
        checkpoint=lambda *_args, **_kwargs: _completed_budget_check(),
        resume_state=None,
        budget_check=None,
    )
    assert _summary(result) == _summary(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("digest", "similarity checkpoint hash mismatch"),
        ("index", "similarity result index is out of range"),
        ("duplicate", "checkpoint contains a duplicate pair"),
    ],
)
async def test_invalid_legacy_results_fail_before_live_pair_lookup(
    monkeypatch: pytest.MonkeyPatch, tamper: str, error: str
) -> None:
    _neurons, config, storage, manifest, state = await _paused_candidate_snapshot(monkeypatch)
    legacy = state["legacy_checkpoint"]
    assert legacy["results"]
    if tamper == "digest":
        legacy["result_prefix_digest"] = "f" * 64
    elif tamper == "index":
        legacy["results"][-1][1] = len(state["candidates"])
    else:
        # Results are source ordered; duplicating the first pair at the next
        # position keeps the serialized row order valid while testing its set.
        legacy["results"][1][0] = legacy["results"][0][0]
        legacy["results"][1][1] = legacy["results"][0][1]

    pair_lookups = 0

    async def observe_pair_lookup(*_args: Any, **_kwargs: Any) -> set[tuple[str, str]]:
        nonlocal pair_lookups
        pair_lookups += 1
        return set()

    storage.find_existing_synapse_pairs = observe_pair_lookup  # type: ignore[method-assign]
    snapshot_count = len(storage.semantic_discovery_states)

    async def reject_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid results must not publish a checkpoint")

    with pytest.raises(RuntimeError, match=error):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=reject_checkpoint,
            resume_state=manifest,
            budget_check=None,
        )
    assert pair_lookups == 0
    assert len(storage.semantic_discovery_states) == snapshot_count
    assert storage.added_synapses == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["source", "lease"])
async def test_legacy_order_repair_preserves_checkpoint_when_source_or_lease_is_lost(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _neurons, _config, storage, manifest, state = await _paused_candidate_snapshot(monkeypatch)
    for candidate, stale_priority in zip(state["candidates"], (0, 2, 4, 1, 3, 5), strict=True):
        candidate[1] = stale_priority
    progress = _Progress("semantic_link")
    progress.strategy_state("semantic_link").update(
        phase="semantic_link_discovery_similarity",
        cursor=manifest["cursor"],
        pending=[json.dumps(manifest)],
    )
    original_progress = deepcopy(progress.state)
    original_snapshot = deepcopy(state)
    if failure == "source":
        storage.semantic_source_generation += 1
    else:

        async def lose_lease(*_args: Any, **_kwargs: Any) -> None:
            raise ConsolidationLeaseLostError("simulated lease loss during repair")

        monkeypatch.setattr(progress, "checkpoint", lose_lease)

    error = ConsolidationProgressError if failure == "source" else ConsolidationLeaseLostError
    with pytest.raises(error, match=r"mutation fence|lease loss"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert progress.state == original_progress
    assert state == original_snapshot
    assert storage.added_synapses == []


@pytest.mark.asyncio
async def test_empty_legacy_results_reject_ambiguous_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _neurons, config, storage, manifest, state = await _paused_candidate_snapshot(monkeypatch)
    for candidate, stale_priority in zip(state["candidates"], (0, 2, 4, 1, 3, 5), strict=True):
        candidate[1] = stale_priority
    state["legacy_checkpoint"].update(
        results=[],
        result_prefix_digest=hashlib.sha256(b"semantic-link-created-v1").hexdigest(),
        synapses_created=0,
        skipped_existing=0,
        skipped_created_this_run=0,
        pairs_evaluated=0,
    )

    pair_lookups = 0

    async def observe_pair_lookup(*_args: Any, **_kwargs: Any) -> set[tuple[str, str]]:
        nonlocal pair_lookups
        pair_lookups += 1
        return set()

    storage.find_existing_synapse_pairs = observe_pair_lookup  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="candidate order is ambiguous"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=lambda *_args, **_kwargs: _completed_budget_check(),
            resume_state=manifest,
            budget_check=None,
        )
    assert pair_lookups == 0


@pytest.mark.asyncio
async def test_valid_legacy_digest_rejects_result_now_present_in_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _neurons, config, storage, manifest, state = await _paused_candidate_snapshot(monkeypatch)
    candidates = state["candidates"]
    for candidate, stale_priority in zip(candidates, (0, 2, 4, 1, 3, 5), strict=True):
        candidate[1] = stale_priority
    source_index, target_index, _similarity = state["legacy_checkpoint"]["results"][0]
    pair = Synapse.create(
        candidates[source_index][0],
        candidates[target_index][0],
        SynapseType.SIMILAR_TO,
        synapse_id="existing-semantic-pair",
    )
    storage.synapses[pair.id] = pair

    pair_lookups = 0
    original_lookup = storage.find_existing_synapse_pairs

    async def observe_pair_lookup(*args: Any, **kwargs: Any) -> set[tuple[str, str]]:
        nonlocal pair_lookups
        pair_lookups += 1
        return await original_lookup(*args, **kwargs)

    storage.find_existing_synapse_pairs = observe_pair_lookup  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="checkpoint contains a duplicate pair"):
        await semantic_discovery_module._discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=lambda *_args, **_kwargs: _completed_budget_check(),
            resume_state=manifest,
            budget_check=None,
        )
    assert pair_lookups == 1


@pytest.mark.asyncio
async def test_mixed_type_bounded_windows_complete_next_window_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_time = utcnow()
    neurons = _grouped_type_neurons() + [
        Neuron.create(
            kind,
            f"semantic candidate {kind.value} {index}",
            metadata={"_embedding": [1.0, 0.0]},
            neuron_id=f"{kind.value}-{index:02d}",
        )
        for kind in (NeuronType.CONCEPT, NeuronType.ENTITY)
        for index in range(3, 4)
    ]
    neurons = [replace(neuron, created_at=reference_time - timedelta(days=1)) for neuron in neurons]
    config = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.0,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )
    _force_semantic_python_fallback(monkeypatch)
    monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 4)
    monkeypatch.setattr(semantic_discovery_module, "_MAX_SIMILARITY_CHECKPOINTS", 4)
    storage = _Storage(neurons=neurons)
    storage.brain.config = config
    progress = _Progress("semantic_link")
    progress.reference_time = reference_time

    original_add_synapse = storage.add_synapse

    async def add_synapse_with_reference_time(synapse: Synapse) -> str:
        return await original_add_synapse(
            replace(synapse, created_at=reference_time + timedelta(seconds=1))
        )

    storage.add_synapse = add_synapse_with_reference_time  # type: ignore[method-assign]

    progress.pause_phase = "semantic_link_discovery_similarity"
    progress.pause_after_phase_calls = 3
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    pending = progress.strategy_state("semantic_link")["pending"]
    manifest = json.loads(pending[0])
    _manifest, paused_state = _staged_semantic_state(storage, pending)
    assert manifest["stage"] == "similarity"
    assert paused_state["bounded"] is True
    assert paused_state["legacy_checkpoint"]["version"] == 2
    progress.pause_phase = None

    next_window_state: dict[str, Any] | None = None
    for _ in range(40):
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            pass
        checkpoint = progress.strategy_state("semantic_link")
        pending = checkpoint.get("pending") or []
        if pending:
            manifest = json.loads(pending[0])
            if "next_window" in manifest:
                _manifest, next_window_state = _staged_semantic_state(storage, pending)
                assert manifest["next_window"] == 0
                break
        if checkpoint.get("phase") == "semantic_link_complete":
            break

    assert next_window_state is not None
    for _ in range(40):
        if progress.strategy_state("semantic_link").get("phase") == "semantic_link_complete":
            break
        try:
            await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
                ConsolidationReport(), dry_run=False
            )
        except ConsolidationPausedError:
            pass
    assert progress.strategy_state("semantic_link").get("phase") == "semantic_link_complete"
    linked_ids = {
        endpoint
        for synapse in storage.synapses.values()
        for endpoint in (synapse.source_id, synapse.target_id)
    }
    assert linked_ids == {neuron.id for neuron in neurons}

"""Synthetic proof of semantic-link batch replay and cursor boundaries."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine import semantic_discovery as discovery
from surreal_memory.engine.semantic_discovery import discover_semantic_synapses

_REFERENCE_TIME = datetime(2026, 10, 24, tzinfo=UTC)


class _FrozenStorage:
    """Small synthetic storage with the source fence used by resumable discovery."""

    current_brain_id = "cursor-review-brain"

    def __init__(self, neurons: list[Neuron], synapses: list[Synapse]) -> None:
        self.neurons = {item.id: item for item in neurons}
        self.synapses = {item.id: item for item in synapses}
        self.states: dict[tuple[str, int], dict[str, Any]] = {}
        self.generation = 0
        self.post_reference_generations: set[int] = set()
        self.pair_lookup_calls = 0
        self.pairs_requested = 0

    async def capture_semantic_source_token(self) -> str:
        return str(self.generation)

    async def assert_semantic_source_unchanged(
        self, token: str, *, created_before: datetime | None = None
    ) -> None:
        if token == str(self.generation):
            return
        if created_before is not None and all(
            value in self.post_reference_generations
            for value in range(int(token) + 1, self.generation + 1)
        ):
            return
        raise RuntimeError("synthetic semantic source mutation fence detected a write")

    async def find_neurons_after_id(
        self,
        cursor: str | None,
        *,
        limit: int,
        ephemeral: bool | None = None,
        include_embedding: bool = False,
        created_before: datetime | None = None,
    ) -> list[Neuron]:
        del ephemeral, include_embedding
        rows = sorted(self.neurons.values(), key=lambda item: item.id)
        if created_before is not None:
            rows = [item for item in rows if item.created_at <= created_before]
        if cursor is not None:
            rows = [item for item in rows if item.id > cursor]
        return rows[:limit]

    async def get_synapses_after_id(
        self,
        cursor: str | None,
        *,
        limit: int,
        created_before: datetime | None = None,
    ) -> list[Synapse]:
        rows = sorted(self.synapses.values(), key=lambda item: item.id)
        if created_before is not None:
            rows = [item for item in rows if item.created_at <= created_before]
        if cursor is not None:
            rows = [item for item in rows if item.id > cursor]
        return rows[:limit]

    async def find_existing_synapse_pairs(
        self,
        pairs: list[tuple[str, str]],
        *,
        created_before: datetime | None = None,
    ) -> set[tuple[str, str]]:
        self.pair_lookup_calls += 1
        self.pairs_requested += len(pairs)
        requested = {tuple(sorted(pair)) for pair in pairs}
        return {
            tuple(sorted((edge.source_id, edge.target_id)))
            for edge in self.synapses.values()
            if created_before is None or edge.created_at <= created_before
            if tuple(sorted((edge.source_id, edge.target_id))) in requested
        }

    async def get_neurons_batch(self, ids: list[str]) -> dict[str, Neuron]:
        return {key: self.neurons[key] for key in ids if key in self.neurons}

    async def save_semantic_discovery_state(
        self, state_id: str, revision: int, payload: dict[str, Any]
    ) -> None:
        self.states[(state_id, revision)] = payload

    async def load_semantic_discovery_state(
        self, state_id: str, revision: int
    ) -> dict[str, Any] | None:
        return self.states.get((state_id, revision))

    def add_batch(self, synapses: list[Synapse]) -> None:
        for edge in synapses:
            self.synapses[edge.id] = replace(
                edge, created_at=_REFERENCE_TIME + timedelta(seconds=1)
            )
            self.generation += 1
            self.post_reference_generations.add(self.generation)


def _fixture(existing_edge: bool = True) -> tuple[list[Neuron], list[Synapse]]:
    neurons = [
        replace(
            Neuron.create(
                NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
                f"synthetic candidate {index}",
                metadata={"_embedding": [1.0, 0.0]},
                neuron_id=f"candidate-{index:02d}",
            ),
            created_at=_REFERENCE_TIME - timedelta(days=1),
        )
        for index in range(8)
    ]
    edges = []
    if existing_edge:
        edges.append(
            replace(
                Synapse.create(
                    neurons[0].id,
                    neurons[1].id,
                    SynapseType.CAUSED_BY,
                    weight=0.5,
                    synapse_id="existing-any-type-edge",
                ),
                created_at=_REFERENCE_TIME - timedelta(days=1),
            )
        )
    return neurons, edges


def _config(pair_cap: int) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="synthetic",
        embedding_model="synthetic",
        semantic_discovery_similarity_threshold=0.5,
        semantic_discovery_max_pairs=pair_cap,
        essence_generator="extractive",
    )


@pytest.mark.asyncio
async def test_frozen_capped_batches_match_full_discovery_and_replay_saved_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    monkeypatch.setattr(discovery, "_EMBEDDING_PAGE_SIZE", 3)
    monkeypatch.setattr(discovery, "_SYNAPSE_PAGE_SIZE", 3)

    original_cosine = discovery._cosine_similarity
    similarity_calls = 0

    def counted_cosine(left: list[float], right: list[float]) -> float:
        nonlocal similarity_calls
        similarity_calls += 1
        return original_cosine(left, right)

    monkeypatch.setattr(discovery, "_cosine_similarity", counted_cosine)
    neurons, initial_edges = _fixture()
    full = _FrozenStorage(neurons, initial_edges)
    full_result = await discover_semantic_synapses(
        full,
        _config(100),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    full_edges = {
        edge.id: (edge.source_id, edge.target_id, edge.type, edge.weight, edge.metadata)
        for edge in full_result.synapses
    }
    full_calls = similarity_calls
    full_pair_lookup_calls = full.pair_lookup_calls
    full_pairs_requested = full.pairs_requested

    storage = _FrozenStorage(neurons, initial_edges)
    resume: dict[str, Any] | None = None
    batch_edges: dict[str, tuple[Any, ...]] = {}
    batch_calls: list[int] = []
    batch_pair_lookup_calls: list[int] = []
    batch_pairs_requested: list[int] = []
    cursors: list[str | None] = []
    first_batch_pairs: list[tuple[str, str]] = []
    for _ in range(20):
        before = similarity_calls
        before_lookup_calls = storage.pair_lookup_calls
        before_pairs_requested = storage.pairs_requested
        result = await discover_semantic_synapses(
            storage,
            _config(6),
            checkpoint=_unused_checkpoint,
            resume_state=resume,
            reference_time=_REFERENCE_TIME,
        )
        batch_calls.append(similarity_calls - before)
        batch_pair_lookup_calls.append(storage.pair_lookup_calls - before_lookup_calls)
        batch_pairs_requested.append(storage.pairs_requested - before_pairs_requested)
        batch_edges.update(
            {
                edge.id: (edge.source_id, edge.target_id, edge.type, edge.weight, edge.metadata)
                for edge in result.synapses
            }
        )
        if not batch_calls[:-1]:
            first_batch_pairs = sorted((edge.source_id, edge.target_id) for edge in result.synapses)
        resume = result.continuation
        if resume is None:
            break
        cursors.append(resume.get("cursor"))
        storage.add_batch(result.synapses)
    else:
        pytest.fail("synthetic capped discovery did not finish within 20 batches")

    assert cursors[0] == "1"  # first cap splits row one after its second peer
    assert cursors[1] == "3"  # batch two finishes rows one and two, then caps in row three
    assert first_batch_pairs == [
        ("candidate-00", "candidate-02"),
        ("candidate-00", "candidate-03"),
        ("candidate-00", "candidate-04"),
        ("candidate-00", "candidate-05"),
        ("candidate-01", "candidate-02"),
        ("candidate-01", "candidate-03"),
    ]  # equal-score peers follow source ID order; candidate-01 is already linked
    assert batch_edges == full_edges
    assert batch_calls[1] == 3 * (len(neurons) - 1)  # rows one through three are visited
    assert batch_pair_lookup_calls[1] == 3
    assert batch_pairs_requested[1] == 15  # three indexed top-K lookups
    assert full_calls == len(neurons) * (len(neurons) - 1)
    assert full_pair_lookup_calls == len(neurons)
    assert full_pairs_requested == len(neurons) * 5
    assert sum(batch_pair_lookup_calls) > full_pair_lookup_calls
    assert sum(batch_pairs_requested) > full_pairs_requested


async def _unused_checkpoint(_stage: str, _cursor: str | None, _details: dict[str, Any]) -> None:
    return None


@pytest.mark.asyncio
async def test_frozen_batch_cursor_still_rejects_pre_reference_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    neurons, initial_edges = _fixture()
    storage = _FrozenStorage(neurons, initial_edges)
    first = await discover_semantic_synapses(
        storage,
        _config(2),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    storage.neurons[neurons[0].id] = neurons[0].with_metadata(_embedding=[0.0, 1.0])
    storage.generation += 1
    with pytest.raises(RuntimeError, match="mutation fence"):
        await discover_semantic_synapses(
            storage,
            _config(2),
            checkpoint=_unused_checkpoint,
            resume_state=first.continuation,
            reference_time=_REFERENCE_TIME,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["malformed", "0", "999", None])
async def test_applied_batch_cursor_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    cursor: str | None,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    neurons, initial_edges = _fixture()
    storage = _FrozenStorage(neurons, initial_edges)
    first = await discover_semantic_synapses(
        storage,
        _config(6),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    first.continuation["cursor"] = cursor
    with pytest.raises(RuntimeError, match="applied-batch similarity cursor"):
        await discover_semantic_synapses(
            storage,
            _config(6),
            checkpoint=_unused_checkpoint,
            resume_state=first.continuation,
            reference_time=_REFERENCE_TIME,
        )


@pytest.mark.asyncio
async def test_applied_batch_matching_cursor_beyond_candidates_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    neurons, initial_edges = _fixture()
    storage = _FrozenStorage(neurons, initial_edges)
    first = await discover_semantic_synapses(
        storage,
        _config(6),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    state_ref = first.continuation["source_state_ref"]
    source_state = storage.states[(state_ref["state_id"], state_ref["revision"])]
    first.continuation["cursor"] = "999"
    source_state["legacy_checkpoint"]["cursor"] = "999"
    source_state["legacy_checkpoint"].pop("similarity_top_k")
    with pytest.raises(RuntimeError, match="cursor exceeds candidate rows"):
        await discover_semantic_synapses(
            storage,
            _config(6),
            checkpoint=_unused_checkpoint,
            resume_state=first.continuation,
            reference_time=_REFERENCE_TIME,
        )


@pytest.mark.asyncio
async def test_applied_batch_backend_change_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    numpy = pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    neurons, initial_edges = _fixture()
    storage = _FrozenStorage(neurons, initial_edges)
    first = await discover_semantic_synapses(
        storage,
        _config(6),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    with pytest.raises(RuntimeError, match="similarity backend changed"):
        await discover_semantic_synapses(
            storage,
            _config(6),
            checkpoint=_unused_checkpoint,
            resume_state=first.continuation,
            reference_time=_REFERENCE_TIME,
        )


@pytest.mark.asyncio
async def test_applied_batch_top_k_change_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    neurons, initial_edges = _fixture()
    storage = _FrozenStorage(neurons, initial_edges)
    first = await discover_semantic_synapses(
        storage,
        _config(6),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    monkeypatch.setattr(discovery, "SEMANTIC_TOP_K", 4)
    with pytest.raises(RuntimeError, match="top-K changed"):
        await discover_semantic_synapses(
            storage,
            _config(6),
            checkpoint=_unused_checkpoint,
            resume_state=first.continuation,
            reference_time=_REFERENCE_TIME,
        )


@pytest.mark.asyncio
async def test_older_applied_batch_checkpoint_replays_from_row_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    neurons, initial_edges = _fixture()
    storage = _FrozenStorage(neurons, initial_edges)
    first = await discover_semantic_synapses(
        storage,
        _config(6),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    state_ref = first.continuation["source_state_ref"]
    source_state = storage.states[(state_ref["state_id"], state_ref["revision"])]
    legacy_checkpoint = source_state["legacy_checkpoint"]
    legacy_checkpoint.pop("similarity_top_k")
    storage.add_batch(first.synapses)

    similarity_calls = 0
    original_cosine = discovery._cosine_similarity

    def counted_cosine(left: list[float], right: list[float]) -> float:
        nonlocal similarity_calls
        similarity_calls += 1
        return original_cosine(left, right)

    monkeypatch.setattr(discovery, "_cosine_similarity", counted_cosine)
    second = await discover_semantic_synapses(
        storage,
        _config(6),
        checkpoint=_unused_checkpoint,
        resume_state=first.continuation,
        reference_time=_REFERENCE_TIME,
    )

    assert second.synapses_created == 6
    assert similarity_calls == 4 * (len(neurons) - 1)
    assert second.continuation is not None
    next_ref = second.continuation["source_state_ref"]
    next_state = storage.states[(next_ref["state_id"], next_ref["revision"])]
    assert next_state["legacy_checkpoint"]["similarity_top_k"] == discovery.SEMANTIC_TOP_K


@pytest.mark.asyncio
async def test_coverage_window_transition_restarts_at_first_window_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    monkeypatch.setattr(discovery, "MAX_NEURONS_TO_LINK", 4)
    neurons, initial_edges = _fixture(existing_edge=False)
    storage = _FrozenStorage(neurons, initial_edges)
    similarity_calls = 0
    original_cosine = discovery._cosine_similarity

    def counted_cosine(left: list[float], right: list[float]) -> float:
        nonlocal similarity_calls
        similarity_calls += 1
        return original_cosine(left, right)

    monkeypatch.setattr(discovery, "_cosine_similarity", counted_cosine)
    first = await discover_semantic_synapses(
        storage,
        _config(100),
        checkpoint=_unused_checkpoint,
        reference_time=_REFERENCE_TIME,
    )
    assert first.continuation is not None
    assert first.continuation.get("next_window") == 0
    storage.add_batch(first.synapses)
    similarity_calls = 0

    second = await discover_semantic_synapses(
        storage,
        _config(100),
        checkpoint=_unused_checkpoint,
        resume_state=first.continuation,
        reference_time=_REFERENCE_TIME,
    )
    assert second.continuation is not None
    assert second.continuation.get("next_window") == 3
    storage.add_batch(second.synapses)
    prior_cursor = second.continuation["cursor"]
    similarity_calls = 0

    third = await discover_semantic_synapses(
        storage,
        _config(100),
        checkpoint=_unused_checkpoint,
        resume_state=second.continuation,
        reference_time=_REFERENCE_TIME,
    )

    assert prior_cursor != "0"
    assert third.coverage_offset == 3
    assert similarity_calls == 4 * 3  # four candidates in the new window, from its row zero

"""Disposable SurrealDB integration for durable semantic coverage batches."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

TEST_SURREALDB_URL = os.getenv("SMEM_TEST_SURREALDB_URL")
TEST_AUTH = ("root", "root")
PAIR_CAP = 2
EMBEDDING_DIMENSIONS = 1024


def _is_loopback_test_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_test_url(TEST_SURREALDB_URL),
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL",
    ),
]


@pytest_asyncio.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SurrealDBStorage]:
    assert TEST_SURREALDB_URL is not None
    for key in tuple(os.environ):
        if key.startswith("SURREALDB_"):
            monkeypatch.delenv(key, raising=False)

    namespace = "semantic_coverage_" + uuid.uuid4().hex[:12]
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace=namespace,
        database="it_" + uuid.uuid4().hex[:12],
        embedding_dim=EMBEDDING_DIMENSIONS,
    )
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage._query(f"REMOVE NAMESPACE {namespace}")
        await storage.close()


async def _add_fixture_brain_and_neurons(
    storage: SurrealDBStorage,
    neuron_count: int,
) -> tuple[Brain, datetime, set[str]]:
    config = BrainConfig(
        embedding_enabled=True,
        semantic_discovery_similarity_threshold=0.99,
        semantic_discovery_max_pairs=PAIR_CAP,
    )
    brain = Brain.create(name="synthetic-semantic-coverage-it", config=config)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    reference_time = utcnow()
    created_at = reference_time - timedelta(days=1)
    embedding = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
    neurons = [
        replace(
            Neuron.create(
                NeuronType.CONCEPT,
                f"synthetic semantic coverage candidate {index}",
                neuron_id=f"semantic-coverage-{index:03d}",
                metadata={"_embedding": embedding},
            ),
            created_at=created_at,
        )
        for index in range(neuron_count)
    ]
    for neuron in neurons:
        await storage.add_neuron(neuron)
    return brain, reference_time, {neuron.id for neuron in neurons}


async def _resume_semantic_link(storage: SurrealDBStorage, run_id: str) -> Any:
    return await ConsolidationEngine(storage, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK],
        resume_run_id=run_id,
    )


async def _run_semantic_link_until_chunk_pause(
    storage: SurrealDBStorage,
    reference_time: datetime,
) -> Any:
    progress = await storage.get_consolidation_progress(storage.current_brain_id)
    initial_pending = None
    run_id = None
    if progress is not None:
        run_id = str(progress["run_id"])
        strategy_state = progress["strategy_states"]["semantic_link"]
        initial_pending = strategy_state.get("pending")

    engine = ConsolidationEngine(storage, ConsolidationConfig())
    check_budget = engine._check_progress_budget

    async def pause_after_committed_chunk() -> None:
        session = engine._progress_session
        state = session.strategy_state("semantic_link") if session is not None else {}
        if (
            state.get("phase") == "semantic_link_discovery_batch"
            and state.get("pending") != initial_pending
        ):
            engine._strategy_deadline = time.perf_counter() - 1
        await check_budget()

    engine._check_progress_budget = pause_after_committed_chunk  # type: ignore[method-assign]
    if run_id is None:
        return await engine.run(
            strategies=[ConsolidationStrategy.SEMANTIC_LINK],
            reference_time=reference_time,
        )
    return await engine.run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK],
        resume_run_id=run_id,
    )


def _track_semantic_chunk_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    storage: SurrealDBStorage,
) -> tuple[list[int], list[dict[str, Any]]]:
    chunk_sizes: list[int] = []
    staged_sources: list[dict[str, Any]] = []
    previous_created = 0
    save = storage.save_consolidation_progress

    async def track_save(brain_id: str, owner_token: str, state: Any) -> Any:
        nonlocal previous_created
        strategy_state = state.get("strategy_states", {}).get("semantic_link", {})
        counters = strategy_state.get("counters", {})
        created = counters.get("semantic_synapses_created", 0)
        if isinstance(created, int) and created > previous_created:
            chunk_sizes.append(created - previous_created)
            previous_created = created

        pending = strategy_state.get("pending")
        if strategy_state.get("phase") == "semantic_link_discovery_batch" and pending:
            manifest = json.loads(pending[0])
            continuation = (
                manifest
                if manifest.get("kind") == "semantic_link_discovery"
                else manifest.get("continuation")
            )
            source_ref = (
                continuation.get("source_state_ref") if isinstance(continuation, dict) else None
            )
            if isinstance(source_ref, dict):
                staged = await storage.load_semantic_discovery_state(
                    source_ref["state_id"], source_ref["revision"]
                )
                if staged is not None:
                    staged_sources.append(dict(staged))
        return await save(brain_id, owner_token, state)

    monkeypatch.setattr(storage, "save_consolidation_progress", track_save)
    return chunk_sizes, staged_sources


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("neuron_count", "candidate_limit"),
    [(5, None), (7, 4)],
)
async def test_semantic_coverage_batches_resume_under_real_progress_leases(
    store: SurrealDBStorage,
    monkeypatch: pytest.MonkeyPatch,
    neuron_count: int,
    candidate_limit: int | None,
) -> None:
    brain, reference_time, expected_ids = await _add_fixture_brain_and_neurons(store, neuron_count)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    if candidate_limit is not None:
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", candidate_limit
        )

    acquired_owners: list[str] = []
    released_owners: list[str] = []
    acquire = store.acquire_consolidation_lease
    release = store.release_consolidation_lease

    async def track_acquire(brain_id: str, owner_token: str, *, lease_seconds: int = 30) -> bool:
        acquired = await acquire(brain_id, owner_token, lease_seconds=lease_seconds)
        if acquired:
            acquired_owners.append(owner_token)
        return bool(acquired)

    async def track_release(brain_id: str, owner_token: str) -> bool:
        released = await release(brain_id, owner_token)
        if released:
            released_owners.append(owner_token)
        return bool(released)

    monkeypatch.setattr(store, "acquire_consolidation_lease", track_acquire)
    monkeypatch.setattr(store, "release_consolidation_lease", track_release)

    chunk_sizes, staged_sources = _track_semantic_chunk_checkpoints(monkeypatch, store)
    last_report = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK],
        reference_time=reference_time,
    )
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    assert last_report.extra["consolidation_status"] == "completed"
    assert len(acquired_owners) == 1
    assert released_owners == acquired_owners
    assert len(set(acquired_owners)) == 1
    assert len(chunk_sizes) >= 2
    assert all(0 < chunk_size <= PAIR_CAP for chunk_size in chunk_sizes)
    assert staged_sources
    assert staged_sources[0]["batch_index"] == 0
    assert staged_sources[0].get("coverage_offset") is None
    assert staged_sources[0]["reference_time"] == reference_time.isoformat(timespec="microseconds")

    assert last_report.extra["semantic_coverage_complete"] is True

    final_progress = await store.get_consolidation_progress(brain.id)
    assert final_progress is not None
    strategy_state = final_progress["strategy_states"]["semantic_link"]
    assert strategy_state["phase"] == "completed"
    assert strategy_state["counters"]["semantic_synapses_created"] == len(
        await store.get_synapses(type=SynapseType.SIMILAR_TO)
    )
    semantic_edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert sum(chunk_sizes) == len(semantic_edges)
    assert len(semantic_edges) == len({edge.id for edge in semantic_edges})
    assert all(edge.created_at > reference_time for edge in semantic_edges)

    covered_ids = {
        endpoint for edge in semantic_edges for endpoint in (edge.source_id, edge.target_id)
    }
    assert covered_ids == expected_ids
    if neuron_count == 5:
        assert len(semantic_edges) == 10
    else:
        assert candidate_limit is not None
        assert len(semantic_edges) < neuron_count * (neuron_count - 1) // 2
        assert last_report.extra["semantic_coverage_offset"] == candidate_limit - 1


@pytest.mark.asyncio
async def test_post_reference_neuron_is_excluded_from_frozen_semantic_coverage(
    store: SurrealDBStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain, reference_time, original_ids = await _add_fixture_brain_and_neurons(store, 5)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )

    first = await _run_semantic_link_until_chunk_pause(store, reference_time)
    assert first.extra["consolidation_status"] == "paused"
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    assert progress["status"] == "paused"
    assert len(await store.get_synapses(type=SynapseType.SIMILAR_TO)) == PAIR_CAP

    late_neuron = replace(
        Neuron.create(
            NeuronType.CONCEPT,
            "synthetic post-reference semantic candidate",
            neuron_id="semantic-coverage-late",
            metadata={"_embedding": [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)},
        ),
        created_at=reference_time + timedelta(seconds=1),
    )
    await store.add_neuron(late_neuron)

    run_id = str(progress["run_id"])
    report = await _resume_semantic_link(store, run_id)
    assert report.extra["consolidation_status"] == "completed"

    edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    linked_ids = {endpoint for edge in edges for endpoint in (edge.source_id, edge.target_id)}
    assert len(edges) == 10
    assert linked_ids == original_ids
    assert late_neuron.id not in linked_ids
    final = await store.get_consolidation_progress(brain.id)
    assert final is not None
    assert final["strategy_states"]["semantic_link"]["counters"]["semantic_synapses_created"] == 10
    assert report.extra["semantic_coverage_complete"] is True


@pytest.mark.asyncio
async def test_pre_reference_mutation_between_windows_fails_closed_without_more_edges(
    store: SurrealDBStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain, reference_time, _expected_ids = await _add_fixture_brain_and_neurons(store, 7)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", 4)

    first = await _run_semantic_link_until_chunk_pause(store, reference_time)
    assert first.extra["consolidation_status"] == "paused"
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    before = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert len(before) == PAIR_CAP

    candidate = await store.get_neuron("semantic-coverage-006")
    assert candidate is not None
    await store.update_neuron(candidate.with_metadata(_embedding=[0.0] + [1.0] * 1023))

    report = await _resume_semantic_link(store, str(progress["run_id"]))
    after = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert len(after) == len(before)
    assert report.extra["consolidation_status"] == "failed"
    final = await store.get_consolidation_progress(brain.id)
    assert final is not None
    assert final["strategy_states"]["semantic_link"]["phase"] != "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate_limit", "tamper_field"),
    [(None, "next_batch"), (None, "cursor"), (4, "next_window")],
)
async def test_tampered_batch_or_window_manifest_fails_closed(
    store: SurrealDBStorage,
    monkeypatch: pytest.MonkeyPatch,
    candidate_limit: int | None,
    tamper_field: str,
) -> None:
    neuron_count = 5 if candidate_limit is None else 7
    brain, reference_time, _expected_ids = await _add_fixture_brain_and_neurons(store, neuron_count)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    if candidate_limit is not None:
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", candidate_limit
        )

    first = await _run_semantic_link_until_chunk_pause(store, reference_time)
    assert first.extra["consolidation_status"] == "paused"
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    assert progress["status"] == "paused"

    manifest: dict[str, Any] | None = None
    continuation: dict[str, Any] | None = None
    for _ in range(20):
        strategy_state = progress["strategy_states"]["semantic_link"]
        pending = strategy_state["pending"]
        assert len(pending) == 1
        manifest = json.loads(pending[0])
        if manifest.get("kind") == "semantic_link_discovery":
            continuation = manifest
        else:
            continuation = manifest.get("continuation")
        if isinstance(continuation, dict) and tamper_field in continuation:
            break
        report = await _run_semantic_link_until_chunk_pause(store, reference_time)
        assert report.extra["consolidation_status"] == "paused"
        progress = await store.get_consolidation_progress(brain.id)
        assert progress is not None
        assert progress["status"] == "paused"
    else:
        pytest.fail(f"semantic continuation never persisted {tamper_field}")

    assert manifest is not None
    assert continuation is not None
    assert tamper_field in continuation
    tamper_owner = "semantic-coverage-tamper-" + uuid.uuid4().hex
    assert await store.acquire_consolidation_lease(brain.id, tamper_owner)
    claimed = await store.claim_consolidation_progress(brain.id, tamper_owner)
    assert claimed is not None
    strategy_state = claimed["strategy_states"]["semantic_link"]
    pending = strategy_state["pending"]
    manifest = json.loads(pending[0])
    if manifest.get("kind") == "semantic_link_discovery":
        continuation = manifest
    else:
        continuation = manifest["continuation"]
    assert tamper_field in continuation
    if tamper_field == "cursor":
        continuation[tamper_field] = "999999"
    else:
        continuation[tamper_field] += 1
    strategy_state["pending"] = [json.dumps(manifest, sort_keys=True, separators=(",", ":"))]
    claimed["strategy_states"]["semantic_link"] = strategy_state
    saved = await store.save_consolidation_progress(brain.id, tamper_owner, claimed)
    assert saved is not None
    assert await store.release_consolidation_lease(brain.id, tamper_owner)
    before = await store.get_synapses(type=SynapseType.SIMILAR_TO)

    report = await _resume_semantic_link(store, str(progress["run_id"]))
    after = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert report.extra["consolidation_status"] == "failed"
    assert len(after) == len(before)
    final = await store.get_consolidation_progress(brain.id)
    assert final is not None
    assert final["strategy_states"]["semantic_link"]["phase"] != "completed"


@pytest.mark.asyncio
async def test_id_order_windows_skip_ineligible_rows_and_cover_single_new_candidate_tail(
    store: SurrealDBStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain, reference_time, eligible_ids = await _add_fixture_brain_and_neurons(store, 5)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr("surreal_memory.engine.semantic_discovery.MAX_NEURONS_TO_LINK", 4)
    ineligible = [
        replace(
            Neuron.create(
                NeuronType.CONCEPT,
                "",
                neuron_id="semantic-coverage-000a",
                metadata={"_embedding": [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)},
            ),
            created_at=reference_time - timedelta(days=1),
        ),
        replace(
            Neuron.create(
                NeuronType.TIME,
                "synthetic ineligible time row",
                neuron_id="semantic-coverage-002a",
                metadata={"_embedding": [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)},
            ),
            created_at=reference_time - timedelta(days=1),
        ),
        replace(
            Neuron.create(
                NeuronType.CONCEPT,
                "synthetic missing-vector row",
                neuron_id="semantic-coverage-003a",
            ),
            created_at=reference_time - timedelta(days=1),
        ),
    ]
    for neuron in ineligible:
        await store.add_neuron(neuron)

    chunk_sizes, _staged_sources = _track_semantic_chunk_checkpoints(monkeypatch, store)
    last_report = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK],
        reference_time=reference_time,
    )
    assert last_report.extra["consolidation_status"] == "completed"
    assert len(chunk_sizes) >= 2
    assert all(0 < chunk_size <= PAIR_CAP for chunk_size in chunk_sizes)

    edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert sum(chunk_sizes) == len(edges)
    covered_ids = {endpoint for edge in edges for endpoint in (edge.source_id, edge.target_id)}
    assert len(edges) == 7
    assert covered_ids == eligible_ids
    assert last_report.extra["semantic_coverage_complete"] is True
    assert last_report.extra["semantic_coverage_offset"] == 3
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    assert (
        progress["strategy_states"]["semantic_link"]["counters"]["semantic_synapses_created"] == 7
    )

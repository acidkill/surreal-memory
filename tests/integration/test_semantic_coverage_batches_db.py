"""Disposable SurrealDB integration for durable semantic coverage batches."""

from __future__ import annotations

import json
import os
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

    run_id: str | None = None
    previous_edge_count = 0
    last_report = None
    completed_in = 0
    for attempt in range(40):
        engine = ConsolidationEngine(store, ConsolidationConfig())
        if run_id is None:
            report = await engine.run(
                strategies=[ConsolidationStrategy.SEMANTIC_LINK],
                reference_time=reference_time,
            )
        else:
            report = await engine.run(
                strategies=[ConsolidationStrategy.SEMANTIC_LINK],
                resume_run_id=run_id,
            )

        progress = await store.get_consolidation_progress(brain.id)
        assert progress is not None
        run_id = str(progress["run_id"])
        edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
        assert len(edges) - previous_edge_count <= PAIR_CAP
        previous_edge_count = len(edges)
        last_report = report

        if attempt == 0:
            assert report.extra["consolidation_status"] == "paused"
            strategy_state = progress["strategy_states"]["semantic_link"]
            pending = strategy_state["pending"]
            assert len(pending) == 1
            manifest = json.loads(pending[0])
            source_ref = manifest["source_state_ref"]
            inspection_owner = "semantic-coverage-inspect-" + uuid.uuid4().hex
            assert await acquire(brain.id, inspection_owner)
            try:
                assert await store.claim_consolidation_progress(brain.id, inspection_owner)
                staged = await store.load_semantic_discovery_state(
                    source_ref["state_id"], source_ref["revision"]
                )
            finally:
                assert await release(brain.id, inspection_owner)
            assert staged is not None
            assert staged["batch_index"] == 0
            assert staged.get("coverage_offset") is None
            assert staged["reference_time"] == reference_time.isoformat(timespec="microseconds")

        status = report.extra.get("consolidation_status")
        if status == "completed":
            completed_in = attempt + 1
            break
        assert status == "paused"
        assert progress["status"] == "paused"
    else:
        pytest.fail("semantic coverage batches did not complete within 40 lease attempts")

    assert last_report is not None
    assert last_report.extra["semantic_coverage_complete"] is True
    assert len(acquired_owners) == completed_in
    assert released_owners == acquired_owners
    assert len(set(acquired_owners)) == len(acquired_owners)

    final_progress = await store.get_consolidation_progress(brain.id)
    assert final_progress is not None
    strategy_state = final_progress["strategy_states"]["semantic_link"]
    assert strategy_state["phase"] == "completed"
    assert strategy_state["counters"]["semantic_synapses_created"] == len(
        await store.get_synapses(type=SynapseType.SIMILAR_TO)
    )
    semantic_edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
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

    first = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK], reference_time=reference_time
    )
    assert first.extra["consolidation_status"] == "paused"
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None

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
    report = first
    for _ in range(20):
        report = await _resume_semantic_link(store, run_id)
        edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
        assert len(edges) <= 10
        if report.extra.get("consolidation_status") == "completed":
            break
    else:
        pytest.fail("semantic coverage did not finish after a post-reference insert")

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

    first = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK], reference_time=reference_time
    )
    assert first.extra["consolidation_status"] == "paused"
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    before = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert len(before) == PAIR_CAP

    candidate = await store.get_neuron("semantic-coverage-006")
    assert candidate is not None
    await store.update_neuron(candidate.with_metadata(_embedding=[0.0] + [1.0] * 1023))

    report = None
    for _ in range(3):
        report = await _resume_semantic_link(store, str(progress["run_id"]))
        after = await store.get_synapses(type=SynapseType.SIMILAR_TO)
        assert len(after) == len(before)
        if report.extra["consolidation_status"] != "paused":
            break
    assert report is not None
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

    first = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.SEMANTIC_LINK], reference_time=reference_time
    )
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
        report = await _resume_semantic_link(store, str(progress["run_id"]))
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

    run_id: str | None = None
    previous_edge_count = 0
    last_report = None
    coverage_reports: list[bool] = []
    for _ in range(20):
        if run_id is None:
            report = await ConsolidationEngine(store, ConsolidationConfig()).run(
                strategies=[ConsolidationStrategy.SEMANTIC_LINK],
                reference_time=reference_time,
            )
        else:
            report = await _resume_semantic_link(store, run_id)
        last_report = report
        coverage_reports.append(report.extra.get("semantic_coverage_complete", False))
        progress = await store.get_consolidation_progress(brain.id)
        assert progress is not None
        run_id = str(progress["run_id"])
        edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
        assert len(edges) - previous_edge_count <= PAIR_CAP
        previous_edge_count = len(edges)
        if report.extra.get("consolidation_status") == "completed":
            break
    else:
        pytest.fail("semantic coverage did not finish across ineligible ID-order rows")

    assert last_report is not None
    edges = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    covered_ids = {endpoint for edge in edges for endpoint in (edge.source_id, edge.target_id)}
    assert len(edges) == 7
    assert covered_ids == eligible_ids
    assert last_report.extra["semantic_coverage_complete"] is True
    assert last_report.extra["semantic_coverage_offset"] == 3
    assert all(not complete for complete in coverage_reports[:-1])
    progress = await store.get_consolidation_progress(brain.id)
    assert progress is not None
    assert (
        progress["strategy_states"]["semantic_link"]["counters"]["semantic_synapses_created"] == 7
    )

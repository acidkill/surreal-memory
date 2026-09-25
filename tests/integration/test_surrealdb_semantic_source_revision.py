"""Disposable local-DB integration tests for semantic source changefeed fences.

These exercise real SurrealQL on SurrealDB 3.2+. This test deliberately refuses
non-loopback URLs: changefeed validation must never write to a shared or
production endpoint.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import uuid
from dataclasses import replace
from datetime import timedelta
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation_progress import (
    CONSOLIDATION_ENGINE_VERSION,
    PROGRESS_FORMAT_VERSION,
)
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION, ensure_schema
from surreal_memory.storage.surrealdb.semantic_discovery_state import (
    SemanticDiscoveryStateConflictError,
)
from surreal_memory.storage.surrealdb.semantic_source_revision import SemanticSourceChangedError
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_fence_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        if hostname == "localhost":
            return True
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires explicit loopback SURREALDB_URL for a disposable SurrealDB >= 3.2",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="fence_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-source-fence-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_changefeed_fence_detects_concurrent_create_update_and_delete(store) -> None:
    token = await store.capture_semantic_source_token()
    await store.assert_semantic_source_unchanged(token)

    neurons = [
        Neuron.create(type=NeuronType.CONCEPT, content=f"fence-probe-{index}")
        for index in range(64)
    ]
    await asyncio.gather(*(store.add_neuron(neuron) for neuron in neurons))
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)

    await asyncio.sleep(1.1)
    token = await store.capture_semantic_source_token()
    changed_neuron = replace(neurons[0], content="fence-probe-updated")
    await store.update_neuron(changed_neuron)
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)

    await asyncio.sleep(1.1)
    token = await store.capture_semantic_source_token()
    assert await store.delete_neuron(neurons[1].id)
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)


@pytest.mark.asyncio
async def test_changefeed_fence_detects_synapse_create_update_and_delete(store) -> None:
    source = Neuron.create(type=NeuronType.CONCEPT, content="source")
    target = Neuron.create(type=NeuronType.CONCEPT, content="target")
    await store.add_neuron(source)
    await store.add_neuron(target)
    await asyncio.sleep(1.1)

    token = await store.capture_semantic_source_token()
    await store.assert_semantic_source_unchanged(token)
    synapse = Synapse.create(
        source_id=source.id,
        target_id=target.id,
        type=SynapseType.RELATED_TO,
    )
    await store.add_synapse(synapse)
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)

    await asyncio.sleep(1.1)
    token = await store.capture_semantic_source_token()
    await store.update_synapse(replace(synapse, weight=0.75))
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)

    await asyncio.sleep(1.1)
    token = await store.capture_semantic_source_token()
    assert await store.delete_synapse(synapse.id)
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)


@pytest.mark.asyncio
async def test_frozen_reference_ignores_late_inserts_in_scans_and_source_fence(store) -> None:
    reference_time = store._parse_datetime(await store._query_response("RETURN time::now()"))
    before = reference_time - timedelta(seconds=2)
    source = replace(
        Neuron.create(type=NeuronType.CONCEPT, content="frozen-source"), created_at=before
    )
    target = replace(
        Neuron.create(type=NeuronType.ENTITY, content="frozen-target"), created_at=before
    )
    await store.add_neuron(source)
    await store.add_neuron(target)
    await asyncio.sleep(1.1)
    reference_time = store._parse_datetime(await store._query_response("RETURN time::now()"))
    token = await store.capture_semantic_source_token()

    late = replace(
        Neuron.create(type=NeuronType.CONCEPT, content="post-reference"),
        created_at=reference_time + timedelta(seconds=5),
    )
    await store.add_neuron(late)
    late_edge = replace(
        Synapse.create(
            source_id=source.id,
            target_id=late.id,
            type=SynapseType.RELATED_TO,
        ),
        created_at=reference_time + timedelta(seconds=5),
    )
    await store.add_synapse(late_edge)

    neurons = await store.find_neurons_after_id(
        None, limit=100, created_before=reference_time, ephemeral=None
    )
    synapses = await store.get_synapses_after_id(None, limit=100, created_before=reference_time)
    degrees = await store.get_synapse_degrees(created_before=reference_time)
    existing = await store.find_existing_synapse_pairs(
        [(source.id, late.id)], created_before=reference_time
    )
    assert {neuron.id for neuron in neurons} == {source.id, target.id}
    assert synapses == []
    assert source.id not in degrees
    assert existing == set()
    await store.assert_semantic_source_unchanged(token, created_before=reference_time)

    await store.update_neuron(replace(source, content="pre-reference-source-edited"))
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token, created_before=reference_time)

    await asyncio.sleep(1.1)
    token = await store.capture_semantic_source_token()
    assert await store.delete_neuron(target.id)
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token, created_before=reference_time)


@pytest.mark.asyncio
async def test_frozen_fence_ignores_unchanged_source_table_ddl_but_rejects_old_update(
    store,
) -> None:
    old_created_at = store._parse_datetime(
        await store._query_response("RETURN time::now()")
    ) - timedelta(days=1)
    old_neuron = replace(
        Neuron.create(type=NeuronType.CONCEPT, content="pre-reference-ddl-fence"),
        created_at=old_created_at,
    )
    await store.add_neuron(old_neuron)

    reference_time = store._parse_datetime(await store._query_response("RETURN time::now()"))
    token = await store.capture_semantic_source_token()
    barrier_stamp, _, _ = store._decode_token(token, store.current_brain_id)

    await store._query("ALTER TABLE neuron CHANGEFEED 7d")
    await store._query("ALTER TABLE synapse CHANGEFEED 7d")

    for table in ("neuron", "synapse"):
        events = await store._query(
            f"SHOW CHANGES FOR TABLE {table} SINCE {barrier_stamp} LIMIT 100"
        )
        assert any(
            isinstance(change, dict)
            and isinstance(change.get("define_table"), dict)
            and change["define_table"].get("name") == table
            for event in events
            for change in event.get("changes", [])
        )

    await store.assert_semantic_source_unchanged(token, created_before=reference_time)

    await store.update_neuron(replace(old_neuron, content="pre-reference-updated"))
    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token, created_before=reference_time)


@pytest.mark.asyncio
async def test_created_at_readonly_converges_existing_rows_and_preserves_ordinary_updates(
    store,
) -> None:
    # Simulate a v12 database whose existing field definitions predate this
    # invariant. The field rewrite must preserve data while upgrading in place.
    await store._query(
        "DEFINE FIELD OVERWRITE created_at ON neuron TYPE datetime DEFAULT time::now()"
    )
    await store._query(
        "DEFINE FIELD OVERWRITE created_at ON synapse TYPE datetime DEFAULT time::now()"
    )
    created_at = utcnow() - timedelta(days=2)
    neurons = [
        replace(
            Neuron.create(type=NeuronType.CONCEPT, content=f"readonly-{i}"), created_at=created_at
        )
        for i in range(2)
    ]
    for neuron in neurons:
        await store.add_neuron(neuron)
    edge = replace(
        Synapse.create(
            source_id=neurons[0].id,
            target_id=neurons[1].id,
            type=SynapseType.RELATED_TO,
        ),
        created_at=created_at,
    )
    await store.add_synapse(edge)

    await ensure_schema(store._ensure_conn())

    original_neuron = await store.get_neuron(neurons[0].id)
    original_edge = await store.get_synapse(edge.id)
    assert original_neuron is not None and original_neuron.created_at == created_at
    assert original_edge is not None and original_edge.created_at == created_at

    await store.update_neuron(replace(neurons[0], content="ordinary neuron update"))
    await store.update_synapse(replace(edge, weight=0.8))
    updated_neuron = await store.get_neuron(neurons[0].id)
    updated_edge = await store.get_synapse(edge.id)
    assert updated_neuron is not None and updated_neuron.created_at == created_at
    assert updated_edge is not None and updated_edge.created_at == created_at

    # SurrealDB accepts these statements but READONLY discards the field write.
    await store._query(
        "UPDATE type::record('neuron', $id) SET created_at = $value",
        id=neurons[0].id.replace("neuron:", ""),
        value=utcnow(),
    )
    await store._query(
        "UPDATE type::record('synapse', $id) SET created_at = $value",
        id=edge.id.replace("synapse:", ""),
        value=utcnow(),
    )
    rejected_neuron_rewrite = await store.get_neuron(neurons[0].id)
    rejected_edge_rewrite = await store.get_synapse(edge.id)
    assert rejected_neuron_rewrite is not None and rejected_neuron_rewrite.created_at == created_at
    assert rejected_edge_rewrite is not None and rejected_edge_rewrite.created_at == created_at


async def test_barrier_discovery_pages_past_retained_events(store, monkeypatch) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_BARRIER_CHANGEFEED_PAGE_SIZE", 2)
    monkeypatch.setattr(revision_module, "_BARRIER_CHANGEFEED_MAX_ROWS", 100)
    now = await store._query_response("RETURN time::now()")
    created_at = store._parse_datetime(now)
    prior_markers = [uuid.uuid4().hex for _ in range(7)]
    for marker_id in prior_markers:
        await store._query(
            "CREATE type::record('semantic_source_barrier', $token_id) "
            "CONTENT $content RETURN AFTER",
            token_id=marker_id,
            content={"brain_id": store.current_brain_id, "created_at": created_at},
        )

    original_query = store._query
    page_queries: list[str] = []

    async def recording_query(sql: str, **params):
        if sql.startswith("SHOW CHANGES FOR TABLE semantic_source_barrier"):
            page_queries.append(sql)
        return await original_query(sql, **params)

    monkeypatch.setattr(store, "_query", recording_query)

    async def before_prior_markers():
        return created_at

    monkeypatch.setattr(store, "_server_now", before_prior_markers)
    token = await store.capture_semantic_source_token()
    await store.assert_semantic_source_unchanged(token)

    assert len(page_queries) >= 4
    assert ' SINCE d"' in page_queries[0]
    cursors = [int(sql.split(" SINCE ", 1)[1].split(" LIMIT ", 1)[0]) for sql in page_queries[1:]]
    assert cursors == sorted(set(cursors))


async def test_raw_versionstamp_is_an_inclusive_barrier_ordered_across_tables(store) -> None:
    before = Neuron.create(type=NeuronType.CONCEPT, content="before-barrier")
    await store.add_neuron(before)
    token = await store.capture_semantic_source_token()
    barrier_stamp, _, readonly_at_capture = store._decode_token(token, store.current_brain_id)
    assert readonly_at_capture is True

    after = Neuron.create(type=NeuronType.CONCEPT, content="after-barrier")
    await store.add_neuron(after)
    events = await store._query("SHOW CHANGES FOR TABLE neuron SINCE 0 LIMIT 10000")

    def event_stamp(record_id: str) -> int:
        marker = record_id.replace("-", "_")
        return next(
            event["versionstamp"]
            for event in events
            if store._contains_marker(event.get("changes"), marker)
        )

    def contains_record(rows: list[dict], record_id: str) -> bool:
        marker = record_id.replace("-", "_")
        return any(store._contains_marker(event.get("changes"), marker) for event in rows)

    before_stamp = event_stamp(before.id)
    after_stamp = event_stamp(after.id)
    assert before_stamp < barrier_stamp < after_stamp

    since_barrier = await store._query(
        f"SHOW CHANGES FOR TABLE neuron SINCE {barrier_stamp} LIMIT 10000"
    )
    since_bounded = await store._query(
        f"SHOW CHANGES FOR TABLE neuron SINCE {barrier_stamp} LIMIT 10"
    )
    assert not contains_record(since_barrier, before.id)
    assert contains_record(since_barrier, after.id)
    assert contains_record(since_bounded, after.id)
    # SINCE is inclusive: the event whose raw stamp is used as the cursor is
    # returned. The barrier itself is on another table, so no source event is
    # skipped by using its exact raw stamp.
    since_before = await store._query(
        f"SHOW CHANGES FOR TABLE neuron SINCE {before_stamp} LIMIT 10000"
    )
    assert contains_record(since_before, before.id)
    assert contains_record(since_before, after.id)
    since_after = await store._query(
        f"SHOW CHANGES FOR TABLE neuron SINCE {after_stamp} LIMIT 10000"
    )
    assert contains_record(since_after, after.id)

    with pytest.raises(SemanticSourceChangedError):
        await store.assert_semantic_source_unchanged(token)


@pytest.mark.asyncio
async def test_snapshot_create_is_exactly_idempotent_on_real_surrealdb(store) -> None:
    """A retry after an interrupted checkpoint must not overwrite a staged revision."""
    state_id = uuid.uuid4().hex
    payload = {
        "brain_id": store._get_brain_id(),
        "run_id": "integration-run",
        "owner_token": "integration-owner",
        "source_token": "integration-source",
        "candidates": [["neuron-one", 0.9, 1]],
    }
    await store.save_semantic_discovery_state(state_id, 1, payload)
    await store.save_semantic_discovery_state(state_id, 1, payload)
    rows = await store._query(
        "SELECT id FROM semantic_discovery_state WHERE state_id = $state_id",
        state_id=state_id,
    )
    assert len(rows) == 1
    with pytest.raises(SemanticDiscoveryStateConflictError):
        await store.save_semantic_discovery_state(
            state_id, 1, {**payload, "candidates": [["neuron-two", 0.9, 1]]}
        )


async def test_snapshot_load_follows_serialized_pending_manifest_after_lease_rotation(
    store,
) -> None:
    state_id = uuid.uuid4().hex
    payload = {
        "brain_id": store._get_brain_id(),
        "run_id": "integration-resume-run",
        "owner_token": "integration-owner-old",
        "source_token": "integration-source-token",
        "candidates": [["neuron-one", 0.9, 1]],
    }
    await store.save_semantic_discovery_state(state_id, 1, payload)

    current_owner = "integration-owner-current"
    brain_id = store._get_brain_id()
    assert await store.acquire_consolidation_lease(brain_id, current_owner, lease_seconds=30)
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": {
            "state_id": state_id,
            "revision": 1,
            "state_revision": 1,
            "source_token": payload["source_token"],
        },
    }
    pending_manifest = json.dumps(
        [json.dumps(manifest, sort_keys=True, separators=(",", ":"))],
        separators=(",", ":"),
    )
    await store.create_consolidation_progress(
        {
            "brain_id": brain_id,
            "run_id": payload["run_id"],
            "schema_version": SCHEMA_VERSION,
            "engine_version": CONSOLIDATION_ENGINE_VERSION,
            "format_version": PROGRESS_FORMAT_VERSION,
            "requested_strategies": ["semantic_link"],
            "completed_strategies": [],
            "options_fingerprint": "integration-options",
            "reference_time": utcnow(),
            "owner_token": current_owner,
            "status": "paused",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_discovery_neurons",
            "cursor": None,
            "strategy_states": {"semantic_link": {"pending": [pending_manifest]}},
            "counters": {},
            "started_at": utcnow(),
            "updated_at": utcnow(),
            "last_error": None,
        }
    )

    assert await store.load_semantic_discovery_state(state_id, 1) == payload

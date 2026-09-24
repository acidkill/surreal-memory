"""Disposable local-DB integration tests for semantic source changefeed fences.

These exercise real SurrealQL on SurrealDB 3.2+. This test deliberately refuses
non-loopback URLs: changefeed validation must never write to a shared or
production endpoint.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import uuid
from dataclasses import replace
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.surrealdb.semantic_source_revision import SemanticSourceChangedError
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

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
    token = await store.capture_semantic_source_token()
    await store.assert_semantic_source_unchanged(token)

    assert len(page_queries) >= 4
    cursors = [int(sql.split(" SINCE ", 1)[1].split(" LIMIT ", 1)[0]) for sql in page_queries]
    assert cursors[0] == 0
    assert cursors == sorted(set(cursors))


async def test_raw_versionstamp_is_an_inclusive_barrier_ordered_across_tables(store) -> None:
    before = Neuron.create(type=NeuronType.CONCEPT, content="before-barrier")
    await store.add_neuron(before)
    token = await store.capture_semantic_source_token()
    barrier_stamp, _ = store._decode_token(token, store.current_brain_id)

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

"""Disposable SurrealDB proof for brain-scoped native neuron record batches."""

from __future__ import annotations

import ipaddress
import json
import os
import uuid
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_neuron_batch_v9_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        return hostname == "localhost" or bool(
            hostname and ipaddress.ip_address(hostname).is_loopback
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires an explicit loopback URL for disposable SurrealDB 3.2.x",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="neuron-record-batch-v9-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


def _captured_queries(store: SurrealDBStorage) -> list[tuple[str, dict[str, object]]]:
    queries: list[tuple[str, dict[str, object]]] = []
    original_query = store._query

    async def capture(sql: str, **params):
        queries.append((sql, params))
        return await original_query(sql, **params)

    store._query = capture  # type: ignore[method-assign]
    return queries


@pytest.mark.asyncio
async def test_get_neurons_batch_reads_typed_record_chunks_with_brain_fence(
    store: SurrealDBStorage,
) -> None:
    numeric_id = "1234567890123456"
    uuid_id = str(uuid.uuid4())
    batch_ids = [f"batch-record-{index:03d}" for index in range(205)]
    active_neurons = [
        Neuron.create(NeuronType.CONCEPT, "numeric string id", neuron_id=numeric_id),
        Neuron.create(NeuronType.CONCEPT, "UUID id", neuron_id=uuid_id),
        *[
            Neuron.create(NeuronType.CONCEPT, f"synthetic batch item {index}", neuron_id=neuron_id)
            for index, neuron_id in enumerate(batch_ids)
        ],
    ]
    await store.add_neurons_batch(active_neurons, record_change=False)

    active_brain_id = store._get_brain_id()
    foreign_brain = Brain.create(name="neuron-record-batch-foreign-v9-it")
    await store.save_brain(foreign_brain)
    store.set_brain(foreign_brain.id)
    foreign_id = str(uuid.uuid4())
    await store.add_neuron(
        Neuron.create(NeuronType.CONCEPT, "foreign synthetic neuron", neuron_id=foreign_id)
    )
    store.set_brain(active_brain_id)

    queries = _captured_queries(store)
    requested = [
        numeric_id,
        uuid_id,
        foreign_id,
        "missing-synthetic-neuron",
        numeric_id,
        *batch_ids,
    ]
    found = await store.get_neurons_batch(requested)

    expected_ids = {numeric_id, uuid_id, *batch_ids}
    assert set(found) == expected_ids
    assert found[numeric_id].content == "numeric string id"
    assert found[uuid_id].content == "UUID id"
    assert found[batch_ids[-1]].content == "synthetic batch item 204"
    assert len(queries) == 3

    for sql, params in queries:
        assert sql.startswith("SELECT * FROM [type::record('neuron',")
        assert sql.endswith("] WHERE brain_id = $brain_id")
        assert "FROM neuron WHERE" not in sql
        assert params["brain_id"] == active_brain_id
        assert 1 <= len(params) - 1 <= 100
        plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN FULL", **params)
        plan_text = json.dumps(plan, ensure_ascii=False, default=str)
        assert "SourceExpr" in plan_text
        assert "TableScan" not in plan_text

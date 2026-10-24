"""Disposable live regression for brain-scoped pending update collapse."""

from __future__ import annotations

import ipaddress
import os
import uuid
from collections import Counter
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_CHANGE_LOG_COLLAPSE_TEST_URL")
SURREALDB_USER = os.getenv("SURREALDB_CHANGE_LOG_COLLAPSE_TEST_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_CHANGE_LOG_COLLAPSE_TEST_PASS", "root")


def _is_authorized_disposable_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.port == 18765
            and parsed.hostname is not None
            and ipaddress.ip_address(parsed.hostname).is_loopback
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_authorized_disposable_url(SURREALDB_URL),
        reason="requires SURREALDB_CHANGE_LOG_COLLAPSE_TEST_URL on loopback port 18765",
    ),
]


@pytest_asyncio.fixture
async def storage():  # type: ignore[no-untyped-def]
    from surrealdb import AsyncSurreal

    namespace = "smem_change_log_collapse_" + uuid.uuid4().hex[:12]
    database = "it_" + uuid.uuid4().hex[:12]
    root = AsyncSurreal(SURREALDB_URL)
    await root.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
    await root.query(f"DEFINE NAMESPACE {namespace}; DEFINE DATABASE {database}")
    with suppress(NotImplementedError):
        await root.close()

    store = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=namespace,
        database=database,
    )
    try:
        await store.initialize()
        brain = Brain.create(name="change-log-collapse-live")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        yield store
    finally:
        await store.close()
        cleanup = AsyncSurreal(SURREALDB_URL)
        await cleanup.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
        await cleanup.query(f"REMOVE NAMESPACE {namespace}")
        with suppress(NotImplementedError):
            await cleanup.close()


def _replay(rows: list[dict[str, Any]]) -> dict[tuple[str, str], Any]:
    state: dict[tuple[str, str], Any] = {}
    for row in sorted(rows, key=lambda item: int(item.get("sequence", 0) or 0)):
        key = (str(row.get("entity_type", "")), str(row.get("entity_id", "")))
        if row.get("operation") == "delete":
            state.pop(key, None)
        else:
            state[key] = row.get("payload")
    return state


@pytest.mark.asyncio
async def test_collapse_preserves_replay_state_brain_scope_and_protected_rows(
    storage: SurrealDBStorage,
) -> None:
    active_brain_id = storage._get_brain_id()
    foreign_brain = Brain.create(name="change-log-collapse-foreign-live")
    await storage.save_brain(foreign_brain)

    large_payload = {"text": "synthetic-" * 8192, "vector": [0.125] * 512}
    synced_sequences = [
        await storage.record_change("synapse", "already-synced", "update", payload=large_payload)
        for _ in range(2)
    ]
    await storage._query(
        "UPDATE change_log SET synced = true WHERE brain_id = $brain_id AND sequence <= $seq",
        brain_id=active_brain_id,
        seq=max(synced_sequences),
    )
    for revision in range(3):
        await storage.record_change(
            "synapse",
            "pending-update",
            "update",
            payload={**large_payload, "revision": revision},
        )
    for _ in range(2):
        await storage.record_change("neuron", "keep-insert", "insert")
    for _ in range(2):
        await storage.record_change("neuron", "keep-delete", "delete")

    storage.set_brain(foreign_brain.id)
    await storage.record_change(
        "synapse", "pending-update", "update", payload={**large_payload, "revision": 99}
    )
    storage.set_brain(active_brain_id)

    before = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id ORDER BY sequence",
        brain_id=active_brain_id,
    )
    queries: list[tuple[str, dict[str, Any]]] = []
    original_query_response = storage._query_response

    async def capture(sql: str, **params: Any) -> Any:
        queries.append((sql, params))
        return await original_query_response(sql, **params)

    storage._query_response = capture  # type: ignore[method-assign]
    cursor = None
    collapsed = 0
    while True:
        page_collapsed, cursor, exhausted = await storage.collapse_pending_updates(cursor=cursor)
        collapsed += page_collapsed
        if exhausted:
            break
    assert collapsed == 2
    storage._query_response = original_query_response  # type: ignore[method-assign]

    after = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id ORDER BY sequence",
        brain_id=active_brain_id,
    )
    foreign_after = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id",
        brain_id=foreign_brain.id,
    )

    assert _replay(after) == _replay(before)
    assert Counter(
        (row["operation"], row["entity_id"], bool(row.get("synced", False))) for row in after
    ) == Counter(
        {
            ("update", "already-synced", True): 2,
            ("update", "pending-update", False): 1,
            ("insert", "keep-insert", False): 2,
            ("delete", "keep-delete", False): 2,
        }
    )
    assert len(foreign_after) == 1
    delete_queries = [
        (sql, params) for sql, params in queries if sql.startswith("DELETE FROM $record_ids")
    ]
    assert len(delete_queries) == 1
    delete_sql, delete_params = delete_queries[0]
    assert delete_sql.endswith("RETURN BEFORE.id")
    assert "brain_id = $brain_id" in delete_sql
    assert "operation = 'update'" in delete_sql
    assert "synced = false" in delete_sql
    assert delete_params["brain_id"] == active_brain_id
    assert len(delete_params["record_ids"]) == 2


@pytest.mark.asyncio
async def test_small_pages_reach_late_duplicates_and_keep_tied_maxima_and_protected_rows(
    storage: SurrealDBStorage,
) -> None:
    active_brain_id = storage._get_brain_id()
    foreign_brain = Brain.create(name="change-log-collapse-small-page-foreign")
    await storage.save_brain(foreign_brain)

    await storage.record_change("synapse", "unique-1", "update", payload={"v": 1})
    await storage.record_change("synapse", "unique-2", "update", payload={"v": 1})
    tail_sequences = [
        await storage.record_change("synapse", "tail", "update", payload={"v": revision})
        for revision in (1, 2, 3)
    ]
    await storage._query(
        "UPDATE change_log SET sequence = $tie_sequence "
        "WHERE brain_id = $brain_id AND sequence IN $sequences",
        brain_id=active_brain_id,
        tie_sequence=max(tail_sequences),
        sequences=tail_sequences[1:],
    )

    synced_sequences = [await storage.record_change("fiber", "synced", "update") for _ in range(2)]
    await storage._query(
        "UPDATE change_log SET synced = true WHERE brain_id = $brain_id AND sequence IN $sequences",
        brain_id=active_brain_id,
        sequences=synced_sequences,
    )
    for operation in ("insert", "delete"):
        for _ in range(2):
            await storage.record_change("neuron", operation, operation)

    storage.set_brain(foreign_brain.id)
    await storage.record_change("synapse", "tail", "update", payload={"v": 99})
    storage.set_brain(active_brain_id)

    cursor = None
    collapsed = 0
    pages = 0
    while True:
        page_collapsed, cursor, exhausted = await storage.collapse_pending_updates(
            max_rows=2, cursor=cursor
        )
        collapsed += page_collapsed
        pages += 1
        if exhausted:
            break

    assert pages >= 3, "unique-key prefix pages must not starve the duplicate tail"
    assert collapsed == 1
    active_rows = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id ORDER BY sequence, id",
        brain_id=active_brain_id,
    )
    tail_rows = [row for row in active_rows if row.get("entity_id") == "tail"]
    assert sorted(row["payload"]["v"] for row in tail_rows) == [2, 3]
    assert (
        sum(
            row.get("entity_id") == "synced" and bool(row.get("synced", False))
            for row in active_rows
        )
        == 2
    )
    assert sum(row.get("operation") == "insert" for row in active_rows) == 2
    assert sum(row.get("operation") == "delete" for row in active_rows) == 2

    foreign_rows = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id",
        brain_id=foreign_brain.id,
    )
    assert len(foreign_rows) == 1
    assert foreign_rows[0]["payload"] == {"v": 99}


@pytest.mark.asyncio
async def test_small_pages_report_empty_tail_and_reject_malformed_cursor(
    storage: SurrealDBStorage,
) -> None:
    brain_id = storage._get_brain_id()
    await storage.record_change("synapse", "one", "update")
    await storage.record_change("synapse", "two", "update")

    deleted, cursor, exhausted = await storage.collapse_pending_updates(max_rows=2)
    assert (deleted, exhausted) == (0, False)
    assert cursor is not None
    assert await storage.collapse_pending_updates(max_rows=2, cursor=cursor) == (
        0,
        cursor,
        True,
    )

    with pytest.raises(ValueError, match="invalid change-log collapse cursor"):
        await storage.collapse_pending_updates(max_rows=2, cursor='{"watermark":2}')
    rows = await storage._query(
        "SELECT id FROM change_log WHERE brain_id = $brain_id AND operation = 'update'",
        brain_id=brain_id,
    )
    assert len(rows) == 2

"""Disposable live regression for brain-scoped pending update collapse."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import pytest
import pytest_asyncio

import surreal_memory.storage.surrealdb.store as store_module
from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import ensure_naive_utc

SURREALDB_URL = os.getenv("SURREALDB_CHANGE_LOG_COLLAPSE_TEST_URL") or os.getenv(
    "SMEM_TEST_SURREALDB_URL"
)
SURREALDB_USER = os.getenv("SURREALDB_CHANGE_LOG_COLLAPSE_TEST_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_CHANGE_LOG_COLLAPSE_TEST_PASS", "root")


def _is_authorized_disposable_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
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
        not _is_authorized_disposable_url(SURREALDB_URL),
        reason="requires an explicit loopback change-log-collapse test URL",
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "_COLLAPSE_READ_PAGE_ROWS", 1)
    active_brain_id = storage._get_brain_id()
    foreign_brain = Brain.create(name="change-log-collapse-small-page-foreign")
    await storage.save_brain(foreign_brain)

    await storage.record_change("synapse", "unique-1", "update", payload={"v": 1})
    await storage.record_change("synapse", "unique-2", "update", payload={"v": 1})
    tail_sequences = [
        await storage.record_change("synapse", "tail", "update", payload={"v": revision})
        for revision in (1, 2, 3)
    ]
    for entity_type in ("neuron", "fiber"):
        for revision in (1, 2):
            await storage.record_change(entity_type, "shared-id", "update", payload={"v": revision})
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
    assert collapsed == 3
    active_rows = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id ORDER BY sequence, id",
        brain_id=active_brain_id,
    )
    tail_rows = [row for row in active_rows if row.get("entity_id") == "tail"]
    assert sorted(row["payload"]["v"] for row in tail_rows) == [2, 3]
    shared_id_rows = [row for row in active_rows if row.get("entity_id") == "shared-id"]
    assert len(shared_id_rows) == 2
    assert {(row["entity_type"], row["payload"]["v"]) for row in shared_id_rows} == {
        ("neuron", 2),
        ("fiber", 2),
    }
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
async def test_collapse_keyset_plans_seek_from_advancing_boundary(
    storage: SurrealDBStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "_COLLAPSE_READ_PAGE_ROWS", 1)
    for entity in ("first", "middle", "last"):
        for revision in (1, 2):
            await storage.record_change("synapse", entity, "update", payload={"v": revision})

    emitted: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    original_query = storage._query

    async def capture(sql: str, **params: Any) -> list[dict[str, Any]]:
        rows = await original_query(sql, **params)
        if "WITH INDEX idx_changelog_seq" in sql or "WITH INDEX idx_changelog_entity_max" in sql:
            emitted.append((sql, dict(params), rows))
        return rows

    monkeypatch.setattr(storage, "_query", capture)
    await storage.collapse_pending_updates(max_rows=3)

    forward = next(item for item in emitted if "last_sequence" in item[1])
    maxima = next(item for item in emitted if "idx_changelog_entity_max" in item[0])

    def scans(plan: Any) -> list[dict[str, Any]]:
        if isinstance(plan, list):
            return [scan for child in plan for scan in scans(child)]
        if not isinstance(plan, dict):
            return []
        own = [plan["attributes"]] if plan.get("operator") == "IndexScan" else []
        return own + [scan for child in plan.values() for scan in scans(child)]

    def has_operator(plan: Any, operator: str) -> bool:
        if isinstance(plan, list):
            return any(has_operator(child, operator) for child in plan)
        if not isinstance(plan, dict):
            return False
        return plan.get("operator") == operator or any(
            has_operator(child, operator) for child in plan.values()
        )

    forward_plan = await storage._query_response(forward[0] + " EXPLAIN", **forward[1])
    forward_index = scans(forward_plan)
    assert len(forward_index) == 1, forward_plan
    assert forward_index[0]["index"] == "idx_changelog_seq", forward_plan
    assert forward_index[0]["direction"] == "Forward", forward_plan
    assert f"MoreThanEqual {forward[1]['last_sequence']}" in forward_index[0]["access"]

    maxima_plan = await storage._query_response(maxima[0] + " EXPLAIN", **maxima[1])
    maxima_index = scans(maxima_plan)
    assert len(maxima_index) == 1, maxima_plan
    assert maxima_index[0]["index"] == "idx_changelog_entity_max", maxima_plan
    assert maxima[1]["entity_type"] == "synapse"
    assert maxima[1]["entity_ids"] == ["first", "middle"]
    access = str(maxima_index[0]["access"])
    assert all(
        value in access for value in (storage._get_brain_id(), "synapse", *maxima[1]["entity_ids"])
    ), access

    noindex_sql = maxima[0].replace("WITH INDEX idx_changelog_entity_max", "WITH NOINDEX")
    noindex_rows = await storage._query(noindex_sql, **maxima[1])
    assert sorted(noindex_rows, key=lambda row: row["entity_id"]) == sorted(
        maxima[2], key=lambda row: row["entity_id"]
    )
    noindex_plan = await storage._query_response(noindex_sql + " EXPLAIN", **maxima[1])
    assert scans(noindex_plan) == []
    assert has_operator(noindex_plan, "TableScan"), noindex_plan


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


@pytest.mark.asyncio
async def test_update_appended_after_watermark_is_left_for_the_next_pass(
    storage: SurrealDBStorage,
) -> None:
    brain_id = storage._get_brain_id()
    await storage.record_change("synapse", "watermark-tail", "update", payload={"v": 1})
    await storage.record_change("synapse", "watermark-tail", "update", payload={"v": 2})

    injected = False
    original_query_response = storage._query_response

    async def inject_after_watermark(sql: str, **params: Any) -> Any:
        nonlocal injected
        rows = await original_query_response(sql, **params)
        if not injected and "ORDER BY sequence DESC LIMIT 1" in sql:
            injected = True
            await storage.record_change("synapse", "watermark-tail", "update", payload={"v": 3})
        return rows

    storage._query_response = inject_after_watermark  # type: ignore[method-assign]
    deleted, cursor, exhausted = await storage.collapse_pending_updates(max_rows=10)
    storage._query_response = original_query_response  # type: ignore[method-assign]

    rows = await storage._query(
        "SELECT * FROM change_log WHERE brain_id = $brain_id AND entity_id = $entity_id "
        "AND operation = 'update' ORDER BY sequence",
        brain_id=brain_id,
        entity_id="watermark-tail",
    )
    assert injected
    assert deleted == 1
    assert exhausted
    assert json.loads(cursor or "")["watermark"] == 2
    assert [row["payload"]["v"] for row in rows] == [2, 3]


@pytest.mark.asyncio
async def test_habit_source_pages_resume_with_physical_ids_and_frozen_upper_bound(
    storage: SurrealDBStorage,
) -> None:
    brain_id = storage._get_brain_id()
    foreign_brain = Brain.create(name="habit-source-foreign-live")
    await storage.save_brain(foreign_brain)
    store_conn = storage._ensure_conn()
    cutoff = datetime(2026, 10, 24, 12, 0, 0)
    until = cutoff
    since = cutoff - timedelta(days=1)

    for record_id, owner_id, created_at in [
        ("action_a", brain_id, cutoff),
        ("action_b", brain_id, cutoff),
        ("action_future", brain_id, cutoff + timedelta(microseconds=1)),
        ("action_foreign", foreign_brain.id, cutoff),
    ]:
        await store_conn.create(
            f"action_log:{record_id}",
            {
                "brain_id": owner_id,
                "action_type": "recall",
                "action_context": record_id,
                "tags": [],
                "session_id": "session-a",
                "created_at": created_at,
            },
        )

    for physical_id, logical_id, owner_id, created_at in [
        ("tool_a", "logical_a", brain_id, cutoff),
        ("tool_b", "logical_b", brain_id, cutoff),
        ("tool_future", "logical_future", brain_id, cutoff + timedelta(microseconds=1)),
        ("tool_foreign", "logical_foreign", foreign_brain.id, cutoff),
    ]:
        await store_conn.create(
            f"tool_events:{physical_id}",
            {
                "event_id": logical_id,
                "brain_id": owner_id,
                "tool_name": physical_id,
                "session_id": "session-a",
                "created_at": created_at,
            },
        )

    action_first = await storage.get_action_sequences(since=since, until=until, limit=1)
    action_second = await storage.get_action_sequences(
        since=since,
        until=until,
        limit=1,
        after=(action_first[-1].created_at, action_first[-1].id),
    )
    tool_first = await storage.get_tool_events_for_mining(
        brain_id, since=since, until=until, limit=1
    )
    tool_second = await storage.get_tool_events_for_mining(
        brain_id,
        since=since,
        until=until,
        limit=1,
        after=(datetime.fromisoformat(tool_first[-1]["created_at"]), tool_first[-1]["id"]),
    )

    action_rows = action_first + action_second
    tool_rows = tool_first + tool_second
    assert [event.id for event in action_rows] == ["action_a", "action_b"]
    assert [row["id"] for row in tool_rows] == ["tool_a", "tool_b"]
    assert all(event.created_at == cutoff for event in action_rows)
    assert all(
        ensure_naive_utc(datetime.fromisoformat(row["created_at"])) == cutoff for row in tool_rows
    )

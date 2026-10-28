from __future__ import annotations

import asyncio
from typing import Any

import pytest

from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport
from surreal_memory.engine.consolidation_delta import run_with_delta
from surreal_memory.storage.surrealdb.store import SurrealDBStorage, _DeadWebSocketReceiverError

ORGANIC_SELECT = "in.metadata.indexed != true"


class OfflineConn:
    """Small deterministic stand-in for the SDK's pending-query cleanup race."""

    def __init__(self, *, fail_organic_read_once: bool = False) -> None:
        self.fail_organic_read_once = fail_organic_read_once
        self.recv_task: asyncio.Task[None] | None = None
        self.qry: dict[str, object] = {"request-id": object()}
        self.queries: list[str] = []

    async def query(self, sql: str, _params: dict[str, Any]) -> list[Any]:
        self.queries.append(sql)
        if self.fail_organic_read_once and ORGANIC_SELECT in sql:
            self.fail_organic_read_once = False

            async def receiver_fails() -> None:
                await asyncio.sleep(0)
                self.qry.clear()
                raise RuntimeError("synthetic websocket receiver failure")

            self.recv_task = asyncio.create_task(receiver_fails())
            await asyncio.sleep(0)
            raise KeyError("request-id")
        return []


def make_storage(conn: OfflineConn) -> SurrealDBStorage:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "default"
    store._reauth_lock = asyncio.Lock()
    store._reauth_lock_loop = asyncio.get_running_loop()
    store._conn_generation = 0
    store._conn_loop = asyncio.get_running_loop()
    store._conn = conn
    return store


def prepare_empty_diagnostics(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    async def empty_stats(_brain_id: str) -> dict[str, int]:
        return {
            "neuron_count": 0,
            "structural_neuron_count": 0,
            "synapse_count": 0,
            "fiber_count": 0,
        }

    monkeypatch.setattr(store, "get_stats", empty_stats)
    engine_runs: list[str] = []

    async def run_engine(self: Any, *args: Any, **kwargs: Any) -> ConsolidationReport:
        engine_runs.append(str(kwargs.get("resume_run_id")))
        return ConsolidationReport()

    monkeypatch.setattr(ConsolidationEngine, "run", run_engine)
    return engine_runs


@pytest.mark.asyncio
async def test_resume_diagnostic_retries_only_read_and_reaches_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = OfflineConn(fail_organic_read_once=True)
    store = make_storage(failed)
    engine_runs = prepare_empty_diagnostics(store, monkeypatch)
    replacement = OfflineConn()

    async def reconnect() -> None:
        store._conn = replacement
        store._conn_generation += 1
        store._conn_loop = asyncio.get_running_loop()

    store._reconnect = reconnect  # type: ignore[method-assign]

    delta = await run_with_delta(store, "default", resume_run_id="synthetic-saved-run")

    assert delta.report is not None
    assert engine_runs == ["synthetic-saved-run"]
    assert sum(ORGANIC_SELECT in query for query in failed.queries) == 1
    assert sum(ORGANIC_SELECT in query for query in replacement.queries) == 2
    assert store._conn is replacement


@pytest.mark.asyncio
async def test_healthy_connection_control_reaches_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = OfflineConn()
    store = make_storage(conn)
    engine_runs = prepare_empty_diagnostics(store, monkeypatch)

    delta = await run_with_delta(store, "default", resume_run_id="synthetic-saved-run")

    assert delta.report is not None
    assert engine_runs == ["synthetic-saved-run"]
    assert sum(ORGANIC_SELECT in query for query in conn.queries) == 2


@pytest.mark.asyncio
async def test_repeated_receiver_failure_stops_after_one_read_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = OfflineConn(fail_organic_read_once=True)
    replacement = OfflineConn(fail_organic_read_once=True)
    store = make_storage(failed)
    engine_runs = prepare_empty_diagnostics(store, monkeypatch)
    reconnects: list[int] = []

    async def reconnect() -> None:
        reconnects.append(1)
        store._conn = replacement
        store._conn_generation += 1
        store._conn_loop = asyncio.get_running_loop()

    store._reconnect = reconnect  # type: ignore[method-assign]
    with pytest.raises(_DeadWebSocketReceiverError):
        await run_with_delta(store, "default", resume_run_id="synthetic-saved-run")

    assert len(reconnects) == 1
    assert sum(ORGANIC_SELECT in query for query in failed.queries) == 1
    assert sum(ORGANIC_SELECT in query for query in replacement.queries) == 1
    assert engine_runs == []


@pytest.mark.asyncio
async def test_mutating_query_is_not_replayed_after_receiver_failure() -> None:
    conn = OfflineConn(fail_organic_read_once=False)
    store = make_storage(conn)
    writes: list[str] = []

    async def failing_write(sql: str, _params: dict[str, Any]) -> list[Any]:
        writes.append(sql)
        conn.qry.clear()

        async def receiver_fails() -> None:
            raise RuntimeError("synthetic websocket receiver failure")

        conn.recv_task = asyncio.create_task(receiver_fails())
        await asyncio.sleep(0)
        raise KeyError("request-id")

    conn.query = failing_write  # type: ignore[method-assign]

    with pytest.raises(_DeadWebSocketReceiverError, match="query was not replayed"):
        await store._query_response("CREATE test CONTENT { value: 1 }")

    assert writes == ["CREATE test CONTENT { value: 1 }"]

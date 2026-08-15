"""Regressions behind the 2026-07-28 `smem consolidate` failure.

Two independent defects made a full consolidation unusable on a ~11k-neuron brain:

1. The per-strategy timeout had regressed to 120s. The heavy passes (compress,
   lifecycle, essence backfill) legitimately need minutes on a large brain, so
   they were aborted mid-run and consolidation never converged. The project
   decision is 600s per strategy — and the total budget must exceed it, or one
   slow strategy starves every later one.

2. ``_lifecycle`` fetched ``find_neurons(limit=10000)`` with embeddings included,
   i.e. 10k rows x a 1024-float vector in a single HTTP response. SurrealDB
   dropped the transfer ("[Errno 104] Connection reset by peer") and the reconnect
   was attempted exactly once, in the same instant, so it hit the same reset and
   the whole pass died.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from surreal_memory.engine.consolidation import ConsolidationConfig


class TestStrategyTimeouts:
    def test_per_strategy_timeout_is_ten_minutes(self) -> None:
        """120s aborted the heavy passes on large brains; the decision is 600s."""
        assert ConsolidationConfig().strategy_timeout_seconds == 600.0

    def test_total_budget_exceeds_a_single_strategy(self) -> None:
        """Otherwise one slow strategy consumes everything and the rest time out."""
        cfg = ConsolidationConfig()
        assert cfg.total_timeout_seconds > cfg.strategy_timeout_seconds


class _FakeNeuron:
    def __init__(self, nid: str) -> None:
        self.id = nid
        self.metadata: dict[str, Any] = {"access_frequency": 1, "priority": 5}


class _RecordingStorage:
    """Records how find_neurons is called; returns two short pages."""

    def __init__(self) -> None:
        self.current_brain_id = "default"
        self.brain_id = "default"
        self.calls: list[dict[str, Any]] = []

    async def find_neurons(self, **kwargs: Any) -> list[_FakeNeuron]:
        self.calls.append(kwargs)
        offset = int(kwargs.get("offset", 0))
        limit = int(kwargs.get("limit", 100))
        if offset >= 10:
            return []
        return [_FakeNeuron(f"n{offset + i}") for i in range(min(10 - offset, limit))]


@pytest.mark.asyncio
async def test_lifecycle_never_fetches_embeddings() -> None:
    """The lifecycle pass reads metadata only — pulling vectors killed the transfer."""
    from surreal_memory.engine.consolidation import ConsolidationEngine

    storage = _RecordingStorage()
    engine = ConsolidationEngine.__new__(ConsolidationEngine)
    engine._storage = storage  # type: ignore[attr-defined]
    engine._config = ConsolidationConfig()  # type: ignore[attr-defined]

    report = type("R", (), {"extra": {}})()
    try:
        await engine._lifecycle(report, None, True)  # type: ignore[attr-defined]
    except Exception:
        # Later stages of the pass need more of the engine than this stub provides;
        # the fetch behaviour is what this test pins down.
        pass

    assert storage.calls, "_lifecycle did not fetch neurons at all"
    for call in storage.calls:
        assert call.get("include_embedding") is False, (
            f"lifecycle fetched embeddings ({call}) — that is the huge response "
            "SurrealDB resets mid-transfer"
        )
        assert int(call.get("limit", 0)) <= 1000, (
            f"lifecycle fetched {call.get('limit')} rows in one response; page it instead"
        )


def _prime_connection_state(store: Any) -> None:
    """Give a ``__new__``-built store the connection bookkeeping ``_query`` needs.

    ``_query`` is single-flight: it reads a generation counter, re-mints the
    re-auth lock when the event loop changes, and compares the connection's loop
    against the running one. A store built with ``__new__`` has none of that, so
    every test that drives ``_query`` directly seeds it here rather than
    repeating four assignments.
    """
    loop = asyncio.get_running_loop()
    store._reauth_lock = asyncio.Lock()
    store._reauth_lock_loop = loop
    store._conn_generation = 0
    store._conn_loop = loop


class _FlakyConn:
    """Fails `attempts_before_success` times with a connection reset, then works."""

    def __init__(self, attempts_before_success: int) -> None:
        self.remaining = attempts_before_success
        self.query_calls = 0

    async def query(self, _sql: str, _params: Any) -> list[Any]:
        self.query_calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ConnectionResetError(104, "Connection reset by peer")
        return [[{"ok": True}]]


@pytest.mark.asyncio
async def test_query_retries_reconnect_after_connection_reset(monkeypatch: Any) -> None:
    """One transient reset must not abort the caller (it killed consolidation)."""
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage.__new__(SurrealDBStorage)
    _prime_connection_state(store)
    conn = _FlakyConn(attempts_before_success=2)
    store._conn = conn  # type: ignore[attr-defined]

    reconnects = {"n": 0}

    async def fake_reconnect() -> None:
        reconnects["n"] += 1

    store._reconnect = fake_reconnect  # type: ignore[assignment,method-assign]
    store._ensure_conn = lambda: conn  # type: ignore[assignment,method-assign]

    # Keep the test fast: the production backoff sleeps between attempts.
    async def _no_sleep(_d: float) -> None:
        return None

    monkeypatch.setattr(
        "surreal_memory.storage.surrealdb.store.asyncio.sleep", _no_sleep, raising=True
    )

    rows = await store._query("SELECT 1")

    assert rows == [{"ok": True}]
    assert reconnects["n"] >= 2, (
        f"only {reconnects['n']} reconnect attempt(s) — a single retry hits the same reset"
    )


class _StubConn:
    """Stands in for an SDK connection: answers, or refuses like a dead socket."""

    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.closed = False
        self.queries = 0

    async def signin(self, _creds: dict[str, str]) -> None:
        # Yield like a real round-trip would, so concurrent callers actually
        # interleave — without this the whole gather runs one caller to
        # completion before the next starts and proves nothing about fan-out.
        await asyncio.sleep(0)

    async def use(self, _ns: str, _db: str) -> None:
        await asyncio.sleep(0)

    async def query(self, _sql: str, _params: Any) -> list[Any]:
        await asyncio.sleep(0)
        self.queries += 1
        if not self.alive:
            raise ConnectionResetError(104, "Connection reset by peer")
        return [[{"ok": True}]]

    async def close(self) -> None:
        self.closed = True


class _ConnFactory:
    """Replacement for ``surrealdb.AsyncSurreal``; counts how often it is called."""

    def __init__(self, *, alive: bool = True) -> None:
        self.alive = alive
        self.calls = 0
        self.built: list[_StubConn] = []

    def __call__(self, _url: str) -> _StubConn:
        self.calls += 1
        conn = _StubConn(alive=self.alive)
        self.built.append(conn)
        return conn


def _store_with(conn: Any) -> Any:
    """A storage object wired to *conn*, with the real _query/_reconnect intact."""
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage.__new__(SurrealDBStorage)
    _prime_connection_state(store)
    store._conn = conn
    store._url = "ws://localhost:8001/rpc"
    store._user = "root"
    store._password = "secret"  # noqa: S105 - test double, never a real credential
    store._namespace = "ns"
    store._database = "db"
    return store


class TestSingleFlightReconnect:
    """One dead transport must cost ONE reconnect, not one per concurrent caller.

    `_reauth_lock` serialised the callers but did not deduplicate them: each of
    the `_BATCH_FETCH_CONCURRENCY` queries fanned out over the shared connection
    built its own replacement and logged its own warning, so a single drop during
    consolidation produced a wall of identical
    "reconnect attempt N/3 failed: [Errno 104] Connection reset by peer" lines
    and leaked a socket per attempt.
    """

    @pytest.mark.asyncio
    async def test_sixteen_concurrent_failures_reconnect_once(self, monkeypatch: Any) -> None:
        from surreal_memory.storage.surrealdb.store import _BATCH_FETCH_CONCURRENCY

        dead = _StubConn(alive=False)
        store = _store_with(dead)
        factory = _ConnFactory(alive=True)
        monkeypatch.setattr("surrealdb.AsyncSurreal", factory, raising=True)

        results = await asyncio.gather(
            *(store._query("SELECT 1") for _ in range(_BATCH_FETCH_CONCURRENCY))
        )

        assert results == [[{"ok": True}]] * _BATCH_FETCH_CONCURRENCY
        assert factory.calls == 1, (
            f"{_BATCH_FETCH_CONCURRENCY} concurrent failures caused {factory.calls} reconnects; "
            "the generation counter should collapse them to one"
        )
        assert store._conn_generation == 1

    @pytest.mark.asyncio
    async def test_reconnect_closes_the_connection_it_replaces(self, monkeypatch: Any) -> None:
        dead = _StubConn(alive=False)
        store = _store_with(dead)
        monkeypatch.setattr("surrealdb.AsyncSurreal", _ConnFactory(alive=True), raising=True)

        await store._query("SELECT 1")

        assert dead.closed is True, (
            "the replaced connection was left open — that leaks a socket (and, on the "
            "WebSocket transport, its receive task) on every reconnect"
        )

    @pytest.mark.asyncio
    async def test_waiters_do_not_each_log_a_warning(self, monkeypatch: Any, caplog: Any) -> None:
        """Every attempt logs at most one warning, whatever the fan-out."""
        import logging as _logging

        from surreal_memory.storage.surrealdb.store import (
            _BATCH_FETCH_CONCURRENCY,
            _RECONNECT_BACKOFF,
        )

        store = _store_with(_StubConn(alive=False))
        # Every replacement is dead too, so all three attempts run and log.
        monkeypatch.setattr("surrealdb.AsyncSurreal", _ConnFactory(alive=False), raising=True)
        # Drop the real backoff waits. Patching asyncio.sleep itself would also
        # neuter the stubs' yields, which is exactly what this test needs.
        monkeypatch.setattr(
            "surreal_memory.storage.surrealdb.store._RECONNECT_BACKOFF",
            (0.0,) * len(_RECONNECT_BACKOFF),
            raising=True,
        )

        with caplog.at_level(_logging.WARNING, logger="surreal_memory.storage.surrealdb.store"):
            outcomes = await asyncio.gather(
                *(store._query("SELECT 1") for _ in range(_BATCH_FETCH_CONCURRENCY)),
                return_exceptions=True,
            )

        assert all(isinstance(o, ConnectionResetError) for o in outcomes)
        warnings = [r for r in caplog.records if "reconnect attempt" in r.getMessage()]
        assert len(warnings) == len(_RECONNECT_BACKOFF), (
            f"{len(warnings)} warnings for {_BATCH_FETCH_CONCURRENCY} callers — expected one per "
            f"attempt ({len(_RECONNECT_BACKOFF)}), not one per caller per attempt"
        )

    @pytest.mark.asyncio
    async def test_bad_credentials_still_fail_fast(self, monkeypatch: Any) -> None:
        """A wrong password must not be retried three times on every caller."""
        from surrealdb.errors import NotAllowedError

        from surreal_memory.storage.surrealdb.connection import StorageAuthError

        class _RefusingConn(_StubConn):
            async def signin(self, _creds: dict[str, str]) -> None:
                await asyncio.sleep(0)
                raise NotAllowedError("auth", "There was a problem with authentication")

        def _factory(_url: str) -> _RefusingConn:
            _factory.calls += 1  # type: ignore[attr-defined]
            return _RefusingConn(alive=True)

        _factory.calls = 0  # type: ignore[attr-defined]
        store = _store_with(_StubConn(alive=False))
        monkeypatch.setattr("surrealdb.AsyncSurreal", _factory, raising=True)

        with pytest.raises(StorageAuthError):
            await store._query("SELECT 1")

        assert _factory.calls == 1, (  # type: ignore[attr-defined]
            "StorageAuthError was retried — bad credentials never fix themselves"
        )


class TestConnectionLoopAffinity:
    """A cached connection belongs to the loop that opened it.

    The SDK creates its response futures on that loop, so reusing the process-wide
    storage singleton from a second ``asyncio.run()`` made every query raise
    ``RuntimeError: … Future … attached to a different loop``. That is neither an
    auth error nor a transport error, so the retry path never caught it and the
    connection stayed broken until the process restarted.
    """

    @pytest.mark.asyncio
    async def test_query_rebuilds_when_the_loop_changed(self, monkeypatch: Any) -> None:
        foreign_loop = asyncio.new_event_loop()
        try:
            healthy_but_foreign = _StubConn(alive=True)
            store = _store_with(healthy_but_foreign)
            store._conn_loop = foreign_loop
            factory = _ConnFactory(alive=True)
            monkeypatch.setattr("surrealdb.AsyncSurreal", factory, raising=True)

            rows = await store._query("SELECT 1")

            assert rows == [{"ok": True}]
            assert factory.calls == 1, "the foreign-loop connection was reused instead of rebuilt"
            assert healthy_but_foreign.queries == 0, (
                "the query ran on the connection owned by the other loop"
            )
            assert store._conn_loop is asyncio.get_running_loop()
        finally:
            foreign_loop.close()

    @pytest.mark.asyncio
    async def test_foreign_connection_is_dropped_not_closed(self, monkeypatch: Any) -> None:
        """``close()`` would await on the other loop's objects — only drop the reference."""
        foreign_loop = asyncio.new_event_loop()
        try:
            healthy_but_foreign = _StubConn(alive=True)
            store = _store_with(healthy_but_foreign)
            store._conn_loop = foreign_loop
            monkeypatch.setattr("surrealdb.AsyncSurreal", _ConnFactory(alive=True), raising=True)

            await store._query("SELECT 1")

            assert healthy_but_foreign.closed is False
        finally:
            foreign_loop.close()

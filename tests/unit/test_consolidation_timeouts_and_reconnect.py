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
    store._reauth_lock = asyncio.Lock()  # type: ignore[attr-defined]
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

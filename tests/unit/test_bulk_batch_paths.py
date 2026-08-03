"""Unit tests for the doc-trainer bulk-path batch optimizations.

Covers the surgical batching wins that cut per-chunk DB round-trips:
- ``increment_keyword_df`` batch UPSERT (was an N+1 per-keyword SELECT+merge).
- ``SurrealDBStorage.add_synapses_batch`` multi-statement INSERT RELATION.
- ``_persist_synapses`` helper (batch with per-synapse fallback).
- ``_skip_change_log`` flag on ``_record_change_internal``.

These are mock/fake based — no live SurrealDB required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.pipeline_steps import _persist_synapses
from surreal_memory.storage.surrealdb.keyword_entity import (
    SurrealDBKeywordEntityMixin,
)
from surreal_memory.utils.timeutils import utcnow

# ---------------------------- keyword DF batch ----------------------------


class _FakeKeywordStore(SurrealDBKeywordEntityMixin):
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.brain = "brain:default"

    def _ensure_conn(self) -> Any:
        return None

    def _get_brain_id(self) -> str:
        return self.brain

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        return []


def test_increment_keyword_df_is_single_round_trip():
    """The N+1 (per-keyword SELECT+merge) must be gone: one _query for N keywords."""
    store = _FakeKeywordStore()
    asyncio.run(store.increment_keyword_df(["react", "vue", "angular", "svelte"]))
    assert len(store.queries) == 1, f"expected 1 _query call, got {len(store.queries)}"
    sql, params = store.queries[0]
    # one UPSERT statement per unique keyword, joined by ;
    assert sql.count("UPSERT") == 4
    assert "fiber_count = (fiber_count ?? 0) + 1" in sql


def test_increment_keyword_df_dedups_and_noops_empty():
    store = _FakeKeywordStore()
    asyncio.run(store.increment_keyword_df(["react", "react", "react"]))
    assert len(store.queries) == 1
    assert store.queries[0][0].count("UPSERT") == 1  # deduped

    store2 = _FakeKeywordStore()
    asyncio.run(store2.increment_keyword_df([]))
    assert store2.queries == []


# ---------------------------- synapse batch ----------------------------


@pytest.fixture(autouse=True)
def _surrealdb_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the fake SurrealDB connection env used by ``_make_storage_spy``.

    Was ``os.environ.setdefault(...)``, which — unlike ``monkeypatch.setenv``
    — mutates the real process environment and never reverts. That leaked a
    dead ``SURREALDB_URL``/placeholder password for the rest of the pytest
    session, which a later unpatched ``setup_mcp_claude_desktop()`` call could
    read via ``SurrealSettings.from_env()`` and write into a real MCP config
    file (#110).
    """
    monkeypatch.setenv("SURREALDB_URL", "http://127.0.0.1:65535")
    monkeypatch.setenv("SURREALDB_USER", "root")
    monkeypatch.setenv("SURREALDB_PASS", "x")
    monkeypatch.setenv("SURREALDB_NS", "ns")
    monkeypatch.setenv("SURREALDB_DB", "db")


def _make_storage_spy() -> Any:
    """A minimal stand-in exposing the methods add_synapses_batch touches."""
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    # construct without connecting (no network); we only exercise add_synapses_batch
    s = SurrealDBStorage()
    s._current_brain_id = "default"  # brain_id field value (alnum+_, not record-id)
    s._skip_change_log = False
    s._query = AsyncMock(return_value=[])
    s._record_change_internal = AsyncMock(return_value=None)
    s._record_changes_bulk = AsyncMock(return_value=None)
    s._conn = AsyncMock()  # backs _ensure_conn() for add_neurons_batch/add_fibers_batch
    return s


def _syn(n: int) -> Synapse:
    return Synapse.create(
        source_id=f"neuron:s{n}",
        target_id=f"neuron:t{n}",
        type=SynapseType.CONTAINS,
    )


def test_add_synapses_batch_single_query_multi_statement():
    s = _make_storage_spy()
    syns = [_syn(i) for i in range(5)]
    added = asyncio.run(s.add_synapses_batch(syns, record_change=False))
    assert added == 5
    assert s._query.await_count == 1, "batch must be ONE _query round-trip"
    sql = s._query.await_args.args[0]
    assert sql.count("INSERT RELATION INTO synapse") == 5
    assert sql.rstrip().endswith(";")


def test_add_synapses_batch_record_change_default_logs_via_bulk():
    """record_change=True must log via ONE _record_changes_bulk call, not N
    _record_change_internal calls — the per-synapse change-log loop was
    ~64% of a batched update's wall clock (see _record_changes_bulk's
    docstring); a "batch" writer reintroducing that loop defeats its own name.
    """
    s = _make_storage_spy()
    syns = [_syn(0), _syn(1)]
    asyncio.run(s.add_synapses_batch(syns))  # record_change defaults True
    assert s._record_change_internal.await_count == 0
    assert s._record_changes_bulk.await_count == 1
    args = s._record_changes_bulk.await_args.args
    assert args[0] == "synapse"
    assert args[1] == "insert"
    assert list(args[2]) == syns


def test_add_synapses_batch_record_change_false_skips_log():
    s = _make_storage_spy()
    asyncio.run(s.add_synapses_batch([_syn(0)], record_change=False))
    assert s._record_changes_bulk.await_count == 0


def test_add_synapses_batch_empty_noop():
    s = _make_storage_spy()
    assert asyncio.run(s.add_synapses_batch([])) == 0
    assert s._query.await_count == 0


# ---------------------------- neuron batch ----------------------------


def _neuron(n: int) -> Neuron:
    return Neuron(id=f"n{n}", type=NeuronType.CONCEPT, content=f"content-{n}", created_at=utcnow())


def test_add_neurons_batch_single_insert_call_per_table():
    """N neurons must cost ONE conn.insert("neuron", [...]) and ONE
    conn.insert("neuron_state", [...]) — not 3*N round-trips."""
    s = _make_storage_spy()
    neurons = [_neuron(i) for i in range(4)]
    added = asyncio.run(s.add_neurons_batch(neurons))
    assert added == 4
    assert s._conn.insert.await_count == 2  # one for "neuron", one for "neuron_state"
    tables_called = {call.args[0] for call in s._conn.insert.await_args_list}
    assert tables_called == {"neuron", "neuron_state"}
    neuron_call = next(c for c in s._conn.insert.await_args_list if c.args[0] == "neuron")
    assert len(neuron_call.args[1]) == 4


def test_add_neurons_batch_logs_via_bulk_change_log():
    s = _make_storage_spy()
    neurons = [_neuron(0), _neuron(1)]
    asyncio.run(s.add_neurons_batch(neurons))
    assert s._record_changes_bulk.await_count == 1
    args = s._record_changes_bulk.await_args.args
    assert args[0] == "neuron"
    assert args[1] == "insert"
    assert list(args[2]) == neurons


def test_add_neurons_batch_record_change_false_skips_log():
    s = _make_storage_spy()
    asyncio.run(s.add_neurons_batch([_neuron(0)], record_change=False))
    assert s._record_changes_bulk.await_count == 0


def test_add_neurons_batch_empty_noop():
    s = _make_storage_spy()
    assert asyncio.run(s.add_neurons_batch([])) == 0
    assert s._conn.insert.await_count == 0


# ---------------------------- fiber batch ----------------------------


def _fiber(n: int) -> Fiber:
    return Fiber.create(
        neuron_ids={f"n{n}"},
        synapse_ids=set(),
        anchor_neuron_id=f"n{n}",
        summary=f"fiber-{n}",
    )


def test_add_fibers_batch_single_insert_call():
    s = _make_storage_spy()
    fibers = [_fiber(i) for i in range(3)]
    added = asyncio.run(s.add_fibers_batch(fibers))
    assert added == 3
    assert s._conn.insert.await_count == 1
    call = s._conn.insert.await_args
    assert call.args[0] == "fiber"
    assert len(call.args[1]) == 3


def test_add_fibers_batch_logs_via_bulk_change_log():
    s = _make_storage_spy()
    fibers = [_fiber(0)]
    asyncio.run(s.add_fibers_batch(fibers))
    assert s._record_changes_bulk.await_count == 1
    args = s._record_changes_bulk.await_args.args
    assert args[0] == "fiber"
    assert args[1] == "insert"
    assert list(args[2]) == fibers


def test_add_fibers_batch_empty_noop():
    s = _make_storage_spy()
    assert asyncio.run(s.add_fibers_batch([])) == 0
    assert s._conn.insert.await_count == 0


# ---------------------------- base-class fallbacks ----------------------------


def test_base_add_neurons_batch_fallback_is_sequential():
    """The base default must fall back to sequential add_neuron for backends
    that do not override (keeps non-SurrealDB backends correct)."""
    from types import SimpleNamespace

    from surreal_memory.storage.base import NeuralStorage

    calls: list[str] = []
    fake = SimpleNamespace(add_neuron=AsyncMock(side_effect=lambda n: calls.append(n.id) or n.id))
    neurons = [_neuron(i) for i in range(3)]
    added = asyncio.run(NeuralStorage.add_neurons_batch(fake, neurons))
    assert added == 3
    assert len(calls) == 3


def test_base_add_fibers_batch_fallback_is_sequential():
    from types import SimpleNamespace

    from surreal_memory.storage.base import NeuralStorage

    calls: list[str] = []
    fake = SimpleNamespace(add_fiber=AsyncMock(side_effect=lambda f: calls.append(f.id) or f.id))
    fibers = [_fiber(i) for i in range(3)]
    added = asyncio.run(NeuralStorage.add_fibers_batch(fake, fibers))
    assert added == 3
    assert len(calls) == 3


# ---------------------------- _persist_synapses helper ----------------------------


class _FakeCtx:
    def __init__(self) -> None:
        self.synapses_created: list[Synapse] = []


class _FakeStorage:
    def __init__(self, *, has_batch: bool = True, batch_raises: bool = False) -> None:
        self._has_batch = has_batch
        self._batch_raises = batch_raises
        self.add_synapse = AsyncMock(side_effect=lambda s: s.id)
        self.batch_calls = 0

    async def add_synapses_batch(self, synapses, *, record_change=True):
        self.batch_calls += 1
        if self._batch_raises:
            raise RuntimeError("boom")
        return len(synapses)


def test_persist_synapses_uses_batch_path():
    storage = _FakeStorage()
    ctx = _FakeCtx()
    syns = [_syn(i) for i in range(4)]
    asyncio.run(_persist_synapses(storage, syns, ctx))
    assert storage.batch_calls == 1
    assert ctx.synapses_created == syns


def test_persist_synapses_falls_back_when_batch_raises():
    storage = _FakeStorage(batch_raises=True)
    ctx = _FakeCtx()
    syns = [_syn(i) for i in range(3)]
    asyncio.run(_persist_synapses(storage, syns, ctx))
    # batch attempted, then per-synapse fallback via gather
    assert storage.batch_calls == 1
    assert storage.add_synapse.await_count == 3
    assert len(ctx.synapses_created) == 3


def test_persist_synapses_empty_is_noop():
    storage = _FakeStorage()
    ctx = _FakeCtx()
    asyncio.run(_persist_synapses(storage, [], ctx))
    assert storage.batch_calls == 0
    assert ctx.synapses_created == []


# ---------------------------- skip_change_log flag ----------------------------


def test_skip_change_log_flag_makes_record_noop():
    s = _make_storage_spy()
    # real _record_change_internal writes via conn.insert; with the flag set it
    # must short-circuit before touching the connection (which is None here).
    s._skip_change_log = True
    n = Neuron(id="neuron:x", type=NeuronType.CONCEPT, content="x", created_at=utcnow())
    # must not raise despite conn being None
    asyncio.run(s._record_change_internal("neuron", n.id, "insert", n))


def test_base_add_synapses_batch_fallback_is_sequential():
    """The base default must fall back to sequential add_synapse for backends
    that do not override (keeps non-SurrealDB backends correct)."""
    from types import SimpleNamespace

    from surreal_memory.storage.base import NeuralStorage

    calls = []
    fake = SimpleNamespace(add_synapse=AsyncMock(side_effect=lambda s: calls.append(s.id) or s.id))
    syns = [_syn(i) for i in range(3)]
    # invoke the unbound default implementation directly on the bare fake
    added = asyncio.run(NeuralStorage.add_synapses_batch(fake, syns))
    assert added == 3
    assert len(calls) == 3

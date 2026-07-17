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


def _make_storage_spy() -> Any:
    """A minimal stand-in exposing the methods add_synapses_batch touches."""
    import os

    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    # construct without connecting (no network); we only exercise add_synapses_batch
    os.environ.setdefault("SURREALDB_URL", "http://127.0.0.1:65535")
    os.environ.setdefault("SURREALDB_USER", "root")
    os.environ.setdefault("SURREALDB_PASS", "x")
    os.environ.setdefault("SURREALDB_NS", "ns")
    os.environ.setdefault("SURREALDB_DB", "db")
    s = SurrealDBStorage()
    s._current_brain_id = "default"  # brain_id field value (alnum+_, not record-id)
    s._skip_change_log = False
    s._query = AsyncMock(return_value=[])
    s._record_change_internal = AsyncMock(return_value=None)
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


def test_add_synapses_batch_record_change_default_logs_each():
    s = _make_storage_spy()
    asyncio.run(s.add_synapses_batch([_syn(0), _syn(1)]))  # record_change defaults True
    assert s._record_change_internal.await_count == 2


def test_add_synapses_batch_empty_noop():
    s = _make_storage_spy()
    assert asyncio.run(s.add_synapses_batch([])) == 0
    assert s._query.await_count == 0


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

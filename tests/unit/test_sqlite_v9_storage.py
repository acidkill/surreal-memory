"""SQLite serialization tests for schema-v9 additions (U1).

Exercises the real SQLite backend (fresh v39 bootstrap schema) for TypedMemory
validity round-trip, Source.trust round-trip, and retrieval-trace CRUD/find/prune.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.core.source import Source
from surreal_memory.storage.sqlite_store import SQLiteStorage
from surreal_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage() -> SQLiteStorage:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStorage(Path(tmpdir) / "test.db")
        await store.initialize()
        brain = Brain.create(name="test_brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        yield store
        await store.close()


async def _add_fiber(store: SQLiteStorage, hint: str = "n") -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"content-{hint}")
    await store.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=f"fiber-{hint}",
    )
    await store.add_fiber(fiber)
    return fiber


class TestSQLiteValidityRoundTrip:
    async def test_validity_survives_round_trip(self, storage: SQLiteStorage) -> None:
        fiber = await _add_fiber(storage, "1")
        now = utcnow()
        tm = TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT).with_validity(
            valid_from=now - timedelta(days=1), valid_until=now, superseded_by="fZ"
        )
        await storage.add_typed_memory(tm)
        got = await storage.get_typed_memory(fiber.id)
        assert got is not None
        assert got.superseded_by == "fZ"
        assert got.is_superseded is True
        assert got.valid_until is not None
        # isoformat round-trip preserves the instant
        assert abs((got.valid_until - now).total_seconds()) < 1

    async def test_defaults_none_when_unset(self, storage: SQLiteStorage) -> None:
        fiber = await _add_fiber(storage, "2")
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT)
        )
        got = await storage.get_typed_memory(fiber.id)
        assert got is not None
        assert got.valid_from is None
        assert got.valid_until is None
        assert got.superseded_by is None


class TestSQLiteSourceTrust:
    async def test_trust_round_trip(self, storage: SQLiteStorage) -> None:
        source = Source.create(brain_id=storage._get_brain_id(), name="doc.pdf", trust=0.8)
        await storage.add_source(source)
        got = await storage.get_source(source.id)
        assert got is not None
        assert got.trust == 0.8

    async def test_trust_none_round_trip(self, storage: SQLiteStorage) -> None:
        source = Source.create(brain_id=storage._get_brain_id(), name="doc2.pdf")
        await storage.add_source(source)
        got = await storage.get_source(source.id)
        assert got is not None
        assert got.trust is None


class TestSQLiteRetrievalTraces:
    async def test_add_get_round_trip(self, storage: SQLiteStorage) -> None:
        trace = RetrievalTrace(
            brain_id=storage._get_brain_id(),
            query="where is Emma",
            depth_used=2,
            mode="deep",
            confidence=0.7,
            latency_ms=11.0,
            fiber_ids=("f1", "f2"),
            anchor_ids=("a1",),
            retrievers=("bm25",),
            fiber_scores=(0.9, 0.4),
            filters={"tags": ["x"]},
            config_snapshot={"trust_weight": 0.0},
        )
        tid = await storage.add_retrieval_trace(trace)
        got = await storage.get_retrieval_trace(tid)
        assert got is not None
        assert got.query == "where is Emma"
        assert got.fiber_ids == ("f1", "f2")
        assert got.fiber_scores == (0.9, 0.4)
        assert got.filters == {"tags": ["x"]}
        assert got.config_snapshot == {"trust_weight": 0.0}

    async def test_find_by_fiber_and_query(self, storage: SQLiteStorage) -> None:
        bid = storage._get_brain_id()
        await storage.add_retrieval_trace(
            RetrievalTrace(brain_id=bid, query="alpha", fiber_ids=("fX",))
        )
        await storage.add_retrieval_trace(
            RetrievalTrace(brain_id=bid, query="beta", fiber_ids=("fY",))
        )
        by_fiber = await storage.find_retrieval_traces(fiber_id="fX")
        assert [t.query for t in by_fiber] == ["alpha"]
        by_query = await storage.find_retrieval_traces(query_contains="BET")
        assert [t.query for t in by_query] == ["beta"]

    async def test_prune_retention_and_max(self, storage: SQLiteStorage) -> None:
        bid = storage._get_brain_id()
        await storage.add_retrieval_trace(
            RetrievalTrace(brain_id=bid, query="old", created_at=utcnow() - timedelta(days=40))
        )
        await storage.add_retrieval_trace(RetrievalTrace(brain_id=bid, query="fresh"))
        removed = await storage.prune_retrieval_traces(retention_days=30)
        assert removed == 1
        remaining = await storage.find_retrieval_traces()
        assert [t.query for t in remaining] == ["fresh"]


class TestSQLiteBatchAndExpiring:
    async def test_batch(self, storage: SQLiteStorage) -> None:
        f1 = await _add_fiber(storage, "1")
        f2 = await _add_fiber(storage, "2")
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f1.id, memory_type=MemoryType.FACT)
        )
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f2.id, memory_type=MemoryType.FACT)
        )
        batch = await storage.get_typed_memories_batch([f1.id, f2.id, "missing"])
        assert set(batch.keys()) == {f1.id, f2.id}

    async def test_expiring(self, storage: SQLiteStorage) -> None:
        f1 = await _add_fiber(storage, "1")
        f2 = await _add_fiber(storage, "2")
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f1.id, memory_type=MemoryType.FACT, expires_in_days=3)
        )
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f2.id, memory_type=MemoryType.FACT, expires_in_days=30)
        )
        expiring = await storage.get_expiring_memories(within_days=7)
        assert [tm.fiber_id for tm in expiring] == [f1.id]

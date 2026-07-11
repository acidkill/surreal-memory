"""Unit tests for schema-v9 storage additions on InMemoryStorage (U1).

Covers retrieval-trace CRUD/find/prune, get_typed_memories_batch,
get_expiring_memories (brain-wide), and TypedMemory validity round-trip.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="test_brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


async def _add_fiber(store: InMemoryStorage, hint: str = "n") -> Fiber:
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


class TestRetrievalTraceStorage:
    async def test_add_and_get(self, storage: InMemoryStorage) -> None:
        trace = RetrievalTrace(brain_id="b", query="hello", fiber_ids=("f1",))
        tid = await storage.add_retrieval_trace(trace)
        assert tid == trace.id
        got = await storage.get_retrieval_trace(tid)
        assert got is not None
        assert got.query == "hello"

    async def test_get_missing_returns_none(self, storage: InMemoryStorage) -> None:
        assert await storage.get_retrieval_trace("nope") is None

    async def test_find_by_fiber_id(self, storage: InMemoryStorage) -> None:
        await storage.add_retrieval_trace(RetrievalTrace(query="a", fiber_ids=("fX",)))
        await storage.add_retrieval_trace(RetrievalTrace(query="b", fiber_ids=("fY",)))
        found = await storage.find_retrieval_traces(fiber_id="fX")
        assert len(found) == 1
        assert found[0].query == "a"

    async def test_find_by_query_contains(self, storage: InMemoryStorage) -> None:
        await storage.add_retrieval_trace(RetrievalTrace(query="where is Emma"))
        await storage.add_retrieval_trace(RetrievalTrace(query="what is Python"))
        found = await storage.find_retrieval_traces(query_contains="emma")
        assert len(found) == 1

    async def test_find_since_and_newest_first(self, storage: InMemoryStorage) -> None:
        old = RetrievalTrace(query="old", created_at=utcnow() - timedelta(days=5))
        new = RetrievalTrace(query="new", created_at=utcnow())
        await storage.add_retrieval_trace(old)
        await storage.add_retrieval_trace(new)
        since = utcnow() - timedelta(days=1)
        found = await storage.find_retrieval_traces(since=since)
        assert [t.query for t in found] == ["new"]
        limited = await storage.find_retrieval_traces(limit=1)
        assert limited[0].query == "new"  # newest first

    async def test_prune_by_retention(self, storage: InMemoryStorage) -> None:
        await storage.add_retrieval_trace(
            RetrievalTrace(query="old", created_at=utcnow() - timedelta(days=40))
        )
        await storage.add_retrieval_trace(RetrievalTrace(query="fresh"))
        removed = await storage.prune_retrieval_traces(retention_days=30)
        assert removed == 1
        remaining = await storage.find_retrieval_traces()
        assert [t.query for t in remaining] == ["fresh"]

    async def test_prune_by_max_traces_keeps_newest(self, storage: InMemoryStorage) -> None:
        for i in range(5):
            await storage.add_retrieval_trace(
                RetrievalTrace(query=f"q{i}", created_at=utcnow() - timedelta(minutes=5 - i))
            )
        removed = await storage.prune_retrieval_traces(max_traces=2)
        assert removed == 3
        remaining = await storage.find_retrieval_traces()
        assert len(remaining) == 2


class TestTypedMemoryBatchAndExpiring:
    async def test_get_typed_memories_batch(self, storage: InMemoryStorage) -> None:
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

    async def test_get_expiring_memories(self, storage: InMemoryStorage) -> None:
        f1 = await _add_fiber(storage, "1")
        f2 = await _add_fiber(storage, "2")
        f3 = await _add_fiber(storage, "3")
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f1.id, memory_type=MemoryType.FACT, expires_in_days=3)
        )
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f2.id, memory_type=MemoryType.FACT, expires_in_days=30)
        )
        await storage.add_typed_memory(
            TypedMemory.create(fiber_id=f3.id, memory_type=MemoryType.FACT)
        )
        expiring = await storage.get_expiring_memories(within_days=7)
        assert [tm.fiber_id for tm in expiring] == [f1.id]

    async def test_get_expiring_respects_limit(self, storage: InMemoryStorage) -> None:
        for i in range(3):
            f = await _add_fiber(storage, str(i))
            await storage.add_typed_memory(
                TypedMemory.create(
                    fiber_id=f.id, memory_type=MemoryType.FACT, expires_in_days=i + 1
                )
            )
        expiring = await storage.get_expiring_memories(within_days=30, limit=2)
        assert len(expiring) == 2


class TestValidityRoundTrip:
    async def test_validity_survives_inmemory_round_trip(self, storage: InMemoryStorage) -> None:
        f1 = await _add_fiber(storage, "1")
        now = utcnow()
        tm = TypedMemory.create(fiber_id=f1.id, memory_type=MemoryType.FACT).with_validity(
            valid_from=now - timedelta(days=1), valid_until=now, superseded_by="fZ"
        )
        await storage.add_typed_memory(tm)
        got = await storage.get_typed_memory(f1.id)
        assert got is not None
        assert got.valid_until == now
        assert got.superseded_by == "fZ"
        assert got.is_superseded is True

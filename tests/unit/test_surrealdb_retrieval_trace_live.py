"""Live-DB round-trip tests for the SurrealDB schema-v9 storage (U1).

Skipped unless SURREALDB_URL points at a running SurrealDB. Verifies the SurrealDB
backend round-trips RetrievalTrace (including the un-sanitised `.id` — regression
for U1 Bug A) and TypedMemory validity fields against the real engine (no mocks).
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.utils.timeutils import utcnow
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name="u1-retrieval-trace-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    # Best-effort cleanup of this throwaway brain's traces on the shared DB.
    try:
        await store.prune_retrieval_traces(max_traces=0)
    except Exception:
        pass
    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await store.close()
    except Exception:
        pass


async def _make_fiber(store, hint: str = "n"):  # type: ignore[no-untyped-def]
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


class TestRetrievalTraceLive:
    async def test_id_round_trips(self, storage) -> None:  # type: ignore[no-untyped-def]
        trace = RetrievalTrace(
            brain_id=storage._get_brain_id(), query="hello world", fiber_ids=("fA",)
        )
        tid = await storage.add_retrieval_trace(trace)
        assert tid == trace.id
        got = await storage.get_retrieval_trace(tid)
        assert got is not None
        # U1 Bug A regression: the original dashed uuid must survive, not the
        # dash-folded record-id form.
        assert got.id == trace.id
        assert got.query == "hello world"
        assert got.fiber_ids == ("fA",)

    async def test_find_by_fiber_and_query(self, storage) -> None:  # type: ignore[no-untyped-def]
        t1 = RetrievalTrace(
            brain_id=storage._get_brain_id(), query="where is Emma", fiber_ids=("fEmma",)
        )
        await storage.add_retrieval_trace(t1)
        by_fiber = await storage.find_retrieval_traces(fiber_id="fEmma")
        assert any(t.id == t1.id for t in by_fiber)
        by_query = await storage.find_retrieval_traces(query_contains="emma")
        assert any(t.id == t1.id for t in by_query)


class TestTypedMemoryValidityLive:
    async def test_validity_round_trips(self, storage) -> None:  # type: ignore[no-untyped-def]
        fiber = await _make_fiber(storage, "v")
        now = utcnow()
        tm = TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT).with_validity(
            valid_until=now, superseded_by="fZ"
        )
        await storage.add_typed_memory(tm)
        got = await storage.get_typed_memory(fiber.id)
        assert got is not None
        assert got.superseded_by == "fZ"
        assert got.is_superseded is True
        assert got.valid_until is not None

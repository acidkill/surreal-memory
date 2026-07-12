"""U4: smem_provenance traces/trace_get query surface over retrieval traces."""

from __future__ import annotations

import pytest

from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.mcp.provenance_handler import ProvenanceHandler
from surreal_memory.storage.memory_store import InMemoryStorage


class _FakeServer(ProvenanceHandler):
    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


@pytest.fixture
async def seeded() -> tuple[_FakeServer, RetrievalTrace, RetrievalTrace]:
    storage = InMemoryStorage()
    storage.set_brain("b")
    t1 = RetrievalTrace(
        brain_id="b",
        query="where does emma live",
        mode="associative",
        confidence=0.9,
        fiber_ids=("f-oslo", "f-bergen"),
    )
    t2 = RetrievalTrace(
        brain_id="b",
        query="what is the capital",
        mode="exact",
        confidence=0.7,
        fiber_ids=("f-paris",),
    )
    await storage.add_retrieval_trace(t1)
    await storage.add_retrieval_trace(t2)
    return _FakeServer(storage), t1, t2


class TestProvenanceTraces:
    async def test_traces_by_fiber_id(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, t1, _ = seeded
        res = await server._provenance({"action": "traces", "fiber_id": "f-oslo"})
        assert res["count"] == 1
        assert res["traces"][0]["id"] == t1.id
        assert res["traces"][0]["query"] == "where does emma live"
        assert res["traces"][0]["fiber_ids"] == ["f-oslo", "f-bergen"]

    async def test_traces_by_query_contains(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, _, t2 = seeded
        res = await server._provenance({"action": "traces", "query_contains": "capital"})
        assert res["count"] == 1
        assert res["traces"][0]["id"] == t2.id

    async def test_traces_no_filter_returns_all(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, _, _ = seeded
        res = await server._provenance({"action": "traces"})
        assert res["count"] == 2

    async def test_traces_no_neuron_id_required(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, _, _ = seeded
        # traces must NOT demand neuron_id (regression: dispatch order).
        res = await server._provenance({"action": "traces", "fiber_id": "nope"})
        assert "error" not in res
        assert res["count"] == 0

    async def test_traces_invalid_since(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, _, _ = seeded
        res = await server._provenance({"action": "traces", "since": "not-a-date"})
        assert "error" in res

    async def test_trace_get_by_id(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, t1, _ = seeded
        res = await server._provenance({"action": "trace_get", "trace_id": t1.id})
        assert res["trace"]["id"] == t1.id
        assert res["trace"]["query"] == "where does emma live"

    async def test_trace_get_missing_id(
        self, seeded: tuple[_FakeServer, RetrievalTrace, RetrievalTrace]
    ) -> None:
        server, _, _ = seeded
        assert "error" in await server._provenance({"action": "trace_get"})
        assert "error" in await server._provenance({"action": "trace_get", "trace_id": "nope"})

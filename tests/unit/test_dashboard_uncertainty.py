"""U6: GET /api/dashboard/uncertainty route (TTL-cached brain-wide overview)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from surreal_memory.core.synapse import SynapseType
from surreal_memory.server.routes.dashboard_api import _UNCERTAINTY_CACHE, get_storage, router


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _UNCERTAINTY_CACHE._store.clear()


def _storage(*, brain_id: str = "b", contradictions: int = 0, total: int = 0) -> Any:
    syns = [
        SimpleNamespace(type=SynapseType.CONTRADICTS, metadata={}) for _ in range(contradictions)
    ]
    ns = SimpleNamespace()
    ns.current_brain_id = brain_id
    ns.brain_id = brain_id
    ns.get_synapses = AsyncMock(return_value=syns)
    ns.get_expiring_memory_count = AsyncMock(return_value=0)
    ns.find_typed_memories = AsyncMock(return_value=[])
    ns.count_typed_memories = AsyncMock(return_value=total)
    # No get_drift_clusters → SurrealDB-like (drift degrades to 0).
    return ns


def _client(storage: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage] = lambda: storage
    return TestClient(app)


class TestUncertaintyRoute:
    def test_returns_overview_shape(self) -> None:
        client = _client(_storage(contradictions=2, total=8))
        resp = client.get("/api/dashboard/uncertainty")
        assert resp.status_code == 200
        body = resp.json()
        assert body["level"] == "high"
        assert body["counts"]["contradictions"] == 2
        assert body["counts"]["drift_clusters"] == 0  # SurrealDB-like degradation
        assert body["contradiction_rate"] == 0.25
        assert body["total_memories"] == 8
        assert "scan" in body

    def test_within_days_validated(self) -> None:
        client = _client(_storage())
        assert client.get("/api/dashboard/uncertainty?within_days=0").status_code == 422
        assert client.get("/api/dashboard/uncertainty?within_days=400").status_code == 422

    def test_cached_second_call_does_not_rehit_storage(self) -> None:
        storage = _storage(contradictions=1, total=4)
        client = _client(storage)
        client.get("/api/dashboard/uncertainty")
        client.get("/api/dashboard/uncertainty")
        # Cached → the aggregation queries ran only once.
        assert storage.count_typed_memories.await_count == 1

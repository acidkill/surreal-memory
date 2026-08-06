"""Tests for dashboard API routes — timeline, fibers, fiber diagram endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from surreal_memory.server.routes.dashboard_api import router


@dataclass
class FakeNeuron:
    """Minimal neuron for testing."""

    id: str
    content: str
    type: Any  # StrEnum-like with .value
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass
class FakeFiber:
    """Minimal fiber for testing."""

    id: str
    summary: str = ""
    neuron_ids: list[str] = field(default_factory=list)


@dataclass
class FakeSynapse:
    """Minimal synapse for testing."""

    id: str
    source_id: str
    target_id: str
    type: Any
    weight: float = 1.0
    direction: Any = None


class FakeType:
    """Mimics a StrEnum with .value."""

    def __init__(self, value: str) -> None:
        self.value = value


class FakeDirection:
    """Mimics SynapseDirection."""

    def __init__(self, value: str) -> None:
        self.value = value


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.get_fibers = AsyncMock(return_value=[])
    storage.get_fiber = AsyncMock(return_value=None)
    storage.get_all_synapses = AsyncMock(return_value=[])
    storage.get_synapses_for_neurons = AsyncMock(return_value={})
    storage.get_neurons_batch = AsyncMock(return_value={})
    return storage


@pytest.fixture()
def client(mock_storage: AsyncMock) -> TestClient:
    app = _make_app()
    app.dependency_overrides = {}

    from surreal_memory.server.routes.dashboard_api import get_storage

    app.dependency_overrides[get_storage] = lambda: mock_storage
    return TestClient(app)


class TestTimelineEndpoint:
    def test_empty_timeline(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []
        assert data["total"] == 0

    def test_timeline_with_neurons(self, client: TestClient, mock_storage: AsyncMock) -> None:
        neurons = [
            FakeNeuron(
                id="n1",
                content="Test memory",
                type=FakeType("concept"),
                metadata={"_created_at": "2026-02-10T10:00:00"},
            ),
            FakeNeuron(
                id="n2",
                content="Another memory",
                type=FakeType("entity"),
                metadata={"_created_at": "2026-02-11T12:00:00"},
            ),
        ]
        mock_storage.find_neurons.return_value = neurons

        resp = client.get("/api/dashboard/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["entries"]) == 2
        # Should be sorted descending by created_at
        assert data["entries"][0]["id"] == "n2"
        assert data["entries"][1]["id"] == "n1"

    def test_timeline_respects_limit(self, client: TestClient, mock_storage: AsyncMock) -> None:
        neurons = [
            FakeNeuron(
                id=f"n{i}",
                content=f"Memory {i}",
                type=FakeType("concept"),
                metadata={"_created_at": f"2026-02-{10 + i:02d}T10:00:00"},
            )
            for i in range(5)
        ]
        mock_storage.find_neurons.return_value = neurons

        resp = client.get("/api/dashboard/timeline?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entries"]) == 2

    def test_timeline_type_filter_in_response(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        neurons = [
            FakeNeuron(
                id="n1",
                content="Concept memory",
                type=FakeType("concept"),
                metadata={"_created_at": "2026-02-10T10:00:00"},
            ),
        ]
        mock_storage.find_neurons.return_value = neurons

        resp = client.get("/api/dashboard/timeline")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert entries[0]["neuron_type"] == "concept"


class TestFibersEndpoint:
    def test_empty_fibers(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/fibers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fibers"] == []

    def test_fibers_list(self, client: TestClient, mock_storage: AsyncMock) -> None:
        fibers = [
            FakeFiber(id="f1", summary="Test fiber", neuron_ids=["n1", "n2"]),
            FakeFiber(id="f2", summary="Another fiber", neuron_ids=["n3"]),
        ]
        mock_storage.get_fibers.return_value = fibers

        resp = client.get("/api/dashboard/fibers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["fibers"]) == 2
        assert data["fibers"][0]["id"] == "f1"
        assert data["fibers"][0]["neuron_count"] == 2
        assert data["fibers"][1]["neuron_count"] == 1

    def test_fibers_limit(self, client: TestClient, mock_storage: AsyncMock) -> None:
        fibers = [FakeFiber(id=f"f{i}", summary=f"Fiber {i}") for i in range(10)]
        mock_storage.get_fibers.return_value = fibers

        resp = client.get("/api/dashboard/fibers?limit=5")
        assert resp.status_code == 200
        # The endpoint passes limit to storage.get_fibers


class TestFiberDiagramEndpoint:
    def test_fiber_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/fiber/nonexistent/diagram")
        assert resp.status_code == 404

    def test_fiber_diagram_success(self, client: TestClient, mock_storage: AsyncMock) -> None:
        fiber = FakeFiber(id="f1", summary="Test", neuron_ids=["n1", "n2"])
        mock_storage.get_fiber.return_value = fiber

        n1 = FakeNeuron(id="n1", content="Node 1", type=FakeType("concept"))
        n2 = FakeNeuron(id="n2", content="Node 2", type=FakeType("entity"))
        mock_storage.get_neurons_batch.return_value = {"n1": n1, "n2": n2}

        syn = FakeSynapse(
            id="s1",
            source_id="n1",
            target_id="n2",
            type=FakeType("temporal"),
            weight=0.8,
            direction=FakeDirection("forward"),
        )
        mock_storage.get_synapses_for_neurons.return_value = {"n1": [syn], "n2": []}

        resp = client.get("/api/dashboard/fiber/f1/diagram")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fiber_id"] == "f1"
        assert len(data["neurons"]) == 2
        assert len(data["synapses"]) == 1
        assert data["synapses"][0]["source_id"] == "n1"
        assert data["synapses"][0]["target_id"] == "n2"

    def test_fiber_diagram_no_synapses(self, client: TestClient, mock_storage: AsyncMock) -> None:
        fiber = FakeFiber(id="f1", summary="Test", neuron_ids=["n1"])
        mock_storage.get_fiber.return_value = fiber

        n1 = FakeNeuron(id="n1", content="Alone", type=FakeType("concept"))
        mock_storage.get_neurons_batch.return_value = {"n1": n1}
        mock_storage.get_synapses_for_neurons.return_value = {"n1": []}

        resp = client.get("/api/dashboard/fiber/f1/diagram")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["neurons"]) == 1
        assert len(data["synapses"]) == 0

    def test_fiber_diagram_filters_external_synapses(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Synapses referencing external neurons should be excluded."""
        fiber = FakeFiber(id="f1", summary="Test", neuron_ids=["n1"])
        mock_storage.get_fiber.return_value = fiber

        n1 = FakeNeuron(id="n1", content="Inside", type=FakeType("concept"))
        mock_storage.get_neurons_batch.return_value = {"n1": n1}

        syn = FakeSynapse(
            id="s1",
            source_id="n1",
            target_id="n_external",  # not in fiber
            type=FakeType("related"),
            direction=FakeDirection("forward"),
        )
        mock_storage.get_synapses_for_neurons.return_value = {"n1": [syn]}

        resp = client.get("/api/dashboard/fiber/f1/diagram")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["synapses"]) == 0  # External synapse filtered


class TestStorageStatusEndpoint:
    """Tests for GET /api/dashboard/storage/status (SurrealDB-only)."""

    def test_returns_200_with_surrealdb_backend(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 10, "synapse_count": 3, "fiber_count": 2}
        )
        mock_storage._url = "http://localhost:8000"
        mock_storage._namespace = "surreal_memory"
        mock_storage._database = "default"
        resp = client.get("/api/dashboard/storage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "surrealdb"
        assert data["healthy"] is True
        assert data["neuron_count"] == 10
        assert data["synapse_count"] == 3
        assert data["fiber_count"] == 2

    def test_healthy_false_when_get_stats_fails(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        mock_storage.get_stats = AsyncMock(side_effect=Exception("db down"))
        resp = client.get("/api/dashboard/storage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "surrealdb"
        assert data["healthy"] is False
        assert data["neuron_count"] == 0

    def test_url_namespace_database_exposed(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 0, "synapse_count": 0, "fiber_count": 0}
        )
        mock_storage._url = "http://surrealdb:8000"
        mock_storage._namespace = "myns"
        mock_storage._database = "mydb"
        resp = client.get("/api/dashboard/storage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["url"] == "http://surrealdb:8000"
        assert data["namespace"] == "myns"
        assert data["database"] == "mydb"


class TestBrainListingUsesActiveBackend:
    """switch_brain and get_brain_files must see SurrealDB-only brains.

    Both previously enumerated brains via cfg.list_brains(), which only globs
    local *.json/*.db fixture files — so a brain that only exists in
    SurrealDB (the only production backend since v2.0.0) was invisible to
    both endpoints: switching to it 404'd, and it never appeared in the
    Settings "Brain Files" panel.
    """

    def test_switch_brain_accepts_surrealdb_only_brain(self, client: TestClient) -> None:
        cfg = MagicMock()
        cfg.switch_brain = MagicMock()
        new_storage = AsyncMock()

        with (
            patch("surreal_memory.unified_config.get_config", return_value=cfg),
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new=AsyncMock(return_value=["surrealdb-only-brain"]),
            ),
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                new=AsyncMock(return_value=new_storage),
            ),
        ):
            resp = client.post(
                "/api/dashboard/brains/switch", json={"brain_name": "surrealdb-only-brain"}
            )

        assert resp.status_code == 200, resp.text
        cfg.switch_brain.assert_called_once_with("surrealdb-only-brain")

    def test_switch_brain_404s_for_truly_missing_brain(self, client: TestClient) -> None:
        cfg = MagicMock()
        with (
            patch("surreal_memory.unified_config.get_config", return_value=cfg),
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new=AsyncMock(return_value=["some-other-brain"]),
            ),
        ):
            resp = client.post("/api/dashboard/brains/switch", json={"brain_name": "nonexistent"})

        assert resp.status_code == 404

    def test_brain_files_lists_surrealdb_only_brain_with_zero_size(
        self, client: TestClient
    ) -> None:
        """A SurrealDB-only brain has no local file, so it must show up with
        size 0 rather than being omitted entirely."""
        cfg = MagicMock()
        cfg.current_brain = "surrealdb-only-brain"
        fake_path = MagicMock()
        fake_path.parent = "/fake/brains"
        fake_path.exists.return_value = False
        cfg.get_brain_db_path = MagicMock(return_value=fake_path)

        with (
            patch("surreal_memory.unified_config.get_config", return_value=cfg),
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new=AsyncMock(return_value=["surrealdb-only-brain"]),
            ),
        ):
            resp = client.get("/api/dashboard/brain-files")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        names = [b["name"] for b in data["brains"]]
        assert "surrealdb-only-brain" in names
        entry = next(b for b in data["brains"] if b["name"] == "surrealdb-only-brain")
        assert entry["size_bytes"] == 0
        assert entry["is_active"] is True
        # #154 finding 5: a brain with no on-disk file must not report a
        # plausible-looking path to something that isn't there.
        assert entry["path"] is None

    def test_brain_files_reports_the_real_path_when_the_file_exists(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The positive case for the same fix: a brain that DOES have a file
        on disk (SQLite-era, or any legacy leftover) still reports its path."""
        real_path = tmp_path / "legacy-brain.db"
        real_path.write_bytes(b"x" * 10)
        cfg = MagicMock()
        cfg.current_brain = "legacy-brain"
        cfg.get_brain_db_path = MagicMock(return_value=str(real_path))

        with (
            patch("surreal_memory.unified_config.get_config", return_value=cfg),
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new=AsyncMock(return_value=["legacy-brain"]),
            ),
        ):
            resp = client.get("/api/dashboard/brain-files")

        assert resp.status_code == 200, resp.text
        entry = next(b for b in resp.json()["brains"] if b["name"] == "legacy-brain")
        assert entry["path"] == str(real_path)
        assert entry["size_bytes"] == 10

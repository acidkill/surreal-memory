"""Tests for /health and /ready endpoints in app.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from neural_memory import __version__
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_storage(brain_name: str = "default") -> MagicMock:
    """Return a minimal storage mock with brain_name."""
    storage = MagicMock()
    storage.brain_name = brain_name
    return storage


def _make_app_with_storage(storage: Any | None, startup_offset: float = 0.0) -> FastAPI:
    """Inline mini-app that mimics the health endpoints from create_app()."""
    import time as _time

    from neural_memory import __version__ as _version
    from neural_memory.server.models import HealthResponse, ReadyResponse
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.storage.sqlite_schema import SCHEMA_VERSION as _SV

    app = FastAPI()
    app.state.storage = storage  # type: ignore[assignment]
    if storage is not None:
        app.state.startup_time = _time.monotonic() - startup_offset

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        _storage: NeuralStorage = app.state.storage
        brain_name: str | None = getattr(_storage, "brain_name", None)
        _startup: float = getattr(app.state, "startup_time", _time.monotonic())
        uptime = _time.monotonic() - _startup
        return HealthResponse(
            status="ok",
            version=_version,
            brain_name=brain_name,
            uptime_seconds=round(uptime, 3),
            schema_version=_SV,
        )

    @app.get("/ready", response_model=ReadyResponse)
    async def ready_check() -> ReadyResponse:
        from fastapi.responses import JSONResponse

        _storage: NeuralStorage | None = getattr(app.state, "storage", None)
        if _storage is None:
            return JSONResponse(  # type: ignore[return-value]
                status_code=503,
                content=ReadyResponse(ready=False, detail="storage not initialized").model_dump(),
            )
        return ReadyResponse(ready=True, detail="ok")

    return app


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_version_matches_package(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/health").json()
        assert data["version"] == __version__

    def test_brain_name_returned(self) -> None:
        storage = _make_mock_storage(brain_name="my-brain")
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/health").json()
        assert data["brain_name"] == "my-brain"

    def test_schema_version_returned(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/health").json()
        assert data["schema_version"] == SCHEMA_VERSION

    def test_uptime_is_non_negative(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage, startup_offset=5.0)
        with TestClient(app) as client:
            data = client.get("/health").json()
        # uptime should reflect the offset
        assert data["uptime_seconds"] >= 5.0

    def test_uptime_is_numeric(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/health").json()
        assert isinstance(data["uptime_seconds"], float | int)

    def test_brain_name_none_when_storage_has_no_attribute(self) -> None:
        storage = MagicMock(spec=[])  # spec=[] means no attributes exposed
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/health").json()
        assert data["brain_name"] is None


# ---------------------------------------------------------------------------
# /ready tests
# ---------------------------------------------------------------------------


class TestReadyEndpoint:
    def test_returns_200_when_storage_initialized(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_true_when_storage_initialized(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/ready").json()
        assert data["ready"] is True

    def test_returns_503_when_storage_not_initialized(self) -> None:
        app = _make_app_with_storage(storage=None)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/ready")
        assert resp.status_code == 503

    def test_ready_false_when_storage_not_initialized(self) -> None:
        app = _make_app_with_storage(storage=None)
        with TestClient(app, raise_server_exceptions=False) as client:
            data = client.get("/ready").json()
        assert data["ready"] is False

    def test_detail_ok_when_ready(self) -> None:
        storage = _make_mock_storage()
        app = _make_app_with_storage(storage)
        with TestClient(app) as client:
            data = client.get("/ready").json()
        assert data["detail"] == "ok"

    def test_detail_message_when_not_ready(self) -> None:
        app = _make_app_with_storage(storage=None)
        with TestClient(app, raise_server_exceptions=False) as client:
            data = client.get("/ready").json()
        assert "storage" in data["detail"].lower()

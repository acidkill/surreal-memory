"""Tests for the hub router's read paths (#152 regression coverage).

`hub_status` and `list_devices` take brain_id from the URL path, not from
X-Brain-ID, so `get_storage`'s header-based resolution never scopes them —
the handlers used to call `storage.set_brain(brain_id)` directly on whatever
storage the dependency handed them: the process-wide shared instance in the
common (SurrealDB) case. Background maintenance loops read
`storage.brain_id` off that same instance on every tick, so a read-only GET
for brain B could redirect the next scheduled consolidation/decay pass onto
B even though the operator never switched brains.

Mirrors test_route_reasoning_training.py's pattern: a bare FastAPI app with
just this router, storage injected via dependency_overrides, and
create_isolated_storage patched per test (refused by default so a forgotten
patch fails loudly rather than reaching a real backend).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from surreal_memory.server.dependencies import get_storage
from surreal_memory.server.routes.hub import router

BOUND_BRAIN = "default"
OTHER_BRAIN = "other-brain"


def _device(device_id: str = "abc123") -> SimpleNamespace:
    return SimpleNamespace(
        device_id=device_id,
        device_name="laptop",
        registered_at=datetime(2026, 1, 1),
        last_sync_sequence=5,
    )


@pytest.fixture(autouse=True)
def _no_real_isolated_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test opens a real storage instead of using a mock.

    Same rationale as test_route_reasoning_training.py: without this, a test
    that forgets to patch create_isolated_storage silently reaches a live
    SURREALDB_URL backend instead of failing.
    """

    async def _refuse(brain_name: str | None = None) -> None:
        raise AssertionError(
            f"create_isolated_storage({brain_name!r}) reached the real backend; "
            "patch it in the test or set mock_storage.brain_id to the request scope"
        )

    monkeypatch.setattr("surreal_memory.unified_config.create_isolated_storage", _refuse)


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.brain_id = BOUND_BRAIN
    storage.get_change_log_stats = AsyncMock(
        return_value={"total": 0, "pending": 0, "synced": 0, "last_sequence": 0}
    )
    storage.list_devices = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def client(mock_storage: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage] = lambda: mock_storage
    return TestClient(app)


def _scoped_storage(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> AsyncMock:
    scoped = AsyncMock()
    scoped.brain_id = OTHER_BRAIN
    scoped.get_change_log_stats = AsyncMock(
        return_value=overrides.get(
            "stats", {"total": 1, "pending": 0, "synced": 1, "last_sequence": 1}
        )
    )
    scoped.list_devices = AsyncMock(return_value=overrides.get("devices", [_device()]))
    scoped.close = AsyncMock()
    monkeypatch.setattr(
        "surreal_memory.unified_config.create_isolated_storage",
        AsyncMock(return_value=scoped),
    )
    return scoped


class TestHubStatusDoesNotLeakBrainState:
    """The read-only status endpoint must never mutate the shared storage's brain."""

    def test_status_for_a_different_brain_does_not_call_set_brain(
        self, client: TestClient, mock_storage: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _scoped_storage(monkeypatch)

        resp = client.get(f"/hub/status/{OTHER_BRAIN}")

        assert resp.status_code == 200
        mock_storage.set_brain.assert_not_called()

    def test_status_for_a_different_brain_leaves_shared_storage_brain_id_unchanged(
        self, client: TestClient, mock_storage: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact scenario in #152: a background loop reads storage.brain_id
        right after this request and must still see the brain it had before.
        """
        _scoped_storage(monkeypatch)

        client.get(f"/hub/status/{OTHER_BRAIN}")

        assert mock_storage.brain_id == BOUND_BRAIN

    def test_status_for_a_different_brain_reads_from_the_scoped_storage(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scoped = _scoped_storage(
            monkeypatch,
            stats={"total": 7, "pending": 2, "synced": 5, "last_sequence": 7},
            devices=[_device("aa"), _device("bb")],
        )

        data = client.get(f"/hub/status/{OTHER_BRAIN}").json()

        assert scoped.get_change_log_stats.await_count == 1
        assert scoped.list_devices.await_count == 1
        assert data["brain_id"] == OTHER_BRAIN
        assert data["device_count"] == 2
        assert data["change_log"]["total"] == 7

    def test_status_for_the_bound_brain_reuses_the_shared_storage(
        self, client: TestClient, mock_storage: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No scope mismatch -> no isolated connection needed."""
        mock_storage.get_change_log_stats.return_value = {
            "total": 3,
            "pending": 1,
            "synced": 2,
            "last_sequence": 3,
        }

        data = client.get(f"/hub/status/{BOUND_BRAIN}").json()

        assert mock_storage.get_change_log_stats.await_count == 1
        assert data["change_log"]["total"] == 3


class TestListDevicesDoesNotLeakBrainState:
    """Same #152 leak, on the /hub/devices/{brain_id} endpoint."""

    def test_devices_for_a_different_brain_does_not_call_set_brain(
        self, client: TestClient, mock_storage: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _scoped_storage(monkeypatch)

        resp = client.get(f"/hub/devices/{OTHER_BRAIN}")

        assert resp.status_code == 200
        mock_storage.set_brain.assert_not_called()

    def test_devices_for_a_different_brain_leaves_shared_storage_brain_id_unchanged(
        self, client: TestClient, mock_storage: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _scoped_storage(monkeypatch)

        client.get(f"/hub/devices/{OTHER_BRAIN}")

        assert mock_storage.brain_id == BOUND_BRAIN

    def test_devices_for_a_different_brain_reads_from_the_scoped_storage(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scoped = _scoped_storage(monkeypatch, devices=[_device("cc")])

        data = client.get(f"/hub/devices/{OTHER_BRAIN}").json()

        assert scoped.list_devices.await_count == 1
        assert data["brain_id"] == OTHER_BRAIN
        assert len(data["devices"]) == 1
        assert data["devices"][0]["device_id"] == "cc"

    def test_devices_for_the_bound_brain_reuses_the_shared_storage(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        mock_storage.list_devices.return_value = [_device("dd")]

        data = client.get(f"/hub/devices/{BOUND_BRAIN}").json()

        assert mock_storage.list_devices.await_count == 1
        assert data["devices"][0]["device_id"] == "dd"


class TestInvalidBrainId:
    def test_status_rejects_invalid_brain_id(self, client: TestClient) -> None:
        resp = client.get("/hub/status/../etc")

        assert resp.status_code in (404, 422)

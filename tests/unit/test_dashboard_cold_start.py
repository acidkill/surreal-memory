"""Cold dashboard responses retain useful counts while one diagnostics job runs."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from surreal_memory.server.dependencies import get_storage
from surreal_memory.server.routes import dashboard_api as api


@pytest.fixture
async def cold_dashboard(monkeypatch):
    await api.invalidate_brain_caches()
    storage = SimpleNamespace(
        brain_id="default",
        get_stats=AsyncMock(
            return_value={"neuron_count": 17, "synapse_count": 29, "fiber_count": 7}
        ),
        find_brain_by_name=AsyncMock(return_value=None),
    )

    @asynccontextmanager
    async def scope(*args):
        yield storage

    monkeypatch.setattr(api, "storage_for_scope", scope)
    monkeypatch.setattr(
        "surreal_memory.unified_config.get_config", lambda: SimpleNamespace(current_brain="default")
    )
    monkeypatch.setattr(
        "surreal_memory.unified_config.list_available_brains", AsyncMock(return_value=["default"])
    )
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_storage] = lambda: storage
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        yield client, storage
    await api.invalidate_brain_caches()


async def test_cold_dashboard_returns_counts_and_pending_without_waiting(
    cold_dashboard, monkeypatch
):
    client, _ = cold_dashboard
    gate = asyncio.Event()
    calls = []

    async def slow(self, name):
        calls.append(name)
        await gate.wait()

    monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze", slow)
    replies = await asyncio.wait_for(
        asyncio.gather(
            client.get("/api/dashboard/stats"),
            client.get("/api/dashboard/health"),
            client.get("/api/dashboard/brains"),
        ),
        timeout=0.2,
    )
    stats, health, brains = replies
    assert stats.status_code == 200
    assert stats.json()["total_neurons"] == 17
    assert stats.json()["total_synapses"] == 29
    assert stats.json()["total_fibers"] == 7
    assert stats.json()["health_grade"] == "—"
    assert stats.json()["purity_score"] is None
    assert health.status_code == 202
    assert health.json() == {"status": "pending"}
    assert brains.json()[0]["grade"] == "—"
    assert brains.json()[0]["purity_score"] is None
    await asyncio.sleep(0)
    assert calls == ["default"]


async def test_failed_health_is_explicit_and_can_be_retried(cold_dashboard, monkeypatch):
    client, _ = cold_dashboard
    calls = 0
    gate = asyncio.Event()

    async def analyze(self, name):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic failure")
        await gate.wait()
        return _report()

    monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze", analyze)
    assert (await client.get("/api/dashboard/health")).status_code == 202
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    failed = await client.get("/api/dashboard/health")
    assert failed.status_code == 503
    assert "unavailable" in failed.json()["detail"]
    # A fresh request observes the retry, rather than remaining stuck on the old error.
    assert (await client.get("/api/dashboard/health")).status_code == 202
    gate.set()
    await asyncio.gather(*api._GRADE_REFRESH_TASKS)
    ready = await client.get("/api/dashboard/health")
    assert ready.status_code == 200
    assert ready.json()["grade"] == "D"
    assert calls == 2


def _report():
    from surreal_memory.engine.diagnostics import BrainHealthReport

    return BrainHealthReport(
        grade="D",
        purity_score=45.7,
        connectivity=0.5,
        diversity=0.5,
        freshness=0.5,
        consolidation_ratio=0.5,
        orphan_rate=0.1,
        activation_efficiency=0.5,
        recall_confidence=0.5,
        neuron_count=17,
        synapse_count=29,
        fiber_count=7,
        warnings=(),
        recommendations=(),
    )


async def test_cancelled_reader_does_not_cancel_shared_analysis(cold_dashboard, monkeypatch):
    _, storage = cold_dashboard
    started, gate = asyncio.Event(), asyncio.Event()

    async def analyze(self, name):
        started.set()
        await gate.wait()
        return _report()

    monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze", analyze)
    reader = asyncio.create_task(api._cached_health_report(storage, "default"))
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    gate.set()
    report = await api._cached_health_report(storage, "default")
    assert report.grade == "D"


async def test_cancelled_analysis_can_restart(cold_dashboard, monkeypatch):
    client, _ = cold_dashboard
    monkeypatch.setattr(
        "surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze",
        AsyncMock(side_effect=[asyncio.CancelledError(), _report()]),
    )
    assert (await client.get("/api/dashboard/health")).status_code == 202
    await asyncio.gather(*api._GRADE_REFRESH_TASKS, return_exceptions=True)
    assert (await client.get("/api/dashboard/health")).status_code == 202
    await asyncio.gather(*api._GRADE_REFRESH_TASKS)
    assert (await client.get("/api/dashboard/health")).json()["grade"] == "D"


async def test_stale_report_survives_failed_refresh(cold_dashboard, monkeypatch):
    client, _ = cold_dashboard
    api._GRADE_LAST_GOOD["health:default"] = _report()
    monkeypatch.setattr(
        "surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze",
        AsyncMock(side_effect=RuntimeError("synthetic failure")),
    )
    first = await client.get("/api/dashboard/health")
    await asyncio.gather(*api._GRADE_REFRESH_TASKS, return_exceptions=True)
    second = await client.get("/api/dashboard/health")
    assert first.status_code == second.status_code == 200
    assert first.json()["purity_score"] == second.json()["purity_score"] == 45.7


async def test_invalidation_prevents_cancel_resistant_task_republishing(
    cold_dashboard, monkeypatch
):
    client, _ = cold_dashboard
    started = asyncio.Event()

    async def analyze(self, name):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return _report()

    monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze", analyze)
    await client.get("/api/dashboard/health")
    await started.wait()
    await api.invalidate_brain_caches()
    assert not api._GRADE_LAST_GOOD
    assert not api._GRADE_INFLIGHT
    assert api._GRADE_CACHE.get("health:default") is None


async def test_each_brain_analysis_owns_its_scoped_lifetime(cold_dashboard, monkeypatch):
    _, storage = cold_dashboard
    open_scopes = set()
    seen = []

    @asynccontextmanager
    async def scope(parent, name):
        assert parent is storage
        scoped = SimpleNamespace(brain_id=name)
        open_scopes.add(name)
        try:
            yield scoped
        finally:
            open_scopes.remove(name)

    async def analyze(self, name):
        await asyncio.sleep(0)
        assert name in open_scopes
        assert self._storage.brain_id == name
        seen.append(name)
        return _report()

    monkeypatch.setattr(api, "storage_for_scope", scope)
    monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze", analyze)
    await asyncio.gather(
        api._cached_health_report(storage, "default"),
        api._cached_health_report(storage, "other"),
    )
    assert sorted(seen) == ["default", "other"]
    assert not open_scopes
    assert storage.brain_id == "default"

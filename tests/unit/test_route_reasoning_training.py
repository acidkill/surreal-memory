"""Tests for the reasoning-training dashboard router.

Mirrors test_dashboard_api.py: a bare FastAPI app with just this router, storage
and brain injected via dependency_overrides. The router-level require_local_request
passes because TestClient's host is "testclient". Config is patched at
surreal_memory.unified_config.* (the handlers import get_config/set_config locally).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import surreal_memory.server.routes.reasoning_training as rt_module
from surreal_memory.core.brain import Brain
from surreal_memory.server.dependencies import get_brain, get_storage
from surreal_memory.server.routes.reasoning_training import router
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

BRAIN_ID = "default"


def _cfg(tmp_path: Path, **rt_kw: Any) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain=BRAIN_ID,
        reasoning_training=ReasoningTrainingConfig(**rt_kw),
    )


def _fiber(
    fid: str,
    model: str,
    category: str,
    title: str = "t",
    confidence: float = 1.0,
    frequency: int = 3,
    *,
    pattern: bool = True,
) -> SimpleNamespace:
    meta: dict[str, Any] = {
        "_source_model": model,
        "_reasoning_category": category,
        "_reasoning_title": title,
        "_reasoning_confidence": confidence,
        "_reasoning_frequency": frequency,
        "_reasoning_signature": f"sig-{fid}",
        "_reasoning_strategy": "verify -> check",
        "_reasoning_description": "medoid description",
    }
    if pattern:
        meta["_reasoning_pattern"] = True
    return SimpleNamespace(id=fid, summary=f"summary-{fid}", metadata=meta)


@pytest.fixture(autouse=True)
def _reset_mining_state() -> Any:
    """Isolate the per-brain mining job state/tasks between tests."""
    rt_module._mining_tasks = {}
    rt_module._mining_states = {}
    yield
    rt_module._mining_tasks = {}
    rt_module._mining_states = {}


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.get_reasoning_stats = AsyncMock(
        return_value={"by_model": {}, "by_category": {}, "total": 0, "unprocessed": 0}
    )
    storage.find_fibers = AsyncMock(return_value=[])
    storage.get_fiber = AsyncMock(return_value=None)
    storage.delete_fiber = AsyncMock(return_value=True)
    storage.delete_reasoning_traces_by_model = AsyncMock(return_value=0)
    return storage


@pytest.fixture
def client(mock_storage: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    brain = Brain.create(BRAIN_ID, brain_id=BRAIN_ID)
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[get_brain] = lambda: brain
    return TestClient(app)


# ── GET /status ───────────────────────────────────────────────────────────────


def test_status_empty(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: _cfg(tmp_path))
    resp = client.get("/api/dashboard/reasoning/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_traces"] == 0
    assert data["total_patterns"] == 0
    assert data["detected_models"] == []
    assert data["mining"]["running"] is False
    assert data["config"]["mining_enabled"] is False


def test_status_with_traces_and_patterns(
    client: TestClient, mock_storage: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: _cfg(tmp_path))
    mock_storage.get_reasoning_stats.return_value = {
        "by_model": {
            "claude-fable-5": {
                "trace_count": 5,
                "unprocessed": 2,
                "last_trace_at": "2026-07-17T10:00:00",
            }
        },
        "by_category": {"debugging": 5},
        "total": 5,
        "unprocessed": 2,
    }
    # 3 debugging patterns (>= min_patterns_per_category=3) → debugging covered.
    mock_storage.find_fibers.return_value = [
        _fiber("p1", "claude-fable-5", "debugging"),
        _fiber("p2", "claude-fable-5", "debugging"),
        _fiber("p3", "claude-fable-5", "debugging"),
        _fiber("p4", "claude-fable-5", "planning"),
    ]
    resp = client.get("/api/dashboard/reasoning/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_traces"] == 5
    assert data["total_patterns"] == 4
    assert "claude-fable-5" in data["detected_models"]
    fable = next(m for m in data["per_model"] if m["model"] == "claude-fable-5")
    assert fable["trace_count"] == 5
    assert fable["pattern_count"] == 4
    assert fable["has_thinking_text"] is True
    assert fable["coverage_percent"] == pytest.approx(12.5)  # 1 of 8 categories
    cov = {c["category"]: c["covered"] for c in data["coverage_by_model"]["claude-fable-5"]}
    assert cov["debugging"] is True
    assert cov["planning"] is False


def test_status_denylisted_model_has_no_thinking(
    client: TestClient, mock_storage: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # opus-4-8 is no longer denylisted, so use a FICTIONAL denylisted model to
    # exercise the has_thinking_text=False path (the mechanism still works).
    monkeypatch.setattr(rt_module, "_MODELS_WITHOUT_THINKING", ("fictional-no-think-model",))
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: _cfg(tmp_path))
    mock_storage.get_reasoning_stats.return_value = {
        "by_model": {
            "fictional-no-think-model": {"trace_count": 0, "unprocessed": 0, "last_trace_at": ""}
        },
        "by_category": {},
        "total": 0,
        "unprocessed": 0,
    }
    resp = client.get("/api/dashboard/reasoning/status")
    data = resp.json()
    model = next(m for m in data["per_model"] if m["model"] == "fictional-no-think-model")
    assert model["has_thinking_text"] is False


def test_status_idle_has_progress_fields(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: _cfg(tmp_path))
    mining = client.get("/api/dashboard/reasoning/status").json()["mining"]
    assert mining["phase"] == "idle"
    assert mining["files_total"] == 0
    assert mining["files_scanned"] == 0
    assert mining["traces_found"] == 0
    assert mining["traces_processed"] == 0
    assert mining["current_model"] is None
    assert mining["models_done"] == 0
    assert mining["models_total"] == 0


async def test_run_mining_wires_progress_and_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surreal_memory.engine.reasoning_progress import (
        PHASE_DISTILLING,
        PHASE_INGESTING,
        MiningProgress,
    )

    cfg = _cfg(tmp_path, mining_enabled=True)
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda *a, **k: cfg)
    monkeypatch.setattr(
        "surreal_memory.unified_config.create_isolated_storage",
        AsyncMock(return_value=AsyncMock()),
    )

    captured: list[dict[str, Any]] = []

    async def _fake_ingest(
        storage: Any, brain_id: str, config: Any, *, backfill: bool = False, progress: Any = None
    ) -> SimpleNamespace:
        if progress is not None:
            progress(
                MiningProgress(
                    phase=PHASE_INGESTING,
                    files_total=10,
                    files_scanned=5,
                    traces_found=3,
                    traces_ingested=2,
                )
            )
            captured.append(dict(rt_module._mining_states[brain_id]))
        return SimpleNamespace(
            traces_ingested=2, traces_scanned=3, files_scanned=10, files_total=10
        )

    async def _fake_distill(
        storage: Any, brain_id: str, config: Any, *, drain: bool = False, progress: Any = None
    ) -> SimpleNamespace:
        if progress is not None:
            progress(
                MiningProgress(
                    phase=PHASE_DISTILLING,
                    current_model="claude-fable-5",
                    models_done=0,
                    models_total=1,
                    patterns_learned=1,
                    traces_processed=2,
                )
            )
        return SimpleNamespace(patterns_learned=1, traces_processed=2, models_seen=1)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", _fake_ingest
    )
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", _fake_distill
    )

    rt_module._mining_states[BRAIN_ID] = rt_module._idle_mining_state()
    rt_module._mining_states[BRAIN_ID].update(running=True)
    await rt_module._run_mining(BRAIN_ID, backfill=False, dry_run=False, models=None)

    # An intermediate ingest snapshot was visible mid-run.
    assert captured and captured[0]["phase"] == PHASE_INGESTING
    assert captured[0]["files_scanned"] == 5
    assert captured[0]["traces_found"] == 3

    final = rt_module._mining_states[BRAIN_ID]
    assert final["phase"] == "done"
    assert final["running"] is False
    assert final["finished_at"] is not None
    assert final["patterns_learned"] == 1
    assert final["traces_processed"] == 2
    assert final["files_scanned"] == 10
    assert final["models_total"] == 1


# ── PUT /config ───────────────────────────────────────────────────────────────


@pytest.fixture
def config_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch get_config/set_config and stub save() so PUT /config never hits disk."""
    holder: dict[str, Any] = {"cfg": _cfg(tmp_path), "saved": None}
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda *_a, **_k: holder["cfg"])
    monkeypatch.setattr("surreal_memory.unified_config.set_config", lambda c: holder.update(cfg=c))
    monkeypatch.setattr(UnifiedConfig, "save", lambda self: holder.update(saved=self))
    return holder


def test_config_toggle_mining(client: TestClient, config_capture: dict[str, Any]) -> None:
    resp = client.put("/api/dashboard/reasoning/config", json={"mining_enabled": True})
    assert resp.status_code == 200
    assert resp.json()["config"]["mining_enabled"] is True
    assert config_capture["cfg"].reasoning_training.mining_enabled is True
    assert config_capture["saved"] is not None  # persisted


def test_config_rejects_model_without_thinking(
    client: TestClient, config_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # opus is mineable now; a genuinely thinking-less model (fictional denylist
    # entry) is still rejected for mining_models.
    monkeypatch.setattr(rt_module, "_MODELS_WITHOUT_THINKING", ("fictional-no-think-model",))
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"mining_models": ["fictional-no-think-model"]},
    )
    assert resp.status_code == 422
    assert "thinking" in resp.json()["detail"].lower()


def test_config_rejects_injection_source_without_thinking(
    client: TestClient, config_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A thinking-less injection SOURCE (fictional denylist entry) is rejected.
    monkeypatch.setattr(rt_module, "_MODELS_WITHOUT_THINKING", ("fictional-no-think-model",))
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"injection_map": {"claude-fable-5": "fictional-no-think-model"}},
    )
    assert resp.status_code == 422


def test_config_allows_denylisted_target(
    client: TestClient, config_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A denylisted (thinking-less) model is still a valid injection TARGET
    # (recipient); the source must have thinking text. opus is no longer
    # denylisted, so pin a fictional denylisted model as the target here.
    monkeypatch.setattr(rt_module, "_MODELS_WITHOUT_THINKING", ("fictional-no-think-model",))
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"injection_map": {"fictional-no-think-model": "claude-fable-5"}},
    )
    assert resp.status_code == 200
    assert config_capture["cfg"].reasoning_training.injection_map == (
        ("fictional-no-think-model", "claude-fable-5"),
    )


def test_config_allows_self_mapping(client: TestClient, config_capture: dict[str, Any]) -> None:
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"injection_map": {"claude-fable-5": "claude-fable-5"}},
    )
    assert resp.status_code == 200


def test_config_min_confidence_out_of_range(
    client: TestClient, config_capture: dict[str, Any]
) -> None:
    resp = client.put("/api/dashboard/reasoning/config", json={"min_confidence": 1.5})
    assert resp.status_code == 422


def test_config_rejects_bad_model_name(client: TestClient, config_capture: dict[str, Any]) -> None:
    resp = client.put("/api/dashboard/reasoning/config", json={"mining_models": ["bad name!;drop"]})
    assert resp.status_code == 422


def test_config_pattern_targets_roundtrip(
    client: TestClient, config_capture: dict[str, Any]
) -> None:
    # Valid targets round-trip; an out-of-range value clamps to 0..100.
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"pattern_targets": {"claude-fable-5": 30, "claude-sonnet-5": 150}},
    )
    assert resp.status_code == 200
    saved = config_capture["cfg"].reasoning_training.pattern_targets
    assert saved == {"claude-fable-5": 30, "claude-sonnet-5": 100}
    assert resp.json()["config"]["pattern_targets"]["claude-fable-5"] == 30


def test_config_rejects_bad_pattern_target_name(
    client: TestClient, config_capture: dict[str, Any]
) -> None:
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"pattern_targets": {"bad name!;drop": 10}},
    )
    assert resp.status_code == 422


def test_config_rejects_glob_injection_source(
    client: TestClient, config_capture: dict[str, Any]
) -> None:
    # Source is matched literally, so a glob source would silently never match.
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"injection_map": {"claude-opus-4-8": "claude-*"}},
    )
    assert resp.status_code == 422
    assert "literal" in resp.json()["detail"].lower()


def test_config_rejects_empty_categories(
    client: TestClient, config_capture: dict[str, Any]
) -> None:
    resp = client.put("/api/dashboard/reasoning/config", json={"categories": []})
    assert resp.status_code == 422


def test_config_rejects_bad_category_name(
    client: TestClient, config_capture: dict[str, Any]
) -> None:
    resp = client.put("/api/dashboard/reasoning/config", json={"categories": ['bad"; drop']})
    assert resp.status_code == 422


def test_config_trace_chars_roundtrip(client: TestClient, config_capture: dict[str, Any]) -> None:
    resp = client.put(
        "/api/dashboard/reasoning/config",
        json={"min_trace_chars": 500, "max_trace_chars": 50_000},
    )
    assert resp.status_code == 200
    body = resp.json()["config"]
    assert body["min_trace_chars"] == 500
    assert body["max_trace_chars"] == 50_000
    assert config_capture["cfg"].reasoning_training.min_trace_chars == 500
    assert config_capture["cfg"].reasoning_training.max_trace_chars == 50_000
    assert config_capture["saved"] is not None  # persisted


# ── POST /mine ────────────────────────────────────────────────────────────────


def test_mine_disabled_returns_400(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: _cfg(tmp_path))
    resp = client.post("/api/dashboard/reasoning/mine", json={})
    assert resp.status_code == 400


def test_mine_starts_when_enabled(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "surreal_memory.unified_config.get_config",
        lambda: _cfg(tmp_path, mining_enabled=True),
    )

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(rt_module, "_run_mining", _noop)
    resp = client.post("/api/dashboard/reasoning/mine", json={"backfill": True})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "started"
    assert body["mining"]["running"] is True


def test_mine_conflict_when_already_running(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "surreal_memory.unified_config.get_config",
        lambda: _cfg(tmp_path, mining_enabled=True),
    )
    monkeypatch.setattr(rt_module, "_mining_tasks", {BRAIN_ID: SimpleNamespace(done=lambda: False)})
    resp = client.post("/api/dashboard/reasoning/mine", json={})
    assert resp.status_code == 409


def test_mine_isolated_per_brain(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A running job on a DIFFERENT brain must neither block nor leak status into
    # the requested brain (this client is scoped to BRAIN_ID="default").
    other_running = rt_module._idle_mining_state()
    other_running.update(running=True, traces_ingested=99)
    monkeypatch.setattr(
        rt_module, "_mining_tasks", {"other-brain": SimpleNamespace(done=lambda: False)}
    )
    monkeypatch.setattr(rt_module, "_mining_states", {"other-brain": other_running})

    # GET /status for `default` does not see other-brain's job.
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: _cfg(tmp_path))
    status = client.get("/api/dashboard/reasoning/status").json()
    assert status["mining"]["running"] is False
    assert status["mining"]["traces_ingested"] == 0

    # POST /mine for `default` is not blocked by other-brain's running job.
    monkeypatch.setattr(
        "surreal_memory.unified_config.get_config",
        lambda: _cfg(tmp_path, mining_enabled=True),
    )

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(rt_module, "_run_mining", _noop)
    assert client.post("/api/dashboard/reasoning/mine", json={}).status_code == 202


async def test_run_mining_forwards_backfill_to_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _run_mining must forward backfill=True into ingest_reasoning_traces (the
    # miner's real full-rescan bypass), on top of the scan_lookback_days=0
    # override it already applies.
    monkeypatch.setattr(
        "surreal_memory.unified_config.get_config",
        lambda: _cfg(tmp_path, mining_enabled=True),
    )
    storage = AsyncMock()
    storage.close = AsyncMock()
    monkeypatch.setattr(
        "surreal_memory.unified_config.create_isolated_storage",
        AsyncMock(return_value=storage),
    )
    ingest_mock = AsyncMock(return_value=SimpleNamespace(traces_ingested=1, traces_scanned=1))
    distill_mock = AsyncMock(
        return_value=SimpleNamespace(patterns_learned=0, traces_processed=1, models_seen=1)
    )
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", ingest_mock
    )
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", distill_mock
    )

    await rt_module._run_mining(BRAIN_ID, True, False, None)

    assert ingest_mock.await_args.kwargs["backfill"] is True


# ── GET/DELETE patterns ───────────────────────────────────────────────────────


def test_list_patterns_filters_and_ranks(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.find_fibers.return_value = [
        _fiber("p1", "claude-fable-5", "debugging", confidence=0.5, frequency=2),
        _fiber("p2", "claude-fable-5", "planning", confidence=1.0, frequency=3),
        _fiber("p3", "claude-sonnet-5", "debugging", confidence=1.0, frequency=1),
    ]
    resp = client.get("/api/dashboard/reasoning/patterns", params={"model": "claude-fable-5"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    # Higher confidence*frequency ranks first (p2: 3.0 > p1: 1.0).
    assert data["patterns"][0]["id"] == "p2"


def test_list_patterns_pagination(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.find_fibers.return_value = [
        _fiber(f"p{i}", "claude-fable-5", "debugging", frequency=i) for i in range(5)
    ]
    resp = client.get("/api/dashboard/reasoning/patterns", params={"limit": 2, "offset": 0})
    data = resp.json()
    assert data["total"] == 5
    assert len(data["patterns"]) == 2
    assert data["limit"] == 2


def test_get_pattern_detail(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.get_fiber.return_value = _fiber(
        "p1", "claude-fable-5", "debugging", title="verify"
    )
    resp = client.get("/api/dashboard/reasoning/patterns/p1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "verify"
    assert data["strategy"] == "verify -> check"
    assert data["description"] == "medoid description"


def test_get_pattern_404_for_non_pattern(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.get_fiber.return_value = _fiber("x", "m", "c", pattern=False)
    resp = client.get("/api/dashboard/reasoning/patterns/x")
    assert resp.status_code == 404


def test_delete_pattern(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.get_fiber.return_value = _fiber("p1", "claude-fable-5", "debugging")
    resp = client.request("DELETE", "/api/dashboard/reasoning/patterns/p1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1
    mock_storage.delete_fiber.assert_awaited_once_with("p1")


def test_delete_pattern_404(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.get_fiber.return_value = None
    resp = client.request("DELETE", "/api/dashboard/reasoning/patterns/nope")
    assert resp.status_code == 404


def test_delete_patterns_by_model(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.find_fibers.return_value = [
        _fiber("p1", "claude-fable-5", "debugging"),
        _fiber("p2", "claude-fable-5", "planning"),
        _fiber("p3", "claude-sonnet-5", "debugging"),
    ]
    resp = client.request(
        "DELETE", "/api/dashboard/reasoning/patterns", params={"model": "claude-fable-5"}
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert mock_storage.delete_fiber.await_count == 2


def test_delete_traces_by_model(client: TestClient, mock_storage: AsyncMock) -> None:
    mock_storage.delete_reasoning_traces_by_model.return_value = 7
    resp = client.request(
        "DELETE", "/api/dashboard/reasoning/traces", params={"model": "claude-fable-5"}
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 7
    mock_storage.delete_reasoning_traces_by_model.assert_awaited_once_with(
        BRAIN_ID, "claude-fable-5"
    )


def test_delete_traces_requires_model(client: TestClient) -> None:
    resp = client.request("DELETE", "/api/dashboard/reasoning/traces")
    assert resp.status_code == 422  # missing required query param


# ── Brain scoping: the NAME, never the record UUID ────────────────────────────

# Every fixture above builds the brain with `brain_id=BRAIN_ID`, so its id and
# its name are the same string -- which is precisely why these bugs shipped: the
# tests could not tell the two apart. Production brains carry a UUID primary key
# and a human name, so this fixture keeps them distinct.
REAL_UUID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def realistic_client(mock_storage: AsyncMock) -> TestClient:
    """A client whose brain has a UUID id and a separate name, as in production."""
    app = FastAPI()
    app.include_router(router)
    brain = Brain.create(BRAIN_ID, brain_id=REAL_UUID)
    assert brain.id != brain.name, "fixture must distinguish the UUID from the name"
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[get_brain] = lambda: brain
    return TestClient(app)


class TestBrainScopeIsTheName:
    """Reasoning rows are written under the brain NAME; the UUID scope is empty.

    Measured on the live brain before the fix: 196 pattern fibers, 92 neurons,
    84 synapses and 6070 traces stranded under the UUID, while 10905 traces sat
    under the name.
    """

    def test_scope_is_the_name_not_the_uuid(self) -> None:
        brain = Brain.create(BRAIN_ID, brain_id=REAL_UUID)
        assert rt_module._brain_scope(brain) == BRAIN_ID
        assert rt_module._brain_scope(brain) != REAL_UUID

    def test_scope_does_not_depend_on_ambient_storage_state(self) -> None:
        """It must not read storage.brain_id -- that is the coupling being removed."""
        import inspect

        src = inspect.getsource(rt_module._brain_scope)
        assert "storage" not in src.split('"""')[-1]

    def test_privacy_wipe_deletes_under_the_name(
        self, realistic_client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """The wipe used the UUID, so traces stored under the name survived it."""
        mock_storage.brain_id = "default"

        resp = realistic_client.request(
            "DELETE", "/api/dashboard/reasoning/traces", params={"model": "claude-fable-5"}
        )

        assert resp.status_code == 200
        scope_used = mock_storage.delete_reasoning_traces_by_model.await_args.args[0]
        assert scope_used == BRAIN_ID
        assert scope_used != REAL_UUID

    def test_mining_job_is_keyed_by_the_name_so_status_can_see_it(
        self, realistic_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keying on the UUID made the status endpoint report idle mid-run."""
        monkeypatch.setattr(
            "surreal_memory.unified_config.get_config",
            lambda: _cfg(tmp_path, mining_enabled=True),
        )
        monkeypatch.setattr(rt_module, "_run_mining", AsyncMock(return_value=None))

        resp = realistic_client.post("/api/dashboard/reasoning/mine", json={})

        assert resp.status_code in (200, 202)
        assert BRAIN_ID in rt_module._mining_states
        assert REAL_UUID not in rt_module._mining_states
        assert BRAIN_ID in rt_module._mining_tasks
        assert REAL_UUID not in rt_module._mining_tasks

    def test_the_mined_scope_passed_to_the_job_is_the_name(
        self, realistic_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_isolated_storage(scope) binds the job's writes -- it must get the name."""
        monkeypatch.setattr(
            "surreal_memory.unified_config.get_config",
            lambda: _cfg(tmp_path, mining_enabled=True),
        )
        spy = AsyncMock(return_value=None)
        monkeypatch.setattr(rt_module, "_run_mining", spy)

        realistic_client.post("/api/dashboard/reasoning/mine", json={})

        assert spy.await_args.args[0] == BRAIN_ID

    def test_no_endpoint_scopes_on_the_record_uuid(self) -> None:
        """Source guard: `brain.id` must never be used as a scope in this router.

        The status endpoint was fixed while mining and the privacy wipe were
        missed. A source-level assertion catches the next one that drifts.
        """
        import ast
        import pathlib

        src = pathlib.Path(rt_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "id"
            and isinstance(node.value, ast.Name)
            and node.value.id == "brain"
        ]
        assert not offenders, (
            f"reasoning_training.py scopes on brain.id (the record UUID) at lines {offenders}; "
            "use _brain_scope(brain, storage) instead — reasoning rows live under the brain name."
        )

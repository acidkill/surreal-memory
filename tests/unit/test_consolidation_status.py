"""Saved evidence and read-only observer contracts; no database or network fixtures."""

from __future__ import annotations

import asyncio
import io
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from surreal_memory.engine.consolidation import ConsolidationStrategy
from surreal_memory.engine.consolidation_receipts import (
    STATE,
    checkpoint_hash,
    confirmed_success,
    success_timestamp,
)
from surreal_memory.server import consolidation_status as observer
from surreal_memory.server.routes.consolidation_status import router

NOW = datetime(2026, 11, 11, 12, tzinfo=UTC)
_OWNER = "synthetic-owner"


def checkpoint(status: str = "completed") -> dict:
    strategies = [s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL]
    return {
        "brain_id": "default",
        "run_id": "synthetic-run",
        "status": status,
        "options_fingerprint": "synthetic-options",
        "requested_strategies": strategies,
        "completed_strategies": strategies if status == "completed" else strategies[:-1],
        "current_strategy": None if status == "completed" else strategies[-1],
        "phase": status,
        "last_error": None,
        "work_progress_seq": 7,
        "last_work_progress_at": NOW - timedelta(minutes=20),
    }


def receipt(cp: dict) -> dict:
    return {
        "phase": "terminal",
        "status": cp["status"],
        "run_id": cp["run_id"],
        "checkpoint_sha256": checkpoint_hash(cp),
        "cli_rc": 0 if cp["status"] == "completed" else 1,
        "finished_at": NOW.isoformat(),
    }


def configmap(value: dict, observation: dict | None = None, proof: dict | None = None) -> dict:
    data = {"receipt.json": json.dumps(value)}
    if observation is not None:
        data["last_observation.json"] = json.dumps(observation)
    if proof is not None:
        data["last_confirmed_success.json"] = json.dumps(proof)
    return {
        "metadata": {"name": STATE, "resourceVersion": "42", "namespace": "synthetic"},
        "data": data,
    }


def summarize(cp: dict | None, cm: dict | None):
    return observer.summarize_consolidation(cp, cm, checkpoint_available=True, now=NOW)


def test_completed_requires_matching_actual_exit_and_all_twenty_strategies():
    cp = checkpoint()
    status = summarize(cp, configmap(receipt(cp)))
    assert status.state == "completed"
    assert status.requested_strategies == status.completed_strategies == 20
    assert status.last_confirmed_success_at == NOW
    assert status.progress_known and status.work_progress_seq == 7


@pytest.mark.parametrize(
    "change", ["missing_exit", "cli_one", "boolean_exit", "run", "hash", "strategy", "missing_time"]
)
def test_unverified_terminal_evidence_cannot_be_green(change):
    cp = checkpoint()
    value = receipt(cp)
    if change == "strategy":
        cp["requested_strategies"] = cp["completed_strategies"] = ["prune"]
        value = receipt(cp)
    else:
        value.update(
            {
                "missing_exit": {"cli_rc": None},
                "cli_one": {"cli_rc": 1},
                "boolean_exit": {"cli_rc": False},
                "run": {"run_id": "wrong"},
                "hash": {"checkpoint_sha256": "wrong"},
                "missing_time": {"finished_at": None},
            }[change]
        )
    status = summarize(cp, configmap(value))
    assert status.state == "unknown"
    assert status.last_confirmed_success_at is None
    assert confirmed_success(value, cp) is None


def test_cli_one_paused_is_progress_not_success():
    cp = checkpoint("paused")
    result = summarize(cp, configmap(receipt(cp)))
    assert result.state == "paused" and result.eligible_work is True
    assert result.completed_strategies == 19
    assert result.last_confirmed_success_at is None


@pytest.mark.parametrize("state", ["running", "paused", "failed"])
def test_finalization_is_eligible_work_after_all_strategies_finish(state):
    cp = checkpoint(state)
    cp["completed_strategies"] = list(cp["requested_strategies"])
    value = (
        receipt(cp)
        if state != "running"
        else {
            "phase": "started",
            "started_at": NOW.isoformat(),
        }
    )
    result = summarize(cp, configmap(value))
    assert result.eligible_work is True
    assert result.completed_strategies == result.requested_strategies == 20
    assert result.state != "completed"
    assert result.last_confirmed_success_at is None


def test_completed_checkpoint_without_actual_terminal_proof_keeps_eligibility_unknown():
    cp = checkpoint()
    value = receipt(cp)
    value["cli_rc"] = None
    result = summarize(cp, configmap(value))
    assert result.state == "unknown" and result.eligible_work is None
    assert 'smem_consolidation_eligible_work{brain="default"}' not in observer.render_metrics(
        result
    )


@pytest.mark.parametrize(
    "outcome,state",
    [("deferred_capacity", "deferred_resource"), ("deferred_active_lease", "deferred_lease")],
)
def test_latest_scheduler_deferrals_are_distinct_from_failure(outcome, state):
    cp = checkpoint("paused")
    result = summarize(
        cp, configmap(receipt(cp), {"outcome": outcome, "observed_at": NOW.isoformat()})
    )
    assert result.state == state and result.eligible_work is True


def test_blocked_observation_keeps_first_failure_and_does_not_publish_raw_receipt():
    cp = checkpoint("failed")
    value = {
        "phase": "blocked",
        "reason": "private CLI payload",
        "cli_output_tail": "secret",
        "blocked_at": (NOW - timedelta(days=2)).isoformat(),
    }
    cm = configmap(value, {"outcome": "blocked", "observed_at": NOW.isoformat()})
    original = deepcopy(cm)
    first = summarize(cp, cm)
    second = observer.summarize_consolidation(
        cp, cm, checkpoint_available=True, now=NOW + timedelta(minutes=10)
    )
    assert first.state == second.state == "blocked"
    assert first.reason == "scheduler_blocked"
    assert first.last_observed_at == second.last_observed_at == NOW
    assert "secret" not in first.model_dump_json() and "private" not in first.model_dump_json()
    assert cm == original


def test_stale_observation_is_unknown_even_when_checkpoint_is_completed():
    cp = checkpoint()
    value = receipt(cp)
    value["finished_at"] = (NOW - timedelta(hours=3)).isoformat()
    result = summarize(cp, configmap(value))
    assert result.state == "unknown" and result.observation_stale
    assert result.last_confirmed_success_at == NOW - timedelta(hours=3)


@pytest.mark.parametrize(
    "cm",
    [
        None,
        {},
        {"data": {}},
        {"metadata": {"name": STATE, "resourceVersion": "1"}, "data": {"receipt.json": "bad"}},
        {
            "metadata": {"name": STATE, "resourceVersion": "1"},
            "data": {"receipt.json": '{"phase":[]}'},
        },
    ],
)
def test_missing_or_malformed_scheduler_evidence_stays_unknown(cm):
    result = summarize(checkpoint(), cm)
    assert result.state == "unknown" and not result.observation_available


def test_malformed_latest_observation_does_not_fall_back_to_old_success():
    cp = checkpoint()
    cm = configmap(receipt(cp))
    cm["data"]["last_observation.json"] = "malformed"
    result = summarize(cp, cm)
    assert result.state == "unknown" and not result.observation_available


def test_legacy_checkpoint_progress_and_unknown_numeric_metrics_are_not_zero_filled():
    cp = checkpoint("paused")
    cp.pop("work_progress_seq")
    cp.pop("last_work_progress_at")
    result = summarize(cp, configmap(receipt(cp)))
    metrics = observer.render_metrics(result)
    assert not result.progress_known and result.work_progress_seq is None
    assert 'smem_consolidation_progress_known{brain="default"} 0' in metrics
    assert "smem_consolidation_work_progress_seq{" not in metrics
    assert "smem_consolidation_last_work_progress_timestamp_seconds{" not in metrics
    assert "smem_consolidation_last_confirmed_success_timestamp_seconds{" not in metrics
    assert "synthetic-run" not in metrics
    assert (
        sum(
            int(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith("smem_consolidation_state{")
        )
        == 1
    )


def test_heartbeat_replay_and_generation_metadata_do_not_invent_work():
    cp = checkpoint("paused")
    first = summarize(cp, configmap(receipt(cp)))
    cp.update(updated_at=NOW.isoformat(), owner_token=_OWNER, source_generation=8)
    second = summarize(cp, configmap(receipt(cp)))
    assert second.work_progress_seq == first.work_progress_seq == 7
    assert second.last_work_progress_at == first.last_work_progress_at


def test_naive_utc_timestamps_and_historical_success_are_validated():
    cp = checkpoint()
    cp["last_work_progress_at"] = "2026-11-11T11:40:00"
    value = receipt(cp)
    proof = confirmed_success(value, cp)
    assert proof is not None and success_timestamp(proof) == NOW
    failed = checkpoint("failed")
    result = summarize(
        failed, configmap({"phase": "blocked", "blocked_at": NOW.isoformat()}, proof=proof)
    )
    assert result.last_confirmed_success_at == NOW
    result = summarize(cp, configmap(value))
    metrics = observer.render_metrics(result)
    expected = datetime(2026, 11, 11, 11, 40, tzinfo=UTC).timestamp()
    assert (
        f'smem_consolidation_last_work_progress_timestamp_seconds{{brain="default"}} {expected}'
        in metrics
    )
    proof["cli_rc"] = None
    assert success_timestamp(proof) is None


@pytest.mark.asyncio
async def test_observer_only_reads_default_checkpoint_and_configmap(monkeypatch):
    cp = checkpoint("paused")
    storage = SimpleNamespace(get_consolidation_progress=AsyncMock(return_value=cp))
    monkeypatch.setattr(observer, "read_scheduler_configmap", lambda: configmap(receipt(cp)))
    result = await observer.observe_consolidation(storage)
    storage.get_consolidation_progress.assert_awaited_once_with("default")
    assert result.work_progress_seq == 7
    assert vars(storage).keys() == {"get_consolidation_progress"}


@pytest.mark.asyncio
async def test_storage_timeout_and_outside_cluster_do_not_initialize_anything(monkeypatch):
    async def slow(_brain):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(observer, "READ_TIMEOUT_SECONDS", 0.001)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    result = await observer.observe_consolidation(SimpleNamespace(get_consolidation_progress=slow))
    assert not result.checkpoint_available and not result.observation_available
    assert result.state == "unknown" and result.eligible_work is None


def test_configmap_reader_is_one_bounded_get_with_ca_and_no_proxy_or_redirect(monkeypatch):
    cp = checkpoint()
    cm = configmap(receipt(cp))
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("POD_NAMESPACE", "synthetic")
    values = iter(("synthetic-a", "synthetic-b"))
    monkeypatch.setattr(observer.Path, "read_text", lambda self: next(values))
    context = MagicMock()
    monkeypatch.setattr(observer.ssl, "create_default_context", MagicMock(return_value=context))
    opener = MagicMock()
    opener.open.side_effect = lambda *args, **kwargs: io.BytesIO(json.dumps(cm).encode())
    build = MagicMock(return_value=opener)
    monkeypatch.setattr(observer.urllib.request, "build_opener", build)
    assert observer.read_scheduler_configmap() == cm
    request = opener.open.call_args.args[0]
    assert request.get_method() == "GET"
    assert (
        request.full_url == f"https://10.0.0.1:443/api/v1/namespaces/synthetic/configmaps/{STATE}"
    )
    assert opener.open.call_args.kwargs["timeout"] == 3
    assert build.call_args.args[0].proxies == {}
    assert isinstance(build.call_args.args[2], observer._NoRedirect)
    assert (
        build.call_args.args[2].redirect_request(None, None, 302, "", {}, "https://elsewhere")
        is None
    )
    observer.ssl.create_default_context.assert_called_once_with(
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )
    assert observer.read_scheduler_configmap() == cm
    first, second = [
        call.args[0].get_header("Authorization") for call in opener.open.call_args_list
    ]
    assert first == "Bearer synthetic-a" and second == "Bearer synthetic-b"


@pytest.mark.asyncio
async def test_observer_access_distinguishes_dashboard_proxy_from_private_metrics(monkeypatch):
    from surreal_memory.utils import config as config_module

    cp = checkpoint()
    storage = SimpleNamespace(get_consolidation_progress=AsyncMock(return_value=cp))
    monkeypatch.setattr(config_module, "get_config", lambda: SimpleNamespace(trusted_networks=[]))
    monkeypatch.setattr(observer, "read_scheduler_configmap", lambda: configmap(receipt(cp)))
    app = FastAPI()
    app.state.storage = storage
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1234)), base_url="http://test"
    ) as client:
        dashboard = await client.get(
            "/api/dashboard/consolidation-status", headers={"X-Forwarded-For": "203.0.113.4"}
        )
        assert dashboard.status_code == 200
        assert dashboard.headers["cache-control"] == "no-store"
        assert "receipt.json" not in dashboard.text and "cli_output" not in dashboard.text
        metrics = await client.get("/metrics/consolidation")
        assert metrics.status_code == 200 and metrics.headers["cache-control"] == "no-store"
        assert (
            await client.get("/metrics/consolidation", headers={"X-Forwarded-For": "127.0.0.1"})
        ).status_code == 403
    storage.get_consolidation_progress.reset_mock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("203.0.113.4", 1234)), base_url="http://test"
    ) as client:
        for path in ("/api/dashboard/consolidation-status", "/metrics/consolidation"):
            assert (
                await client.get(path, headers={"X-Forwarded-For": "127.0.0.1"})
            ).status_code == 403
    storage.get_consolidation_progress.assert_not_awaited()


@pytest.mark.parametrize("status", ["paused", "completed"])
def test_valid_recovery_archive_is_historical_in_observer_and_runner(status):
    from tests.unit.test_cluster_consolidation_hourly import recovered_checkpoint

    cp = recovered_checkpoint()
    cp["status"] = status
    if status == "completed":
        cp["completed_strategies"] = cp["requested_strategies"].copy()
    cp["phase"] = "semantic_link_discovery_neurons"
    value = receipt(cp)
    original = deepcopy(cp)
    assert not observer.checkpoint_errors(cp)
    result = summarize(cp, configmap(value))
    assert result.state == status
    assert (result.last_confirmed_success_at is not None) is (status == "completed")
    assert cp == original


@pytest.mark.parametrize("mutation", ["active", "unknown_field", "wrong_run", "v1", "outside_path"])
def test_recovery_archive_cannot_hide_active_or_invalid_evidence(mutation):
    from tests.unit.test_cluster_consolidation_hourly import recovered_checkpoint

    cp = recovered_checkpoint()
    audit = cp["strategy_states"]["semantic_link"]["recovery"]
    if mutation == "active":
        cp["last_error"] = "active failure"
    elif mutation == "unknown_field":
        audit["unexpected"] = "field"
    elif mutation == "wrong_run":
        audit["run_id"] = "different"
    elif mutation == "v1":
        audit["version"] = 1
    else:
        cp["unrelated_archive"] = audit
    assert observer.checkpoint_errors(cp)
    assert summarize(cp, configmap(receipt(cp))).state == "failed"

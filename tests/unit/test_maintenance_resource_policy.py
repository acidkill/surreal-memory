"""Synthetic instance policy and planned-restart receipt boundaries."""

from __future__ import annotations

import io
import json
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.engine.consolidation import ConsolidationStrategy
from surreal_memory.engine.consolidation_receipts import (
    resource_rebaseline_matches,
    resource_rebaseline_seed,
)
from surreal_memory.server.consolidation_status import summarize_consolidation
from tests.unit.test_cluster_consolidation_hourly import hourly


def safety_fixture(monkeypatch, tmp_path, limit=16):
    monkeypatch.delenv("SURREAL_MEMORY_MAINTENANCE_CONFIG", raising=False)
    if limit != 16:
        path = tmp_path / "maintenance.json"
        path.write_text(json.dumps({"memory_limit_bytes": limit * hourly.GIB}))
        monkeypatch.setenv("SURREAL_MEMORY_MAINTENANCE_CONFIG", str(path))
    for key in ("current", "working", "available", "oom", "failcnt"):
        monkeypatch.setenv("PROM_QUERY_" + key.upper(), key)
    monkeypatch.setenv("PROMETHEUS_URL", "http://metrics:9090")
    monkeypatch.setenv("DB_POD", "database")
    pod = {
        "metadata": {"uid": "db-uid"},
        "spec": {
            "containers": [{"name": "surrealdb", "resources": {"limits": {"memory": f"{limit}Gi"}}}]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"name": "database", "containerID": "container", "restartCount": 0, "ready": True}
            ],
        },
    }
    api = MagicMock()
    api.request.return_value = pod
    values = {
        "current": (limit - 0.25) * hourly.GIB,
        "working": (limit - 4) * hourly.GIB,
        "available": 6 * hourly.GIB,
        "oom": 0,
        "failcnt": 0,
    }

    def response(url, **kwargs):
        q = hourly.urllib.parse.parse_qs(hourly.urllib.parse.urlsplit(url).query)["query"][0]
        value = time.time() if q.startswith("timestamp(") else values[q]
        return io.BytesIO(
            json.dumps(
                {"status": "success", "data": {"result": [{"value": [time.time(), str(value)]}]}}
            ).encode()
        )

    monkeypatch.setattr(hourly.urllib.request, "urlopen", response)
    return hourly.Safety(api), values, pod


@pytest.mark.parametrize("limit", [16, 18, 24])
def test_warm_cache_admission_uses_resident_headroom(monkeypatch, tmp_path, limit):
    safety, values, _ = safety_fixture(monkeypatch, tmp_path, limit)
    safety.sample(initial=True)
    values["working"] = (limit - 3) * hourly.GIB
    with pytest.raises(hourly.DeferredError):
        safety.sample(initial=True)
    values["current"] = (limit - 0.75) * hourly.GIB
    with pytest.raises(hourly.CapacityError):
        safety.sample()
    values["current"] = (limit - 1) * hourly.GIB
    values["working"] = (limit - 1.5) * hourly.GIB
    with pytest.raises(hourly.CapacityError):
        safety.sample()
    values["working"] = (limit - 3.01) * hourly.GIB
    values["current"] = limit * hourly.GIB
    safety.sample(initial=True)
    values["current"] += 1
    with pytest.raises(hourly.DeferredError):
        safety.sample(initial=True)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"memory_limit_bytes": True},
        {"memory_limit_bytes": "24Gi"},
        {"memory_limit_bytes": 3 * 1024**3},
        {"memory_limit_bytes": 4 * 1024**3 + 1},
        {"memory_limit_bytes": 24 * 1024**3, "unknown": True},
        [],
        None,
    ],
)
def test_instance_config_is_strict(monkeypatch, tmp_path, config):
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SURREAL_MEMORY_MAINTENANCE_CONFIG", str(path))
    with pytest.raises(hourly.BlockedError):
        hourly.Safety(MagicMock())


@pytest.mark.parametrize("content", ["not json", " " * 4097])
def test_unreadable_or_oversized_config_blocks(monkeypatch, tmp_path, content):
    path = tmp_path / "maintenance.json"
    path.write_text(content)
    monkeypatch.setenv("SURREAL_MEMORY_MAINTENANCE_CONFIG", str(path))
    with pytest.raises(hourly.BlockedError):
        hourly.Safety(MagicMock())
    path.unlink()
    with pytest.raises(hourly.BlockedError):
        hourly.Safety(MagicMock())


def test_default_and_instance_guard_fences(monkeypatch, tmp_path):
    safety, values, pod = safety_fixture(monkeypatch, tmp_path)
    assert safety.memory_limit_bytes == 16 * hourly.GIB
    safety.sample(initial=True)
    values["working"] = values["current"] + 1
    with pytest.raises(hourly.BlockedError, match="working-set"):
        safety.sample(initial=True)
    values["working"] = 12 * hourly.GIB
    values["available"] = 4 * hourly.GIB
    with pytest.raises(hourly.DeferredError):
        safety.sample(initial=True)
    values["available"] = 6 * hourly.GIB
    pod["spec"]["containers"][0]["resources"]["limits"]["memory"] = "24Gi"
    with pytest.raises(hourly.BlockedError, match="16Gi"):
        safety.sample(initial=True)


def restart_fixture():
    strategies = [s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL]
    counters = {
        "semantic_links_created": 1,
        "semantic_links_skipped": 2,
        "semantic_link_failures": 0,
    }
    cp = {
        "brain_id": "default",
        "run_id": "synthetic-run",
        "status": "paused",
        "options_fingerprint": "synthetic-options",
        "reference_time": "2026-01-01T00:00:00Z",
        "requested_strategies": strategies,
        "completed_strategies": [s for s in strategies if s != "semantic_link"],
        "current_strategy": "semantic_link",
        "strategy_states": {s: {} for s in strategies},
        "counters": {"semantic_link": counters},
    }
    cp["strategy_states"]["semantic_link"] = {
        "phase": "validation",
        "cursor": 7,
        "counters": counters,
    }
    old = {
        "phase": "terminal",
        "status": "paused",
        "cli_rc": 1,
        "run_id": cp["run_id"],
        "checkpoint_sha256": hourly.checkpoint_hash(cp),
        "resource_baseline": [
            ["old-uid", hourly.canonical([["database", "old-container", 2]])],
            0,
            0,
        ],
        "finished_at": "2026-01-02T00:00:00Z",
    }
    baseline = [["new-uid", hourly.canonical([["database", "new-container", 0]])], 0, 0]
    log = "clean budget pause\n" + hourly.MARKER + hourly.canonical(old)
    return cp, old, baseline, log


def make_seed(cp, old, baseline, log):
    return resource_rebaseline_seed(
        old, cp, log, baseline, 16 * hourly.GIB, 24 * hourly.GIB, "2026-01-03T00:00:00Z"
    )


def test_distinct_seed_preserves_all_state_and_observer_agrees():
    cp, old, baseline, log = restart_fixture()
    before = deepcopy(cp)
    seed = make_seed(cp, old, baseline, log)
    assert cp == before and seed["cli_rc"] is None
    assert hourly.select_command(seed, cp, None)[-1] == "--resume"
    assert resource_rebaseline_matches(seed, cp)
    cm = {
        "metadata": {"name": hourly.STATE, "resourceVersion": "2"},
        "data": {"receipt.json": hourly.canonical(seed)},
    }
    assert (
        summarize_consolidation(
            cp, cm, checkpoint_available=True, now=hourly.datetime(2026, 1, 3, tzinfo=hourly.UTC)
        ).state
        == "paused"
    )
    assert seed["resource_rebaseline"]["previous_receipt"] == old
    assert seed["resource_rebaseline"]["previous_log"] == log


@pytest.mark.parametrize(
    "change",
    [
        "run",
        "options",
        "reference",
        "state",
        "counter",
        "audit",
        "extra",
        "approval",
        "rc",
        "baseline",
        "policy",
    ],
)
def test_seed_mutation_never_authorizes_resume(change):
    cp, old, baseline, log = restart_fixture()
    seed = make_seed(cp, old, baseline, log)
    if change in {"run", "options", "reference"}:
        cp[
            {"run": "run_id", "options": "options_fingerprint", "reference": "reference_time"}[
                change
            ]
        ] = "changed"
    elif change == "state":
        cp["strategy_states"]["prune"]["cursor"] = 8
    elif change == "counter":
        cp["counters"]["semantic_link"]["semantic_links_created"] += 1
    elif change == "audit":
        seed["resource_rebaseline"]["version"] = 2
    elif change == "extra":
        seed["unknown"] = True
    elif change == "approval":
        seed["approved"] = False
    elif change == "rc":
        seed["cli_rc"] = 0
    elif change == "baseline":
        seed["resource_baseline"][1] = 1
    else:
        seed["resource_policy_sha256"] = "wrong"
    assert not resource_rebaseline_matches(seed, cp)
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(seed, cp, None)


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "started",
        "completed",
        "missing-log",
        "error-log",
        "old-identity",
        "old-container",
        "restart",
        "oom",
        "failcnt",
        "missing-state",
        "unfinished",
        "counter-mismatch",
        "counter-bool",
        "future",
        "limit",
    ],
)
def test_restart_constructor_rejects_incomplete_or_unsafe_proof(change):
    cp, old, baseline, log = restart_fixture()
    if change == "failed":
        old["cli_rc"] = 2
    elif change == "started":
        old["phase"] = "started"
    elif change == "completed":
        cp["status"] = "completed"
    elif change == "missing-log":
        log = ""
    elif change == "error-log":
        log = "ERROR backend\n" + log
    elif change == "old-identity":
        baseline[0][0] = "old-uid"
    elif change == "old-container":
        baseline[0][1] = hourly.canonical([["database", "old-container", 0]])
    elif change == "restart":
        baseline[0][1] = hourly.canonical([["database", "new-container", 1]])
    elif change == "oom":
        baseline[1] = 1
    elif change == "failcnt":
        baseline[2] = 1
    elif change == "missing-state":
        cp["strategy_states"].pop("prune")
    elif change == "unfinished":
        cp["completed_strategies"].pop()
    elif change == "counter-mismatch":
        cp["counters"] = {}
    elif change == "counter-bool":
        cp["strategy_states"]["semantic_link"]["counters"]["semantic_link_failures"] = False
    # Match the predecessor to changed CP, so validation cannot stop at a stale hash alone.
    if change not in {"missing-log", "error-log"}:
        old["checkpoint_sha256"] = hourly.checkpoint_hash(cp)
        log = "clean pause\n" + hourly.MARKER + hourly.canonical(old)
    with pytest.raises(hourly.BlockedError):
        resource_rebaseline_seed(
            old,
            cp,
            log,
            baseline,
            16 * hourly.GIB,
            (3 if change == "limit" else 24) * hourly.GIB,
            "2999-01-03T00:00:00Z" if change == "future" else "2026-01-03T00:00:00Z",
        )


def operator_fixture():
    cp, old, baseline, log = restart_fixture()
    raw = json.dumps(old)
    observation = {"outcome": "deferred_capacity", "observed_at": "2026-01-02T01:00:00Z"}
    data = {
        "receipt.json": raw,
        "last_observation.json": json.dumps(observation),
        "first_failure.json": "preserve",
        "last_confirmed_success.json": "preserve",
    }
    api = MagicMock()
    current = {"metadata": {"resourceVersion": "1"}, "data": data}
    api.state.side_effect = lambda: deepcopy(current)
    api.identity.return_value = {"job_uid": "operator-uid"}
    api.previous_log.side_effect = lambda r: (
        log if r == old else log + "\nHOURLY_OBSERVATION " + hourly.canonical(observation)
    )

    def cas(cm, seed):
        assert cm["metadata"]["resourceVersion"] == "1"
        current["metadata"]["resourceVersion"] = "2"
        current["data"] = dict(cm["data"], **{"receipt.json": hourly.canonical(seed)})
        return deepcopy(current)

    api.cas.side_effect = cas
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp), _query=AsyncMock(return_value=[])
    )
    safety = MagicMock(
        baseline=baseline,
        memory_limit_bytes=24 * hourly.GIB,
        policy_sha256=hourly.checkpoint_hash({"memory_limit_bytes": 24 * hourly.GIB}),
    )
    args = SimpleNamespace(
        previous_receipt_sha256=hourly.hashlib.sha256(raw.encode()).hexdigest(),
        checkpoint_sha256=hourly.checkpoint_hash(cp),
        confirm_run_id=cp["run_id"],
        previous_memory_limit_bytes=16 * hourly.GIB,
    )
    return api, storage, safety, args, current


@pytest.mark.asyncio
async def test_operator_archives_latest_observation_cas_and_readback_without_checkpoint_write():
    api, storage, safety, args, current = operator_fixture()
    before = deepcopy(await storage.get_consolidation_progress())
    assert await hourly.rebaseline_resources(api, storage, safety, args) == 0
    assert await storage.get_consolidation_progress() == before
    assert api.cas.call_count == 1 and api.no_other_jobs.call_count == 3
    assert "last_observation.json" not in current["data"]
    archive = json.loads(
        next(v for k, v in current["data"].items() if k.startswith("resource_rebaseline_"))
    )
    assert json.loads(archive["observation_raw"])["outcome"] == "deferred_capacity"
    assert (
        current["data"]["first_failure.json"]
        == current["data"]["last_confirmed_success.json"]
        == "preserve"
    )
    assert storage._query.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["hash", "run", "lease", "jobs", "403", "drift", "cas", "ack", "observation", "safety"],
)
async def test_operator_fails_closed_without_retry(failure):
    api, storage, safety, args, current = operator_fixture()
    if failure == "hash":
        args.previous_receipt_sha256 = "wrong"
    elif failure == "run":
        args.confirm_run_id = "wrong"
    elif failure == "lease":
        storage._query.return_value = [{}]
    elif failure in {"jobs", "403"}:
        api.no_other_jobs.side_effect = hourly.BlockedError(failure)
    elif failure == "drift":
        changed = deepcopy(await storage.get_consolidation_progress())
        changed["cursor"] = 100
        storage.get_consolidation_progress.side_effect = [
            await storage.get_consolidation_progress(),
            changed,
        ]
    elif failure == "cas":
        api.cas.side_effect = RuntimeError("409 Conflict")
    elif failure == "ack":
        api.state.side_effect = [deepcopy(current), {"data": {}}]
    elif failure == "observation":
        api.previous_log.side_effect = [api.previous_log({}), "stale"]
    else:
        safety.sample.side_effect = hourly.BlockedError("unsafe")
    with pytest.raises((hourly.BlockedError, RuntimeError)):
        await hourly.rebaseline_resources(api, storage, safety, args)
    assert api.cas.call_count <= 1
    if failure not in {"cas", "ack"}:
        api.cas.assert_not_called()


@pytest.mark.parametrize(
    "items,allowed",
    [
        ([], False),
        ([{"metadata": {"uid": "own"}}], True),
        ([{"metadata": {"uid": "name-only"}}], False),
        (
            [{"metadata": {"uid": "own"}}, {"metadata": {"uid": "other"}, "status": {"active": 1}}],
            False,
        ),
        ([{"metadata": {"uid": "own"}}, {"metadata": {"uid": "other"}, "status": {}}], False),
        (
            [
                {"metadata": {"uid": "own"}},
                {
                    "metadata": {"uid": "old"},
                    "status": {"conditions": [{"type": "Complete", "status": "True"}]},
                },
            ],
            True,
        ),
    ],
)
def test_actual_job_inventory_requires_own_uid_and_no_other_unknown_worker(items, allowed):
    api = object.__new__(hourly.Kubernetes)
    api.namespace = "synthetic"
    api.request = MagicMock(return_value={"items": items})
    if allowed:
        api.no_other_jobs("own")
    else:
        with pytest.raises(hourly.BlockedError):
            api.no_other_jobs("own")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["hash", "jobs", "lease", "postidentity", "postworker", "success"]
)
async def test_real_main_operator_entrypoint_never_initializes_or_migrates(failure, monkeypatch):
    from surreal_memory import unified_config
    from surreal_memory.storage.surrealdb import migrations as migrations_module

    api, storage, safety, args, current = operator_fixture()
    storage._reconnect = AsyncMock()
    storage.close = AsyncMock()
    storage.set_brain = MagicMock()
    storage.initialize = AsyncMock(side_effect=AssertionError("schema initialization forbidden"))
    storage.save_brain = AsyncMock(side_effect=AssertionError("brain write forbidden"))
    if failure == "hash":
        args.previous_receipt_sha256 = "wrong"
    elif failure == "jobs":
        api.no_other_jobs.side_effect = hourly.BlockedError("active worker")
    elif failure == "lease":
        storage._query.return_value = [{}]
    elif failure in {"postidentity", "postworker"}:
        target = safety.sample if failure == "postidentity" else api.no_other_jobs
        target.side_effect = [None, None, hourly.BlockedError("post-CAS drift")]
    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
    monkeypatch.setattr(hourly, "Kubernetes", MagicMock(return_value=api))
    monkeypatch.setattr(hourly, "Safety", MagicMock(return_value=safety))
    monkeypatch.setattr(
        "surreal_memory.storage.surrealdb.SurrealDBStorage", MagicMock(return_value=storage)
    )
    shared = AsyncMock(side_effect=AssertionError("shared storage factory forbidden"))
    migrate = AsyncMock(side_effect=AssertionError("brain migration forbidden"))
    schema = AsyncMock(side_effect=AssertionError("schema migration forbidden"))
    monkeypatch.setattr(unified_config, "get_shared_storage", shared)
    monkeypatch.setattr(unified_config, "_migrate_brain_runtime_config", migrate)
    monkeypatch.setattr(migrations_module, "apply_migrations", schema)
    monkeypatch.setattr(unified_config, "get_config", MagicMock())
    argv = [
        "--rebaseline-resources",
        "--confirm-run-id",
        args.confirm_run_id,
        "--checkpoint-sha256",
        args.checkpoint_sha256,
        "--previous-receipt-sha256",
        args.previous_receipt_sha256,
        "--previous-memory-limit-bytes",
        str(args.previous_memory_limit_bytes),
    ]
    if failure == "success":
        assert await hourly.main(argv) == 0
    else:
        with pytest.raises(hourly.BlockedError):
            await hourly.main(argv)
    storage.initialize.assert_not_called()
    storage.save_brain.assert_not_called()
    shared.assert_not_called()
    migrate.assert_not_called()
    schema.assert_not_called()
    storage.close.assert_awaited_once()
    storage._reconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["identity", "worker"])
async def test_post_cas_resource_or_worker_change_never_acknowledges_success(change, capsys):
    api, storage, safety, args, current = operator_fixture()
    target = safety.sample if change == "identity" else api.no_other_jobs
    target.side_effect = [
        None,
        None,
        hourly.BlockedError("post-CAS changed; inspect without retry"),
    ]
    with pytest.raises(hourly.BlockedError):
        await hourly.rebaseline_resources(api, storage, safety, args)
    assert api.cas.call_count == 1
    assert "HOURLY_RESOURCE_REBASELINE " not in capsys.readouterr().out


@pytest.mark.parametrize("old_limit,new_limit", [(16, 16), (24, 16), (16, 24)])
def test_explicit_terminal_resource_replacement_can_raise_keep_or_lower_limit(old_limit, new_limit):
    cp, old, baseline, log = restart_fixture()
    old["resource_policy_sha256"] = hourly.checkpoint_hash(
        {"memory_limit_bytes": old_limit * hourly.GIB}
    )
    log = "clean pause\n" + hourly.MARKER + hourly.canonical(old)
    seed = resource_rebaseline_seed(
        old,
        cp,
        log,
        baseline,
        old_limit * hourly.GIB,
        new_limit * hourly.GIB,
        "2026-01-03T00:00:00Z",
    )
    assert hourly.select_command(seed, cp, None)[-1] == "--resume"
    with pytest.raises(hourly.BlockedError, match="previous limit"):
        resource_rebaseline_seed(
            old,
            cp,
            log,
            baseline,
            (old_limit + 1) * hourly.GIB,
            new_limit * hourly.GIB,
            "2026-01-03T00:00:00Z",
        )
    with pytest.raises(hourly.BlockedError, match="terminal predecessor"):
        resource_rebaseline_seed(
            seed,
            cp,
            "",
            baseline,
            new_limit * hourly.GIB,
            old_limit * hourly.GIB,
            "2026-01-03T00:00:00Z",
        )


@pytest.mark.parametrize("second", [16, 24])
def test_duplicate_config_fields_fail_closed_even_when_identical(monkeypatch, tmp_path, second):
    path = tmp_path / "maintenance.json"
    path.write_text(
        '{"memory_limit_bytes":'
        + str(16 * hourly.GIB)
        + ',"memory_limit_bytes":'
        + str(second * hourly.GIB)
        + "}"
    )
    monkeypatch.setenv("SURREAL_MEMORY_MAINTENANCE_CONFIG", str(path))
    with pytest.raises(hourly.BlockedError, match="malformed"):
        hourly.Safety(MagicMock())

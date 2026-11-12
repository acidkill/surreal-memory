"""Offline authorization and process-supervision tests for the cluster runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "hourly", Path(__file__).parents[2] / "scripts" / "cluster_consolidation_hourly.py"
)
assert SPEC and SPEC.loader
hourly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hourly)


def checkpoint(status="completed", run_id="run"):
    return {
        "brain_id": "default",
        "run_id": run_id,
        "status": status,
        "last_error": None,
        "options_fingerprint": "options",
        "requested_strategies": ["prune"],
        "completed_strategies": ["prune"] if status == "completed" else [],
    }


def receipt(cp, phase="terminal"):
    return {
        "phase": phase,
        "checkpoint_sha256": hourly.checkpoint_hash(cp),
        "status": cp["status"],
        "cli_rc": 0 if cp["status"] == "completed" else 1,
        "approved": True,
        "resource_baseline": [["uid", "[]"], 0],
    }


@pytest.mark.parametrize("status,resume", [("completed", False), ("paused", True)])
def test_selection_requires_log_receipt_and_live_checkpoint(status, resume):
    cp = checkpoint(status)
    state = receipt(cp)
    command = hourly.select_command(state, cp, hourly.MARKER + hourly.canonical(state))
    assert ("--resume" in command) is resume
    assert command[:4] == ["smem", "consolidate", "--brain", "default"]


@pytest.mark.parametrize("log", [None, "", "unrelated", hourly.MARKER + "{}"])
def test_missing_or_stale_logs_block(log):
    cp = checkpoint()
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(receipt(cp), cp, log)


@pytest.mark.parametrize("phase", ["started", "blocked", "awaiting_manual", "unknown"])
def test_unknown_previous_attempt_never_retries(phase):
    cp = checkpoint()
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(receipt(cp, phase), cp, None)


def test_bootstrap_requires_exact_approved_completed_checkpoint():
    cp = checkpoint()
    state = receipt(cp, "bootstrap")
    assert "--resume" not in hourly.select_command(state, cp, None)
    state["checkpoint_sha256"] = "stale"
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(state, cp, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("last_error", "failure"),
        ("status", "running"),
        ("brain_id", "wrong"),
        ("options_fingerprint", None),
    ],
)
def test_bad_checkpoint_blocks(field, value):
    cp = checkpoint()
    cp[field] = value
    with pytest.raises(hourly.BlockedError):
        hourly.clean_checkpoint(cp)


@pytest.mark.parametrize("status,rc", [("completed", 0), ("paused", 1)])
def test_terminal_actual_cli_exit_and_checkpoint(status, rc):
    hourly.terminal(checkpoint(status, "new"), rc, checkpoint(), False, "ordinary budget pause")
    with pytest.raises(hourly.BlockedError):
        hourly.terminal(checkpoint(status, "new"), 1 - rc, checkpoint(), False, "")


@pytest.mark.parametrize(
    "output", ["ERROR database gone", "Traceback", "lease_lost", "failed strategy"]
)
def test_cli_errors_block_even_with_completed_checkpoint(output):
    with pytest.raises(hourly.BlockedError):
        hourly.terminal(checkpoint(run_id="new"), 0, checkpoint(), False, output)


def test_resume_requires_same_run_and_fingerprint():
    with pytest.raises(hourly.BlockedError):
        hourly.terminal(checkpoint(run_id="other"), 0, checkpoint(), True, "")
    cp = checkpoint()
    cp["options_fingerprint"] = "changed"
    with pytest.raises(hourly.BlockedError):
        hourly.terminal(cp, 0, checkpoint(), True, "")


@pytest.mark.asyncio
async def test_active_lease_defers_without_cli_and_preserves_seed():
    seed = {"phase": "awaiting_manual", "seed_run_id": "manual"}
    api = MagicMock()
    api.state.return_value = {"data": {"receipt.json": json.dumps(seed)}}
    api.identity.return_value = {"pod_name": "self"}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=checkpoint("running")),
        _query=AsyncMock(return_value=[{"owner_token": "manual"}]),
    )
    with (
        patch.object(hourly, "datetime") as clock,
        patch.object(hourly.subprocess, "Popen") as popen,
    ):
        clock.now.return_value = __import__("datetime").datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = __import__("datetime").datetime.fromisoformat
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 0
    popen.assert_not_called()
    assert api.cas.call_args.args[1] == seed
    assert api.cas.call_args.args[2]["outcome"] == "deferred_active_lease"


@pytest.mark.asyncio
async def test_duplicate_cas_loser_never_spawns():
    cp = checkpoint()
    state = receipt(cp, "bootstrap")
    api = MagicMock()
    api.state.return_value = {
        "metadata": {"resourceVersion": "1"},
        "data": {"receipt.json": json.dumps(state)},
    }
    api.identity.return_value = {}
    api.cas.side_effect = RuntimeError("409 Conflict")
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp), _query=AsyncMock(return_value=[])
    )
    with (
        patch.object(hourly, "datetime") as clock,
        patch.object(hourly.subprocess, "Popen") as popen,
    ):
        clock.now.return_value = __import__("datetime").datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = __import__("datetime").datetime.fromisoformat
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 2
    popen.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_blocked_ticks_preserve_initial_cause_and_cli_evidence(capsys):
    import copy

    original = {
        "phase": "blocked",
        "blocked_at": "2026-11-05T18:15:00+00:00",
        "reason": "CLI exit disagrees with terminal checkpoint",
        "cli_rc": 2,
        "cli_output_sha256": "original-output-hash",
        "pod_name": "original-failed-pod",
    }
    api = MagicMock()
    cm = {"data": {"receipt.json": hourly.canonical(original)}}
    api.state.side_effect = lambda: copy.deepcopy(cm)
    api.identity.return_value = {"pod_name": "later-check"}

    def cas(old, decision, observation=None):
        cm["data"]["receipt.json"] = hourly.canonical(decision)
        cm["data"]["last_observation.json"] = hourly.canonical(observation)

    api.cas.side_effect = cas
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=checkpoint("failed")),
        _query=AsyncMock(return_value=[]),
    )
    safety = MagicMock()
    with patch.object(hourly.subprocess, "Popen") as popen:
        for _ in range(3):
            assert await hourly.run(api, storage, safety) == 2
            assert json.loads(cm["data"]["receipt.json"]) == original
            observed = json.loads(cm["data"]["last_observation.json"])
            assert observed["outcome"] == "blocked"
            assert observed["reason"] == original["reason"]
            assert "HOURLY_OBSERVATION " + hourly.canonical(observed) in capsys.readouterr().out
    popen.assert_not_called()
    api.previous_log.assert_not_called()
    storage.get_consolidation_progress.assert_not_awaited()
    storage._query.assert_not_awaited()
    safety.sample.assert_not_called()


@pytest.mark.asyncio
async def test_new_cli_failure_latches_actual_exit_and_output_before_validation():
    cp = checkpoint()
    state = receipt(cp, "bootstrap")
    api = MagicMock()
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": json.dumps(state)}}
    api.state.return_value = cm
    api.cas.return_value = cm
    api.identity.return_value = {"pod_name": "first-failure"}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp),
        _query=AsyncMock(return_value=[]),
    )
    child = MagicMock()
    child.poll.return_value = 2
    child.wait.return_value = 2
    output = "stale semantic source\n" * 300

    def popen(command, **kwargs):
        kwargs["stdout"].write(output)
        return child

    with patch.object(hourly.subprocess, "Popen", side_effect=popen):
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 2
    blocked = api.cas.call_args.args[1]
    assert blocked["phase"] == "blocked"
    assert blocked["reason"] == "CLI exit disagrees with terminal checkpoint"
    assert blocked["blocked_at"]
    assert blocked["cli_rc"] == 2
    assert blocked["cli_output_sha256"] == hourly.hashlib.sha256(output.encode()).hexdigest()
    assert blocked["cli_output_tail"] == output[-4096:]


def test_stop_terminates_then_kills_unresponsive_child():
    child = MagicMock()
    child.poll.return_value = None
    child.wait.side_effect = [
        subprocess.TimeoutExpired("smem", 20),
        subprocess.TimeoutExpired("smem", 10),
        -9,
    ]
    hourly.stop(child)
    child.terminate.assert_called_once()
    child.kill.assert_called_once()


def test_cas_preserves_seed_and_unrelated_data():
    api = hourly.Kubernetes.__new__(hourly.Kubernetes)
    api.namespace = "test"
    api.request = MagicMock()
    cm = {"metadata": {"resourceVersion": "7"}, "data": {"other": "retain"}}
    api.cas(cm, {"phase": "awaiting_manual"}, {"outcome": "deferred"})
    body = api.request.call_args.args[1]
    assert body["metadata"]["resourceVersion"] == "7"
    assert body["data"]["other"] == "retain"
    assert json.loads(body["data"]["last_observation.json"])["outcome"] == "deferred"


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status,rc", [("completed", 0), ("paused", 1)])
async def test_immediate_log_chain_survives_deferrals_then_clears(final_status, rc, capsys):
    import copy
    import datetime

    cp = checkpoint()
    state = receipt(cp)
    state.update(pod_name="original")
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": json.dumps(state)}}
    logs = {"original": hourly.MARKER + hourly.canonical(state)}
    api = MagicMock()
    api.state.side_effect = lambda: copy.deepcopy(cm)
    api.previous_log.side_effect = lambda item: logs[item["pod_name"]]
    api.identity.return_value = {"pod_name": "deferred-1"}

    def cas(old, decision, observation=None):
        cm["metadata"]["resourceVersion"] = str(int(cm["metadata"]["resourceVersion"]) + 1)
        cm["data"]["receipt.json"] = hourly.canonical(decision)
        if observation is not None:
            cm["data"]["last_observation.json"] = hourly.canonical(observation)
        elif decision["phase"] in {"started", "terminal"}:
            cm["data"].pop("last_observation.json", None)
        return copy.deepcopy(cm)

    api.cas.side_effect = cas
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp), _query=AsyncMock(return_value=[])
    )
    safety = MagicMock(baseline=[["uid", "[]"], 0])
    with patch.object(hourly, "datetime") as clock:
        clock.now.return_value = datetime.datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = datetime.datetime.fromisoformat
        for index in range(2):
            name = "deferred-" + str(index)
            api.identity.return_value = {"pod_name": name}
            safety.sample.side_effect = hourly.DeferredError("capacity")
            assert await hourly.run(api, storage, safety) == 0
            logs[name] = capsys.readouterr().out
            assert hourly.MARKER + hourly.canonical(state) in logs[name]
        after = checkpoint(final_status, "new")
        storage.get_consolidation_progress.side_effect = [cp, cp, cp, after]
        safety.sample.side_effect = None
        api.identity.return_value = {"pod_name": "terminal"}
        child = MagicMock()
        child.poll.return_value = rc
        child.wait.return_value = rc
        with patch.object(hourly.subprocess, "Popen", return_value=child):
            assert await hourly.run(api, storage, safety) == 0
        final_log = capsys.readouterr().out
    final = json.loads(cm["data"]["receipt.json"])
    assert "last_observation.json" not in cm["data"]
    assert final["phase"] == "terminal"
    assert ("--resume" in hourly.select_command(final, after, final_log)) is (
        final_status == "paused"
    )


@pytest.mark.asyncio
async def test_signal_stops_child_and_persists_blocked():
    import datetime

    cp = checkpoint()
    state = receipt(cp, "bootstrap")
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": json.dumps(state)}}
    api = MagicMock()
    api.state.return_value = cm
    api.cas.return_value = cm
    api.identity.return_value = {}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp), _query=AsyncMock(return_value=[])
    )
    child = MagicMock()
    child.poll.side_effect = [None, None, -15]
    child.returncode = -15
    handlers = {}

    def register(sig, handler):
        handlers[sig] = handler
        return hourly.signal.SIG_DFL

    async def interrupted_sleep(seconds):
        handlers[hourly.signal.SIGTERM](hourly.signal.SIGTERM, None)

    with (
        patch.object(hourly, "datetime") as clock,
        patch.object(hourly.subprocess, "Popen", return_value=child),
        patch.object(hourly.signal, "signal", side_effect=register),
        patch.object(hourly.asyncio, "sleep", side_effect=interrupted_sleep),
    ):
        clock.now.return_value = datetime.datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = datetime.datetime.fromisoformat
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 2
    child.send_signal.assert_called_once_with(hourly.signal.SIGINT)
    assert api.cas.call_args.args[1]["phase"] == "blocked"
    assert api.cas.call_args.args[1]["cli_rc"] == -15
    assert api.cas.call_args.args[1]["cli_output_sha256"] == hourly.hashlib.sha256(b"").hexdigest()


@pytest.mark.asyncio
async def test_active_started_owner_never_changes_cas_version():
    import datetime

    api = MagicMock()
    api.state.return_value = {"data": {"receipt.json": '{"phase":"started"}'}}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=checkpoint("running")),
        _query=AsyncMock(return_value=[{"owner_token": "active"}]),
    )
    with (
        patch.object(hourly, "datetime") as clock,
        patch.object(hourly.subprocess, "Popen") as popen,
    ):
        clock.now.return_value = datetime.datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = datetime.datetime.fromisoformat
        assert await hourly.run(api, storage, MagicMock()) == 0
    api.cas.assert_not_called()
    popen.assert_not_called()


@pytest.mark.asyncio
async def test_main_reloads_environment_before_storage(monkeypatch):
    from surreal_memory import unified_config

    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
    events = []
    with (
        patch.object(
            unified_config, "get_config", side_effect=lambda **kw: events.append(("config", kw))
        ),
        patch.object(
            unified_config,
            "get_shared_storage",
            new=AsyncMock(side_effect=lambda brain: events.append(("storage", brain))),
        ),
        patch.object(hourly, "Kubernetes"),
        patch.object(hourly, "Safety"),
        patch.object(hourly, "run", new=AsyncMock(return_value=0)),
    ):
        assert await hourly.main() == 0
    assert events == [("config", {}), ("config", {"reload": True}), ("storage", "default")]


def test_interrupted_bootstrap_never_invents_historical_exit():
    cp = checkpoint("paused")
    state = receipt(cp, "bootstrap")
    state.update(source="user_interrupted", cli_rc=None, next_action="resume")
    assert "--resume" in hourly.select_command(state, cp, None)
    state["approved"] = False
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(state, cp, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,rc,resume", [("completed", 0, False), ("paused", 1, True)])
async def test_real_runner_terminal_branches_persist_actual_exit(status, rc, resume):
    before = checkpoint("paused" if resume else "completed")
    state = receipt(before, "bootstrap")
    if resume:
        state.update(source="user_interrupted", cli_rc=None, next_action="resume")
    after = checkpoint(status, "run" if resume else "new")
    api = MagicMock()
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": json.dumps(state)}}
    api.state.return_value = cm
    api.cas.return_value = cm
    api.identity.return_value = {"pod_name": "self"}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(side_effect=[before, before, before, after]),
        _query=AsyncMock(return_value=[]),
    )
    child = MagicMock()
    child.poll.return_value = rc
    child.wait.return_value = rc
    with (
        patch.object(hourly, "datetime") as clock,
        patch.object(hourly.subprocess, "Popen", return_value=child),
    ):
        clock.now.return_value = __import__("datetime").datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = __import__("datetime").datetime.fromisoformat
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 0
    final = api.cas.call_args.args[1]
    assert final["phase"] == "terminal"
    assert final["cli_rc"] == rc
    assert final["run_id"] == after["run_id"]


@pytest.mark.asyncio
async def test_prestart_low_capacity_defers_preserving_decision():
    cp = checkpoint()
    state = receipt(cp, "bootstrap")
    api = MagicMock()
    api.state.return_value = {"data": {"receipt.json": json.dumps(state)}}
    api.identity.return_value = {}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp), _query=AsyncMock(return_value=[])
    )
    safety = MagicMock()
    safety.sample.side_effect = hourly.DeferredError("low capacity")
    with (
        patch.object(hourly, "datetime") as clock,
        patch.object(hourly.subprocess, "Popen") as popen,
    ):
        clock.now.return_value = __import__("datetime").datetime(2026, 11, 5, 14, tzinfo=hourly.UTC)
        clock.fromisoformat.side_effect = __import__("datetime").datetime.fromisoformat
        assert await hourly.run(api, storage, safety) == 0
    popen.assert_not_called()
    assert api.cas.call_args.args[1] == state
    assert api.cas.call_args.args[2]["outcome"] == "deferred_capacity"


def test_safety_new_restart_and_new_oom_block(monkeypatch):
    import io
    import time

    monkeypatch.setenv("DB_POD", "db")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus:9090")
    for key in ("CURRENT", "WORKING", "AVAILABLE", "OOM", "FAILCNT"):
        monkeypatch.setenv("PROM_QUERY_" + key, key.lower())
    api = MagicMock()
    pod = {
        "metadata": {"uid": "uid"},
        "spec": {
            "containers": [{"name": "surrealdb", "resources": {"limits": {"memory": "16Gi"}}}]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"name": "db", "ready": True, "containerID": "cid", "restartCount": 4}
            ],
        },
    }
    api.request.return_value = pod
    safety = hourly.Safety(api)
    oom = [0]
    source_stamp = [time.time()]
    current_memory = [9 * hourly.GIB]
    working_memory = [8 * hourly.GIB]

    def response(url, **kwargs):
        query = hourly.urllib.parse.parse_qs(hourly.urllib.parse.urlsplit(url).query)["query"][0]
        values = {
            "current": current_memory[0],
            "working": working_memory[0],
            "available": 6 * hourly.GIB,
            "oom": oom[0],
            "failcnt": 0,
        }
        value = source_stamp[0] if query.startswith("timestamp(") else values[query]
        return io.BytesIO(
            json.dumps(
                {"status": "success", "data": {"result": [{"value": [time.time(), str(value)]}]}}
            ).encode()
        )

    with patch.object(hourly.urllib.request, "urlopen", side_effect=response):
        safety.sample(initial=True)
        current_memory[0] = 14 * hourly.GIB
        with pytest.raises(hourly.DeferredError):
            safety.sample(initial=True)
        current_memory[0] = 15.25 * hourly.GIB
        working_memory[0] = 13 * hourly.GIB
        with pytest.raises(hourly.BlockedError):
            safety.sample()
        current_memory[0] = 9 * hourly.GIB
        working_memory[0] = 8 * hourly.GIB
        pod["spec"]["containers"][0]["resources"]["limits"]["memory"] = "12Gi"
        with pytest.raises(hourly.BlockedError, match="16Gi"):
            safety.sample()
        pod["spec"]["containers"][0]["resources"]["limits"]["memory"] = "16Gi"
        source_stamp[0] = time.time() - 120
        with pytest.raises(hourly.BlockedError, match="scrape stale"):
            safety.sample()
        source_stamp[0] = time.time()
        oom[0] = 1
        with pytest.raises(hourly.BlockedError, match="new OOM"):
            safety.sample()
        oom[0] = 0
        pod["status"]["containerStatuses"][0]["restartCount"] = 5
        with pytest.raises(hourly.BlockedError, match="restarted"):
            safety.sample()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,rc", [("completed", 0), ("paused", 1)])
async def test_all_strategy_terminal_success_proof_requires_actual_cli_zero(status, rc):
    from tests.unit.test_consolidation_status import checkpoint as all_checkpoint

    before = all_checkpoint()
    after = all_checkpoint(status)
    after["run_id"] = "new-run"
    state = receipt(before, "bootstrap")
    api = MagicMock()
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": json.dumps(state)}}
    api.state.return_value = api.cas.return_value = cm
    api.identity.return_value = {}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(side_effect=[before, before, before, after]),
        _query=AsyncMock(return_value=[]),
    )
    child = MagicMock()
    child.poll.return_value = child.wait.return_value = rc
    with patch.object(hourly.subprocess, "Popen", return_value=child):
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 0
    final = api.cas.call_args.args[1]
    assert final["cli_rc"] == rc
    assert ("confirmed_success" in final) is (rc == 0)
    if rc == 0:
        proof = final["confirmed_success"]
        assert proof["cli_rc"] == 0
        assert proof["run_id"] == after["run_id"]
        assert proof["checkpoint_sha256"] == hourly.checkpoint_hash(after)
        assert len(proof["completed_strategies"]) == 20


def test_success_history_only_updates_under_terminal_cas_and_survives_blocked_receipt():
    from surreal_memory.engine.consolidation_receipts import confirmed_success
    from tests.unit.test_consolidation_status import checkpoint as all_checkpoint
    from tests.unit.test_consolidation_status import receipt as terminal_receipt

    cp = all_checkpoint()
    value = terminal_receipt(cp)
    proof = confirmed_success(value, cp)
    assert proof is not None
    value["confirmed_success"] = proof
    api = hourly.Kubernetes.__new__(hourly.Kubernetes)
    api.namespace = "synthetic"
    api.request = MagicMock()
    cm = {"metadata": {"resourceVersion": "7"}, "data": {"other": "retain"}}
    api.cas(cm, value)
    body = api.request.call_args.args[1]
    assert json.loads(body["data"]["last_confirmed_success.json"]) == proof
    assert body["metadata"]["resourceVersion"] == "7"
    for invalid in (
        {"phase": "blocked", "confirmed_success": proof, "reason": "original"},
        {**value, "cli_rc": 1},
        {**value, "run_id": "wrong"},
    ):
        api.cas(cm, invalid)
        assert "last_confirmed_success.json" not in api.request.call_args.args[1]["data"]
    old = dict(cm, data={"last_confirmed_success.json": json.dumps(proof)})
    blocked = {"phase": "blocked", "reason": "first-failure"}
    api.cas(old, blocked, {"outcome": "blocked"})
    saved = api.request.call_args.args[1]["data"]
    assert json.loads(saved["receipt.json"]) == blocked
    assert json.loads(saved["last_confirmed_success.json"]) == proof


def recovered_checkpoint():
    from surreal_memory.engine.consolidation import ConsolidationStrategy
    from surreal_memory.engine.consolidation_progress import options_fingerprint

    cp = checkpoint("paused")
    reference = "2026-11-05T12:00:00"
    recovered_at = "2026-11-08T21:00:00"
    token = hourly.canonical(
        {
            "version": 1,
            "brain_id": "default",
            "versionstamp": 1,
            "captured_at": reference,
            "created_at_readonly": True,
        }
    )
    new_token = token.replace('"versionstamp":1', '"versionstamp":2')
    old_ref = {"state_id": "old", "revision": 4, "state_revision": 4, "source_token": token}
    new_ref = {
        "state_id": "generation-new",
        "revision": 1,
        "state_revision": 1,
        "source_token": new_token,
    }
    prior = {
        "kind": "semantic_link_discovery",
        "version": 5,
        "stage": "validation",
        "cursor": "old-cursor",
        "discovery": {
            "kind": "semantic_link_discovery",
            "version": 5,
            "stage": "neurons",
            "cursor": "old-cursor",
            "source_state_ref": old_ref,
        },
        "source_validation_ref": {
            "state_id": "validation",
            "revision": 1,
            "state_revision": 1,
            "source_token": token,
        },
    }
    bootstrap = {
        "kind": "semantic_link_discovery",
        "version": 5,
        "stage": "neurons",
        "cursor": None,
        "source_state_ref": new_ref,
    }
    counters = {
        "semantic_synapses_created": 7625,
        "semantic_synapses_skipped": 77982,
        "semantic_link_failures": 0,
        "pairs_evaluated": 94223,
    }
    audit = {
        "kind": "stale_semantic_source_restart",
        "version": 2,
        "brain_id": "default",
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": reference,
        "expected_status": "failed",
        "previous_phase": "semantic_link_discovery_validation",
        "generation": 1,
        "semantics": "cumulative_write_events",
        "source_changed_verified": True,
        "source_change": "neuron:deleted DELETE versionstamp=42",
        "prior_manifest": prior,
        "manifest_sha256": options_fingerprint(prior),
        "prior_counters": counters.copy(),
        "counters_sha256": options_fingerprint(counters),
        "retired_source_state_ref": old_ref,
        "retired_cursor": "old-cursor",
        "source_token_sha256": hourly.hashlib.sha256(token.encode()).hexdigest(),
        "new_source_token_sha256": hourly.hashlib.sha256(new_token.encode()).hexdigest(),
        "source_state_ref": new_ref,
        "bootstrap_sha256": options_fingerprint(bootstrap),
        "resolved_error": "semantic_link (progress coordination: ConsolidationProgressError)",
        "resolved_nested_error": "semantic_link source changed; discovery checkpoint preserved: source changed",
        "owner_token": "recovery-owner",
        "recovered_at": recovered_at,
    }
    strategies = [s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL]
    cp.update(
        requested_strategies=strategies,
        completed_strategies=[s for s in strategies if s != "semantic_link"],
        reference_time=reference,
        current_strategy="semantic_link",
        phase="semantic_link_restart_pending",
        cursor=None,
        owner_token=audit["owner_token"],
        updated_at=recovered_at,
        counters={"semantic_link": counters.copy()},
        strategy_states={
            "semantic_link": {
                "phase": "semantic_link_restart_pending",
                "cursor": None,
                "updated_at": recovered_at,
                "counters": counters.copy(),
                "pending": [hourly.canonical(bootstrap)],
                "recovery": audit,
                "recovery_history": [audit],
            }
        },
    )
    cp["strategy_states"].update(
        {
            strategy: {"phase": "completed", "counters": {}}
            for strategy in cp["completed_strategies"]
        }
    )
    return cp


def recovery_receipt(cp):
    state = receipt(cp, "bootstrap")
    audit = cp["strategy_states"]["semantic_link"]["recovery"]
    state.update(
        source="operator_recovery",
        cli_rc=None,
        next_action="resume",
        run_id=cp["run_id"],
        options_fingerprint=cp["options_fingerprint"],
        reference_time=cp["reference_time"],
        no_active_worker_verified=True,
        no_active_lease_verified=True,
        source_generation=audit["generation"],
        recovery_audit_sha256=hourly.checkpoint_hash(audit),
    )
    return state


def test_operator_recovery_selects_same_run_resume_without_inventing_cli_exit():
    import copy

    cp = recovered_checkpoint()
    original = copy.deepcopy(cp)
    state = recovery_receipt(cp)
    assert hourly.select_command(state, cp, None) == [
        "smem",
        "consolidate",
        "--brain",
        "default",
        "--resume",
    ]
    assert state["cli_rc"] is None
    assert cp == original
    assert cp["counters"]["semantic_link"]["semantic_synapses_created"] == 7625


def test_operator_recovery_accepts_equivalent_sdk_datetime_serializations():
    from datetime import datetime

    cp = recovered_checkpoint()
    cp["reference_time"] = datetime.fromisoformat(cp["reference_time"])
    cp["updated_at"] = datetime.fromisoformat(cp["updated_at"])
    cp["strategy_states"]["semantic_link"]["updated_at"] += "Z"
    state = recovery_receipt(cp)
    state["reference_time"] = cp["reference_time"].isoformat() + "Z"
    assert "--resume" in hourly.select_command(state, cp, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("approved", False),
        ("run_id", "other"),
        ("options_fingerprint", "other"),
        ("reference_time", "2026-11-06T12:00:00"),
        ("checkpoint_sha256", "other"),
        ("recovery_audit_sha256", "other"),
        ("source_generation", 2),
        ("source_generation", True),
        ("no_active_worker_verified", False),
        ("no_active_lease_verified", False),
        ("cli_rc", 0),
        ("next_action", "fresh"),
    ],
)
def test_operator_recovery_refuses_mismatched_seed(field, value):
    cp = recovered_checkpoint()
    state = recovery_receipt(cp)
    state[field] = value
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(state, cp, None)


@pytest.mark.parametrize(
    "where,field,value",
    [
        ("cp", "last_error", "unrelated active failure"),
        ("cp", "completed_strategies", ["prune"]),
        ("cp", "requested_strategies", ["semantic_link"]),
        ("cp", "run_id", "other"),
        ("cp", "options_fingerprint", "other"),
        ("cp", "reference_time", "2026-11-06T12:00:00"),
        ("semantic", "last_error", "unrelated active failure"),
        ("semantic", "cursor", "old-cursor"),
        ("semantic", "recovery_history", ["malformed historical failure"]),
        ("semantic", "recovery_history", "malformed history"),
        ("audit", "resolved_error", "unrelated failure"),
        ("audit", "source_changed_verified", False),
        ("audit", "generation", 0),
        ("audit", "manifest_sha256", "other"),
        ("audit", "source_token_sha256", "other"),
        ("audit", "unexpected_error", "active failure"),
        ("audit", "owner_token", "other"),
    ],
)
def test_operator_recovery_refuses_active_errors_or_malformed_audit(where, field, value):
    cp = recovered_checkpoint()
    state = recovery_receipt(cp)
    semantic = cp["strategy_states"]["semantic_link"]
    target = {"cp": cp, "semantic": semantic, "audit": semantic["recovery"]}[where]
    target[field] = value
    state["checkpoint_sha256"] = hourly.checkpoint_hash(cp)
    state["recovery_audit_sha256"] = hourly.checkpoint_hash(semantic["recovery"])
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(state, cp, None)


@pytest.mark.parametrize("status", ["paused", "completed"])
def test_valid_recovery_history_survives_later_terminal_receipts(status):
    cp = recovered_checkpoint()
    cp["status"] = status
    if status == "completed":
        cp["completed_strategies"] = cp["requested_strategies"].copy()
    cp["phase"] = "semantic_link_discovery_neurons"
    cp["owner_token"] += "-later"
    cp["strategy_states"]["semantic_link"]["phase"] = cp["phase"]
    state = receipt(cp)
    command = hourly.select_command(state, cp, hourly.MARKER + hourly.canonical(state))
    assert ("--resume" in command) is (status == "paused")
    cp["unrelated_history"] = cp["strategy_states"]["semantic_link"]["recovery"]
    with pytest.raises(hourly.BlockedError):
        hourly.clean_checkpoint(cp)


@pytest.mark.parametrize("failures", [1, False, None])
def test_operator_recovery_requires_explicit_numeric_zero_backend_failures(failures):
    from surreal_memory.engine.consolidation_progress import options_fingerprint

    cp = recovered_checkpoint()
    semantic = cp["strategy_states"]["semantic_link"]
    for counters in (
        cp["counters"]["semantic_link"],
        semantic["counters"],
        semantic["recovery"]["prior_counters"],
    ):
        counters["semantic_link_failures"] = failures
    semantic["recovery"]["counters_sha256"] = options_fingerprint(semantic["counters"])
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(recovery_receipt(cp), cp, None)


def test_missing_runtime_recovery_validator_blocks_historical_error_exemption():
    from surreal_memory.engine import consolidation_progress

    cp = recovered_checkpoint()
    with patch.object(
        consolidation_progress, "validate_semantic_source_generation_audit", None, create=True
    ):
        with pytest.raises(hourly.BlockedError):
            hourly.clean_checkpoint(cp)


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight_window", [False, True])
async def test_operator_recovery_active_lease_never_invokes_cli(preflight_window):
    cp = recovered_checkpoint()
    state = recovery_receipt(cp)
    api = MagicMock()
    api.state.return_value = {"data": {"receipt.json": hourly.canonical(state)}}
    api.identity.return_value = {}
    active = [{"owner_token": "active"}]
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp),
        _query=AsyncMock(side_effect=[[], active] if preflight_window else [active]),
    )
    with patch.object(hourly.subprocess, "Popen") as popen:
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 0
    popen.assert_not_called()
    assert api.cas.call_args.args[1] == state
    assert api.cas.call_args.args[2]["outcome"] == "deferred_active_lease"
    assert state["cli_rc"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status,rc", [("completed", 0), ("paused", 1)])
async def test_operator_recovery_runner_records_actual_terminal_exit(status, rc, capsys):
    import copy

    before = recovered_checkpoint()
    state = recovery_receipt(before)
    after = copy.deepcopy(before)
    after["status"] = status
    after["phase"] = "semantic_link_discovery_neurons"
    after["strategy_states"]["semantic_link"]["phase"] = after["phase"]
    if status == "completed":
        after["completed_strategies"] = after["requested_strategies"].copy()
    for counters in (
        after["counters"]["semantic_link"],
        after["strategy_states"]["semantic_link"]["counters"],
    ):
        counters["semantic_synapses_created"] += 2
    api = MagicMock()
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": hourly.canonical(state)}}
    api.state.return_value = cm
    api.cas.return_value = cm
    api.identity.return_value = {}
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(side_effect=[before, before, before, after]),
        _query=AsyncMock(return_value=[]),
    )
    child = MagicMock()
    child.poll.return_value = rc
    child.wait.return_value = rc
    with patch.object(hourly.subprocess, "Popen", return_value=child) as popen:
        assert await hourly.run(api, storage, MagicMock(baseline=[["uid", "[]"], 0])) == 0
    assert popen.call_args.args[0] == ["smem", "consolidate", "--brain", "default", "--resume"]
    terminal_state = api.cas.call_args.args[1]
    assert terminal_state["phase"] == "terminal"
    assert terminal_state["cli_rc"] == rc
    assert terminal_state["run_id"] == before["run_id"]
    assert after["counters"]["semantic_link"]["semantic_synapses_created"] == 7627
    command = hourly.select_command(terminal_state, after, capsys.readouterr().out)
    assert ("--resume" in command) is (status == "paused")


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "seed_status",
        "baseline",
        "baseline_type",
        "owner",
        "updated_at",
        "phase",
        "cursor",
        "duplicate_requested",
        "duplicate_completed",
        "untyped_requested",
        "untyped_completed",
        "negative_counter",
        "boolean_counter",
        "aggregate_counter",
        "missing_failure_counter",
        "history_empty",
        "history_tip",
        "history_invalid",
        "history_v1",
        "audit_missing",
        "audit_float_version",
        "pending_empty",
        "pending_multiple",
        "pending_json",
        "pending_scalar",
        "pending_changed",
        "pending_oversized",
        "missing_context",
    ],
)
def test_operator_recovery_rejects_drift_and_malformed_evidence(mutation):
    import copy

    cp = recovered_checkpoint()
    seed = recovery_receipt(cp)
    semantic = cp["strategy_states"]["semantic_link"]
    if mutation == "source":
        seed["source"] = "automatic_recovery"
    elif mutation == "seed_status":
        seed["status"] = "completed"
    elif mutation in {"baseline", "baseline_type"}:
        seed["resource_baseline"] = [] if mutation == "baseline" else True
    elif mutation in {"owner", "updated_at", "phase", "cursor"}:
        field = "owner_token" if mutation == "owner" else mutation
        cp[field] = "wrong"
    elif mutation.startswith("duplicate_"):
        field = mutation.removeprefix("duplicate_") + "_strategies"
        cp[field].append(cp[field][0])
    elif mutation.startswith("untyped_"):
        cp[mutation.removeprefix("untyped_") + "_strategies"] = "semantic_link"
    elif mutation in {"negative_counter", "boolean_counter"}:
        semantic["counters"]["pairs_evaluated"] = -1 if mutation == "negative_counter" else True
    elif mutation == "aggregate_counter":
        cp["counters"]["semantic_link"]["pairs_evaluated"] += 1
    elif mutation == "missing_failure_counter":
        del semantic["counters"]["semantic_link_failures"]
    elif mutation == "history_empty":
        semantic["recovery_history"] = []
    elif mutation == "history_tip":
        semantic["recovery_history"] = [{**semantic["recovery"], "owner_token": "other"}]
    elif mutation in {"history_invalid", "history_v1"}:
        old = dict(semantic["recovery"], version=1) if mutation == "history_v1" else {}
        semantic["recovery_history"].insert(0, old)
    elif mutation == "audit_missing":
        del semantic["recovery"]
    elif mutation == "audit_float_version":
        semantic["recovery"]["version"] = 2.0
    elif mutation == "pending_empty":
        semantic["pending"] = []
    elif mutation == "pending_multiple":
        semantic["pending"] *= 2
    elif mutation == "pending_json":
        semantic["pending"] = ["{"]
    elif mutation == "pending_scalar":
        semantic["pending"] = ["42"]
    elif mutation == "pending_changed":
        semantic["pending"] = ["{}"]
    elif mutation == "pending_oversized":
        semantic["pending"] = [" " * (512 * 1024 + 1)]
    elif mutation == "missing_context":
        del cp["reference_time"]
    seed["checkpoint_sha256"] = hourly.checkpoint_hash(cp)
    if "recovery" in semantic:
        seed["recovery_audit_sha256"] = hourly.checkpoint_hash(semantic["recovery"])
    original = copy.deepcopy(cp)
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(seed, cp, None)
    assert cp == original


def test_recovery_bootstrap_is_not_confirmed_success():
    from surreal_memory.engine.consolidation_receipts import confirmed_success

    cp = recovered_checkpoint()
    assert confirmed_success(recovery_receipt(cp), cp) is None
    assert len(cp["strategy_states"]) == 20
    assert len(cp["completed_strategies"]) == 19


@pytest.mark.parametrize("rc", [None, 0, False, True, 2])
def test_recovered_paused_terminal_requires_actual_cli_one(rc):
    cp = recovered_checkpoint()
    state = receipt(cp)
    state["cli_rc"] = rc
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(state, cp, hourly.MARKER + hourly.canonical(state))


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "unrelated"),
        ("version", 4),
        ("version", 5.0),
        ("stage", "similarity"),
        ("cursor", "stale"),
        ("unexpected", "field"),
    ],
)
def test_hash_consistent_forged_bootstrap_is_not_recovery_proof(field, value):
    cp = recovered_checkpoint()
    semantic = cp["strategy_states"]["semantic_link"]
    manifest = json.loads(semantic["pending"][0])
    manifest[field] = value
    semantic["pending"] = [hourly.canonical(manifest)]
    semantic["recovery"]["bootstrap_sha256"] = hourly.checkpoint_hash(manifest)
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(recovery_receipt(cp), cp, None)


@pytest.mark.parametrize("status,rc", [("paused", True), ("completed", False)])
def test_terminal_validation_rejects_boolean_exit_codes(status, rc):
    cp = recovered_checkpoint()
    cp["status"] = status
    if status == "completed":
        cp["completed_strategies"] = cp["requested_strategies"].copy()
    with pytest.raises(hourly.BlockedError):
        hourly.terminal(cp, rc, cp, True, "")


def test_optional_empty_recovery_is_absent_but_cannot_authorize_operator_resume():
    cp = checkpoint("paused")
    cp["strategy_states"] = {"semantic_link": {"recovery": None}}
    assert not hourly.checkpoint_errors(cp)
    state = receipt(cp, "bootstrap")
    state.update(source="operator_recovery", cli_rc=None, next_action="resume")
    with pytest.raises(hourly.BlockedError):
        hourly.select_command(state, cp, None)

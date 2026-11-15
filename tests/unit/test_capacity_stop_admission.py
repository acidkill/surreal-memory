"""Exact-state capacity admission, without reclassifying a historical CLI130."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.engine import consolidation_receipts as receipts
from surreal_memory.engine.consolidation import ConsolidationStrategy
from tests.unit.test_cluster_consolidation_hourly import hourly


def evidence():
    completed = ["dedup", "learn_habits", "process_reasoning_traces", "process_tool_events"]
    counters = {"synapses_pruned": 0, "synapse_pages": 1, "neurons_pruned": 0, "neuron_pages": 0}
    prune = {
        "phase": "synapse_pending",
        "cursor": "saved-cursor",
        "pending": ["saved-edge-" + str(i) for i in range(1583)],
        "counters": counters,
    }
    states = {name: {"phase": "completed", "counters": {}} for name in completed}
    states["prune"] = prune
    cp = {
        "brain_id": "default",
        "run_id": "saved-capacity-run",
        "status": "paused",
        "last_error": None,
        "current_strategy": "prune",
        "phase": "synapse_pending",
        "owner_token": "saved-owner",
        "options_fingerprint": "same-options",
        "reference_time": "2026-01-01T00:00:00Z",
        "requested_strategies": [
            s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL
        ],
        "completed_strategies": completed,
        "strategy_states": states,
        "counters": {"prune": counters},
    }
    cp["native_handoff"] = {
        "version": 1,
        "lineage_root_sha256": "a" * 64,
        "attempt_seq": 1,
        "run_seq": 0,
        "payload_sha256": receipts.native_payload_hash(cp),
    }
    output = "Resuming saved consolidation checkpoint for brain 'default'...\nConsolidation: prune progress state found - resuming...\n"
    baseline = [["same-db-uid", '[["surrealdb","same-container",2]]'], 0.0, 0.0]
    previous = {
        "phase": "blocked",
        "cli_rc": 130,
        "reason": "CLI exit disagrees with terminal checkpoint",
        "capacity_stop_reason": "database memory exceeds safety threshold",
        "before_run_id": cp["run_id"],
        "before_sha256": "b" * 64,
        "before_resources": baseline,
        "pod_name": "failed-pod",
        "pod_uid": "failed-pod-uid",
        "job_name": "failed-job",
        "job_uid": "failed-job-uid",
        "blocked_at": "2026-01-01T01:00:00Z",
        "cli_output_tail": output,
        "cli_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }
    raw = receipts.canonical(previous)
    log = output + "HOURLY_BLOCKED " + previous["reason"] + "\n"
    limit = 18 * 1024**3
    policy = receipts.checkpoint_hash({"memory_limit_bytes": limit})
    approval = {
        "approved_at": "2026-01-01T02:00:00Z",
        "checkpoint_sha256": receipts.checkpoint_hash(cp),
        "previous_receipt_raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "previous_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
        "first_blocked_sha256": receipts.checkpoint_hash(previous),
        "completed_states_sha256": receipts.checkpoint_hash(
            {name: states[name] for name in completed}
        ),
        "prune_state_sha256": receipts.checkpoint_hash(prune),
        "counters_sha256": receipts.checkpoint_hash(cp["counters"]),
        "native_handoff_sha256": receipts.checkpoint_hash(cp["native_handoff"]),
        **{
            key: cp[key]
            for key in ["run_id", "owner_token", "options_fingerprint", "reference_time"]
        },
        "failed_pod": {
            **{key: previous[key] for key in ["pod_name", "pod_uid", "job_name", "job_uid"]},
            "exit_code": 2,
        },
        "memory_limit_bytes": limit,
        "resource_policy_sha256": policy,
        "previous_resource_policy_sha256": policy,
        "resource_baseline": baseline,
        "runtime": {
            "source_sha": "c" * 40,
            "image_digest": "sha256:" + "d" * 64,
            "runner_sha256": hashlib.sha256(Path(hourly.__file__).read_bytes()).hexdigest(),
            "receipts_sha256": hashlib.sha256(Path(receipts.__file__).read_bytes()).hexdigest(),
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "numpy_version": version("numpy"),
            "package_version": version("surreal-memory"),
        },
        "reviewed_source_sha": "c" * 40,
        "source_fix_reviewed": True,
        "resource_root_cause_addressed": True,
        "no_active_worker_verified": True,
        "no_active_lease_verified": True,
    }
    return raw, log, cp, approval


def seed(raw, log, cp, approval):
    return receipts.capacity_resume_seed(raw, log, cp, approval=approval)


def test_admission_preserves_old130_and_saved_checkpoint_verbatim():
    raw, log, cp, approval = evidence()
    before = copy.deepcopy(cp)
    admitted = seed(raw, log, cp, approval)
    assert cp == before
    assert admitted["cli_rc"] is None and admitted["status"] == "paused"
    assert admitted["phase"] == "bootstrap" and admitted["next_action"] == "resume"
    assert admitted["capacity_resume"]["previous_receipt_raw"] == raw
    assert json.loads(raw)["cli_rc"] == 130
    assert receipts.confirmed_success(admitted, cp) is None
    assert receipts.select_command(admitted, cp, None) == [
        "smem",
        "consolidate",
        "--brain",
        "default",
        "--resume",
    ]
    with pytest.raises(receipts.BlockedError):
        receipts.select_command(json.loads(raw), cp, log)


@pytest.mark.parametrize(
    "key",
    [
        "checkpoint_sha256",
        "previous_receipt_raw_sha256",
        "previous_log_sha256",
        "first_blocked_sha256",
        "completed_states_sha256",
        "prune_state_sha256",
        "counters_sha256",
        "native_handoff_sha256",
        "run_id",
        "owner_token",
        "options_fingerprint",
        "reference_time",
        "resource_policy_sha256",
        "previous_resource_policy_sha256",
        "reviewed_source_sha",
    ],
)
def test_changed_approval_binding_is_rejected(key):
    raw, log, cp, approval = evidence()
    approval[key] = "changed"
    with pytest.raises(receipts.BlockedError):
        seed(raw, log, cp, approval)


@pytest.mark.parametrize(
    "key",
    [
        "source_fix_reviewed",
        "resource_root_cause_addressed",
        "no_active_worker_verified",
        "no_active_lease_verified",
    ],
)
def test_required_review_and_safety_attestations_are_not_inferred(key):
    raw, log, cp, approval = evidence()
    approval[key] = False
    with pytest.raises(receipts.BlockedError):
        seed(raw, log, cp, approval)


@pytest.mark.parametrize(
    "fault",
    ["owner", "run", "reference", "options", "counter", "states", "native", "error", "status"],
)
def test_saved_state_drift_is_rejected_without_mutation(fault):
    raw, log, cp, approval = evidence()
    if fault == "counter":
        cp["counters"]["prune"]["synapses_pruned"] += 1
    elif fault == "states":
        cp["strategy_states"]["dedup"]["phase"] = "changed"
    elif fault == "native":
        cp["native_handoff"]["attempt_seq"] += 1
    else:
        cp[
            {
                "owner": "owner_token",
                "run": "run_id",
                "reference": "reference_time",
                "options": "options_fingerprint",
                "error": "last_error",
                "status": "status",
            }[fault]
        ] = "changed"
    before = copy.deepcopy(cp)
    with pytest.raises(receipts.BlockedError):
        seed(raw, log, cp, approval)
    assert cp == before


@pytest.mark.parametrize(
    "fault", ["cli", "wrapper", "pod", "job", "reason", "capacity", "log", "tail", "oom", "restart"]
)
def test_unreviewed_failure_or_resource_evidence_is_rejected(fault):
    raw, log, cp, approval = evidence()
    previous = json.loads(raw)
    if fault == "cli":
        previous["cli_rc"] = 1
    elif fault == "wrapper":
        approval["failed_pod"]["exit_code"] = 1
    elif fault == "pod":
        approval["failed_pod"]["pod_uid"] = "other"
    elif fault == "job":
        approval["failed_pod"]["job_uid"] = "other"
    elif fault == "reason":
        previous["reason"] = "engine lease absent or lost"
    elif fault == "capacity":
        previous["capacity_stop_reason"] = "unknown"
    elif fault == "log":
        log = "unknown prefix\n" + log
    elif fault == "tail":
        previous["cli_output_tail"] = "other"
    elif fault == "oom":
        approval["resource_baseline"][1] = 1
    elif fault == "restart":
        approval["resource_baseline"][0][1] = '[["surrealdb","same-container",3]]'
    raw = receipts.canonical(previous)
    # Rebind predecessor bytes so the relevant semantic guard also has to refuse it.
    approval["previous_receipt_raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    approval["first_blocked_sha256"] = receipts.checkpoint_hash(previous)
    with pytest.raises(receipts.BlockedError):
        seed(raw, log, cp, approval)


def operator_fixture(tmp_path):
    raw, log, cp, approval = evidence()
    file = tmp_path / "approval.json"
    file.write_text(json.dumps(approval))
    cm = {"metadata": {"resourceVersion": "1"}, "data": {"receipt.json": raw}}
    api = MagicMock()
    api.state.side_effect = lambda: copy.deepcopy(cm)
    api.identity.return_value = {
        "pod_name": "operator",
        "pod_uid": "operator-pod",
        "job_name": "operator",
        "job_uid": "operator-job",
    }
    api.previous_log.return_value = log
    digest = approval["runtime"]["image_digest"]
    api.request.return_value = {
        "metadata": {"uid": "operator-pod"},
        "spec": {"containers": [{"image": "registry/smem@" + digest}]},
        "status": {"containerStatuses": [{"imageID": "registry/smem@" + digest}]},
    }

    def cas(value, receipt):
        assert value["metadata"]["resourceVersion"] == cm["metadata"]["resourceVersion"]
        cm["data"] = {**value["data"], "receipt.json": receipts.canonical(receipt)}
        cm["metadata"]["resourceVersion"] = str(int(cm["metadata"]["resourceVersion"]) + 1)
        return copy.deepcopy(cm)

    api.cas.side_effect = cas
    storage = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=cp), _query=AsyncMock(return_value=[])
    )
    safety = MagicMock(
        baseline=approval["resource_baseline"],
        policy_sha256=approval["resource_policy_sha256"],
        memory_limit_bytes=approval["memory_limit_bytes"],
    )
    args = SimpleNamespace(approval_file=str(file))
    return api, storage, safety, args, cm, cp, approval


async def test_explicit_operator_archives_and_cas_seeds_without_launch_or_checkpoint_write(
    tmp_path,
):
    api, storage, safety, args, cm, cp, _ = operator_fixture(tmp_path)
    before = copy.deepcopy(cp)
    old = cm["data"]["receipt.json"]
    with patch.object(hourly.subprocess, "Popen") as launch:
        assert await hourly.admit_capacity_stop(api, storage, safety, args) == 0
    launch.assert_not_called()
    assert cp == before
    admitted = json.loads(cm["data"]["receipt.json"])
    key = "capacity_resume_" + receipts.checkpoint_hash(admitted) + ".json"
    assert cm["data"][key] == receipts.canonical(admitted)
    assert admitted["capacity_resume"]["previous_receipt_raw"] == old
    assert admitted["cli_rc"] is None
    api.previous_log.assert_called_once_with(json.loads(old), expected_exit_code=2)


@pytest.mark.parametrize("fault", ["lease", "worker", "pressure", "image", "runtime", "cas"])
async def test_operator_preflight_and_cas_loss_preserve_predecessor(tmp_path, fault):
    api, storage, safety, args, cm, cp, approval = operator_fixture(tmp_path)
    old = copy.deepcopy(cm)
    if fault == "lease":
        storage._query.return_value = [{"owner_token": "other"}]
    elif fault == "worker":
        api.no_other_jobs.side_effect = receipts.BlockedError("active worker")
    elif fault == "pressure":
        safety.sample.side_effect = hourly.DeferredError("capacity")
    elif fault == "image":
        api.request.return_value["status"]["containerStatuses"][0]["imageID"] = "other"
    elif fault == "runtime":
        approval["runtime"]["runner_sha256"] = "0" * 64
        Path(args.approval_file).write_text(json.dumps(approval))
    elif fault == "cas":
        api.cas.side_effect = RuntimeError("409 conflict")
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_capacity_stop(api, storage, safety, args)
    assert cm == old


async def test_admitted_seed_launches_exactly_one_native_resume_then_preserves_archive(tmp_path):
    api, storage, safety, args, cm, cp, _ = operator_fixture(tmp_path)
    await hourly.admit_capacity_stop(api, storage, safety, args)
    admitted = json.loads(cm["data"]["receipt.json"])
    key = "capacity_resume_" + receipts.checkpoint_hash(admitted) + ".json"
    archive = cm["data"][key]
    child = MagicMock(returncode=1)
    child.poll.return_value = child.wait.return_value = 1
    with patch.object(hourly.subprocess, "Popen", return_value=child) as launch:
        assert await hourly.run(api, storage, safety) == 0
    launch.assert_called_once()
    assert launch.call_args.args[0] == [
        "smem",
        "consolidate",
        "--brain",
        "default",
        "--resume",
        "--strategy-timeout",
        "1800",
        "--total-timeout",
        "3000",
    ]
    terminal = json.loads(cm["data"]["receipt.json"])
    assert terminal["cli_rc"] == 1 and terminal["status"] == "paused"
    assert terminal["capacity_resume_sha256"] == receipts.checkpoint_hash(admitted)
    assert cm["data"][key] == archive
    assert json.loads(admitted["capacity_resume"]["previous_receipt_raw"])["cli_rc"] == 130


async def test_seed_without_immutable_archive_never_launches(tmp_path):
    api, storage, safety, args, cm, cp, approval = operator_fixture(tmp_path)
    admitted = seed(cm["data"]["receipt.json"], api.previous_log.return_value, cp, approval)
    cm["data"]["receipt.json"] = receipts.canonical(admitted)
    with patch.object(hourly.subprocess, "Popen") as launch:
        assert await hourly.run(api, storage, safety) == 2
    launch.assert_not_called()


@pytest.mark.parametrize("fault", [None, "hash", "log", "identity"])
def test_later_blocked_observation_must_be_bound_and_archived(fault):
    raw, log, cp, approval = evidence()
    previous = json.loads(raw)
    observed = {
        "outcome": "blocked",
        "reason": previous["reason"],
        "pod_name": "observed",
        "pod_uid": "observed-pod",
        "job_name": "observed",
        "job_uid": "observed-job",
    }
    observation_raw = receipts.canonical(observed)
    observation_log = (
        receipts.MARKER
        + receipts.canonical(previous)
        + "\nHOURLY_OBSERVATION "
        + receipts.canonical(observed)
        + "\n"
    )
    approval["observation_raw_sha256"] = hashlib.sha256(observation_raw.encode()).hexdigest()
    approval["observation_log_sha256"] = hashlib.sha256(observation_log.encode()).hexdigest()
    if fault == "hash":
        approval["observation_raw_sha256"] = "changed"
    elif fault == "log":
        observation_log = "other"
    elif fault == "identity":
        observation_raw = observation_raw.replace("observed-pod", "different-pod")
    if fault:
        with pytest.raises(receipts.BlockedError):
            receipts.capacity_resume_seed(
                raw,
                log,
                cp,
                approval=approval,
                observation_raw=observation_raw,
                observation_log=observation_log,
            )
    else:
        admitted = receipts.capacity_resume_seed(
            raw,
            log,
            cp,
            approval=approval,
            observation_raw=observation_raw,
            observation_log=observation_log,
        )
        assert admitted["capacity_resume"]["observation_raw"] == observation_raw


@pytest.mark.parametrize("fault", [None, "pod", "job", "exit", "active", "oom"])
def test_failed_predecessor_identity_and_actual_wrapper_two_checked(fault):
    raw, _, _, _ = evidence()
    previous = json.loads(raw)
    pod = {
        "metadata": {
            "uid": previous["pod_uid"],
            "ownerReferences": [{"kind": "Job", "uid": previous["job_uid"]}],
        },
        "status": {
            "phase": "Failed",
            "containerStatuses": [{"state": {"terminated": {"exitCode": 2, "reason": "Error"}}}],
        },
    }
    job = {"metadata": {"uid": previous["job_uid"]}, "status": {"failed": 1}}
    if fault == "pod":
        pod["metadata"]["uid"] = "other"
    elif fault == "job":
        job["metadata"]["uid"] = "other"
    elif fault == "exit":
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 1
    elif fault == "active":
        job["status"]["active"] = 1
    elif fault == "oom":
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["reason"] = "OOMKilled"
    api = hourly.Kubernetes.__new__(hourly.Kubernetes)
    api.namespace = "synthetic"
    api.request = MagicMock(side_effect=[pod, job, "actual log"])
    if fault:
        with pytest.raises(receipts.BlockedError):
            api.previous_log(previous, expected_exit_code=2)
    else:
        assert api.previous_log(previous, expected_exit_code=2) == "actual log"

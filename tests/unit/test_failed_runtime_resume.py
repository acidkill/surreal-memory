"""Synthetic evidence for the explicitly approved failed-runtime resume."""

import copy
import hashlib
import json

import pytest

from surreal_memory.engine import consolidation_receipts as receipts
from surreal_memory.engine.consolidation import ConsolidationStrategy


def runtime_resume_fixture():
    strategies = [s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL]
    ref = {"state_id": "synthetic-source", "revision": 1, "source_token": "synthetic-token"}
    manifest = {
        "kind": "semantic_link_discovery",
        "stage": "similarity",
        "version": 5,
        "source_state_ref": ref,
    }
    counters = {"semantic_link_failures": 0, "pairs_evaluated": 3}
    states = {
        name: {"phase": "completed", "counters": {}}
        for name in strategies
        if name != "semantic_link"
    }
    states["semantic_link"] = {
        "phase": "semantic_link_discovery_validation",
        "counters": counters.copy(),
        "pending": [json.dumps(manifest)],
    }
    cp = {
        "brain_id": "default",
        "run_id": "synthetic-run",
        "status": "failed",
        "last_error": "semantic_link (progress coordination: ConsolidationProgressError)",
        "options_fingerprint": "synthetic-options",
        "reference_time": "2026-01-01T00:00:00Z",
        "requested_strategies": strategies,
        "completed_strategies": [s for s in strategies if s != "semantic_link"],
        "current_strategy": "semantic_link",
        "phase": "semantic_link_discovery_validation",
        "strategy_states": states,
        "counters": {"semantic_link": counters.copy()},
    }
    cp["native_handoff"] = {
        "version": 1,
        "lineage_root_sha256": "a" * 64,
        "attempt_seq": 1,
        "run_seq": 0,
        "payload_sha256": receipts.native_payload_hash(cp),
    }
    previous = {
        "phase": "blocked",
        "reason": "live checkpoint reports failure",
        "before_run_id": cp["run_id"],
        "cli_rc": 130,
        "pod_uid": "synthetic-pod",
        "job_uid": "synthetic-job",
        "blocked_at": "2026-01-01T01:00:00Z",
    }
    log = (
        "RuntimeError: semantic_link similarity backend changed after its checkpoint\n"
        "HOURLY_BLOCKED live checkpoint reports failure\n"
    )
    approval = {
        "approved_at": "2026-01-01T02:00:00Z",
        "checkpoint_sha256": receipts.checkpoint_hash(cp),
        "previous_receipt_sha256": receipts.checkpoint_hash(previous),
        "previous_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
        "first_blocked_sha256": receipts.checkpoint_hash(previous),
        "other_states_sha256": receipts.checkpoint_hash(
            {k: v for k, v in states.items() if k != "semantic_link"}
        ),
        "counters_sha256": receipts.checkpoint_hash(cp["counters"]),
        "source_state_ref": ref.copy(),
        "source_state_sha256": receipts.checkpoint_hash(
            {
                "source_token": "synthetic-token",
                "legacy_checkpoint": {"similarity_backend": "numpy_float32:2.5.4"},
            }
        ),
        "source_token": "synthetic-token",
        "source_backend": "numpy_float32:2.5.4",
        "resource_policy_sha256": "c" * 64,
        "resource_baseline": [["synthetic-db", '[["db","container",0]]'], 0, 0],
        "runtime": {
            "source_sha": "d" * 40,
            "image_digest": "sha256:" + "e" * 64,
            "package_version": "1.0",
            "numpy_version": "2.5.4",
            "python_version": "3.12.1",
            "selected_backend": "numpy_float32:2.5.4",
        },
        "no_active_worker_verified": True,
        "no_active_lease_verified": True,
        "failed_pod": {"pod_uid": "synthetic-pod", "job_uid": "synthetic-job", "exit_code": 2},
    }
    return cp, previous, log, approval


@pytest.fixture
def evidence():
    cp, previous, log, approval = runtime_resume_fixture()
    return previous, log, cp, approval


def seed(evidence):
    previous, log, cp, approval = evidence
    return receipts.failed_runtime_resume_seed(previous, log, previous, cp, approval=approval)


def test_approved_failure_resumes_without_changing_checkpoint(evidence):
    previous, _, cp, _ = evidence
    before = copy.deepcopy(cp)
    result = seed(evidence)
    assert receipts.failed_runtime_resume_matches(result, cp)
    assert receipts.select_command(result, cp, None)[-1] == "--resume"
    assert result["status"] == "failed" and result["cli_rc"] is None
    assert (
        previous["cli_rc"] == 130
        and result["failed_runtime_resume"]["approval"]["failed_pod"]["exit_code"] == 2
    )
    assert receipts.confirmed_success(result, cp) is None
    assert cp == before
    with pytest.raises(receipts.BlockedError):
        receipts.select_command({"phase": "blocked"}, cp, None)


@pytest.mark.parametrize(
    "path,value",
    [
        (("checkpoint_sha256",), "0" * 64),
        (("source_state_ref", "state_id"), "other"),
        (("runtime", "selected_backend"), "other"),
        (("runtime", "python_version"), "3.11.9"),
        (("no_active_worker_verified",), False),
        (("no_active_lease_verified",), False),
        (("resource_baseline",), [["old", "[]"], 0, 0]),
        (("failed_pod", "pod_uid"), "other"),
    ],
)
def test_rejects_unapproved_or_incompatible_evidence(evidence, path, value):
    approval = evidence[3]
    target = approval
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(receipts.BlockedError):
        seed(evidence)


@pytest.mark.parametrize(
    "field,value",
    [("run_id", "other"), ("options_fingerprint", "other"), ("last_error", "unrelated error")],
)
def test_rejects_checkpoint_drift(evidence, field, value):
    result = seed(evidence)
    cp = copy.deepcopy(evidence[2])
    cp[field] = value
    assert not receipts.failed_runtime_resume_matches(result, cp)
    with pytest.raises(receipts.BlockedError):
        receipts.select_command(result, cp, None)


def test_rejects_unrelated_log_even_if_hash_is_approved(evidence):
    previous, _, cp, approval = evidence
    log = "unrelated failure\nHOURLY_BLOCKED live checkpoint reports failure\n"
    approval["previous_log_sha256"] = hashlib.sha256(log.encode()).hexdigest()
    with pytest.raises(receipts.BlockedError):
        receipts.failed_runtime_resume_seed(previous, log, previous, cp, approval=approval)

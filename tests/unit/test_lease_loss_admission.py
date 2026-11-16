"""Reviewed one-shot recovery from a blocked lease-loss receipt."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.engine import consolidation_receipts as receipts
from tests.unit.test_capacity_stop_admission import (
    evidence as capacity_evidence,
)
from tests.unit.test_capacity_stop_admission import (
    operator_fixture,
)
from tests.unit.test_capacity_stop_admission import (
    synthetic_runtime_versions as synthetic_runtime_versions,
)
from tests.unit.test_cluster_consolidation_hourly import hourly


def lease_loss_evidence():
    raw, _, cp, approval = capacity_evidence()
    previous = json.loads(raw)
    previous["reason"] = "engine lease absent or lost"
    previous.pop("capacity_stop_reason", None)
    previous["resource_policy_sha256"] = approval["resource_policy_sha256"]
    previous["cli_output_tail"] = "Resuming saved consolidation checkpoint for brain default...\n"
    previous["cli_output_sha256"] = hashlib.sha256(previous["cli_output_tail"].encode()).hexdigest()
    cp["status"] = "running"
    cp["strategy_states"]["prune"]["counters"]["prune_atomic_journal"] = 1
    cp["native_handoff"]["payload_sha256"] = receipts.native_payload_hash(cp)
    raw = receipts.canonical(previous)
    log = previous["cli_output_tail"] + "HOURLY_BLOCKED engine lease absent or lost\n"
    brain = Brain(id="synthetic-brain-uuid", name="default", config=BrainConfig())
    approval.update(
        lease_loss_resume_version=1,
        checkpoint_sha256=receipts.checkpoint_hash(cp),
        previous_receipt_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        previous_log_sha256=hashlib.sha256(log.encode()).hexdigest(),
        original_engine_outcome_unknown=True,
        brain_id=brain.id,
        brain_config_sha256=receipts.checkpoint_hash(asdict(brain.config)),
    )
    return raw, log, cp, approval, brain


def seed(evidence):
    raw, log, cp, approval, _ = evidence
    return receipts.lease_loss_resume_seed(raw, log, cp, approval=approval)


def operator(tmp_path):
    api, storage, safety, args, cm, _, _ = operator_fixture(tmp_path)
    raw, log, cp, approval, brain = lease_loss_evidence()
    cm["data"]["receipt.json"] = raw
    api.previous_log.return_value = log
    storage.get_consolidation_progress.return_value = cp
    storage.get_brain = AsyncMock(return_value=brain)
    Path(args.approval_file).write_text(json.dumps(approval))
    return api, storage, safety, args, cm, cp, approval


def test_approved_seed_preserves_running_checkpoint_and_archive():
    evidence = lease_loss_evidence()
    before = copy.deepcopy(evidence[2])
    result = seed(evidence)
    assert evidence[2] == before
    assert receipts.lease_loss_resume_matches(result, before)
    assert result["source"] == "operator_lease_loss_resume"
    assert result["cli_rc"] is None
    assert result["lease_loss_resume"]["previous_receipt_raw"] == evidence[0]
    assert receipts.select_command(result, before, None)[-1] == "--resume"


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sha256",
        "previous_receipt_raw_sha256",
        "previous_log_sha256",
        "run_id",
        "options_fingerprint",
        "brain_config_sha256",
    ],
)
def test_unapproved_binding_is_rejected(field):
    evidence = lease_loss_evidence()
    evidence[3][field] = "different"
    with pytest.raises(receipts.BlockedError):
        seed(evidence)


@pytest.mark.parametrize(
    "field",
    [
        "source_fix_reviewed",
        "original_engine_outcome_unknown",
        "no_active_worker_verified",
        "no_active_lease_verified",
    ],
)
def test_review_acknowledgements_are_required(field):
    evidence = lease_loss_evidence()
    evidence[3][field] = False
    with pytest.raises(receipts.BlockedError):
        seed(evidence)


@pytest.mark.parametrize("change", ["status", "run_id", "pending", "counter", "handoff"])
def test_seed_rejects_checkpoint_drift(change):
    evidence = lease_loss_evidence()
    result = seed(evidence)
    cp = copy.deepcopy(evidence[2])
    if change == "status":
        cp["status"] = "paused"
    elif change == "run_id":
        cp["run_id"] = "different"
    elif change == "pending":
        cp["strategy_states"]["prune"]["pending"].pop()
    elif change == "counter":
        cp["strategy_states"]["prune"]["counters"]["prune_atomic_journal"] = 0
    else:
        cp["native_handoff"]["attempt_seq"] += 1
    assert not receipts.lease_loss_resume_matches(result, cp)


async def test_admission_archives_and_cas_without_checkpoint_write(tmp_path):
    api, storage, safety, args, cm, cp, _ = operator(tmp_path)
    before = copy.deepcopy(cp)
    previous = cm["data"]["receipt.json"]
    assert await hourly.admit_lease_loss_resume(api, storage, safety, args) == 0
    assert cp == before
    admitted = json.loads(cm["data"]["receipt.json"])
    archive_key = "lease_loss_resume_" + receipts.checkpoint_hash(admitted) + ".json"
    assert cm["data"][archive_key] == receipts.canonical(admitted)
    assert admitted["lease_loss_resume"]["previous_receipt_raw"] == previous
    assert admitted["cli_rc"] is None


@pytest.mark.parametrize("fault", ["lease", "worker", "pressure", "image", "runtime", "cas"])
async def test_preflight_or_cas_failure_preserves_receipt(tmp_path, fault):
    api, storage, safety, args, cm, _, approval = operator(tmp_path)
    before = copy.deepcopy(cm)
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
    else:
        api.cas.side_effect = RuntimeError("409 conflict")
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_lease_loss_resume(api, storage, safety, args)
    assert cm == before


def test_explicit_native_failure_is_not_unknown_historical_outcome():
    raw, _, cp, approval, _ = lease_loss_evidence()
    previous = json.loads(raw)
    output = "Consolidation engine outcome: lease_lost; synthetic lease lost\n"
    previous["cli_output_tail"] = output
    previous["cli_output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    raw = receipts.canonical(previous)
    log = output + "HOURLY_BLOCKED engine lease absent or lost\n"
    approval["previous_receipt_raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    approval["previous_log_sha256"] = hashlib.sha256(log.encode()).hexdigest()
    with pytest.raises(receipts.BlockedError):
        receipts.lease_loss_resume_seed(raw, log, cp, approval=approval)


@pytest.mark.parametrize("mismatch", [None, "0" * 64])
def test_historical_receipt_may_omit_resource_policy_but_mismatch_is_rejected(mismatch):
    raw, log, cp, approval, brain = lease_loss_evidence()
    previous = json.loads(raw)
    previous.pop("resource_policy_sha256")
    raw = receipts.canonical(previous)
    approval["previous_receipt_raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    assert seed((raw, log, cp, approval, brain))
    previous["resource_policy_sha256"] = mismatch
    raw = receipts.canonical(previous)
    approval["previous_receipt_raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(receipts.BlockedError):
        seed((raw, log, cp, approval, brain))


async def test_brain_configuration_drift_blocks_preflight(tmp_path):
    api, storage, safety, args, cm, _, _ = operator(tmp_path)
    before = copy.deepcopy(cm)
    brain = storage.get_brain.return_value
    storage.get_brain.return_value = replace(
        brain, config=replace(brain.config, activation_strategy="different")
    )
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_lease_loss_resume(api, storage, safety, args)
    assert cm == before


async def test_admission_cannot_be_replayed(tmp_path):
    api, storage, safety, args, cm, cp, _ = operator(tmp_path)
    await hourly.admit_lease_loss_resume(api, storage, safety, args)
    admitted = copy.deepcopy(cm)
    before = copy.deepcopy(cp)
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_lease_loss_resume(api, storage, safety, args)
    assert cm == admitted and cp == before


async def test_missing_archive_never_launches_resume(tmp_path, monkeypatch):
    api, storage, safety, args, cm, _, _ = operator(tmp_path)
    await hourly.admit_lease_loss_resume(api, storage, safety, args)
    admitted = json.loads(cm["data"]["receipt.json"])
    del cm["data"]["lease_loss_resume_" + receipts.checkpoint_hash(admitted) + ".json"]
    launches = []
    monkeypatch.setattr(hourly.subprocess, "Popen", lambda *a, **k: launches.append(a))
    assert await hourly.run(api, storage, safety) == 2
    assert launches == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--admit-lease-loss-resume"],
        ["--approval-file", "synthetic.json"],
        ["--admit-lease-loss-resume", "--admit-capacity-stop", "--approval-file", "synthetic.json"],
        ["--admit-lease-loss-resume", "--approval-file", "synthetic.json", "--confirm-run-id", "x"],
    ],
)
async def test_admission_arguments_fail_before_database_access(argv, monkeypatch):
    from surreal_memory import unified_config

    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
    monkeypatch.setattr(unified_config, "get_config", lambda **kwargs: None)
    with pytest.raises(SystemExit) as exc:
        await hourly.main(argv)
    assert exc.value.code == 2


def lease_loss_v2_evidence():
    old = lease_loss_evidence()
    prior = seed(old)
    prior_raw = receipts.canonical(prior)
    prior_hash = receipts.checkpoint_hash(prior)
    raw, _, original_cp, approval, brain = old
    cp = copy.deepcopy(original_cp)
    cp["owner_token"] = "synthetic-new-owner"  # noqa: S105 - synthetic identity
    cp["native_handoff"]["attempt_seq"] += 1
    cp["native_handoff"]["payload_sha256"] = receipts.native_payload_hash(cp)
    previous = json.loads(raw)
    previous.update(
        reason="failed runtime resume native claim differs",
        cli_rc=1,
        before_sha256=receipts.checkpoint_hash(original_cp),
        lease_loss_resume_sha256=prior_hash,
        blocked_at="2026-01-01T02:30:00Z",
        pod_name="synthetic-v2-pod",
        pod_uid="synthetic-v2-pod-uid",
        job_name="synthetic-v2-job",
        job_uid="synthetic-v2-job-uid",
    )
    message = (
        "Consolidation stopped safely in prune: lease-fenced page outcome is unconfirmed; "
        "this session cannot replay or checkpoint"
    )
    output = (
        f"Consolidation engine outcome: lease_lost; {message}\n"
        f"  {message}\n"
        "  Consolidation status: lease_lost\n"
        "  Stages failed: prune (progress coordination: ConsolidationCheckpointUnconfirmedError)\n"
    )
    previous["cli_output_tail"] = output
    previous["cli_output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    raw = receipts.canonical(previous)
    log = output + "HOURLY_BLOCKED failed runtime resume native claim differs\n"
    approval.pop("original_engine_outcome_unknown")
    approval.update(
        lease_loss_resume_version=2,
        approved_at="2026-01-01T03:00:00Z",
        checkpoint_sha256=receipts.checkpoint_hash(cp),
        previous_receipt_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        previous_log_sha256=hashlib.sha256(log.encode()).hexdigest(),
        known_engine_status="lease_lost",
        known_engine_error="ConsolidationCheckpointUnconfirmedError",
        transaction_commit_outcome_unknown=True,
        original_cause_unknown=True,
        prior_admission_sha256=prior_hash,
        failed_pod={
            **{k: previous[k] for k in ("pod_name", "pod_uid", "job_name", "job_uid")},
            "exit_code": 2,
        },
    )
    return raw, log, cp, approval, brain, prior_raw


def v2_seed(evidence):
    raw, log, cp, approval, _, prior_raw = evidence
    return receipts.lease_loss_resume_seed(
        raw, log, cp, approval=approval, previous_intent_raw=prior_raw
    )


def test_v2_preserves_known_failure_and_unchanged_checkpoint():
    evidence = lease_loss_v2_evidence()
    cp_before = copy.deepcopy(evidence[2])
    result = v2_seed(evidence)
    assert evidence[2] == cp_before
    assert receipts.lease_loss_resume_matches(result, cp_before)
    assert result["lease_loss_resume"]["version"] == 2
    assert result["lease_loss_resume"]["previous_intent_raw"] == evidence[5]
    assert json.loads(result["lease_loss_resume"]["previous_receipt_raw"])["cli_rc"] == 1
    assert result["cli_rc"] is None
    assert receipts.select_command(result, cp_before, None)[-1] == "--resume"


@pytest.mark.parametrize(
    "field,value",
    [
        ("known_engine_status", "failed"),
        ("known_engine_error", "OtherError"),
        ("transaction_commit_outcome_unknown", False),
        ("original_cause_unknown", False),
        ("prior_admission_sha256", "0" * 64),
        ("checkpoint_sha256", "0" * 64),
        ("run_id", "other"),
        ("options_fingerprint", "other"),
    ],
)
def test_v2_rejects_mismatched_review_or_bindings(field, value):
    evidence = lease_loss_v2_evidence()
    evidence[3][field] = value
    with pytest.raises(receipts.BlockedError):
        v2_seed(evidence)


@pytest.mark.parametrize("fault", ["missing", "noncanonical", "tampered", "v1_ack"])
def test_v2_rejects_broken_prior_intent_and_v1_ack(fault):
    evidence = list(lease_loss_v2_evidence())
    if fault == "missing":
        evidence[5] = None
    elif fault == "noncanonical":
        evidence[5] += " "
    elif fault == "tampered":
        prior = json.loads(evidence[5])
        prior["run_id"] = "different"
        evidence[5] = receipts.canonical(prior)
    else:
        evidence[3]["original_engine_outcome_unknown"] = True
    with pytest.raises(receipts.BlockedError):
        v2_seed(evidence)


@pytest.mark.parametrize(
    "fault",
    [
        "cli130",
        "wrong_reason",
        "missing_line",
        "duplicate",
        "extra_error",
        "status_completed",
        "status_paused",
        "outcome_completed",
    ],
)
def test_v2_rejects_other_failures_even_if_hashes_are_reapproved(fault):
    evidence = list(lease_loss_v2_evidence())
    previous = json.loads(evidence[0])
    output = previous["cli_output_tail"]
    if fault == "cli130":
        previous["cli_rc"] = 130
    elif fault == "wrong_reason":
        previous["reason"] = "other failure"
    elif fault == "missing_line":
        output = output.replace("  Consolidation status: lease_lost\n", "")
    elif fault == "duplicate":
        output += "  Consolidation status: lease_lost\n"
    elif fault == "status_completed":
        output += "  Consolidation status: completed\n"
    elif fault == "status_paused":
        output += "  Consolidation status: paused\n"
    elif fault == "outcome_completed":
        output += "Consolidation engine outcome: completed\n"
    else:
        output += "RuntimeError: unrelated synthetic error\n"
    previous["cli_output_tail"] = output
    previous["cli_output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    evidence[0] = receipts.canonical(previous)
    evidence[1] = output + "HOURLY_BLOCKED " + previous["reason"] + "\n"
    evidence[3]["previous_receipt_raw_sha256"] = hashlib.sha256(evidence[0].encode()).hexdigest()
    evidence[3]["previous_log_sha256"] = hashlib.sha256(evidence[1].encode()).hexdigest()
    with pytest.raises(receipts.BlockedError):
        v2_seed(evidence)


def v2_operator(tmp_path):
    api, storage, safety, args, cm, _, _ = operator(tmp_path)
    raw, log, cp, approval, brain, prior_raw = lease_loss_v2_evidence()
    cm["data"]["receipt.json"] = raw
    cm["data"]["lease_loss_resume_" + approval["prior_admission_sha256"] + ".json"] = prior_raw
    api.previous_log.return_value = log
    storage.get_consolidation_progress.return_value = cp
    storage.get_brain.return_value = brain
    Path(args.approval_file).write_text(json.dumps(approval))
    return api, storage, safety, args, cm, cp, approval


async def test_v2_operator_admission_cas_preserves_known_failure_and_prior(tmp_path):
    api, storage, safety, args, cm, cp, approval = v2_operator(tmp_path)
    previous = cm["data"]["receipt.json"]
    prior_key = "lease_loss_resume_" + approval["prior_admission_sha256"] + ".json"
    prior = cm["data"][prior_key]
    before = copy.deepcopy(cp)
    assert await hourly.admit_lease_loss_resume(api, storage, safety, args) == 0
    assert cp == before
    admitted = json.loads(cm["data"]["receipt.json"])
    assert admitted["lease_loss_resume"]["previous_receipt_raw"] == previous
    assert admitted["lease_loss_resume"]["previous_intent_raw"] == prior
    assert cm["data"][prior_key] == prior
    assert cm["data"][
        "lease_loss_resume_" + receipts.checkpoint_hash(admitted) + ".json"
    ] == receipts.canonical(admitted)
    assert json.loads(previous)["cli_rc"] == 1 and admitted["cli_rc"] is None
    saved = copy.deepcopy(cm)
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_lease_loss_resume(api, storage, safety, args)
    assert cm == saved


@pytest.mark.parametrize(
    "fault", ["prior_missing", "prior_changed", "lease", "worker", "brain", "runtime", "cas"]
)
async def test_v2_operator_rejects_stale_or_contended_admission(tmp_path, fault):
    api, storage, safety, args, cm, _, approval = v2_operator(tmp_path)
    key = "lease_loss_resume_" + approval["prior_admission_sha256"] + ".json"
    if fault == "prior_missing":
        del cm["data"][key]
    elif fault == "prior_changed":
        cm["data"][key] += " "
    elif fault == "lease":
        storage._query.return_value = [{"owner_token": "other"}]
    elif fault == "worker":
        api.no_other_jobs.side_effect = receipts.BlockedError("active worker")
    elif fault == "brain":
        brain = storage.get_brain.return_value
        storage.get_brain.return_value = replace(brain, id="other")
    elif fault == "runtime":
        approval["runtime"]["runner_sha256"] = "0" * 64
        Path(args.approval_file).write_text(json.dumps(approval))
    else:
        api.cas.side_effect = RuntimeError("synthetic CAS conflict")
    before = copy.deepcopy(cm)
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_lease_loss_resume(api, storage, safety, args)
    assert cm == before


async def test_v2_inconsistent_readback_reports_uncertainty_without_retry(tmp_path):
    api, storage, safety, args, cm, cp, _ = v2_operator(tmp_path)
    before = copy.deepcopy(cp)
    reads = 0

    def state():
        nonlocal reads
        reads += 1
        result = copy.deepcopy(cm)
        if reads == 4:
            result["data"]["unexpected.json"] = "synthetic drift"
        return result

    api.state.side_effect = state
    with pytest.raises(receipts.BlockedError, match="uncertain"):
        await hourly.admit_lease_loss_resume(api, storage, safety, args)
    assert api.cas.call_count == 1
    assert cp == before


async def test_v2_worker_rejects_missing_prior_archive_before_start(tmp_path, monkeypatch):
    api, storage, safety, args, cm, _, approval = v2_operator(tmp_path)
    assert await hourly.admit_lease_loss_resume(api, storage, safety, args) == 0
    del cm["data"]["lease_loss_resume_" + approval["prior_admission_sha256"] + ".json"]
    launches = []
    monkeypatch.setattr(hourly.subprocess, "Popen", lambda *a, **k: launches.append(a))
    assert await hourly.run(api, storage, safety) == 2
    assert launches == []


@pytest.mark.parametrize("fault", ["prior_run", "prior_checkpoint", "mixed_origin"])
def test_v2_rejects_rehashed_broken_intent_chain(fault):
    evidence = list(lease_loss_v2_evidence())
    previous = json.loads(evidence[0])
    prior = json.loads(evidence[5])
    if fault == "prior_run":
        prior["run_id"] = "different"
    elif fault == "prior_checkpoint":
        prior["checkpoint_sha256"] = "0" * 64
    else:
        previous["capacity_resume_sha256"] = "0" * 64
    prior_hash = receipts.checkpoint_hash(prior)
    evidence[5] = receipts.canonical(prior)
    previous["lease_loss_resume_sha256"] = prior_hash
    evidence[0] = receipts.canonical(previous)
    evidence[3]["prior_admission_sha256"] = prior_hash
    evidence[3]["previous_receipt_raw_sha256"] = hashlib.sha256(evidence[0].encode()).hexdigest()
    with pytest.raises(receipts.BlockedError):
        v2_seed(evidence)

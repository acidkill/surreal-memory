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

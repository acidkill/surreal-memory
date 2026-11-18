"""Offline native session to scheduled receipt handoff regressions."""

import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.engine import consolidation_receipts as receipts
from surreal_memory.engine.consolidation import ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressSession,
    options_fingerprint,
)
from surreal_memory.server.consolidation_status import summarize_consolidation
from tests.unit.test_cluster_consolidation_hourly import hourly
from tests.unit.test_consolidation_progress import FakeProgressStorage
from tests.unit.test_consolidation_status import configmap

REFERENCE = datetime(2026, 11, 13)
STRATEGIES = [
    strategy.value for strategy in ConsolidationStrategy if strategy != ConsolidationStrategy.ALL
]
FINGERPRINT = options_fingerprint({"strategies": STRATEGIES})


def managed_receipt(cp):
    return {
        "phase": "terminal",
        "status": cp["status"],
        "cli_rc": 1 if cp["status"] == "paused" else 0,
        "run_id": cp["run_id"],
        "checkpoint_sha256": receipts.checkpoint_hash(cp),
        "native_handoff_anchor": receipts.native_handoff_anchor(cp),
    }


def select(receipt, cp):
    return receipts.select_command(receipt, cp, receipts.MARKER + receipts.canonical(receipt))


async def open_session(storage, day=0):
    session = await ConsolidationProgressSession.open(
        storage, STRATEGIES, FINGERPRINT, REFERENCE + timedelta(days=day)
    )
    assert session is not None
    return session


@pytest.mark.asyncio
async def test_repeated_native_resume_and_fresh_runs_reach_scheduler():
    storage = FakeProgressStorage()
    first = await open_session(storage)
    await first.checkpoint("prune", "synapse_scan", cursor="synapse:1")
    await first.pause()
    await first.close()
    old = deepcopy(storage.progress)
    receipt = managed_receipt(old)
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    assert len(STRATEGIES) == 20
    fresh_state = deepcopy(old)
    fresh_state["run_id"] = "unexpected-fresh"
    with pytest.raises(receipts.BlockedError):
        receipts.prepare_native_handoff(old, fresh_state, fresh=True)
    root = old["native_handoff"]["lineage_root_sha256"]
    previous_attempt = old["native_handoff"]["attempt_seq"]
    for cursor in ("synapse:2", "synapse:3"):
        resumed = await open_session(storage)
        assert resumed.resumed
        await resumed.checkpoint("prune", "synapse_scan", cursor=cursor)
        await resumed.pause()
        await resumed.close()
        cp = deepcopy(storage.progress)
        assert cp["native_handoff"]["attempt_seq"] > previous_attempt
        assert cp["native_handoff"]["lineage_root_sha256"] == root
        assert select(receipt, cp)[-1] == "--resume"
        previous_attempt = cp["native_handoff"]["attempt_seq"]
    for run_seq in (1, 2):
        resumed = await open_session(storage)
        for strategy in STRATEGIES:
            await resumed.complete_strategy(strategy)
        await resumed.complete()
        await resumed.close()
        fresh = await open_session(storage, run_seq)
        assert not fresh.resumed
        await fresh.pause()
        await fresh.close()
        cp = deepcopy(storage.progress)
        assert cp["run_id"] != old["run_id"]
        assert cp["native_handoff"]["lineage_root_sha256"] == root
        assert cp["native_handoff"]["run_seq"] == run_seq
        assert select(receipt, cp)[-1] == "--resume"
    final = await open_session(storage)
    for strategy in STRATEGIES:
        await final.complete_strategy(strategy)
    await final.complete()
    await final.close()
    assert select(receipt, storage.progress) == ["smem", "consolidate", "--brain", "default"]
    assert receipt["cli_rc"] == 1
    assert receipts.confirmed_success(receipt, storage.progress) is None
    status = summarize_consolidation(
        storage.progress, configmap(receipt), checkpoint_available=True
    )
    assert status.state == "unknown"
    assert status.reason == "native_completed_external_cli_unknown"
    assert status.eligible_work is None
    assert status.last_confirmed_success_at is None


@pytest.mark.asyncio
async def test_unproved_or_tampered_native_progress_is_blocked():
    storage = FakeProgressStorage()
    first = await open_session(storage)
    await first.pause()
    await first.close()
    old = deepcopy(storage.progress)
    receipt = managed_receipt(old)
    resumed = await open_session(storage)
    await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:2")
    await resumed.pause()
    await resumed.close()
    current = deepcopy(storage.progress)
    assert select(receipt, current)[-1] == "--resume"
    for field, value in (
        ("lineage_root_sha256", "0" * 64),
        ("attempt_seq", 0),
        ("run_seq", -1),
        ("payload_sha256", "0" * 64),
    ):
        bad = deepcopy(current)
        bad["native_handoff"][field] = value
        with pytest.raises(receipts.BlockedError):
            select(receipt, bad)
    for field, value in (
        ("options_fingerprint", "different"),
        ("reference_time", "2020-01-01T00:00:00Z"),
    ):
        bad = deepcopy(current)
        bad[field] = value
        bad["native_handoff"]["payload_sha256"] = receipts.native_payload_hash(bad)
        with pytest.raises(receipts.BlockedError):
            select(receipt, bad)
    unproved = deepcopy(current)
    unproved.pop("native_handoff")
    with pytest.raises(receipts.BlockedError):
        select(receipt, unproved)


@pytest.mark.asyncio
async def test_active_native_session_defers_before_scheduler_selection():
    storage = FakeProgressStorage()
    first = await open_session(storage)
    await first.pause()
    await first.close()
    receipt = managed_receipt(deepcopy(storage.progress))
    active = await open_session(storage)
    api = MagicMock()
    api.state.return_value = {"data": {"receipt.json": receipts.canonical(receipt)}}
    api.identity.return_value = {"pod_name": "self"}
    api.previous_log.return_value = receipts.MARKER + receipts.canonical(receipt)
    backend = SimpleNamespace(
        get_consolidation_progress=AsyncMock(return_value=deepcopy(storage.progress)),
        _query=AsyncMock(return_value=[{"owner_token": active.owner_token}]),
    )
    try:
        with patch.object(hourly.subprocess, "Popen") as popen:
            assert await hourly.run(api, backend, MagicMock()) == 0
        popen.assert_not_called()
        assert api.cas.call_args.args[1] == receipt
        assert api.cas.call_args.args[2]["outcome"] == "deferred_active_lease"
    finally:
        await active.close()


@pytest.mark.asyncio
async def test_completed_strategy_regression_is_rejected_even_with_rebound_digest():
    storage = FakeProgressStorage()
    strategies = ["prune", "semantic_link"]
    fingerprint = options_fingerprint({"strategies": strategies})
    first = await ConsolidationProgressSession.open(storage, strategies, fingerprint, REFERENCE)
    assert first is not None
    await first.complete_strategy("prune")
    await first.pause()
    await first.close()
    receipt = managed_receipt(deepcopy(storage.progress))
    resumed = await ConsolidationProgressSession.open(storage, strategies, fingerprint, REFERENCE)
    assert resumed is not None
    await resumed.pause()
    await resumed.close()
    bad = deepcopy(storage.progress)
    bad["strategy_states"]["prune"]["phase"] = "starting"
    bad["native_handoff"]["payload_sha256"] = receipts.native_payload_hash(bad)
    with pytest.raises(receipts.BlockedError):
        select(receipt, bad)


@pytest.mark.asyncio
async def test_native_payload_storage_roundtrip_normalization_and_owner_binding():
    storage = FakeProgressStorage()
    session = await open_session(storage)
    await session.pause()
    await session.close()
    cp = deepcopy(storage.progress)
    digest = receipts.native_payload_hash(cp)
    roundtrip = deepcopy(cp)
    roundtrip["reference_time"] = (
        cp["reference_time"].replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    )
    roundtrip["started_at"] = cp["started_at"].replace(tzinfo=UTC)
    roundtrip["id"] = "consolidation_progress:synthetic"
    roundtrip["updated_at"] = "2030-01-01T00:00:00Z"
    roundtrip.pop("last_error", None)
    roundtrip.pop("cursor", None)
    assert receipts.native_payload_hash(roundtrip) == digest
    roundtrip["owner_token"] = cp["owner_token"] + "-different"
    assert receipts.native_payload_hash(roundtrip) != digest


@pytest.mark.asyncio
async def test_reviewed_bridge_archives_evidence_and_rejects_mutations():
    storage = FakeProgressStorage()
    session = await open_session(storage)
    for strategy in STRATEGIES:
        if strategy != "semantic_link":
            await session.complete_strategy(strategy)
    await session.checkpoint("semantic_link", "validation", counters={"semantic_link_failures": 0})
    await session.pause()
    await session.close()
    cp = deepcopy(storage.progress)
    cp["reference_time"] = cp["reference_time"].isoformat() + "Z"
    previous = managed_receipt(cp)
    previous.update(
        checkpoint_sha256="a" * 64,
        finished_at="2026-11-13T10:00:00Z",
        resource_baseline=[["synthetic", "[]"], 0],
        resource_policy_sha256="b" * 64,
    )
    log = (
        "  Consolidation status: paused\n  Stages timed out: semantic_link\n"
        + receipts.MARKER
        + receipts.canonical(previous)
    )
    blocked = {
        **previous,
        "phase": "blocked",
        "reason": "live checkpoint diverged from receipt",
        "blocked_at": "2026-11-13T10:01:00Z",
    }
    options = {
        "approved_at": "2026-11-13T10:02:00Z",
        "approved_checkpoint_sha256": receipts.checkpoint_hash(cp),
        "approved_other_states_sha256": receipts.checkpoint_hash(
            {k: v for k, v in cp["strategy_states"].items() if k != "semantic_link"}
        ),
    }
    seed = receipts.native_handoff_bridge_seed(previous, log, blocked, cp, **options)
    assert seed["cli_rc"] is None
    assert seed["native_handoff_bridge"]["previous_receipt"] == previous
    assert seed["native_handoff_bridge"]["first_blocked_receipt"] == blocked
    assert receipts.select_command(seed, cp, None)[-1] == "--resume"
    status = summarize_consolidation(
        cp,
        configmap(seed),
        checkpoint_available=True,
        now=datetime(2026, 11, 13, 10, 3, tzinfo=UTC),
    )
    assert status.state == "paused"
    for changes in (
        {"approved_checkpoint_sha256": "0" * 64},
        {"approved_other_states_sha256": "0" * 64},
        {"approved_at": "2026-11-13T09:59:00Z"},
    ):
        with pytest.raises(receipts.BlockedError):
            receipts.native_handoff_bridge_seed(
                previous, log, blocked, cp, **{**options, **changes}
            )
    with pytest.raises(receipts.BlockedError):
        receipts.native_handoff_bridge_seed(
            previous, log + "\nERROR synthetic", blocked, cp, **options
        )
    with pytest.raises(receipts.BlockedError):
        receipts.native_handoff_bridge_seed(
            previous, log, {**blocked, "reason": "different"}, cp, **options
        )


@pytest.mark.asyncio
async def test_failed_runtime_approval_remains_failed_until_native_claim_clears_error():
    from tests.unit.test_failed_runtime_resume import runtime_resume_fixture

    cp, previous, log, approval = runtime_resume_fixture()
    seed = receipts.failed_runtime_resume_seed(previous, log, previous, cp, approval=approval)
    before = deepcopy(cp)
    status = summarize_consolidation(
        cp,
        configmap(seed),
        checkpoint_available=True,
        now=datetime(2026, 1, 1, 2, 1, tzinfo=UTC),
    )
    assert status.state == "failed"
    assert status.reason == "runtime_resume_pending"
    assert receipts.confirmed_success(seed, cp) is None
    assert cp == before

    storage = FakeProgressStorage()
    session = await open_session(storage)
    for strategy in STRATEGIES:
        if strategy != "semantic_link":
            await session.complete_strategy(strategy)
    await session.checkpoint(
        "semantic_link",
        "semantic_link_discovery_validation",
        counters={"semantic_link_failures": 0},
    )
    await session.fail("semantic_link (progress coordination: ConsolidationProgressError)")
    await session.close()
    failed = deepcopy(storage.progress)
    resumed = await open_session(storage)
    running = deepcopy(storage.progress)
    assert running["status"] == "running"
    assert running["last_error"] is None
    assert running["owner_token"] != failed["owner_token"]
    for key in (
        "run_id",
        "reference_time",
        "completed_strategies",
        "strategy_states",
        "counters",
        "work_progress_seq",
    ):
        assert running.get(key) == failed.get(key)
    assert receipts.native_handoff_valid(running)
    await resumed.close()


def completed_semantic_fixture():
    cp = {
        "brain_id": "default",
        "run_id": "synthetic-semantic",
        "status": "completed",
        "phase": "completed",
        "last_error": None,
        "current_strategy": None,
        "cursor": None,
        "options_fingerprint": "synthetic-semantic-options",
        "reference_time": "2026-01-01T00:00:00Z",
        "requested_strategies": ["semantic_link"],
        "completed_strategies": ["semantic_link"],
        "schema_version": 12,
        "format_version": 1,
        "engine_version": "3.11.0:checkpoint-v1",
        "strategy_states": {"semantic_link": {"phase": "completed", "counters": {}}},
        "counters": {},
    }
    cp["native_handoff"] = {
        "version": 1,
        "lineage_root_sha256": "a" * 64,
        "attempt_seq": 1,
        "run_seq": 0,
        "payload_sha256": receipts.native_payload_hash(cp),
    }
    identity = {
        k: cp[k]
        for k in (
            "brain_id",
            "run_id",
            "options_fingerprint",
            "reference_time",
            "schema_version",
            "engine_version",
            "format_version",
        )
    }
    cp_hash = receipts._native_digest(cp)
    tail = "[frozen candidate coverage complete]"
    semantic_receipt = {
        "mode": "semantic_link",
        "status": "completed",
        "phase": "terminal",
        "identity": identity,
        "checkpoint_sha256": cp_hash,
        "finished_at": "2026-01-01T01:00:00Z",
        "attempts": [
            {
                "cli_rc": 0,
                "status": "completed",
                "checkpoint_sha256": cp_hash,
                "output_tail": tail,
                "output_sha256": hashlib.sha256(tail.encode()).hexdigest(),
            }
        ],
    }
    previous = {"phase": "blocked", "cli_rc": 130, "reason": "synthetic first failure"}
    observation = {"outcome": "blocked", "reason": "synthetic observation"}
    previous_raw = json.dumps(previous, sort_keys=True)
    observation_raw = json.dumps(observation, sort_keys=True)
    semantic_raw = json.dumps(semantic_receipt, sort_keys=True)
    log = "synthetic terminal output\n" + tail

    def sha(value):
        return hashlib.sha256(value.encode()).hexdigest()

    approval = {
        "version": 1,
        "kind": "completed_semantic_to_fresh_all",
        "approved_at": "2026-01-01T02:00:00Z",
        "previous_receipt_sha256": sha(previous_raw),
        "observation_sha256": sha(observation_raw),
        "checkpoint_sha256": cp_hash,
        "identity_sha256": receipts._native_digest(identity),
        "semantic_receipt_sha256": sha(semantic_raw),
        "semantic_log_sha256": sha(log),
        "resource_baseline": [["synthetic-db", '[["db","container",0]]'], 0, 0],
        "memory_limit_bytes": 18 * 1024**3,
        "brain_uuid": "synthetic-brain",
        "brain_config_sha256": "b" * 64,
        "state_resource_version": "synthetic-rv",
        "runtime": {
            "source_sha": "c" * 40,
            "image_digest": "sha256:" + "d" * 64,
            "package_version": "1.0",
            "numpy_version": "2.5.4",
            "python_version": "3.12.1",
            "selected_backend": "numpy_float32:2.5.4",
        },
        "semantic_source": {
            "state_name": "synthetic-state",
            "state_uid": "synthetic-state-uid",
            "job_name": "synthetic-job",
            "job_uid": "synthetic-job-uid",
            "pod_name": "synthetic-pod",
            "pod_uid": "synthetic-pod-uid",
            "image_digest": "sha256:" + "e" * 64,
        },
    }
    return cp, previous_raw, observation_raw, semantic_raw, log, approval


def test_completed_semantic_seed_archives_history_and_cannot_be_replayed_with_drift():
    cp, previous_raw, observation_raw, semantic_raw, log, approval = completed_semantic_fixture()
    original = deepcopy(cp)
    seed = receipts.completed_semantic_handoff_seed(
        previous_raw,
        cp,
        approval=approval,
        observation_raw=observation_raw,
        semantic_receipt_raw=semantic_raw,
        semantic_log=log,
    )
    assert cp == original
    assert seed["phase"] == "bootstrap"
    assert seed["status"] == "completed"
    assert seed["cli_rc"] is None
    assert seed["next_action"] == "fresh_all"
    assert seed["expected_all_options_fingerprint"] == receipts.hourly_options_fingerprint()
    assert receipts.completed_semantic_handoff_matches(seed, cp)
    archive = seed["completed_semantic_handoff"]
    assert archive["previous_raw"] == previous_raw
    assert archive["observation_raw"] == observation_raw
    altered = deepcopy(cp)
    altered["run_id"] = "synthetic-other"
    assert not receipts.completed_semantic_handoff_matches(seed, altered)


@pytest.mark.parametrize(
    "fault", ["prior", "observation", "receipt", "log", "checkpoint", "approval"]
)
def test_completed_semantic_seed_refuses_stale_evidence(fault):
    cp, previous, observation, semantic, log, approval = completed_semantic_fixture()
    if fault == "prior":
        previous += " "
    elif fault == "observation":
        observation += " "
    elif fault == "receipt":
        semantic += " "
    elif fault == "log":
        log += " "
    elif fault == "checkpoint":
        cp["run_id"] = "synthetic-drift"
    else:
        approval["resource_baseline"] = None
    with pytest.raises(receipts.BlockedError):
        receipts.completed_semantic_handoff_seed(
            previous,
            cp,
            approval=approval,
            observation_raw=observation,
            semantic_receipt_raw=semantic,
            semantic_log=log,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "cli_rc",
        "receipt_status",
        "mode",
        "coverage",
        "tail",
        "log_error",
        "paused_checkpoint",
        "native_handoff",
    ],
)
def test_semantic_handoff_refuses_invalid_completion_even_with_matching_approval(fault):
    cp, previous, observation, semantic_raw, log, approval = completed_semantic_fixture()
    semantic = json.loads(semantic_raw)
    if fault == "cli_rc":
        semantic["attempts"][-1]["cli_rc"] = 1
    elif fault == "receipt_status":
        semantic["status"] = "paused"
    elif fault == "mode":
        semantic["mode"] = "all"
    elif fault == "coverage":
        log = "synthetic log without coverage marker"
    elif fault == "tail":
        semantic["attempts"][-1]["output_tail"] = "absent from log"
    elif fault == "log_error":
        semantic["attempts"][-1]["output_tail"] += "\nERROR synthetic failure"
        log += "\nERROR synthetic failure"
        semantic["attempts"][-1]["output_sha256"] = hashlib.sha256(
            semantic["attempts"][-1]["output_tail"].encode()
        ).hexdigest()
    elif fault == "paused_checkpoint":
        cp.update(status="paused", completed_strategies=[], phase="semantic_link_discovery")
        cp["native_handoff"]["payload_sha256"] = receipts.native_payload_hash(cp)
        approval["checkpoint_sha256"] = receipts._native_digest(cp)
    else:
        cp["native_handoff"]["payload_sha256"] = "0" * 64
        approval["checkpoint_sha256"] = receipts._native_digest(cp)
    semantic_raw = json.dumps(semantic, sort_keys=True)
    approval["semantic_receipt_sha256"] = hashlib.sha256(semantic_raw.encode()).hexdigest()
    approval["semantic_log_sha256"] = hashlib.sha256(log.encode()).hexdigest()
    with pytest.raises(receipts.BlockedError):
        receipts.completed_semantic_handoff_seed(
            previous,
            cp,
            approval=approval,
            observation_raw=observation,
            semantic_receipt_raw=semantic_raw,
            semantic_log=log,
        )


@pytest.mark.parametrize("outcome", ["success", "cas_conflict", "lost_response"])
def test_completed_semantic_admission_cas_preserves_history_without_retry(tmp_path, outcome):
    cp, previous, observation, semantic, log, approval = completed_semantic_fixture()
    approval_file = tmp_path / "approval.json"
    approval_file.write_text(json.dumps(approval))
    history = '{"synthetic":"previous-success"}'
    original = {
        "metadata": {"resourceVersion": approval["state_resource_version"]},
        "data": {
            "receipt.json": previous,
            "last_observation.json": observation,
            "last_confirmed_success.json": history,
        },
    }
    stored = deepcopy(original)
    api = MagicMock(namespace="synthetic")
    api.path.side_effect = lambda resource: resource
    api.request.side_effect = lambda path: (
        {"spec": {"suspend": True}} if "cronjobs/" in path else {"data": {"receipt.json": semantic}}
    )
    api.previous_log.return_value = log
    api.state.side_effect = lambda: deepcopy(stored)

    def cas(cm, seed):
        if outcome == "cas_conflict":
            stored["data"]["receipt.json"] = '{"phase":"blocked","reason":"competing writer"}'
            raise RuntimeError("synthetic conflict")
        stored["data"] = dict(cm["data"], **{"receipt.json": hourly.canonical(seed)})
        if outcome == "lost_response":
            raise TimeoutError("synthetic lost acknowledgement")

    api.cas.side_effect = cas
    safety = MagicMock()
    args = SimpleNamespace(approval_file=str(approval_file))
    with (
        patch.object(hourly, "live", AsyncMock(return_value=(cp, []))),
        patch.object(hourly, "verify_completed_semantic_handoff", AsyncMock()),
    ):
        if outcome == "success":
            assert (
                asyncio.run(hourly.admit_completed_semantic_handoff(api, None, safety, args)) == 0
            )
            assert stored["data"]["last_confirmed_success.json"] == history
            assert "last_observation.json" not in stored["data"]
            assert any(key.startswith("completed_semantic_observation_") for key in stored["data"])
            assert any(key.startswith("completed_semantic_handoff_") for key in stored["data"])
        else:
            with pytest.raises(hourly.BlockedError, match="outcome uncertain"):
                asyncio.run(hourly.admit_completed_semantic_handoff(api, None, safety, args))
            if outcome == "cas_conflict":
                assert (
                    stored["data"]["receipt.json"]
                    == '{"phase":"blocked","reason":"competing writer"}'
                )
                assert stored["data"]["last_confirmed_success.json"] == history
            else:
                assert stored["data"]["last_confirmed_success.json"] == history
                with pytest.raises(hourly.BlockedError):
                    asyncio.run(hourly.admit_completed_semantic_handoff(api, None, safety, args))
        assert api.cas.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "state_uid",
        "state_receipt",
        "log",
        "image",
        "image_id",
        "exit",
        "restart",
        "checkpoint",
        "lease",
    ],
)
def test_completed_semantic_verifier_rechecks_source_and_current_state(change):
    cp, previous, observation, semantic, log, approval = completed_semantic_fixture()
    seed = receipts.completed_semantic_handoff_seed(
        previous,
        cp,
        approval=approval,
        semantic_receipt_raw=semantic,
        semantic_log=log,
        observation_raw=observation,
    )
    source = approval["semantic_source"]
    state = {"metadata": {"uid": source["state_uid"]}, "data": {"receipt.json": semantic}}
    pod = {
        "spec": {"containers": [{"image": "example@" + source["image_digest"]}]},
        "status": {
            "containerStatuses": [
                {
                    "imageID": "example@" + source["image_digest"],
                    "restartCount": 0,
                    "state": {"terminated": {"exitCode": 0}},
                }
            ]
        },
    }
    current, leases = deepcopy(cp), []
    if change == "state_uid":
        state["metadata"]["uid"] = "changed"
    if change == "state_receipt":
        state["data"]["receipt.json"] = "changed"
    if change == "log":
        log += " changed"
    if change == "image":
        pod["spec"]["containers"][0]["image"] = "changed"
    status = pod["status"]["containerStatuses"][0]
    if change == "image_id":
        status["imageID"] = "changed"
    if change == "exit":
        status["state"]["terminated"]["exitCode"] = 1
    if change == "restart":
        status["restartCount"] = 1
    if change == "checkpoint":
        current["run_id"] = "changed"
    if change == "lease":
        leases = [{"owner": "synthetic"}]
    api = MagicMock()
    api.path.side_effect = lambda value: value
    api.request.side_effect = lambda value: state if value.startswith("configmaps/") else pod
    api.previous_log.return_value = log
    safety = MagicMock()
    with (
        patch.object(hourly, "live", AsyncMock(return_value=(current, leases))),
        patch.object(hourly, "_verify_handoff_brain", AsyncMock()),
        patch.object(hourly, "_verify_reviewed_resume_runtime") as runtime,
    ):
        if change == "none":
            asyncio.run(hourly.verify_completed_semantic_handoff(api, None, safety, seed, cp))
            runtime.assert_called_once_with(
                api, safety, seed, approval, label="completed semantic handoff"
            )
            api.previous_log.assert_called_once_with(source)
        else:
            with pytest.raises(hourly.BlockedError):
                asyncio.run(hourly.verify_completed_semantic_handoff(api, None, safety, seed, cp))


def test_handoff_brain_rejects_config_drift():
    from dataclasses import dataclass

    @dataclass
    class Config:
        enabled: bool = True

    config = Config()
    approval = {
        "brain_uuid": "synthetic",
        "brain_config_sha256": receipts._native_digest({"enabled": True}),
    }
    storage = SimpleNamespace(
        get_brain=AsyncMock(
            return_value=SimpleNamespace(name="default", id="synthetic", config=config)
        )
    )
    asyncio.run(hourly._verify_handoff_brain(storage, approval))
    config.enabled = False
    with pytest.raises(hourly.BlockedError):
        asyncio.run(hourly._verify_handoff_brain(storage, approval))


def test_handoff_flag_uses_operator_storage_and_never_runs_cli():
    storage = MagicMock()
    storage._reconnect = AsyncMock()
    storage.close = AsyncMock()
    with (
        patch("surreal_memory.storage.surrealdb.SurrealDBStorage", return_value=storage),
        patch.object(hourly, "Kubernetes"),
        patch.object(hourly, "Safety"),
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock()) as shared,
        patch("surreal_memory.unified_config.get_config"),
        patch.dict("os.environ", {"SURREAL_MEMORY_STORAGE": "surrealdb"}),
        patch.object(hourly, "run", AsyncMock()) as run,
        patch.object(
            hourly, "admit_completed_semantic_handoff", AsyncMock(return_value=0)
        ) as admit,
    ):
        assert (
            asyncio.run(
                hourly.main(
                    ["--admit-completed-semantic-handoff", "--approval-file", "synthetic.json"]
                )
            )
            == 0
        )
        admit.assert_awaited_once()
        shared.assert_not_awaited()
        run.assert_not_awaited()
        storage._reconnect.assert_awaited_once()
        storage.close.assert_awaited_once()


def test_semantic_final_attempt_allows_only_exact_health_advisory():
    cp, previous, observation, raw, _, approval = completed_semantic_fixture()
    semantic = json.loads(raw)
    output = "[frozen candidate coverage complete]\n  Health delta unavailable: diagnostic snapshot failed"
    semantic["attempts"][-1]["output_tail"] = output
    semantic["attempts"][-1]["output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    raw = json.dumps(semantic, sort_keys=True)
    log = "prior attempt: ERROR synthetic prior failure\n" + output
    approval["semantic_receipt_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    approval["semantic_log_sha256"] = hashlib.sha256(log.encode()).hexdigest()
    seed = receipts.completed_semantic_handoff_seed(
        previous,
        cp,
        approval=approval,
        observation_raw=observation,
        semantic_receipt_raw=raw,
        semantic_log=log,
    )
    assert seed["next_action"] == "fresh_all"


def test_hourly_cli_native_engine_fingerprint_at_progress_open():
    from surreal_memory.cli.commands import tools as cli_tools
    from surreal_memory.engine import consolidation as engine_module
    from surreal_memory.engine import consolidation_delta

    class StopBeforeProgress(BaseException):
        pass

    captured = []
    budgets = []
    storage = FakeProgressStorage()

    async def capture_open(store, strategies, fingerprint, reference_time, **kwargs):
        assert store is storage
        captured.append((strategies, fingerprint))
        raise StopBeforeProgress()

    async def run_delta(store, brain_id, **kwargs):
        assert store is storage and brain_id == "default"
        assert kwargs["strategies"] == [ConsolidationStrategy.ALL]
        budgets.append((kwargs["strategy_timeout_seconds"], kwargs["total_timeout_seconds"]))
        config = kwargs["config"]
        assert (
            config.prune_weight_threshold,
            config.prune_min_inactive_days,
            config.merge_overlap_threshold,
        ) == (0.05, 7.0, 0.5)
        engine = engine_module.ConsolidationEngine(store, config)
        assert engine._dream_decay_multiplier == 10.0 and engine._tier_config is None
        await engine.run(
            strategies=kwargs["strategies"],
            dry_run=kwargs["dry_run"],
            strategy_timeout_seconds=kwargs["strategy_timeout_seconds"],
            total_timeout_seconds=kwargs["total_timeout_seconds"],
        )
        pytest.fail("engine continued beyond progress-open boundary")

    with (
        patch.object(cli_tools, "get_config", return_value=object()),
        patch.object(cli_tools, "resolve_brain", return_value="default"),
        patch.object(cli_tools, "get_storage", AsyncMock(return_value=storage)),
        patch.object(cli_tools, "run_async", side_effect=asyncio.run),
        patch.object(consolidation_delta, "run_with_delta", side_effect=run_delta),
        patch.object(engine_module.ConsolidationProgressSession, "open", side_effect=capture_open),
    ):
        for strategy_timeout, total_timeout in [(1800, 3000), (17, 31)]:
            with pytest.raises(StopBeforeProgress):
                cli_tools.consolidate(
                    brain="default", strategy_timeout=strategy_timeout, total_timeout=total_timeout
                )
    expected_names = sorted(
        x.value for x in ConsolidationStrategy if x != ConsolidationStrategy.ALL
    )
    assert [item[0] for item in captured] == [expected_names, expected_names]
    assert [item[1] for item in captured] == [receipts.hourly_options_fingerprint()] * 2

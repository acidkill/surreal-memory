"""Offline native session to scheduled receipt handoff regressions."""

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
    log = receipts.MARKER + receipts.canonical(previous)
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

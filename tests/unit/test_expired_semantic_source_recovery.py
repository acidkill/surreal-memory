"""Expired discovery is retired explicitly without claiming a source mutation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressError,
    _restart_semantic_source_generation,
    recover_stale_semantic_link_discovery,
    validate_semantic_source_generation_audit,
)
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceFenceExpiredError,
    SurrealDBSemanticSourceRevisionMixin,
)
from tests.unit.test_stale_semantic_recovery_diagnostic_counters import PartialRecoveryStorage
from tests.unit.test_stale_semantic_recovery_repeat import (
    BRAIN_ID,
    REFERENCE_TIME,
    RUN_ID,
    _arguments,
)


class ExpiredStorage(PartialRecoveryStorage, SurrealDBSemanticSourceRevisionMixin):
    _create_semantic_source_fence_session = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.now = REFERENCE_TIME + timedelta(days=8)
        self.new_token = json.dumps(
            {
                **json.loads(self.new_token),
                "captured_at": (self.now + timedelta(seconds=1)).isoformat(),
            }
        )
        self.progress["strategy_states"].update(
            {
                name: {"phase": "completed", "counters": {"retained": 1}}
                for name in self.progress["completed_strategies"]
            }
        )

    def _get_brain_id(self) -> str:
        return BRAIN_ID

    async def _server_now(self) -> datetime:
        return self.now

    async def assert_semantic_source_unchanged(self, token: str, *, created_before: Any) -> None:
        await SurrealDBSemanticSourceRevisionMixin.assert_semantic_source_unchanged(
            self, token, created_before=created_before
        )


@pytest.mark.asyncio
async def test_manual_expired_generation_preserves_work_and_has_truthful_audit() -> None:
    storage = ExpiredStorage()
    before = deepcopy(storage.progress)
    snapshots = deepcopy(storage.states)
    preview = await recover_stale_semantic_link_discovery(storage, **_arguments())
    assert preview["source_expired"] is True
    assert storage.progress == before and storage.cas_calls == []
    result = await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert result["source_expired"] is True
    after = storage.progress
    for key in (
        "run_id",
        "reference_time",
        "options_fingerprint",
        "requested_strategies",
        "completed_strategies",
        "counters",
    ):
        assert after[key] == before[key]
    for name in before["completed_strategies"]:
        assert after["strategy_states"][name] == before["strategy_states"][name]
    assert all(storage.states[key] == value for key, value in snapshots.items())
    audit = after["strategy_states"]["semantic_link"]["recovery"]
    assert audit["version"] == 3 and audit["source_changed_verified"] is False
    assert audit["source_expiry_verified_at"] == storage.now
    assert audit["prior_counters"]["semantic_synapses_created"] == 7625
    assert validate_semantic_source_generation_audit(
        audit,
        brain_id=BRAIN_ID,
        run_id=RUN_ID,
        options_fingerprint=after["options_fingerprint"],
        reference_time=REFERENCE_TIME,
    )
    retry = await recover_stale_semantic_link_discovery(
        storage, **{**_arguments(), "expected_status": "paused"}, dry_run=False
    )
    assert retry["already_recovered"] is True and len(storage.cas_calls) == 1


@pytest.mark.asyncio
async def test_automatic_generation_still_refuses_native_expiry() -> None:
    storage = ExpiredStorage()
    before = deepcopy(storage.progress)
    storage.lease_owner = "automatic-worker"
    with pytest.raises((ConsolidationProgressError, SemanticSourceFenceExpiredError)):
        await _restart_semantic_source_generation(
            storage, storage.progress, storage.lease_owner, dry_run=False
        )
    assert storage.progress == before and storage.cas_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "backend",
        "apply",
        "unknown",
        "lease",
        "cas",
        "fresh_scope",
        "fresh_time",
        "unproven_age",
        "old_negative",
        "old_readonly",
        "old_future",
    ],
)
async def test_expired_recovery_retains_failure_fences(failure: str) -> None:
    storage = ExpiredStorage(cas_race=failure == "cas")
    semantic = storage.progress["strategy_states"]["semantic_link"]
    if failure == "backend":
        semantic["counters"]["semantic_link_failures"] = 1
        storage.progress["counters"]["semantic_link"]["semantic_link_failures"] = 1
    elif failure == "apply":
        semantic["phase"] = storage.progress["phase"] = "semantic_link_apply"
    elif failure == "unknown":
        storage.progress["last_error"] = "unknown write failure"
    elif failure == "lease":
        storage.lease_owner = "someone-else"
    elif failure in {"fresh_scope", "fresh_time"}:
        token = json.loads(storage.new_token)
        token["brain_id" if failure == "fresh_scope" else "captured_at"] = (
            "other" if failure == "fresh_scope" else REFERENCE_TIME.isoformat()
        )
        storage.new_token = json.dumps(token)
    elif failure.startswith("old_"):
        wrapper = json.loads(semantic["pending"][0])
        decoded = json.loads(wrapper["discovery"]["source_state_ref"]["source_token"])
        if failure == "old_negative":
            decoded["versionstamp"] = -1
        elif failure == "old_readonly":
            decoded["created_at_readonly"] = False
        else:
            decoded["captured_at"] = (storage.now + timedelta(days=1)).isoformat()
        token = json.dumps(decoded)
        wrapper["discovery"]["source_state_ref"]["source_token"] = token
        wrapper["source_validation_ref"]["source_token"] = token
        semantic["pending"] = [json.dumps(wrapper)]
        for snapshot in storage.states.values():
            snapshot["source_token"] = token
    elif failure == "unproven_age":
        storage.now = REFERENCE_TIME

        async def expired(*args: Any, **kwargs: Any) -> None:
            raise SemanticSourceFenceExpiredError("arbitrary expiry message")

        storage.assert_semantic_source_unchanged = expired
    before = deepcopy(storage.progress)
    with pytest.raises((ConsolidationProgressError, SemanticSourceFenceExpiredError)):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert storage.progress == before
    if failure != "cas":
        assert storage.cas_calls == []


@pytest.mark.asyncio
async def test_expired_generation_reuses_live_pair_and_deleted_neuron_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit import test_stale_semantic_recovery_diagnostic_counters as existing

    monkeypatch.setattr(existing, "PartialRecoveryStorage", ExpiredStorage)
    await existing.test_recovered_bootstrap_scans_live_population_and_skips_live_pairs(monkeypatch)


def expired_checkpoint() -> dict[str, Any]:
    from surreal_memory.engine.consolidation_receipts import canonical, checkpoint_hash
    from tests.unit.test_cluster_consolidation_hourly import recovered_checkpoint

    cp = recovered_checkpoint()
    semantic = cp["strategy_states"]["semantic_link"]
    audit = semantic["recovery"]
    audit.update(
        version=3, source_changed_verified=False, source_expiry_verified_at="2026-11-12T12:00:00"
    )
    new_ref = audit["source_state_ref"]
    new_ref["source_token"] = canonical(
        {**json.loads(new_ref["source_token"]), "captured_at": "2026-11-12T12:00:01"}
    )
    audit["new_source_token_sha256"] = hashlib.sha256(new_ref["source_token"].encode()).hexdigest()
    bootstrap = json.loads(semantic["pending"][0])
    bootstrap["source_state_ref"] = new_ref
    semantic["pending"] = [canonical(bootstrap)]
    audit["bootstrap_sha256"] = checkpoint_hash(bootstrap)
    audit["recovered_at"] = semantic["updated_at"] = cp["updated_at"] = "2026-11-12T12:00:02"
    return cp


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing",
        "float_version",
        "claimed_change",
        "unexpired",
        "future_proof",
        "unknown",
        "backend",
        "wrong_scope",
        "v2_expiry",
    ],
)
def test_expired_audit_agrees_with_seed_and_observer(mutation: str | None) -> None:
    from surreal_memory.engine.consolidation_receipts import (
        BlockedError,
        checkpoint_errors,
        select_command,
    )
    from tests.unit.test_cluster_consolidation_hourly import recovery_receipt
    from tests.unit.test_consolidation_status import configmap, receipt, summarize

    cp = expired_checkpoint()
    audit = cp["strategy_states"]["semantic_link"]["recovery"]
    if mutation == "missing":
        audit.pop("source_expiry_verified_at")
    elif mutation == "float_version":
        audit["version"] = 3.0
    elif mutation == "claimed_change":
        audit["source_changed_verified"] = True
    elif mutation == "unexpired":
        audit["source_expiry_verified_at"] = "2026-11-05T13:00:00"
    elif mutation == "future_proof":
        audit["source_expiry_verified_at"] = "2026-11-13T12:00:00"
    elif mutation == "unknown":
        audit["unexpected"] = True
    elif mutation == "backend":
        cp["last_error"] = "unrelated backend failure"
    elif mutation == "wrong_scope":
        audit["brain_id"] = "another-brain"
    elif mutation == "v2_expiry":
        audit["version"] = 2
    state = recovery_receipt(cp)
    if mutation:
        assert checkpoint_errors(cp)
        with pytest.raises(BlockedError):
            select_command(state, cp, None)
    else:
        assert not checkpoint_errors(cp)
        assert select_command(state, cp, None) == [
            "smem",
            "consolidate",
            "--brain",
            "default",
            "--resume",
        ]
    cp["phase"] = "semantic_link_discovery_neurons"
    result = summarize(cp, configmap(receipt(cp)))
    assert result.state == ("failed" if mutation else "paused")
    assert result.last_confirmed_success_at is None


@pytest.mark.asyncio
async def test_expired_validation_session_path_uses_actual_native_age_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    storage = ExpiredStorage()
    before = deepcopy(storage.progress)
    old_token = next(iter(storage.states.values()))["source_token"]
    session = SimpleNamespace(
        token=old_token,
        brain_id=BRAIN_ID,
        token_versionstamp=12,
        created_before=REFERENCE_TIME,
        validation_token=old_token,
        watermark=12,
        target_token=None,
    )

    async def verify(**kwargs: Any) -> str:
        return await storage._assert_source_session_unchanged(session)

    session.verify = verify

    class Validation:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.session = session

        async def restore(self) -> None:
            pass

    monkeypatch.setattr(storage, "_create_semantic_source_fence_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_validation.SemanticSourceValidation", Validation
    )
    result = await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert result["source_expired"] is True
    assert storage.progress["counters"] == before["counters"]


@pytest.mark.asyncio
async def test_expired_generation_readback_normalizes_sdk_datetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ExpiredStorage()
    original = storage.compare_and_swap_semantic_link_recovery

    async def sdk_response(**kwargs: Any) -> dict[str, Any] | None:
        result = await original(**kwargs)
        for audit in [
            result["strategy_states"]["semantic_link"]["recovery"],
            *result["strategy_states"]["semantic_link"]["recovery_history"],
        ]:
            audit["source_expiry_verified_at"] = (
                audit["source_expiry_verified_at"].isoformat() + "Z"
            )
        return result

    monkeypatch.setattr(storage, "compare_and_swap_semantic_link_recovery", sdk_response)
    result = await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert result["source_expired"] is True


def test_expired_recovery_cli_reports_manual_outcome(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from surreal_memory.cli.commands import tools

    storage = ExpiredStorage()
    monkeypatch.setattr(tools, "get_config", MagicMock())
    monkeypatch.setattr(tools, "resolve_brain", lambda *a: BRAIN_ID)
    monkeypatch.setattr(tools, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(tools, "run_async", asyncio.run)
    tools.recover_semantic_discovery(run_id=RUN_ID, stale_source=True)
    preview = capsys.readouterr().out
    assert "would be retired; no source mutation was claimed" in preview
    assert storage.cas_calls == []
    tools.recover_semantic_discovery(
        run_id=RUN_ID, stale_source=True, execute=True, confirm_run_id=RUN_ID
    )
    assert "retired; no source mutation was claimed" in capsys.readouterr().out
    assert storage.progress["strategy_states"]["semantic_link"]["recovery"]["version"] == 3

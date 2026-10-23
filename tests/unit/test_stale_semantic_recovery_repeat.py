"""Focused operator recovery tests without a live SurrealDB service."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationResumeMismatchError,
    recover_stale_semantic_link_discovery,
)
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceChangedError,
)

RUN_ID = "repeat-stale-recovery-run"
FINGERPRINT = "repeat-stale-recovery-options"
BRAIN_ID = "repeat-stale-recovery-brain"
REFERENCE_TIME = datetime(2026, 9, 25, 12, 0, 0)


class StaleRecoveryStorage:
    current_brain_id = BRAIN_ID

    def __init__(self, *, source_changed: bool = True, cas_race: bool = False) -> None:
        self.progress = _failed_repeat_checkpoint()
        self.source_changed = source_changed
        self.cas_race = cas_race
        self.lease_owner: str | None = None
        self.cas_calls: list[dict[str, Any]] = []
        self.source_checks: list[tuple[str, datetime]] = []

    async def acquire_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int
    ) -> bool:
        if self.lease_owner is not None:
            return False
        self.lease_owner = owner_token
        return True

    async def renew_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int
    ) -> bool:
        return self.lease_owner == owner_token

    async def release_consolidation_lease(self, brain_id: str, owner_token: str) -> bool:
        if self.lease_owner == owner_token:
            self.lease_owner = None
            return True
        return False

    async def get_consolidation_progress(self, brain_id: str) -> dict[str, Any]:
        return deepcopy(self.progress)

    async def assert_semantic_source_unchanged(
        self, source_token: str, *, created_before: datetime
    ) -> None:
        self.source_checks.append((source_token, created_before))
        if self.source_changed:
            raise SemanticSourceChangedError("verified source mutation")

    async def compare_and_swap_semantic_link_recovery(self, **kwargs: Any) -> dict[str, Any] | None:
        self.cas_calls.append(deepcopy(kwargs))
        if self.cas_race:
            return None
        self.progress.update(
            status=kwargs["new_status"],
            phase="semantic_link_restart_pending",
            cursor=None,
            owner_token=kwargs["new_owner_token"],
            updated_at=kwargs["updated_at"],
            strategy_states=deepcopy(kwargs["strategy_states"]),
            counters=deepcopy(kwargs["counters"]),
        )
        return deepcopy(self.progress)


def _failed_repeat_checkpoint() -> dict[str, Any]:
    strategies = {
        strategy.value
        for strategy in ConsolidationStrategy
        if strategy is not ConsolidationStrategy.ALL
    }
    completed = sorted(strategies - {"semantic_link"})
    source_token = json.dumps(
        {
            "version": 1,
            "brain_id": BRAIN_ID,
            "versionstamp": 12,
            "captured_at": (REFERENCE_TIME - timedelta(minutes=2)).isoformat(),
            "created_at_readonly": True,
        },
        sort_keys=True,
    )
    legacy_marker = {
        "kind": "legacy_semantic_discovery_restart",
        "version": 1,
        "run_id": RUN_ID,
        "options_fingerprint": FINGERPRINT,
        "expected_status": "paused",
        "previous_phase": "semantic_link_discovery_neurons",
    }
    stale_marker = {
        "kind": "stale_semantic_source_restart",
        "version": 1,
        "run_id": RUN_ID,
        "options_fingerprint": FINGERPRINT,
        "expected_status": "running",
        "previous_phase": "semantic_link_discovery_neurons",
        "manifest_sha256": "a" * 64,
        "source_token_sha256": "b" * 64,
        "prior_recovery_sha256": "c" * 64,
        "source_changed_verified": True,
        "owner_token": "earlier-recovery-owner",
        "recovered_at": REFERENCE_TIME - timedelta(minutes=1),
    }
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": {
            "state_id": "staged-semantic-source",
            "revision": 1,
            "state_revision": 1,
            "source_token": source_token,
            "brain_id": BRAIN_ID,
            "run_id": RUN_ID,
        },
    }
    semantic_state = {
        "phase": "semantic_link_discovery_neurons",
        "cursor": "neuron:123",
        "pending": [json.dumps([json.dumps(manifest)])],
        "counters": {},
        "recovery": stale_marker,
        "recovery_history": [legacy_marker, stale_marker],
        "updated_at": REFERENCE_TIME - timedelta(seconds=2),
    }
    return {
        "brain_id": BRAIN_ID,
        "run_id": RUN_ID,
        "options_fingerprint": FINGERPRINT,
        "reference_time": REFERENCE_TIME,
        "status": "failed",
        "current_strategy": "semantic_link",
        "phase": "semantic_link_discovery_neurons",
        "cursor": "neuron:123",
        "owner_token": "failed-worker-owner",
        "updated_at": REFERENCE_TIME - timedelta(seconds=1),
        "requested_strategies": sorted(strategies),
        "completed_strategies": completed,
        "strategy_states": {"semantic_link": semantic_state},
        "counters": {"semantic_link": {}},
    }


def _arguments() -> dict[str, str]:
    return {
        "run_id": RUN_ID,
        "expected_options_fingerprint": FINGERPRINT,
        "expected_status": "failed",
        "confirmation_run_id": RUN_ID,
    }


@pytest.mark.asyncio
async def test_failed_repeat_recovery_preflights_then_pauses_and_is_idempotent() -> None:
    storage = StaleRecoveryStorage()
    original = deepcopy(storage.progress)

    preview = await recover_stale_semantic_link_discovery(storage, **_arguments())
    assert preview["dry_run"] is True
    assert storage.progress == original
    assert storage.cas_calls == []
    assert storage.source_checks[0][1] == REFERENCE_TIME

    result = await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert result["already_recovered"] is False
    assert storage.progress["status"] == "paused"
    assert storage.progress["reference_time"] == original["reference_time"]
    assert storage.progress["completed_strategies"] == original["completed_strategies"]
    assert storage.progress["counters"] == original["counters"]
    history = storage.progress["strategy_states"]["semantic_link"]["recovery_history"]
    assert history[:2] == original["strategy_states"]["semantic_link"]["recovery_history"]
    assert len(history) == 3
    assert history[-1]["source_changed_verified"] is True
    assert storage.cas_calls[0]["expected_status"] == "failed"
    assert storage.cas_calls[0]["new_status"] == "paused"

    retry = await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert retry["already_recovered"] is True
    cli_retry_args = {**_arguments(), "expected_status": "paused"}
    cli_retry = await recover_stale_semantic_link_discovery(
        storage, **cli_retry_args, dry_run=False
    )
    assert cli_retry["already_recovered"] is True
    assert len(storage.cas_calls) == 1


@pytest.mark.asyncio
async def test_paused_discovery_cannot_start_a_new_stale_source_recovery() -> None:
    storage = StaleRecoveryStorage()
    storage.progress["status"] = "paused"

    with pytest.raises(ConsolidationResumeMismatchError, match="only verifies"):
        await recover_stale_semantic_link_discovery(
            storage, **{**_arguments(), "expected_status": "paused"}, dry_run=False
        )

    assert storage.source_checks == []
    assert storage.cas_calls == []


@pytest.mark.asyncio
async def test_failed_repeat_without_a_fresh_source_change_is_refused() -> None:
    storage = StaleRecoveryStorage(source_changed=False)
    original = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError, match="not proven"):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)

    assert storage.progress == original
    assert storage.cas_calls == []


@pytest.mark.asyncio
async def test_failed_repeat_with_graph_effects_is_refused_before_source_check() -> None:
    storage = StaleRecoveryStorage()
    storage.progress["counters"]["semantic_link"] = {"semantic_synapses_created": 1}
    storage.progress["strategy_states"]["semantic_link"]["counters"] = {
        "semantic_synapses_created": 1
    }
    original = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError, match=r"counters.*nonzero"):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)

    assert storage.progress == original
    assert storage.source_checks == []
    assert storage.cas_calls == []


@pytest.mark.asyncio
async def test_failed_repeat_cas_race_does_not_reset_checkpoint() -> None:
    storage = StaleRecoveryStorage(cas_race=True)
    original = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError, match="changed during recovery"):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)

    assert storage.progress == original
    assert len(storage.cas_calls) == 1
    assert storage.lease_owner is None


@pytest.mark.asyncio
async def test_first_stale_recovery_preserves_other_strategies_and_can_repeat() -> None:
    storage = StaleRecoveryStorage()
    semantic = storage.progress["strategy_states"]["semantic_link"]
    semantic.pop("recovery")
    semantic.pop("recovery_history")
    storage.progress["strategy_states"]["prune"] = {
        "phase": "completed",
        "counters": {"synapses_pruned": 12},
    }
    storage.progress["counters"]["prune"] = {"synapses_pruned": 12}
    original = deepcopy(storage.progress)

    await recover_stale_semantic_link_discovery(storage, **_arguments())
    assert storage.progress == original
    assert storage.cas_calls == []
    await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert storage.progress["status"] == "paused"
    assert storage.progress["run_id"] == original["run_id"]
    assert storage.progress["reference_time"] == original["reference_time"]
    assert storage.progress["completed_strategies"] == original["completed_strategies"]
    assert storage.progress["counters"] == original["counters"]
    assert storage.progress["strategy_states"]["prune"] == original["strategy_states"]["prune"]
    assert len(storage.progress["strategy_states"]["semantic_link"]["recovery_history"]) == 1
    retry = await recover_stale_semantic_link_discovery(
        storage, **{**_arguments(), "expected_status": "paused"}, dry_run=False
    )
    assert retry["already_recovered"] is True
    assert len(storage.cas_calls) == 1

    # A later attempt can encounter a new mutation without a legacy repair ancestor.
    repaired = storage.progress["strategy_states"]["semantic_link"]
    repaired.update(phase=semantic["phase"], cursor=semantic["cursor"], pending=semantic["pending"])
    storage.progress.update(status="failed", phase=semantic["phase"], cursor=semantic["cursor"])
    await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert len(storage.progress["strategy_states"]["semantic_link"]["recovery_history"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["graph_effect", "source_unchanged", "orphan_history", "lease_lost", "cas_race"]
)
async def test_first_stale_recovery_keeps_safety_guards(failure: str) -> None:
    storage = StaleRecoveryStorage(
        source_changed=failure != "source_unchanged", cas_race=failure == "cas_race"
    )
    semantic = storage.progress["strategy_states"]["semantic_link"]
    semantic.pop("recovery")
    semantic.pop("recovery_history")
    if failure == "graph_effect":
        semantic["counters"] = {"semantic_synapses_created": 1}
        storage.progress["counters"]["semantic_link"] = dict(semantic["counters"])
    elif failure == "orphan_history":
        semantic["recovery_history"] = [{"kind": "unknown"}]
    elif failure == "lease_lost":

        async def lost(*args: Any, **kwargs: Any) -> bool:
            return False

        storage.renew_consolidation_lease = lost  # type: ignore[method-assign]
    original = deepcopy(storage.progress)
    from surreal_memory.engine.consolidation_progress import ConsolidationProgressError

    with pytest.raises(ConsolidationProgressError):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert storage.progress == original
    assert storage.lease_owner is None


def test_failed_report_does_not_suggest_ineligible_data() -> None:
    report = ConsolidationReport(started_at=REFERENCE_TIME)
    report.extra["consolidation_status"] = "failed"
    report.extra["failed_strategies"] = [
        "semantic_link (progress coordination: ConsolidationProgressError)"
    ]
    assert "Why nothing changed" not in report.summary()
    assert "Stages failed: semantic_link" in report.summary()


@pytest.mark.asyncio
async def test_stale_discovery_error_points_to_read_only_recovery_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine.consolidation_progress import ConsolidationProgressError
    from tests.unit.test_consolidation_mid_strategies_resume import _engine, _Progress, _Storage

    storage = _Storage()
    progress = _Progress("semantic_link")
    progress.strategy_state("semantic_link").update(
        phase="semantic_link_discovery_neurons",
        pending=[json.dumps({"kind": "semantic_link_discovery", "version": 3})],
        counters={},
    )

    async def changed(*args: Any, **kwargs: Any) -> None:
        raise SemanticSourceChangedError("neuron changed since capture")

    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery.discover_semantic_synapses", changed
    )
    with pytest.raises(ConsolidationProgressError) as caught:
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    message = str(caught.value)
    assert "recover-semantic-discovery" in message
    assert "--run-id test-run" in message
    assert "--stale-source" in message
    assert "--execute" not in message
    assert storage.added_synapses == []
    assert progress.writes == []

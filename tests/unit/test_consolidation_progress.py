from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

import surreal_memory.engine.consolidation_progress as progress_module
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationLeaseBusyError,
    ConsolidationLeaseLostError,
    ConsolidationProgressError,
    ConsolidationProgressSession,
    ConsolidationResumeMismatchError,
    options_fingerprint,
    stored_form,
)

REFERENCE_TIME = datetime(2026, 9, 20, 12, 30)


class FakeProgressStorage:
    current_brain_id = "default"

    def __init__(self) -> None:
        self.progress: dict[str, Any] | None = None
        self.lease_owner: str | None = None
        self.lease_available = True
        self.reject_saves = False
        self.started = 0
        self.released: list[str] = []

    async def acquire_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int
    ) -> bool:
        if not self.lease_available:
            return False
        if self.lease_owner is not None:
            return False
        self.lease_owner = owner_token
        return True

    async def renew_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int
    ) -> bool:
        return self.lease_owner == owner_token

    async def release_consolidation_lease(self, brain_id: str, owner_token: str) -> bool:
        if self.lease_owner != owner_token:
            return False
        self.released.append(owner_token)
        self.lease_owner = None
        return True

    async def get_consolidation_progress(self, brain_id: str) -> dict[str, Any] | None:
        return deepcopy(self.progress)

    async def start_consolidation_progress(self, state: dict[str, Any]) -> dict[str, Any]:
        self.started += 1
        self.progress = deepcopy(state)
        return deepcopy(self.progress)

    async def claim_consolidation_progress(
        self, brain_id: str, owner_token: str
    ) -> dict[str, Any] | None:
        if self.progress is None:
            return None
        self.progress["owner_token"] = owner_token
        return deepcopy(self.progress)

    async def save_consolidation_progress(
        self, brain_id: str, owner_token: str, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self.reject_saves or self.lease_owner != owner_token:
            return None
        self.progress = deepcopy(state)
        return deepcopy(self.progress)


def test_stored_form_omits_none_valued_keys_at_every_depth_and_keeps_array_items() -> None:
    value = {
        "gone": None,
        "nested": {"gone": None, "kept": 1},
        "items": [{"gone": None, "kept": 2}, None, 3],
        "zero": 0,
        "empty": "",
        "false": False,
    }

    assert stored_form(value) == {
        "nested": {"kept": 1},
        "items": [{"kept": 2}, None, 3],
        "zero": 0,
        "empty": "",
        "false": False,
    }
    assert value["gone"] is None
    assert value["nested"] == {"gone": None, "kept": 1}


@pytest.mark.asyncio
async def test_new_run_persists_committed_checkpoints_and_completion() -> None:
    storage = FakeProgressStorage()
    fingerprint = options_fingerprint({"strategies": ["prune"]})

    session = await ConsolidationProgressSession.open(
        storage, ["prune"], fingerprint, REFERENCE_TIME
    )

    assert session is not None
    assert not session.resumed
    assert session.state["status"] == "running"
    assert session.state["schema_version"] > 0
    assert session.state["engine_version"] == progress_module.CONSOLIDATION_ENGINE_VERSION
    assert session.state["engine_version"] == "3.11.0:checkpoint-v1"
    assert storage.started == 1

    await session.checkpoint(
        "prune",
        "synapse_delete",
        cursor="synapse:abc",
        pending=["synapse:a", "synapse:b"],
        counters={"deleted": 2},
    )
    assert session.last_checkpoint == "prune phase=synapse_delete cursor=synapse:abc"
    assert session.strategy_state("prune")["pending"] == ["synapse:a", "synapse:b"]

    await session.complete_strategy("prune")
    await session.complete()
    assert storage.progress is not None
    assert storage.progress["status"] == "completed"
    assert storage.progress["completed_strategies"] == ["prune"]
    await session.close()
    assert storage.lease_owner is None


@pytest.mark.asyncio
async def test_resume_reclaims_unfinished_run_and_freezes_reference_time() -> None:
    storage = FakeProgressStorage()
    fingerprint = options_fingerprint({"strategies": ["prune"]})
    first = await ConsolidationProgressSession.open(storage, ["prune"], fingerprint, REFERENCE_TIME)
    assert first is not None
    run_id = first.state["run_id"]
    await first.checkpoint("prune", "synapse_scan", cursor="synapse:10")
    await first.pause()
    await first.close()

    second = await ConsolidationProgressSession.open(
        storage,
        ["prune"],
        fingerprint,
        REFERENCE_TIME + timedelta(days=2),
    )
    assert second is not None
    assert second.resumed
    assert second.state["run_id"] == run_id
    assert second.reference_time == REFERENCE_TIME
    assert second.last_checkpoint == "prune phase=synapse_scan cursor=synapse:10"
    assert storage.started == 1
    await second.close()


@pytest.mark.asyncio
async def test_mismatched_options_leave_checkpoint_untouched() -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(
        storage, ["prune"], options_fingerprint({"min_age": 7}), REFERENCE_TIME
    )
    assert first is not None
    await first.pause()
    await first.close()
    before = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError, match="different strategy/configuration"):
        await ConsolidationProgressSession.open(
            storage, ["prune"], options_fingerprint({"min_age": 14}), REFERENCE_TIME
        )

    assert storage.progress == before
    assert storage.lease_owner is None


@pytest.mark.asyncio
async def test_code_version_mismatch_leaves_checkpoint_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None
    await first.checkpoint("prune", "synapse_scan", cursor="synapse:10")
    await first.pause()
    await first.close()
    before = deepcopy(storage.progress)

    monkeypatch.setattr(progress_module, "CONSOLIDATION_ENGINE_VERSION", "3.11.1:checkpoint-v2")
    with pytest.raises(ConsolidationResumeMismatchError, match="different code version"):
        await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)

    assert storage.progress == before
    assert storage.lease_owner is None


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["different_run", "completed", "missing"])
async def test_explicit_resume_refuses_changed_checkpoint_without_starting_run(
    replacement: str,
) -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None
    run_id = first.state["run_id"]
    await first.pause()
    await first.close()
    assert storage.progress is not None
    if replacement == "different_run":
        storage.progress["run_id"] = "replacement-run"
    elif replacement == "completed":
        storage.progress["status"] = "completed"
    else:
        storage.progress = None
    before = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError):
        await ConsolidationProgressSession.open(
            storage, ["prune"], "same", REFERENCE_TIME, resume_run_id=run_id
        )

    assert storage.progress == before
    assert storage.started == 1
    assert storage.lease_owner is None


@pytest.mark.asyncio
async def test_explicit_resume_preserves_completed_strategies_and_reference_time() -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(
        storage, ["mature", "merge", "prune"], "same", REFERENCE_TIME
    )
    assert first is not None
    run_id = first.state["run_id"]
    await first.complete_strategy("prune")
    await first.complete_strategy("mature")
    await first.pause()
    await first.close()

    second = await ConsolidationProgressSession.open(
        storage,
        ["mature", "merge", "prune"],
        "same",
        REFERENCE_TIME + timedelta(days=2),
        resume_run_id=run_id,
    )
    assert second is not None
    assert second.resumed
    assert second.completed_strategies == {"prune", "mature"}
    assert second.reference_time == REFERENCE_TIME
    assert storage.started == 1
    await second.close()


def test_blocked_report_explains_no_execution_without_eligibility_hints() -> None:
    report = ConsolidationReport()
    report.extra["consolidation_status"] = "blocked"
    report.extra["failed_strategies"] = ["progress coordination (ConsolidationResumeMismatchError)"]

    summary = report.summary()

    assert "Strategies were not executed; checkpoint preserved" in summary
    assert "Why nothing changed" not in summary
    assert "store more memories" not in summary


def test_partial_failed_report_preserves_changes_without_eligibility_hints() -> None:
    report = ConsolidationReport(synapses_pruned=3)
    report.extra["consolidation_status"] = "failed"
    assert "Synapses pruned: 3" in report.summary()
    assert "Strategies were not executed" not in report.summary()


@pytest.mark.asyncio
async def test_explicit_reference_time_mismatch_is_rejected() -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None
    await first.pause()
    await first.close()

    with pytest.raises(ConsolidationResumeMismatchError, match="explicit reference_time"):
        await ConsolidationProgressSession.open(
            storage,
            ["prune"],
            "same",
            REFERENCE_TIME + timedelta(days=1),
            explicit_reference_time=REFERENCE_TIME + timedelta(days=1),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_after_failure", [False, True])
async def test_resumed_attempt_clears_prior_error_without_resetting_work(
    paused_after_failure: bool,
) -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(
        storage, ["prune", "semantic_link"], "same", REFERENCE_TIME
    )
    assert first is not None
    await first.checkpoint("prune", "done", counters={"synapses_pruned": 17})
    await first.complete_strategy("prune")
    await first.checkpoint(
        "semantic_link", "discovery_neurons", cursor="neuron:example", pending=["saved-source"]
    )
    await first.fail("previous receiver failure")
    if paused_after_failure:
        await first.pause()
    await first.close()
    before = deepcopy(storage.progress)
    assert before is not None

    resumed = await ConsolidationProgressSession.open(
        storage,
        ["prune", "semantic_link"],
        "same",
        REFERENCE_TIME + timedelta(days=3),
        resume_run_id=before["run_id"],
    )
    assert resumed is not None
    try:
        assert storage.progress is not None
        assert storage.progress["status"] == "running"
        assert storage.progress.get("last_error") is None
        for key in (
            "run_id",
            "reference_time",
            "options_fingerprint",
            "requested_strategies",
            "completed_strategies",
            "strategy_states",
            "phase",
            "cursor",
            "counters",
        ):
            assert storage.progress[key] == before[key]
        await resumed.pause()
        assert storage.progress["status"] == "paused"
        assert storage.progress.get("last_error") is None
        await resumed.fail("new coordination failure")
        assert storage.progress["last_error"] == "new coordination failure"
    finally:
        await resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_save", [False, True])
async def test_rejected_resume_preserves_prior_error(reject_save: bool) -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None
    await first.checkpoint("prune", "synapse_pending", cursor="example", pending=["pending"])
    await first.fail("unresolved prior failure")
    await first.close()
    before = deepcopy(storage.progress)
    storage.reject_saves = reject_save

    with pytest.raises(
        ConsolidationLeaseLostError if reject_save else ConsolidationResumeMismatchError
    ):
        await ConsolidationProgressSession.open(
            storage, ["prune"], "same" if reject_save else "different", REFERENCE_TIME
        )
    assert storage.progress is not None and before is not None
    for key in ("last_error", "phase", "cursor", "strategy_states", "counters", "run_id"):
        assert storage.progress[key] == before[key]
    assert storage.lease_owner is None


@pytest.mark.asyncio
async def test_live_lease_rejects_second_worker() -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None

    with pytest.raises(ConsolidationLeaseBusyError):
        await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)

    await first.close()


@pytest.mark.asyncio
async def test_checkpoint_is_fenced_after_lease_loss() -> None:
    storage = FakeProgressStorage()
    session = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert session is not None
    storage.reject_saves = True

    with pytest.raises(ConsolidationLeaseLostError, match="rejected the checkpoint"):
        await session.checkpoint("prune", "scan", cursor="synapse:1")

    await session.close()


@pytest.mark.asyncio
async def test_engine_stops_before_next_unit_when_lease_is_lost() -> None:
    engine = object.__new__(ConsolidationEngine)
    engine._progress_session = type(
        "LostLease", (), {"lease_lost": True, "brain_id": "test-brain"}
    )()
    engine._strategy_deadline = None
    engine._total_deadline = None

    with pytest.raises(ConsolidationLeaseLostError, match="stopping before another work unit"):
        await engine._check_progress_budget()


@pytest.mark.asyncio
async def test_engine_resumes_after_strategy_timeout_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(
        storage,
        ConsolidationConfig(strategy_timeout_seconds=0.3, total_timeout_seconds=2.0),
    )
    resumed_cursors: list[str | None] = []
    resume_events: list[str] = []

    async def run_strategy(
        strategy: ConsolidationStrategy,
        report: Any,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        cursor = engine._strategy_resume_cursor()
        if cursor is None:
            await engine._checkpoint_progress(
                "synapse_scan",
                cursor="synapse:10",
                counters={"scanned": 10},
            )
            await asyncio.sleep(0.6)
            return
        resumed_cursors.append(cursor)
        resume_events.append("strategy")
        await engine._checkpoint_progress(
            "synapse_delete",
            cursor="synapse:20",
            counters={"scanned": 20, "deleted": 3},
        )

    monkeypatch.setattr(engine, "_run_strategy", run_strategy)

    first = await engine.run([ConsolidationStrategy.PRUNE])
    assert first.extra["consolidation_status"] == "paused"
    assert first.extra["last_checkpoint"] == ("prune phase=synapse_scan cursor=synapse:10")
    assert "The next smem consolidate --resume will resume" in first.summary()
    assert storage.progress is not None
    run_id = storage.progress["run_id"]
    frozen_reference_time = storage.progress["reference_time"]
    assert storage.progress["status"] == "paused"
    assert storage.progress["completed_strategies"] == []

    def on_progress(message: str) -> None:
        resume_events.append(f"progress:{message}")

    second = await engine.run([ConsolidationStrategy.PRUNE], on_progress=on_progress)
    assert second.extra["consolidation_status"] == "completed"
    assert resumed_cursors == ["synapse:10"]
    assert storage.progress is not None
    assert storage.progress["run_id"] == run_id
    assert storage.progress["reference_time"] == frozen_reference_time
    assert storage.progress["completed_strategies"] == ["prune"]
    assert "Consolidation: prune progress state found - resuming..." in second.summary()
    assert resume_events == [
        "progress:Consolidation: prune progress state found - resuming...",
        "strategy",
    ]


@pytest.mark.asyncio
async def test_engine_failure_retries_last_committed_phase_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage)
    seen_phases: list[str | None] = []

    async def run_strategy(
        strategy: ConsolidationStrategy,
        report: Any,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        state = engine._progress_session.strategy_state(strategy.value)
        seen_phases.append(state.get("phase"))
        if len(seen_phases) == 1:
            await engine._checkpoint_progress(
                "retention_traces",
                cursor="neuron:last",
                counters={"synapse_pages": 124, "neuron_pages": 61},
            )
            raise RuntimeError("transient retention failure")

    monkeypatch.setattr(engine, "_run_strategy", run_strategy)

    first = await engine.run([ConsolidationStrategy.PRUNE])
    assert first.extra["consolidation_status"] == "failed"
    assert storage.progress is not None
    assert storage.progress["status"] == "failed"
    committed = storage.progress["strategy_states"]["prune"]
    assert committed["phase"] == "retention_traces"
    assert committed["cursor"] == "neuron:last"

    second = await engine.run([ConsolidationStrategy.PRUNE])
    assert second.extra["consolidation_status"] == "completed"
    assert seen_phases == ["starting", "retention_traces"]


@pytest.mark.asyncio
async def test_progress_validation_failure_is_persisted_not_misreported_as_lease_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage)

    async def run_strategy(
        strategy: ConsolidationStrategy,
        report: Any,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        await engine._checkpoint_progress("semantic_link_discovery_neurons", cursor="neuron:1")
        raise ConsolidationProgressError("neuron changed since token captured")

    monkeypatch.setattr(engine, "_run_strategy", run_strategy)
    report = await engine.run([ConsolidationStrategy.SEMANTIC_LINK])

    assert report.extra["consolidation_status"] == "failed"
    assert "neuron changed since token captured" in report.summary()
    assert storage.progress is not None
    assert storage.progress["status"] == "failed"
    assert storage.progress["phase"] == "semantic_link_discovery_neurons"
    assert storage.progress["cursor"] == "neuron:1"


@pytest.mark.asyncio
async def test_progress_failure_stops_later_tiers_and_resume_preserves_committed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage)
    strategies = [
        ConsolidationStrategy.PRUNE,
        ConsolidationStrategy.MATURE,
        ConsolidationStrategy.SUMMARIZE,
        ConsolidationStrategy.ENRICH,
        ConsolidationStrategy.SEMANTIC_LINK,
    ]
    seen: list[str] = []

    async def fail_summarize(
        strategy: ConsolidationStrategy,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        seen.append(strategy.value)
        if strategy == ConsolidationStrategy.SUMMARIZE:
            await engine._checkpoint_progress(
                "fiber_census", cursor="synthetic-census-page", counters={"fiber_pages": 2}
            )
            raise ConsolidationProgressError("synthetic census integrity failure")

    monkeypatch.setattr(engine, "_run_strategy", fail_summarize)
    first = await engine.run(strategies, reference_time=REFERENCE_TIME)

    assert seen == ["prune", "mature", "summarize"]
    assert first.extra["consolidation_status"] == "failed"
    assert "synthetic census integrity failure" in first.summary()
    assert storage.progress is not None
    assert storage.progress["status"] == "failed"
    assert storage.progress["completed_strategies"] == ["prune", "mature"]
    committed = deepcopy(storage.progress["strategy_states"])
    assert committed["summarize"]["phase"] == "fiber_census"
    assert committed["summarize"]["cursor"] == "synthetic-census-page"
    assert committed["summarize"]["counters"] == {"fiber_pages": 2}
    run_id = storage.progress["run_id"]
    seen.clear()

    async def recovered(
        strategy: ConsolidationStrategy,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        assert reference_time == REFERENCE_TIME
        seen.append(strategy.value)
        if strategy == ConsolidationStrategy.SUMMARIZE:
            assert engine._progress_session.strategy_state("summarize") == committed["summarize"]

    monkeypatch.setattr(engine, "_run_strategy", recovered)
    second = await engine.run(strategies, resume_run_id=run_id)
    assert seen == ["summarize", "enrich", "semantic_link"]
    assert second.extra["consolidation_status"] == "completed"
    assert storage.progress["run_id"] == run_id
    assert storage.progress["strategy_states"]["prune"] == committed["prune"]
    assert storage.progress["strategy_states"]["mature"] == committed["mature"]


@pytest.mark.asyncio
async def test_explicit_engine_resume_skips_finished_strategies_in_original_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage)
    strategies = [
        ConsolidationStrategy.PRUNE,
        ConsolidationStrategy.MATURE,
        ConsolidationStrategy.MERGE,
    ]
    executed: list[str] = []

    async def initial_run(
        strategy: ConsolidationStrategy,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        if strategy == ConsolidationStrategy.MERGE:
            raise ConsolidationProgressError("synthetic census failure")

    monkeypatch.setattr(engine, "_run_strategy", initial_run)
    first = await engine.run(strategies, reference_time=REFERENCE_TIME)
    assert first.extra["consolidation_status"] == "failed"
    assert storage.progress is not None
    run_id = storage.progress["run_id"]
    assert set(storage.progress["completed_strategies"]) == {"prune", "mature"}

    async def resumed_run(
        strategy: ConsolidationStrategy,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        assert reference_time == REFERENCE_TIME
        executed.append(strategy.value)
        report.fibers_merged += 1

    monkeypatch.setattr(engine, "_run_strategy", resumed_run)
    second = await engine.run(strategies, resume_run_id=run_id)

    assert executed == ["merge"]
    assert second.fibers_merged == 1
    assert second.extra["consolidation_status"] == "completed"
    assert storage.progress["run_id"] == run_id
    assert storage.progress["reference_time"] == REFERENCE_TIME
    assert storage.started == 1


def test_report_summary_renders_saved_checkpoint_details() -> None:
    report = ConsolidationReport()
    report.extra["dedup_resumed_checkpoint"] = "dedup_window_complete"
    report.extra["last_checkpoint"] = "prune phase=synapse_scan cursor=synapse:100"

    summary = report.summary()

    assert "Dedup resumed from saved checkpoint: dedup_window_complete" in summary
    assert "Last committed checkpoint: prune phase=synapse_scan cursor=synapse:100" in summary


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("readable", [False, True])
async def test_terminal_write_acknowledgement_is_reconciled_without_replay(
    committed: bool, readable: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surreal_memory.engine.consolidation_progress import ConsolidationCompletionUnconfirmedError

    storage = FakeProgressStorage()
    session = await ConsolidationProgressSession.open(
        storage, ["prune"], "fingerprint", REFERENCE_TIME
    )
    assert session is not None
    await session.complete_strategy("prune")
    saved = storage.save_consolidation_progress
    writes: list[str] = []

    async def lose_ack(brain_id: str, owner_token: str, state: dict[str, Any]) -> None:
        writes.append(str(state["status"]))
        if committed:
            await saved(brain_id, owner_token, state)
        raise ConnectionError("synthetic lost acknowledgement")

    async def read(brain_id: str) -> dict[str, Any] | None:
        if not readable:
            raise ConnectionError("synthetic read outage")
        return deepcopy(storage.progress)

    monkeypatch.setattr(storage, "save_consolidation_progress", lose_ack)
    monkeypatch.setattr(storage, "get_consolidation_progress", read)
    try:
        if committed and readable:
            await session.complete()
            assert session.state["status"] == "completed"
        else:
            with pytest.raises(ConsolidationCompletionUnconfirmedError):
                await session.complete()
            assert session.state["status"] != "completed"
        assert writes == ["completed"]
        assert storage.progress is not None
        assert storage.progress["status"] == ("completed" if committed else "running")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_auto_tier_ambiguous_apply_is_not_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.unified_config import TierConfig

    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage, tier_config=TierConfig(auto_enabled=True))
    strategies = AsyncMock()
    tiers = AsyncMock(side_effect=ConnectionError("synthetic partial tier apply"))
    monkeypatch.setattr(engine, "_run_strategy", strategies)
    monkeypatch.setattr(engine, "_run_auto_tier", tiers)

    first = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert first.extra["consolidation_status"] == "failed"
    assert storage.progress is not None
    assert storage.progress["phase"] == "auto_tier_started"
    second = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert second.extra["consolidation_status"] == "failed"
    assert "outcome is unknown" in second.summary()
    assert strategies.await_count == 1
    assert tiers.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", ["auto_tier_completed", "strategy_completed"])
async def test_auto_tier_receipt_allows_finalization_only_resume(
    monkeypatch: pytest.MonkeyPatch,
    receipt: str,
) -> None:
    from surreal_memory.unified_config import TierConfig

    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage, tier_config=TierConfig(auto_enabled=True))
    strategies = AsyncMock()
    tiers = AsyncMock()
    monkeypatch.setattr(engine, "_run_strategy", strategies)
    monkeypatch.setattr(engine, "_run_auto_tier", tiers)
    saved = storage.save_consolidation_progress

    async def fail_terminal(
        brain_id: str, token: str, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        if state["status"] == "completed":
            raise ConnectionError("synthetic final write failure")
        return await saved(brain_id, token, state)

    monkeypatch.setattr(storage, "save_consolidation_progress", fail_terminal)
    first = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert first.extra["consolidation_status"] == "completion_unconfirmed"
    assert storage.progress is not None
    assert storage.progress["phase"] == "auto_tier_completed"
    storage.progress["phase"] = receipt
    storage.progress.pop("native_handoff", None)  # Model the legacy finalization receipt.
    monkeypatch.setattr(storage, "save_consolidation_progress", saved)
    second = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert second.extra["consolidation_status"] == (
        "completed" if receipt == "auto_tier_completed" else "failed"
    )
    if receipt != "auto_tier_completed":
        assert "outcome is unknown" in second.summary()
    assert strategies.await_count == 1
    assert tiers.await_count == 1


@pytest.mark.asyncio
async def test_terminal_lease_rejection_is_reported_without_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage)
    monkeypatch.setattr(engine, "_run_strategy", AsyncMock())
    saved = storage.save_consolidation_progress

    async def reject_terminal(
        brain_id: str, token: str, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        if state["status"] == "completed":
            return None
        return await saved(brain_id, token, state)

    monkeypatch.setattr(storage, "save_consolidation_progress", reject_terminal)
    report = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert report.extra["consolidation_status"] == "lease_lost"
    assert storage.progress is not None
    assert storage.progress["status"] == "running"


@pytest.mark.asyncio
async def test_terminal_readback_refuses_corrupt_empty_strategy_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine.consolidation_progress import ConsolidationCompletionUnconfirmedError

    storage = FakeProgressStorage()
    session = await ConsolidationProgressSession.open(
        storage, ["prune"], "fingerprint", REFERENCE_TIME
    )
    assert session is not None
    await session.complete_strategy("prune")
    saved = storage.save_consolidation_progress

    async def corrupt_ack(brain_id: str, token: str, state: dict[str, Any]) -> None:
        await saved(
            brain_id, token, {**state, "requested_strategies": [], "completed_strategies": []}
        )
        raise ConnectionError("synthetic lost acknowledgement")

    monkeypatch.setattr(storage, "save_consolidation_progress", corrupt_ack)
    try:
        with pytest.raises(ConsolidationCompletionUnconfirmedError):
            await session.complete()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_cancellation_after_terminal_commit_does_not_pause_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    engine = ConsolidationEngine(storage)
    monkeypatch.setattr(engine, "_run_strategy", AsyncMock())
    monkeypatch.setattr(
        engine, "_prune_semantic_discovery_state", AsyncMock(side_effect=asyncio.CancelledError())
    )
    with pytest.raises(asyncio.CancelledError):
        await engine.run([ConsolidationStrategy.SEMANTIC_LINK], reference_time=REFERENCE_TIME)
    assert storage.progress is not None
    assert storage.progress["status"] == "completed"


@pytest.mark.asyncio
async def test_useful_work_marker_survives_resume_without_counting_coordination() -> None:
    storage = FakeProgressStorage()
    session = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert session is not None
    assert "work_progress_seq" not in session.state
    await session.checkpoint("prune", "starting")
    await session.pause()
    await session.close()
    resumed = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert resumed is not None
    assert "work_progress_seq" not in resumed.state
    await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:9")
    assert resumed.state["work_progress_seq"] == 1
    timestamp = resumed.state["last_work_progress_at"]
    await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:9")
    await resumed.finalization_checkpoint("report")
    await resumed.pause()
    assert resumed.state["work_progress_seq"] == 1
    assert resumed.state["last_work_progress_at"] == timestamp
    await resumed.close()
    restarted = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert restarted is not None
    await restarted.checkpoint("prune", "synapse_scan", cursor="synapse:10")
    assert restarted.state["work_progress_seq"] == 2
    await restarted.complete_strategy("prune")
    await restarted.complete_strategy("prune")
    assert restarted.state["work_progress_seq"] == 3
    await restarted.close()


@pytest.mark.asyncio
async def test_legacy_work_checkpoint_stays_unknown_until_new_committed_unit() -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None
    await first.checkpoint("prune", "synapse_scan", cursor="synapse:10", counters={"deleted": 5})
    await first.close()
    assert storage.progress is not None
    storage.progress.pop("native_handoff")
    storage.progress.pop("work_progress_seq")
    storage.progress.pop("last_work_progress_at")
    storage.progress["strategy_states"]["prune"].pop("work_progress")
    legacy = deepcopy(storage.progress)
    resumed = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert resumed is not None
    for key in (
        "run_id",
        "reference_time",
        "engine_version",
        "schema_version",
        "options_fingerprint",
        "completed_strategies",
        "counters",
        "strategy_states",
    ):
        assert resumed.state[key] == legacy[key]
    await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:10", counters={"deleted": 5})
    assert "work_progress_seq" not in resumed.state
    await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:11", counters={"deleted": 6})
    assert resumed.state["work_progress_seq"] == 1
    await resumed.close()


@pytest.mark.asyncio
async def test_scoped_semantic_work_does_not_count_replay_revision_or_generation_reset() -> None:
    storage = FakeProgressStorage()
    session = await ConsolidationProgressSession.open(
        storage, ["semantic_link"], "same", REFERENCE_TIME
    )
    assert session is not None
    await session.checkpoint(
        "semantic_link",
        "semantic_link_discovery_neurons",
        cursor="changed",
        pending=['{"source_state_ref":{"revision":999}}'],
        counters={"validation_cursor": 999, "state_revision": 999},
    )
    assert "work_progress_seq" not in session.state
    for axis, scope, before, after in (
        ("validation", "target-a", 10, 20),
        ("neuron_count", "source-a", 0, 100),
        ("similarity", "fingerprint-a", 0, 9),
        ("apply", "batch-a", "", "synapse:2"),
    ):
        await session.checkpoint("semantic_link", "work", work={axis: (scope, before, after)})
    assert session.state["work_progress_seq"] == 4
    timestamp = session.state["last_work_progress_at"]
    await session.close()
    resumed = await ConsolidationProgressSession.open(
        storage, ["semantic_link"], "same", REFERENCE_TIME
    )
    assert resumed is not None
    for work in (
        {"validation": ("target-a", 10, 20)},
        {"neuron_count": ("source-a", 0, 100)},
        {"similarity": ("fingerprint-a", 0, 9)},
        {"apply": ("batch-a", "", "synapse:2")},
        {"neuron_count": ("source-b", 0, 0)},
        {"similarity": ("fingerprint-b", 0, 0)},
        {"apply": ("batch-b", "", "")},
    ):
        await resumed.checkpoint("semantic_link", "work", work=work)
    assert resumed.state["work_progress_seq"] == 4
    assert resumed.state["last_work_progress_at"] == timestamp
    await resumed.checkpoint("semantic_link", "work", work={"neuron_count": ("source-b", 0, 1)})
    assert resumed.state["work_progress_seq"] == 5
    await resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_uncertain_work_save_fences_retry_and_restart_does_not_double_count(
    committed: bool,
) -> None:
    storage = FakeProgressStorage()
    first = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert first is not None
    original = storage.save_consolidation_progress

    async def uncertain(*args: Any, **kwargs: Any) -> Any:
        if committed:
            await original(*args, **kwargs)
        raise OSError("lost acknowledgement")

    storage.save_consolidation_progress = uncertain  # type: ignore[method-assign]
    with pytest.raises(OSError, match="acknowledgement"):
        await first.checkpoint("prune", "synapse_scan", cursor="synapse:1")
    with pytest.raises(progress_module.ConsolidationCheckpointUnconfirmedError):
        await first.pause()
    storage.save_consolidation_progress = original  # type: ignore[method-assign]
    await first.close()
    resumed = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert resumed is not None
    assert resumed.state.get("work_progress_seq") == (1 if committed else None)
    await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:1")
    assert resumed.state["work_progress_seq"] == 1
    before = deepcopy(storage.progress)
    storage.reject_saves = True
    with pytest.raises(ConsolidationLeaseLostError):
        await resumed.checkpoint("prune", "synapse_scan", cursor="synapse:2")
    assert storage.progress == before
    await resumed.close()


@pytest.mark.asyncio
async def test_semantic_engine_publishes_real_coverage_and_apply_work_across_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_consolidation_mid_strategies_resume import _engine
    from tests.unit.test_semantic_discovery_cap_continuation import _semantic_link_fixture

    graph, fixture_progress, reference = _semantic_link_fixture(monkeypatch)
    storage = FakeProgressStorage()
    storage.current_brain_id = graph.current_brain_id
    checkpoints: list[dict[str, Any]] = []
    save = storage.save_consolidation_progress

    async def capture(*args: Any, **kwargs: Any) -> Any:
        saved = await save(*args, **kwargs)
        checkpoints.append(deepcopy(saved))
        return saved

    monkeypatch.setattr(storage, "save_consolidation_progress", capture)
    for _ in range(15):
        session = await ConsolidationProgressSession.open(
            storage, ["semantic_link"], "same", reference
        )
        assert session is not None
        engine = _engine(graph, ConsolidationStrategy.SEMANTIC_LINK, fixture_progress)
        engine._progress_session = session
        try:
            await engine._semantic_link(ConsolidationReport(), dry_run=False)
        except progress_module.ConsolidationPausedError:
            pass
        finally:
            await session.close()
        if session.strategy_state("semantic_link").get("phase") == "semantic_link_complete":
            break
    else:
        pytest.fail("capped semantic work did not finish")
    assert graph.added_synapses
    assert storage.progress is not None
    assert storage.progress["work_progress_seq"] > 1
    assert isinstance(storage.progress["last_work_progress_at"], datetime)
    previous_seq = 0
    progressed_phases = set()
    for checkpoint in checkpoints:
        seq = checkpoint.get("work_progress_seq", 0)
        assert seq >= previous_seq
        if seq > previous_seq:
            progressed_phases.add(checkpoint["phase"])
        previous_seq = seq
    assert "semantic_link_discovery_neurons" in progressed_phases
    assert "semantic_link_discovery_similarity" in progressed_phases
    assert "semantic_link_apply" in progressed_phases


@pytest.mark.asyncio
async def test_structural_metadata_and_phase_changes_are_not_work() -> None:
    storage = FakeProgressStorage()
    session = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert session is not None
    await session.checkpoint(
        "prune",
        "report",
        cursor='{"hash":"a","state_revision":1}',
        counters={"state_revision": 1, "heartbeat": 1, "generation": 1},
    )
    await session.checkpoint(
        "prune",
        "another_report",
        cursor='{"hash":"b","state_revision":2}',
        counters={"state_revision": 2, "heartbeat": 2, "generation": 2},
    )
    assert "work_progress_seq" not in session.state
    await session.checkpoint("prune", "synapse_scan", cursor="synapse:10")
    seq = session.state["work_progress_seq"]
    await session.checkpoint("prune", "neuron_scan", cursor="synapse:10")
    assert session.state["work_progress_seq"] == seq
    await session.close()

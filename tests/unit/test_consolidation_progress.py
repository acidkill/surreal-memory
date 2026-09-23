from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

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
    ConsolidationProgressSession,
    ConsolidationResumeMismatchError,
    options_fingerprint,
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
    assert "The next smem consolidate will resume" in first.summary()
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


def test_report_summary_renders_saved_checkpoint_details() -> None:
    report = ConsolidationReport()
    report.extra["dedup_resumed_checkpoint"] = "dedup_window_complete"
    report.extra["last_checkpoint"] = "prune phase=synapse_scan cursor=synapse:100"

    summary = report.summary()

    assert "Dedup resumed from saved checkpoint: dedup_window_complete" in summary
    assert "Last committed checkpoint: prune phase=synapse_scan cursor=synapse:100" in summary

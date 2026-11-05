"""Compression preservation summaries count only the confirmed run prefix."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.engine.compression import CompressionEngine, CompressionTier
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressSession,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow
from tests.unit.test_compression_intent_resume import _compression_engine, _Progress, _scan_store
from tests.unit.test_consolidation_progress import FakeProgressStorage


async def _preserved_store(count: int, monkeypatch: pytest.MonkeyPatch) -> InMemoryStorage:
    store = await _scan_store(count, varied_skips=False)
    for index in range(count):
        fiber = await store.get_fiber(f"f-{index:05d}")
        assert fiber is not None
        await store.update_fiber(replace(fiber, compression_tier=1))
    monkeypatch.setattr(
        CompressionEngine,
        "determine_target_tier",
        lambda _self, _fiber, _now, *, heat_score: CompressionTier.GRAPH_ONLY,
    )
    return store


@pytest.mark.asyncio
async def test_preservation_details_are_debug_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = await _preserved_store(1, monkeypatch)
    fiber = await store.get_fiber("f-00000")
    assert fiber is not None
    engine = CompressionEngine(store)
    with caplog.at_level(logging.INFO, logger="surreal_memory.engine.compression"):
        result = await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)
    assert result.skip_reason == "preservation_conflict"
    assert "Preserving existing compression originals" not in caplog.text

    with caplog.at_level(logging.DEBUG, logger="surreal_memory.engine.compression"):
        await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)
    assert any(
        record.levelno == logging.DEBUG and fiber.id in record.message for record in caplog.records
    )
    assert await store.get_fiber(fiber.id) == fiber
    assert await store.get_compression_backup(fiber.id) is None


@pytest.mark.asyncio
async def test_direct_report_counts_preservation_subset_without_changing_fibers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _preserved_store(3, monkeypatch)
    pinned = await store.get_fiber("f-00002")
    assert pinned is not None
    await store.update_fiber(replace(pinned, pinned=True))
    before = await store.get_fibers(limit=10)

    report = await CompressionEngine(store).run()

    assert report.fibers_preserved == 2
    assert report.fibers_skipped == 3
    assert report.fibers_compressed == report.tokens_saved == 0
    assert await store.get_fibers(limit=10) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
async def test_engine_counts_preservation_without_durable_progress(
    monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    store = await _preserved_store(3, monkeypatch)
    report = ConsolidationReport()
    await ConsolidationEngine(store)._compress(report, utcnow(), dry_run)
    assert report.fibers_preserved == (0 if dry_run else 3)
    assert report.compression_counts_complete is True


@pytest.mark.asyncio
async def test_page_pause_and_repeated_resume_do_not_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _preserved_store(300, monkeypatch)
    progress = _Progress()
    progress.fail = False
    engine = _compression_engine(store, progress)
    checked = 0
    pause = True

    async def budget() -> None:
        nonlocal checked
        checked += 1
        if pause and checked == 275:
            raise ConsolidationPausedError("midpage pause")

    monkeypatch.setattr(engine, "_check_progress_budget", budget)
    partial = ConsolidationReport()
    with pytest.raises(ConsolidationPausedError, match="midpage"):
        await engine._compress(partial, utcnow(), False)
    assert partial.fibers_preserved == 250
    saved = progress.strategy_state("compress")
    assert saved["cursor"] == "f-00249"
    assert saved["counters"]["fibers_preserved"] == 250
    assert saved["counters"]["compression_counts_complete"] == 1

    pause = False
    for _attempt in range(2):
        report = ConsolidationReport()
        await engine._compress(report, utcnow(), False)
        assert report.fibers_preserved == 300
        assert report.compression_counts_complete is True
    assert progress.writes == [("compress_fibers", "f-00249"), ("compress_fibers", "f-00299")]


@pytest.mark.asyncio
async def test_cancel_before_checkpoint_commit_restores_confirmed_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedProgress(_Progress):
        interrupt = True

        async def checkpoint(self, strategy: str, phase: str, **kwargs: Any) -> None:
            if self.interrupt:
                self.interrupt = False
                # Like the real session, stage counters before storage acknowledges.
                states = dict(self.state["strategy_states"])
                states[strategy] = {**self.strategy_state(strategy), "phase": phase, **kwargs}
                self.state["strategy_states"] = states
                raise asyncio.CancelledError()
            await super().checkpoint(strategy, phase, **kwargs)

    store = await _preserved_store(300, monkeypatch)
    progress = InterruptedProgress()
    progress.fail = False
    engine = _compression_engine(store, progress)
    partial = ConsolidationReport()
    with pytest.raises(asyncio.CancelledError):
        await engine._compress(partial, utcnow(), False)
    assert partial.fibers_preserved == 0
    assert not progress.strategy_state("compress").get("cursor")

    report = ConsolidationReport()
    await engine._compress(report, utcnow(), False)
    assert report.fibers_preserved == 300


@pytest.mark.asyncio
@pytest.mark.parametrize(("cancel_cursor", "confirmed"), [("f-00249", 0), ("f-00299", 250)])
async def test_real_session_cancellation_resumes_the_durable_prefix(
    monkeypatch: pytest.MonkeyPatch, cancel_cursor: str, confirmed: int
) -> None:
    class Storage(FakeProgressStorage):
        interrupt = True

        async def save_consolidation_progress(
            self, brain_id: str, owner_token: str, state: dict[str, Any]
        ) -> dict[str, Any] | None:
            if self.interrupt and state.get("cursor") == cancel_cursor:
                self.interrupt = False
                raise asyncio.CancelledError()
            return await super().save_consolidation_progress(brain_id, owner_token, state)

    store = await _preserved_store(300, monkeypatch)
    durable = Storage()
    reference_time = utcnow()
    session = await ConsolidationProgressSession.open(
        durable, ["compress"], "preservation-test", reference_time
    )
    assert session is not None
    engine = ConsolidationEngine(store)
    engine._active_strategy = ConsolidationStrategy.COMPRESS
    engine._progress_session = session
    partial = ConsolidationReport()
    with pytest.raises(asyncio.CancelledError):
        await engine._compress(partial, reference_time, False)
    assert partial.fibers_preserved == confirmed
    assert durable.progress is not None
    saved_counters = durable.progress.get("counters", {}).get("compress", {})
    assert saved_counters.get("fibers_preserved", 0) == confirmed
    await session.close()

    resumed_session = await ConsolidationProgressSession.open(
        durable, ["compress"], "preservation-test", reference_time
    )
    assert resumed_session is not None
    engine = ConsolidationEngine(store)
    engine._active_strategy = ConsolidationStrategy.COMPRESS
    engine._progress_session = resumed_session
    report = ConsolidationReport()
    await engine._compress(report, reference_time, False)
    assert report.fibers_preserved == 300
    assert durable.progress["counters"]["compress"]["fibers_preserved"] == 300
    await resumed_session.close()


@pytest.mark.asyncio
async def test_pause_after_checkpoint_commit_reports_committed_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _preserved_store(300, monkeypatch)
    progress = _Progress()
    progress.fail = False
    engine = _compression_engine(store, progress)

    async def budget() -> None:
        if progress.writes:
            raise ConsolidationPausedError("committed pause")

    monkeypatch.setattr(engine, "_check_progress_budget", budget)
    report = ConsolidationReport()
    with pytest.raises(ConsolidationPausedError, match="committed"):
        await engine._compress(report, utcnow(), False)
    assert report.fibers_preserved == 250
    assert progress.strategy_state("compress")["counters"]["fibers_preserved"] == 250


@pytest.mark.asyncio
async def test_old_progressed_checkpoint_keeps_partial_coverage_across_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _preserved_store(12, monkeypatch)
    progress = _Progress()
    progress.fail = False
    progress.strategy_state("compress").update(
        phase="compress_fibers",
        cursor="f-00009",
        counters={"fibers_compressed": 0, "tokens_saved": 0},
    )
    engine = _compression_engine(store, progress)
    for _attempt in range(2):
        report = ConsolidationReport()
        await engine._compress(report, utcnow(), False)
        assert report.fibers_preserved == 2
        assert report.compression_counts_complete is False
    assert progress.strategy_state("compress")["counters"]["compression_counts_complete"] == 0


@pytest.mark.asyncio
async def test_empty_new_scan_persists_an_exact_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    store = await _preserved_store(0, monkeypatch)
    progress = _Progress()
    progress.fail = False
    report = ConsolidationReport()
    await _compression_engine(store, progress)._compress(report, utcnow(), False)
    assert report.fibers_preserved == 0
    assert report.compression_counts_complete is True
    assert progress.strategy_state("compress")["counters"]["fibers_preserved"] == 0
    assert progress.strategy_state("compress")["counters"]["compression_counts_complete"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("compression_completed", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_early_resume_pause_restores_saved_compression_counts(
    monkeypatch: pytest.MonkeyPatch, compression_completed: bool, legacy: bool
) -> None:
    from copy import deepcopy

    store = FakeProgressStorage()
    resume = False
    calls: list[str] = []

    async def strategy(
        engine: ConsolidationEngine, step: ConsolidationStrategy, *_args: Any
    ) -> None:
        calls.append(step.value)
        if step is ConsolidationStrategy.PRUNE:
            if resume:
                raise ConsolidationPausedError("earlier strategy paused")
            raise RuntimeError("earlier strategy failed")
        session = engine._progress_session
        assert session is not None
        counters = {"fibers_compressed": 3, "tokens_saved": 20}
        if not legacy:
            counters.update(fibers_preserved=7, compression_counts_complete=1)
        await session.checkpoint(step.value, "compress_fibers", cursor="f-00006", counters=counters)
        if not compression_completed:
            raise ConsolidationPausedError("compression paused")

    monkeypatch.setattr(ConsolidationEngine, "_run_strategy", strategy)
    requested = [ConsolidationStrategy.PRUNE, ConsolidationStrategy.COMPRESS]
    first = await ConsolidationEngine(store).run(requested)  # type: ignore[arg-type]
    assert first.extra["consolidation_status"] == ("failed" if compression_completed else "paused")
    assert calls == ["prune", "compress"]
    assert store.progress is not None
    saved_compression = deepcopy(store.progress["strategy_states"]["compress"])

    resume = True
    calls.clear()
    report = await ConsolidationEngine(store).run(requested)  # type: ignore[arg-type]

    assert report.extra["consolidation_status"] == "paused"
    assert calls == ["prune"]
    assert report.fibers_compressed == 3
    assert report.tokens_saved == 20
    assert report.fibers_preserved == (0 if legacy else 7)
    assert report.compression_counts_complete is not legacy
    assert store.progress["strategy_states"]["compress"] == saved_compression


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_twenty_completed_strategies_restore_compression_without_rescanning(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    class Storage(FakeProgressStorage):
        offline = True

        async def save_consolidation_progress(
            self, brain_id: str, owner_token: str, state: dict[str, Any]
        ) -> dict[str, Any] | None:
            if self.offline and state["status"] == "completed":
                raise ConnectionError("terminal response unavailable")
            return await super().save_consolidation_progress(brain_id, owner_token, state)

    store = Storage()
    called: list[str] = []

    async def strategy(
        engine: ConsolidationEngine, step: ConsolidationStrategy, *_args: Any
    ) -> None:
        called.append(step.value)
        session = engine._progress_session
        assert session is not None
        counters = {"fibers_compressed": 3, "tokens_saved": 20}
        if not legacy:
            counters.update(fibers_preserved=7, compression_counts_complete=1)
        await session.checkpoint(step.value, "batch", counters=counters)

    monkeypatch.setattr(ConsolidationEngine, "_run_strategy", strategy)
    monkeypatch.setattr(ConsolidationEngine, "_run_auto_tier", AsyncMock())
    first = await ConsolidationEngine(store).run()  # type: ignore[arg-type]
    assert first.extra["consolidation_status"] == "completion_unconfirmed"
    assert len(called) == 20
    store.offline = False
    called.clear()

    resumed = await ConsolidationEngine(store).run()  # type: ignore[arg-type]
    assert resumed.extra["consolidation_status"] == "completed"
    assert called == []
    assert resumed.fibers_compressed == 3
    assert resumed.tokens_saved == 20
    assert resumed.fibers_preserved == (0 if legacy else 7)
    assert resumed.compression_counts_complete is not legacy

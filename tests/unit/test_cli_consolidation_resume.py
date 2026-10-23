"""Tests for honest CLI reporting of resumable consolidation runs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from surreal_memory.cli.commands import tools


def test_paused_consolidation_prints_checkpoint_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta = MagicMock()
    delta.summary.return_value = (
        "Consolidation paused: prune phase=synapse_scan cursor=synapse:100. "
        "The next smem consolidate will resume this checkpoint."
    )
    delta.report.extra = {
        "consolidation_status": "paused",
        "last_checkpoint": "prune phase=synapse_scan cursor=synapse:100",
    }

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=MagicMock()),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate(strategy="prune")

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "last checkpoint" in output.lower() or "phase=synapse_scan" in output
    assert "next smem consolidate will resume" in output


def test_resume_notice_is_echoed_before_consolidation_returns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta = MagicMock()
    delta.summary.return_value = "Consolidation Report"
    delta.report.extra = {"consolidation_status": "completed"}
    notice = "Consolidation: prune progress state found - resuming..."
    callback_events: list[str] = []

    async def fake_run_with_delta(
        _storage: object,
        _brain_id: str,
        *,
        on_progress: Callable[[str], None] | None = None,
        **_kwargs: object,
    ) -> MagicMock:
        assert on_progress is not None
        on_progress(notice)
        callback_events.append("notice emitted before result")
        return delta

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=MagicMock()),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            side_effect=fake_run_with_delta,
        ),
    ):
        tools.consolidate(strategy="prune")

    output = capsys.readouterr().out
    assert output.index("Consolidating brain") < output.index(notice)
    assert output.index(notice) < output.index("Consolidation Report")
    assert callback_events == ["notice emitted before result"]


def test_consolidation_hints_reuse_after_snapshot_without_extra_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta = MagicMock()
    delta.summary.return_value = "Consolidation Report\n\nHealth Delta:"
    delta.after = SimpleNamespace(orphan_rate=0.25, neuron_count=100, consolidation_ratio=0.0)
    delta.report.extra = {"consolidation_status": "completed"}

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=MagicMock()),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ),
        patch(
            "surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze", new_callable=AsyncMock
        ) as analyze,
    ):
        tools.consolidate()

    analyze.assert_not_awaited()
    output = capsys.readouterr().out
    assert "Health Delta:" in output
    assert "Hint: 25 orphan neurons detected (25% of total)." in output
    assert "Run with --strategy prune to clean up" in output
    assert "Hint: No memories have reached SEMANTIC stage yet." in output
    assert "Run with --strategy mature to advance episodic memories." in output


def test_explicit_resume_forwards_saved_strategies_for_engine_checkpoint_skip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta = MagicMock()
    delta.summary.return_value = "Consolidation Report"
    delta.after = SimpleNamespace(orphan_rate=0.0, neuron_count=0, consolidation_ratio=1.0)
    delta.report.extra = {"consolidation_status": "completed"}
    state = {
        "run_id": "resume-run-123",
        "status": "paused",
        "requested_strategies": ["prune", "mature", "merge"],
        "completed_strategies": ["prune", "mature"],
    }

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock) as get_storage,
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ) as run_with_delta,
    ):
        get_storage.return_value.get_consolidation_progress = AsyncMock(return_value=state)
        tools.consolidate(resume=True)

    get_storage.return_value.get_consolidation_progress.assert_awaited_once_with("test-brain")
    kwargs = run_with_delta.await_args.kwargs
    assert [strategy.value for strategy in kwargs["strategies"]] == [
        "prune",
        "mature",
        "merge",
    ]
    assert kwargs["resume_run_id"] == "resume-run-123"
    capsys.readouterr()


def test_failed_resume_reports_saved_checkpoint_counters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy

    completed = [
        item.value
        for item in ConsolidationStrategy
        if item not in {ConsolidationStrategy.ALL, ConsolidationStrategy.SEMANTIC_LINK}
    ]
    state = {
        "run_id": "resume-run-123",
        "status": "paused",
        "requested_strategies": [*completed, "semantic_link"],
        "completed_strategies": completed,
        "counters": {
            "prune": {"synapses_pruned": 10641},
            "semantic_link": {"semantic_synapses_created": 4},
        },
    }
    delta = MagicMock()
    delta.report = ConsolidationReport()
    delta.after = SimpleNamespace(orphan_rate=0.25, neuron_count=100, consolidation_ratio=0.0)
    delta.report.extra["consolidation_status"] = "failed"
    delta.report.extra["failed_strategies"] = [
        "semantic_link (progress coordination: ConsolidationProgressError)"
    ]
    delta.summary.return_value = delta.report.summary()

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock) as get_storage,
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ),
    ):
        get_storage.return_value.get_consolidation_progress = AsyncMock(return_value=state)
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate(resume=True)

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "19 completed strategies" in output
    assert "prune.synapses_pruned=10641" in output
    assert "semantic_link.semantic_synapses_created=4" in output
    assert "Hint:" not in output
    assert "Stages failed: semantic_link" in output


@pytest.mark.parametrize(
    "state",
    [
        None,
        {
            "run_id": "finished-run",
            "requested_strategies": ["prune"],
            "completed_strategies": ["prune"],
        },
        {"run_id": "", "requested_strategies": ["prune", "bogus"], "completed_strategies": []},
    ],
)
def test_explicit_resume_refuses_when_checkpoint_is_missing_complete_or_invalid(
    state: object, capsys: pytest.CaptureFixture[str]
) -> None:

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock) as get_storage,
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
        ) as run_with_delta,
    ):
        get_storage.return_value.get_consolidation_progress = AsyncMock(return_value=state)
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate(resume=True)

    assert exc_info.value.exit_code == 1
    assert "no valid incomplete consolidation checkpoint" in capsys.readouterr().err.lower()
    run_with_delta.assert_not_awaited()


@pytest.mark.parametrize(
    ("strategy", "dry_run"),
    [("all", False), ("prune", False), (None, True)],
)
def test_explicit_resume_rejects_strategy_selection_and_dry_run(
    strategy: str | None, dry_run: bool
) -> None:
    with patch.object(tools, "run_async") as run_async:
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate(resume=True, strategy=strategy, dry_run=dry_run)

    assert exc_info.value.exit_code == 1
    run_async.assert_not_called()


def test_blocked_consolidation_suppresses_health_suggestions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta = MagicMock()
    delta.summary.return_value = "Consolidation blocked"
    delta.after = SimpleNamespace(orphan_rate=0.5, neuron_count=20, consolidation_ratio=0.0)
    delta.report.extra = {"consolidation_status": "blocked"}

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=MagicMock()),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate()

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Consolidation blocked" in output
    assert "Hint:" not in output

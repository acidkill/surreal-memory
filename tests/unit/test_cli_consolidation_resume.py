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
        tools.consolidate(strategy="all")

    analyze.assert_not_awaited()
    output = capsys.readouterr().out
    assert "Health Delta:" in output
    assert "Hint: 25 orphan neurons detected (25% of total)." in output
    assert "Run with --strategy prune to clean up" in output
    assert "Hint: No memories have reached SEMANTIC stage yet." in output
    assert "Run with --strategy mature to advance episodic memories." in output

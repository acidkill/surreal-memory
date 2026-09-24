"""Operator-only CLI guardrails for legacy semantic discovery recovery."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from surreal_memory.cli.commands import tools


@pytest.fixture
def recovery_storage() -> MagicMock:
    storage = MagicMock()
    storage.get_consolidation_progress = AsyncMock(
        return_value={
            "run_id": "run-123",
            "status": "paused",
            "options_fingerprint": "fingerprint-123",
        }
    )
    return storage


def test_recovery_cli_preview_is_read_only(
    recovery_storage: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {"run_id": "run-123", "reference_time": "2026-09-24T00:00:00Z"}
    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="default"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=recovery_storage),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_progress.recover_legacy_semantic_link_discovery",
            new_callable=AsyncMock,
            return_value=result,
        ) as recovery,
    ):
        tools.recover_semantic_discovery(run_id="run-123")
    recovery.assert_awaited_once()
    assert recovery.await_args.kwargs["dry_run"] is True
    assert recovery.await_args.kwargs["expected_options_fingerprint"] == "fingerprint-123"
    assert "No changes were made" in capsys.readouterr().out


def test_recovery_cli_requires_typed_run_id_before_execute(
    recovery_storage: MagicMock,
) -> None:
    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="default"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=recovery_storage),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_progress.recover_legacy_semantic_link_discovery",
            new_callable=AsyncMock,
        ) as recovery,
        pytest.raises(typer.Exit),
    ):
        tools.recover_semantic_discovery(run_id="run-123", confirm_run_id="different", execute=True)
    recovery.assert_not_awaited()


def test_recovery_cli_execute_uses_current_run_fingerprint(
    recovery_storage: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="default"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=recovery_storage),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_progress.recover_legacy_semantic_link_discovery",
            new_callable=AsyncMock,
            return_value={"run_id": "run-123", "reference_time": "2026-09-24T00:00:00Z"},
        ) as recovery,
    ):
        tools.recover_semantic_discovery(run_id="run-123", confirm_run_id="run-123", execute=True)
    recovery.assert_awaited_once()
    assert recovery.await_args.kwargs["dry_run"] is False
    assert recovery.await_args.kwargs["expected_status"] == "paused"
    assert "completed strategies were preserved" in capsys.readouterr().out

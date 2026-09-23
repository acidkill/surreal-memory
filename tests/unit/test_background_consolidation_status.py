from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.server.app import _consolidation_loop


@pytest.mark.asyncio
async def test_background_paused_run_is_not_logged_as_complete(
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = SimpleNamespace(brain_id="default")
    maintenance = SimpleNamespace(
        scheduled_consolidation_interval_hours=1,
        scheduled_consolidation_strategies=("prune",),
    )
    report = SimpleNamespace(
        extra={"consolidation_status": "paused"},
        summary=lambda: "paused at prune phase=synapse_scan",
    )
    engine = SimpleNamespace(run=AsyncMock(return_value=report))
    sleeps = 0

    async def stop_after_one_run(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    with (
        patch("asyncio.sleep", stop_after_one_run),
        patch("surreal_memory.engine.consolidation.ConsolidationEngine", return_value=engine),
        caplog.at_level("WARNING"),
        pytest.raises(asyncio.CancelledError),
    ):
        await _consolidation_loop(storage, maintenance)

    assert "Background consolidation paused" in caplog.text
    assert "Background consolidation complete" not in caplog.text

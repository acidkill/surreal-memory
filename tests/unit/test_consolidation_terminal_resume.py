"""A CLI resume must finish the same 20/20 run after an interrupted terminal write."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from typer.testing import CliRunner

from surreal_memory.cli.commands import tools
from surreal_memory.cli.main import app
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.diagnostics import DiagnosticsEngine
from tests.unit.test_consolidation_progress import FakeProgressStorage

if TYPE_CHECKING:
    import pytest


def test_all_completed_resume_after_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Storage(FakeProgressStorage):
        offline_on_complete = True

        async def save_consolidation_progress(
            self, brain_id: str, owner_token: str, state: dict[str, Any]
        ) -> dict[str, Any] | None:
            if self.offline_on_complete and state["status"] == "completed":
                raise ConnectionError("response unavailable")
            return await super().save_consolidation_progress(brain_id, owner_token, state)

    storage = Storage()
    called: list[str] = []

    async def strategy(
        engine: ConsolidationEngine, step: ConsolidationStrategy, *args: Any
    ) -> None:
        called.append(step.value)
        session = engine._progress_session
        assert session is not None
        await session.checkpoint(step.value, "batch", counters={"processed": 17})

    # Simulate a gateway outage for optional health reads in both invocations.
    monkeypatch.setattr(DiagnosticsEngine, "analyze", AsyncMock(side_effect=ConnectionError("502")))
    monkeypatch.setattr(ConsolidationEngine, "_run_strategy", strategy)
    monkeypatch.setattr(tools, "get_storage", AsyncMock(return_value=storage))
    runner = CliRunner()
    interrupted = runner.invoke(app, ["consolidate", "--brain", "default"])
    assert interrupted.exit_code == 1
    assert "completion_unconfirmed" in interrupted.output
    assert storage.progress is not None and storage.progress["status"] == "running"
    before = deepcopy(storage.progress)
    assert set(called) == set(before["completed_strategies"])
    assert len(called) == len(before["requested_strategies"]) == 20
    assert len(before["counters"]) == 20

    storage.offline_on_complete = False
    called.clear()
    resumed = runner.invoke(app, ["consolidate", "--resume", "--brain", "default"])
    assert resumed.exit_code == 0, (resumed.output, resumed.exception)
    assert called == []
    assert storage.progress["status"] == "completed"
    for key in (
        "run_id",
        "reference_time",
        "requested_strategies",
        "completed_strategies",
        "strategy_states",
        "options_fingerprint",
        "counters",
    ):
        assert storage.progress[key] == before[key], key
    assert storage.started == 1
    assert storage.lease_owner is None

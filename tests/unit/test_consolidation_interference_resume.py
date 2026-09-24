"""Durable fan-effect census and aggregation resume without replaying a scan."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

REFERENCE = datetime(2026, 9, 20)


class _Storage:
    current_brain_id = "default"

    def __init__(self) -> None:
        self.neurons = [
            SimpleNamespace(
                id=f"n-{index:04d}",
                created_at=REFERENCE - timedelta(days=1),
                metadata={"tags": [f"tag-{index % 130:03d}"]},
            )
            for index in range(520)
        ]
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.read_cursors: list[str | None] = []
        self.fail_tag_write_once = False

    async def _query(self, sql: str, **params: Any) -> list[Any]:
        raise AssertionError("the test plan handles durable rows in memory")

    async def get_brain(self, brain_id: str) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(interference_detection_enabled=True, fan_effect_threshold=4)
        )

    async def find_neurons_after_id(
        self,
        cursor: str | None,
        *,
        limit: int,
        created_before: datetime,
        include_embedding: bool,
    ) -> list[Any]:
        assert not include_embedding
        self.read_cursors.append(cursor)
        return [
            row
            for row in self.neurons
            if (cursor is None or row.id > cursor) and row.created_at <= created_before
        ][:limit]


class _Plan:
    def __init__(self, storage: _Storage, **identity: Any) -> None:
        self.storage = storage

    async def put_item(self, kind: str, item_key: str, fields: dict[str, Any]) -> None:
        await self.put_items([(kind, item_key, fields)])

    async def put_items(self, items: list[tuple[str, str, dict[str, Any]]]) -> None:
        if self.storage.fail_tag_write_once and items[0][0] == "interference_tag_page":
            self.storage.fail_tag_write_once = False
            raise OSError("simulated failure after immutable page marker")
        for kind, key, fields in items:
            previous = self.storage.items.setdefault((kind, key), fields.copy())
            if previous != fields:
                raise ValueError("immutable interference source changed")

    async def iter_items(
        self, kind: str, *, after: str = ""
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        for (row_kind, key), value in sorted(self.storage.items.items()):
            if row_kind == kind and key > after:
                yield key, value


class _Progress:
    def __init__(self, pause_on: tuple[str, int] | None = None) -> None:
        self.state: dict[str, Any] = {"run_id": "run", "strategy_states": {}}
        self.pause_on = pause_on
        self.phases: dict[str, int] = {}

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return self.state["strategy_states"].setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        self.strategy_state(strategy).update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )
        self.phases[phase] = self.phases.get(phase, 0) + 1
        if self.pause_on == (phase, self.phases[phase]):
            raise ConsolidationPausedError("test interruption after committed checkpoint")


def _engine(storage: _Storage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage)  # type: ignore[arg-type]
    engine._active_strategy = ConsolidationStrategy.INTERFERENCE
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest.mark.parametrize("pause_on", [("interference_scan", 1), ("interference_aggregate", 1)])
@pytest.mark.asyncio
async def test_interference_resumes_scan_or_aggregation_without_double_counting(
    monkeypatch: pytest.MonkeyPatch,
    pause_on: tuple[str, int],
) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.consolidation.SurrealDBConsolidationGroupPlan", _Plan
    )
    storage = _Storage()
    progress = _Progress(pause_on)
    with pytest.raises(ConsolidationPausedError, match="test interruption"):
        await _engine(storage, progress)._interference(
            ConsolidationReport(), REFERENCE, dry_run=False
        )
    cursor = progress.strategy_state("interference")["cursor"]
    storage.read_cursors.clear()
    progress.pause_on = None
    report = ConsolidationReport()
    await _engine(storage, progress)._interference(report, REFERENCE, dry_run=False)
    assert report.extra["interference_fan_effects"] == 130
    assert progress.strategy_state("interference")["phase"] == "completed"
    if pause_on[0] == "interference_scan":
        assert storage.read_cursors[0] == cursor == "n-0249"
    else:
        assert storage.read_cursors == []


@pytest.mark.asyncio
async def test_interference_detects_source_change_between_marker_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.consolidation.SurrealDBConsolidationGroupPlan", _Plan
    )
    storage = _Storage()
    storage.fail_tag_write_once = True
    progress = _Progress()
    with pytest.raises(OSError, match="simulated failure"):
        await _engine(storage, progress)._interference(
            ConsolidationReport(), REFERENCE, dry_run=False
        )
    storage.neurons[0].metadata["tags"] = ["changed-tag"]
    with pytest.raises(ValueError, match="source changed"):
        await _engine(storage, progress)._interference(
            ConsolidationReport(), REFERENCE, dry_run=False
        )
    assert progress.strategy_state("interference").get("phase") != "completed"

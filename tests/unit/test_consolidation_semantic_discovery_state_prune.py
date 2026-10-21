"""Consolidation deletes semantic-discovery snapshots once semantic_link completes."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from surreal_memory import unified_config
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    INCOMPLETE_STATUSES,
    ConsolidationPausedError,
)
from surreal_memory.storage.surrealdb.consolidation_state import (
    SurrealDBConsolidationStateMixin,
)
from surreal_memory.unified_config import MaintenanceConfig, UnifiedConfig
from tests.unit.test_consolidation_progress import FakeProgressStorage


class _PruneStorage(FakeProgressStorage):
    def __init__(self) -> None:
        super().__init__()
        self.unfinished_run_ids = ["run-paused-elsewhere"]
        self.prune_calls: list[dict[str, Any]] = []
        self.prune_result = 3
        self.prune_error: Exception | None = None

    async def get_consolidation_run_ids(
        self, *, statuses: Collection[str], brain_id: str | None = None
    ) -> list[str]:
        assert set(statuses) == INCOMPLETE_STATUSES
        return list(self.unfinished_run_ids)

    async def prune_semantic_discovery_state(
        self, *, older_than: timedelta, keep_run_ids: Collection[str]
    ) -> int:
        assert self.progress is not None
        self.prune_calls.append(
            {
                "older_than": older_than,
                "keep_run_ids": set(keep_run_ids),
                "completed_strategies": list(self.progress["completed_strategies"]),
            }
        )
        if self.prune_error is not None:
            raise self.prune_error
        return self.prune_result


def _retention(monkeypatch: pytest.MonkeyPatch, days: int) -> None:
    cfg = unified_config.get_config()
    maintenance = replace(cfg.maintenance, semantic_discovery_state_retention_days=days)
    monkeypatch.setattr(unified_config, "get_config", lambda: replace(cfg, maintenance=maintenance))


def _engine(
    storage: FakeProgressStorage,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pause: bool = False,
) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage)

    async def run_strategy(
        strategy: ConsolidationStrategy,
        report: Any,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        if pause:
            raise ConsolidationPausedError("simulated budget stop")

    monkeypatch.setattr(engine, "_run_strategy", run_strategy)
    return engine


@pytest.mark.asyncio
async def test_completed_semantic_link_prunes_snapshots_of_finished_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retention(monkeypatch, 7)
    storage = _PruneStorage()

    report = await _engine(storage, monkeypatch).run([ConsolidationStrategy.SEMANTIC_LINK])

    assert report.extra["consolidation_status"] == "completed"
    assert storage.progress is not None
    run_id = storage.progress["run_id"]
    assert storage.prune_calls == [
        {
            "older_than": timedelta(days=7),
            "keep_run_ids": {"run-paused-elsewhere", run_id},
            # The prune follows the durable completion marker, never precedes it.
            "completed_strategies": ["semantic_link"],
        }
    ]
    assert report.semantic_discovery_states_pruned == 3
    assert "Semantic discovery snapshots pruned: 3" in report.summary()


@pytest.mark.asyncio
async def test_other_strategies_do_not_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    _retention(monkeypatch, 7)
    storage = _PruneStorage()

    report = await _engine(storage, monkeypatch).run([ConsolidationStrategy.PRUNE])

    assert report.extra["consolidation_status"] == "completed"
    assert storage.prune_calls == []
    assert report.semantic_discovery_states_pruned == 0


@pytest.mark.asyncio
async def test_paused_semantic_link_keeps_every_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _retention(monkeypatch, 7)
    storage = _PruneStorage()

    report = await _engine(storage, monkeypatch, pause=True).run(
        [ConsolidationStrategy.SEMANTIC_LINK]
    )

    assert report.extra["consolidation_status"] == "paused"
    assert storage.prune_calls == []


@pytest.mark.asyncio
async def test_zero_retention_disables_the_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    _retention(monkeypatch, 0)
    storage = _PruneStorage()

    report = await _engine(storage, monkeypatch).run([ConsolidationStrategy.SEMANTIC_LINK])

    assert report.extra["consolidation_status"] == "completed"
    assert storage.prune_calls == []
    assert "Semantic discovery snapshots pruned: 0" in report.summary()


@pytest.mark.asyncio
async def test_prune_failure_is_reported_without_failing_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retention(monkeypatch, 7)
    storage = _PruneStorage()
    storage.prune_error = RuntimeError("datastore unavailable")

    report = await _engine(storage, monkeypatch).run([ConsolidationStrategy.SEMANTIC_LINK])

    assert report.extra["consolidation_status"] == "completed"
    assert storage.progress is not None
    assert storage.progress["status"] == "completed"
    assert report.semantic_discovery_states_pruned == 0
    assert report.extra["semantic_discovery_state_prune_error"] == (
        "RuntimeError: datastore unavailable"
    )
    assert (
        "Semantic discovery snapshots pruned: 0 [prune FAILED: RuntimeError: datastore unavailable]"
        in report.summary()
    )


@pytest.mark.asyncio
async def test_dry_run_never_prunes(monkeypatch: pytest.MonkeyPatch) -> None:
    _retention(monkeypatch, 7)
    storage = _PruneStorage()

    report = await _engine(storage, monkeypatch).run(
        [ConsolidationStrategy.SEMANTIC_LINK], dry_run=True
    )

    assert report.extra["consolidation_status"] == "dry_run"
    assert storage.prune_calls == []


@pytest.mark.asyncio
async def test_backend_without_prune_support_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retention(monkeypatch, 7)
    storage = FakeProgressStorage()

    report = await _engine(storage, monkeypatch).run([ConsolidationStrategy.SEMANTIC_LINK])

    assert report.extra["consolidation_status"] == "completed"
    assert "semantic_discovery_state_prune_error" not in report.extra


class _RunIdStorage(SurrealDBConsolidationStateMixin):
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return "brain-a"

    async def _query(self, sql: str, **params: Any) -> list[Any]:
        self.queries.append((sql, params))
        return ["run-paused", None, "run-failed"]


@pytest.mark.asyncio
async def test_run_ids_are_read_by_brain_and_status() -> None:
    storage = _RunIdStorage()

    run_ids = await storage.get_consolidation_run_ids(statuses=INCOMPLETE_STATUSES)

    assert run_ids == ["run-paused", "run-failed"]
    sql, params = storage.queries[-1]
    assert "brain_id = $brain_id AND status IN $statuses" in sql
    assert params == {"brain_id": "brain-a", "statuses": sorted(INCOMPLETE_STATUSES)}
    with pytest.raises(TypeError, match="not a string"):
        await storage.get_consolidation_run_ids(statuses="paused")


def test_retention_defaults_to_seven_days_and_round_trips(tmp_path: Path) -> None:
    assert MaintenanceConfig().semantic_discovery_state_retention_days == 7
    assert MaintenanceConfig.from_dict({}).semantic_discovery_state_retention_days == 7
    disabled = MaintenanceConfig(semantic_discovery_state_retention_days=0)
    assert MaintenanceConfig.from_dict(disabled.to_dict()) == disabled

    UnifiedConfig(data_dir=tmp_path, maintenance=disabled).save()
    loaded = UnifiedConfig.load(tmp_path / "config.toml")
    assert loaded.maintenance.semantic_discovery_state_retention_days == 0

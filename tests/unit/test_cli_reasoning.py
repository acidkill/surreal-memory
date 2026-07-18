"""Tests for the `smem reasoning` CLI (status / mine / patterns / clear).

Drives the real Typer app via CliRunner with the reasoning command's storage +
config patched (no live DB). The command imports get_config/get_storage from its
own module, and get_unified_config / the engine fns lazily inside handlers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from surreal_memory.cli.main import app
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

runner = CliRunner()

CLI = "surreal_memory.cli.commands.reasoning"


def _fiber(
    fid: str, model: str, category: str, title: str = "t", conf: float = 1.0, freq: int = 3
) -> SimpleNamespace:
    return SimpleNamespace(
        id=fid,
        summary=title,
        metadata={
            "_reasoning_pattern": True,
            "_source_model": model,
            "_reasoning_category": category,
            "_reasoning_title": title,
            "_reasoning_confidence": conf,
            "_reasoning_frequency": freq,
        },
    )


def _storage(stats: dict | None = None, fibers: list | None = None, deleted: int = 0) -> AsyncMock:
    s = AsyncMock()
    s.brain_id = "default"
    s.get_reasoning_stats = AsyncMock(
        return_value=stats or {"by_model": {}, "by_category": {}, "total": 0, "unprocessed": 0}
    )
    s.find_fibers = AsyncMock(return_value=fibers or [])
    s.delete_reasoning_traces_by_model = AsyncMock(return_value=deleted)
    s.close = AsyncMock()
    return s


def _ucfg(tmp_path: Path, **rt: object) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(**rt),  # type: ignore[arg-type]
    )


def test_status_json(tmp_path: Path) -> None:
    storage = _storage(
        stats={
            "by_model": {
                "claude-fable-5": {"trace_count": 5, "unprocessed": 2, "last_trace_at": "x"}
            },
            "by_category": {},
            "total": 5,
            "unprocessed": 2,
        },
        fibers=[_fiber("p1", "claude-fable-5", "debugging")],
    )
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
        patch("surreal_memory.unified_config.get_config", return_value=_ucfg(tmp_path)),
        patch(
            "surreal_memory.engine.reasoning_distiller.reasoning_coverage",
            new=AsyncMock(
                return_value={"by_category": {}, "covered": {}, "coverage_percent": 12.5}
            ),
        ),
    ):
        result = runner.invoke(app, ["reasoning", "status", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total_traces"] == 5
    assert data["total_patterns"] == 1
    assert data["models"][0]["model"] == "claude-fable-5"
    assert data["models"][0]["coverage_percent"] == 12.5


def test_patterns_filter_by_model(tmp_path: Path) -> None:
    storage = _storage(
        fibers=[
            _fiber("p1", "claude-fable-5", "debugging", "verify"),
            _fiber("p2", "claude-sonnet-5", "planning", "plan"),
        ]
    )
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
    ):
        result = runner.invoke(
            app, ["reasoning", "patterns", "--model", "claude-fable-5", "--json"]
        )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 1
    assert data["patterns"][0]["source_model"] == "claude-fable-5"


def test_clear_force(tmp_path: Path) -> None:
    storage = _storage(deleted=3)
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
    ):
        result = runner.invoke(
            app, ["reasoning", "clear", "--model", "claude-fable-5", "--force", "--json"]
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["deleted"] == 3
    storage.delete_reasoning_traces_by_model.assert_awaited_once_with("default", "claude-fable-5")


def test_mine_disabled_exits_nonzero(tmp_path: Path) -> None:
    storage = _storage()
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
        patch("surreal_memory.unified_config.get_config", return_value=_ucfg(tmp_path)),
    ):
        result = runner.invoke(app, ["reasoning", "mine"])

    assert result.exit_code == 1  # mining disabled, no --force


def test_mine_runs_when_enabled(tmp_path: Path) -> None:
    storage = _storage()
    ingest = SimpleNamespace(traces_ingested=4, traces_scanned=10, files_scanned=3, files_total=3)
    distill = SimpleNamespace(patterns_learned=2, traces_processed=4, models_seen=1)
    ingest_mock = AsyncMock(return_value=ingest)
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
        patch(
            "surreal_memory.unified_config.get_config",
            return_value=_ucfg(tmp_path, mining_enabled=True),
        ),
        patch("surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", new=ingest_mock),
        patch(
            "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns",
            new=AsyncMock(return_value=distill),
        ),
    ):
        result = runner.invoke(app, ["reasoning", "mine", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["traces_ingested"] == 4
    assert data["patterns_learned"] == 2
    # No --backfill flag → backfill must default to False (never a silent full rescan).
    assert ingest_mock.await_args.kwargs["backfill"] is False


def test_mine_dry_run(tmp_path: Path) -> None:
    storage = _storage()
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
        patch(
            "surreal_memory.unified_config.get_config",
            return_value=_ucfg(tmp_path, mining_enabled=True),
        ),
    ):
        result = runner.invoke(app, ["reasoning", "mine", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["dry_run"] is True
    assert data["traces_ingested"] == 0


def test_mine_force_bypasses_disabled(tmp_path: Path) -> None:
    # --force runs mining even when mining_enabled is False (the privacy escape hatch).
    storage = _storage()
    ingest = SimpleNamespace(traces_ingested=1, traces_scanned=1, files_scanned=1, files_total=1)
    distill = SimpleNamespace(patterns_learned=0, traces_processed=1, models_seen=1)
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
        patch("surreal_memory.unified_config.get_config", return_value=_ucfg(tmp_path)),
        patch(
            "surreal_memory.engine.reasoning_miner.ingest_reasoning_traces",
            new=AsyncMock(return_value=ingest),
        ),
        patch(
            "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns",
            new=AsyncMock(return_value=distill),
        ),
    ):
        result = runner.invoke(app, ["reasoning", "mine", "--force", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["traces_ingested"] == 1


def test_mine_applies_overrides(tmp_path: Path) -> None:
    # --backfill + --models must flow into the run config passed to the engine.
    storage = _storage()
    ingest_mock = AsyncMock(
        return_value=SimpleNamespace(
            traces_ingested=1, traces_scanned=1, files_scanned=1, files_total=1
        )
    )
    distill_mock = AsyncMock(
        return_value=SimpleNamespace(patterns_learned=0, traces_processed=1, models_seen=1)
    )
    with (
        patch(f"{CLI}.get_config", MagicMock()),
        patch(f"{CLI}.get_storage", new=AsyncMock(return_value=storage)),
        patch(
            "surreal_memory.unified_config.get_config",
            return_value=_ucfg(tmp_path, mining_enabled=True),
        ),
        patch("surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", new=ingest_mock),
        patch(
            "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", new=distill_mock
        ),
    ):
        result = runner.invoke(
            app, ["reasoning", "mine", "--backfill", "--models", "a,b", "--json"]
        )

    assert result.exit_code == 0, result.output
    run_cfg = ingest_mock.await_args.args[2]  # (storage, brain_id, config)
    assert run_cfg.reasoning_training.scan_lookback_days == 0
    assert run_cfg.reasoning_training.mining_models == ("a", "b")
    # --backfill must also flow as the real backfill kwarg (full-rescan bypass),
    # not just the scan_lookback_days=0 override.
    assert ingest_mock.await_args.kwargs["backfill"] is True

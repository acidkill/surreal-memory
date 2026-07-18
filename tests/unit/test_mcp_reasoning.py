"""Tests for the smem_reasoning MCP tool (status / mine / patterns / config).

Builds an MCPServer with a real UnifiedConfig (the mine action does dc_replace on
config, which needs a real dataclass) and a mocked storage, then drives
server.call_tool directly (mirrors test_mcp.py's TestMCPToolCalls).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig


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


def _make_server(tmp_path: Path, **rt: object) -> MCPServer:
    cfg = UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="test-brain",
        reasoning_training=ReasoningTrainingConfig(**rt),  # type: ignore[arg-type]
    )
    with patch("surreal_memory.mcp.server.get_config", return_value=cfg):
        return MCPServer()


def _storage(stats: dict | None = None, fibers: list | None = None) -> AsyncMock:
    s = AsyncMock()
    s.brain_id = "test-brain"
    s.get_brain = AsyncMock(return_value=MagicMock(id="test-brain"))
    s.get_reasoning_stats = AsyncMock(
        return_value=stats or {"by_model": {}, "by_category": {}, "total": 0, "unprocessed": 0}
    )
    s.find_fibers = AsyncMock(return_value=fibers or [])
    return s


@pytest.mark.asyncio
async def test_reasoning_config_action(tmp_path: Path) -> None:
    server = _make_server(tmp_path, mining_enabled=True)
    result = await server.call_tool("smem_reasoning", {"action": "config"})
    assert result["config"]["mining_enabled"] is True


@pytest.mark.asyncio
async def test_reasoning_status(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
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
        patch.object(server, "get_storage", return_value=storage),
        patch(
            "surreal_memory.engine.reasoning_distiller.reasoning_coverage",
            new=AsyncMock(
                return_value={"by_category": {}, "covered": {}, "coverage_percent": 12.5}
            ),
        ),
    ):
        result = await server.call_tool("smem_reasoning", {"action": "status"})

    assert result["total_traces"] == 5
    assert result["total_patterns"] == 1
    assert result["models"][0]["model"] == "claude-fable-5"
    assert result["models"][0]["coverage_percent"] == 12.5


@pytest.mark.asyncio
async def test_reasoning_patterns_filter(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    storage = _storage(
        fibers=[
            _fiber("p1", "claude-fable-5", "debugging", "verify"),
            _fiber("p2", "claude-sonnet-5", "planning", "plan"),
        ]
    )
    with patch.object(server, "get_storage", return_value=storage):
        result = await server.call_tool(
            "smem_reasoning", {"action": "patterns", "model": "claude-fable-5"}
        )

    assert result["count"] == 1
    assert result["patterns"][0]["source_model"] == "claude-fable-5"


@pytest.mark.asyncio
async def test_reasoning_mine_disabled(tmp_path: Path) -> None:
    server = _make_server(tmp_path)  # mining_enabled defaults False
    storage = _storage()
    with patch.object(server, "get_storage", return_value=storage):
        result = await server.call_tool("smem_reasoning", {"action": "mine"})

    assert "error" in result
    assert "disabled" in result["error"].lower()


@pytest.mark.asyncio
async def test_reasoning_mine_runs(tmp_path: Path) -> None:
    server = _make_server(tmp_path, mining_enabled=True)
    storage = _storage()
    ingest = SimpleNamespace(traces_ingested=4, traces_scanned=10, files_scanned=3, files_total=3)
    distill = SimpleNamespace(patterns_learned=2, traces_processed=4, models_seen=1)
    ingest_mock = AsyncMock(return_value=ingest)
    with (
        patch.object(server, "get_storage", return_value=storage),
        # mine runs on an isolated (non-cached) storage to avoid the shared-singleton race.
        patch(
            "surreal_memory.unified_config.create_isolated_storage",
            new=AsyncMock(return_value=storage),
        ),
        patch("surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", new=ingest_mock),
        patch(
            "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns",
            new=AsyncMock(return_value=distill),
        ),
    ):
        result = await server.call_tool("smem_reasoning", {"action": "mine"})

    assert result["traces_ingested"] == 4
    assert result["patterns_learned"] == 2
    # No backfill arg → backfill must default to False (never a silent full rescan).
    assert ingest_mock.await_args.kwargs["backfill"] is False


@pytest.mark.asyncio
async def test_reasoning_mine_applies_overrides(tmp_path: Path) -> None:
    # backfill + models must flow into the run config passed to the engine.
    server = _make_server(tmp_path, mining_enabled=True)
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
        patch.object(server, "get_storage", return_value=storage),
        patch(
            "surreal_memory.unified_config.create_isolated_storage",
            new=AsyncMock(return_value=storage),
        ),
        patch("surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", new=ingest_mock),
        patch(
            "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", new=distill_mock
        ),
    ):
        await server.call_tool(
            "smem_reasoning", {"action": "mine", "backfill": True, "models": ["a", "b"]}
        )

    run_cfg = ingest_mock.await_args.args[2]  # (storage, brain_id, config)
    assert run_cfg.reasoning_training.scan_lookback_days == 0
    assert run_cfg.reasoning_training.mining_models == ("a", "b")
    # backfill must also flow as the real backfill kwarg (full-rescan bypass),
    # not just the scan_lookback_days=0 override.
    assert ingest_mock.await_args.kwargs["backfill"] is True


@pytest.mark.asyncio
async def test_reasoning_unknown_action(tmp_path: Path) -> None:
    server = _make_server(tmp_path)
    storage = _storage()
    with patch.object(server, "get_storage", return_value=storage):
        result = await server.call_tool("smem_reasoning", {"action": "bogus"})

    assert "error" in result
    assert "Unknown action" in result["error"]

"""U4: recall persists a RetrievalTrace only when enabled (or per-call trace=true).

Neutral default (trace.enabled=false) MUST be a true no-op — no build, no task, no
storage write. Per-call trace=true forces one synchronous trace and returns its id.
Config-enabled sampling fires-and-forgets in the background.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig, TraceConfig


def _make_result() -> RetrievalResult:
    return RetrievalResult(
        answer="Emma lives in Bergen.",
        confidence=0.9,
        depth_used=DepthLevel.INSTANT,
        neurons_activated=2,
        fibers_matched=["f-bergen"],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=["a-bergen"]),
        context="Emma lives in Bergen.",
        latency_ms=3.0,
        tokens_used=8,
        synthesis_method="single",
    )


def _make_server(trace_cfg: TraceConfig) -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
        cfg = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
            trace=trace_cfg,
        )
        cfg.write_gate.enabled = False
        mock_get_config.return_value = cfg
        return MCPServer()


async def _run_recall(server: MCPServer, extra_args: dict[str, Any]) -> dict[str, Any]:
    mock_storage = AsyncMock()
    mock_storage.get_brain = AsyncMock(
        return_value=MagicMock(
            id="test-brain", config=MagicMock(trust_weight=0.0, recency_weight=1.0)
        )
    )
    mock_storage._current_brain_id = "test-brain"
    mock_storage.brain_id = "test-brain"
    mock_storage.get_typed_memory = AsyncMock(return_value=None)
    mock_storage.add_retrieval_trace = AsyncMock(return_value="trace-xyz")

    result_obj = _make_result()
    with (
        patch.object(server, "get_storage", return_value=mock_storage),
        patch("surreal_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
        patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
        patch.object(server, "_fire_eternal_trigger"),
        patch.object(server, "_record_tool_action", new_callable=AsyncMock),
        patch.object(server, "_passive_capture", new_callable=AsyncMock),
    ):
        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(return_value=result_obj)
        mock_pipeline_cls.return_value = mock_pipeline
        res = await server.call_tool("smem_recall", {"query": "where does emma live", **extra_args})
        # Drain any fire-and-forget trace tasks before asserting.
        for task in list(getattr(server, "_trace_tasks", set())):
            await task
    res["_storage"] = mock_storage  # smuggle for assertions
    return res


class TestRecallTracePersist:
    @pytest.mark.asyncio
    async def test_default_disabled_is_noop(self) -> None:
        server = _make_server(TraceConfig())  # enabled=False
        res = await _run_recall(server, {})
        storage = res.pop("_storage")
        storage.add_retrieval_trace.assert_not_called()
        assert "trace_id" not in res

    @pytest.mark.asyncio
    async def test_per_call_trace_true_persists_and_returns_id(self) -> None:
        server = _make_server(TraceConfig())  # disabled globally...
        res = await _run_recall(server, {"trace": True})  # ...but forced per-call
        storage = res.pop("_storage")
        storage.add_retrieval_trace.assert_awaited_once()
        assert res.get("trace_id")

    @pytest.mark.asyncio
    async def test_config_enabled_fires_background_no_id(self) -> None:
        server = _make_server(TraceConfig(enabled=True, sample_rate=1.0))
        res = await _run_recall(server, {})
        storage = res.pop("_storage")
        storage.add_retrieval_trace.assert_awaited_once()
        # Background telemetry does NOT surface a trace_id in the response.
        assert "trace_id" not in res

    @pytest.mark.asyncio
    async def test_trace_failure_never_breaks_recall(self) -> None:
        server = _make_server(TraceConfig(enabled=True, sample_rate=1.0))
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(
            return_value=MagicMock(id="test-brain", config=MagicMock())
        )
        mock_storage.brain_id = "test-brain"
        mock_storage.get_typed_memory = AsyncMock(return_value=None)
        mock_storage.add_retrieval_trace = AsyncMock(side_effect=RuntimeError("db down"))
        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("surreal_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=_make_result())
            mock_pipeline_cls.return_value = mock_pipeline
            res = await server.call_tool("smem_recall", {"query": "q"})
            for task in list(getattr(server, "_trace_tasks", set())):
                await task
        # Recall still succeeds despite the trace backend failing.
        assert res["answer"] == "Emma lives in Bergen."
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_per_call_persist_failure_surfaces_trace_error(self) -> None:
        # The caller explicitly asked for trace_id via trace=true; a persist failure
        # must surface as trace_error (not be swallowed silently), and recall still works.
        server = _make_server(TraceConfig())  # disabled globally; per-call forces it
        mock_storage = AsyncMock()
        mock_storage.get_brain = AsyncMock(
            return_value=MagicMock(id="test-brain", config=MagicMock())
        )
        mock_storage.brain_id = "test-brain"
        mock_storage.get_typed_memory = AsyncMock(return_value=None)
        mock_storage.add_retrieval_trace = AsyncMock(side_effect=RuntimeError("db down"))
        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch("surreal_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
        ):
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=_make_result())
            mock_pipeline_cls.return_value = mock_pipeline
            res = await server.call_tool("smem_recall", {"query": "q", "trace": True})
        assert res["answer"] == "Emma lives in Bergen."
        assert "trace_id" not in res
        assert "trace_error" in res

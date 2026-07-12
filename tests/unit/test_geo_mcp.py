"""U8: geospatial args at the MCP boundary (smem_recall `near`, smem_remember `location`).

Verifies the handler-level contract that the pipeline-level tests (test_geo_recall.py)
cannot: that a `near` object is parsed into a GeoFilter and handed to
ReflexPipeline.query(near=...), that malformed `near`/`location` yield clean MCP error
dicts, and that a valid `location` reaches the encoder's fiber metadata.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig
from surreal_memory.utils.geo import GeoFilter


class _StopEncodeError(Exception):
    """Sentinel to stop _remember right after encode is invoked."""


def _make_server() -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
        cfg = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
        )
        cfg.write_gate.enabled = False
        cfg.encryption.enabled = False
        cfg.safety.auto_redact_min_severity = 3
        mock_get_config.return_value = cfg
        return MCPServer()


def _make_result() -> RetrievalResult:
    return RetrievalResult(
        answer="A cafe in Oslo.",
        confidence=0.9,
        depth_used=DepthLevel.INSTANT,
        neurons_activated=1,
        fibers_matched=["f-oslo"],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
        context="A cafe in Oslo.",
        latency_ms=1.0,
        tokens_used=10,
        metadata={},
        score_breakdown=None,
    )


def _brain_storage() -> AsyncMock:
    mock_storage = AsyncMock()
    mock_storage.get_brain = AsyncMock(return_value=MagicMock(id="test-brain", config=MagicMock()))
    mock_storage._current_brain_id = "test-brain"
    mock_storage.brain_id = "test-brain"
    return mock_storage


async def _run_recall(server: MCPServer, extra_args: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Run smem_recall with a mocked pipeline; return (response, captured query call)."""
    mock_pipeline = AsyncMock()
    mock_pipeline.query = AsyncMock(return_value=_make_result())
    with (
        patch.object(server, "get_storage", return_value=_brain_storage()),
        patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=mock_pipeline),
        patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
        patch.object(server, "_fire_eternal_trigger"),
        patch.object(server, "_record_tool_action", new_callable=AsyncMock),
        patch.object(server, "_passive_capture", new_callable=AsyncMock),
    ):
        response = await server.call_tool("smem_recall", {"query": "cafe in oslo", **extra_args})
    return response, mock_pipeline.query.call_args


class TestRecallNearBoundary:
    @pytest.fixture
    def server(self) -> MCPServer:
        return _make_server()

    async def test_near_is_parsed_into_geofilter_and_passed_to_pipeline(
        self, server: MCPServer
    ) -> None:
        _, call = await _run_recall(
            server, {"near": {"lat": 59.9139, "lon": 10.7522, "radius_m": 50000}}
        )
        geo = call.kwargs["near"]
        assert isinstance(geo, GeoFilter)
        assert geo.center.lat == pytest.approx(59.9139)
        assert geo.center.lon == pytest.approx(10.7522)
        assert geo.radius_m == pytest.approx(50000)

    async def test_no_near_passes_none(self, server: MCPServer) -> None:
        _, call = await _run_recall(server, {})
        assert call.kwargs["near"] is None

    async def test_missing_radius_returns_error(self, server: MCPServer) -> None:
        response, _ = await _run_recall(server, {"near": {"lat": 59.9, "lon": 10.7}})
        assert "error" in response
        assert "near" in response["error"].lower()

    async def test_non_object_near_returns_error(self, server: MCPServer) -> None:
        response, _ = await _run_recall(server, {"near": "not-an-object"})
        assert "error" in response
        assert "near" in response["error"].lower()

    async def test_out_of_range_radius_returns_error(self, server: MCPServer) -> None:
        response, _ = await _run_recall(
            server, {"near": {"lat": 59.9, "lon": 10.7, "radius_m": -1}}
        )
        assert "error" in response
        assert "near" in response["error"].lower()


class TestRememberLocationBoundary:
    @pytest.fixture
    def server(self) -> MCPServer:
        return _make_server()

    async def _run_remember(self, server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
        with (
            patch.object(server, "get_storage", return_value=_brain_storage()),
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
        ):
            return await server.call_tool("smem_remember", {"content": "a cafe", **args})

    async def test_invalid_location_coords_returns_error(self, server: MCPServer) -> None:
        response = await self._run_remember(server, {"location": {"lat": "north", "lon": 10.7}})
        assert "error" in response
        assert "location" in response["error"].lower()

    async def test_out_of_range_location_returns_error(self, server: MCPServer) -> None:
        response = await self._run_remember(server, {"location": {"lat": 999, "lon": 10.7}})
        assert "error" in response
        assert "location" in response["error"].lower()

    async def test_valid_location_reaches_encoder_metadata(self, server: MCPServer) -> None:
        # A valid location must be handed to encoder.encode() as metadata['location'].
        captured: dict[str, Any] = {}

        async def _capture_encode(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise _StopEncodeError  # short-circuit the rest of _remember; assert wiring only

        with (
            patch.object(server, "get_storage", return_value=_brain_storage()),
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
            patch("surreal_memory.mcp.remember_handler.MemoryEncoder") as mock_encoder_cls,
        ):
            mock_encoder_cls.return_value.encode = _capture_encode
            with pytest.raises(_StopEncodeError):
                await server._remember(
                    {
                        "content": "a cafe",
                        "location": {"lat": 59.9139, "lon": 10.7522, "label": "Oslo"},
                    }
                )

        assert captured["metadata"]["location"] == {
            "lat": 59.9139,
            "lon": 10.7522,
            "label": "Oslo",
        }

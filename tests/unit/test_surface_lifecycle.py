"""Tests for Knowledge Surface lifecycle — regeneration, show, MCP tool.

Tests the regenerate_surface, show_surface functions and the
SurfaceHandler MCP mixin.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.surface.lifecycle import regenerate_surface, show_surface
from neural_memory.surface.models import (
    Cluster,
    DepthHint,
    DepthLevel,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SignalLevel,
    SurfaceEdge,
    SurfaceFrontmatter,
    SurfaceMeta,
    SurfaceNode,
)
from neural_memory.surface.serializer import serialize

# ── Helpers ────────────────────────────────────────────


def _make_surface() -> KnowledgeSurface:
    """Build a minimal valid KnowledgeSurface."""
    return KnowledgeSurface(
        frontmatter=SurfaceFrontmatter(
            brain="testbrain",
            updated="2026-03-16T10:00:00",
            neurons=50,
            synapses=120,
        ),
        graph=(
            GraphEntry(
                node=SurfaceNode(id="d1", content="Chose Redis", node_type="decision", priority=7),
                edges=(
                    SurfaceEdge(edge_type="caused", target_id="f1", target_text="Faster caching"),
                ),
            ),
        ),
        clusters=(Cluster(name="caching", node_ids=("d1", "f1"), description="Cache layer"),),
        signals=(Signal(level=SignalLevel.WATCHING, text="Redis migration"),),
        depth_map=(DepthHint(node_id="d1", level=DepthLevel.SUFFICIENT, context="well covered"),),
        meta=SurfaceMeta(coverage=0.6, staleness=0.15),
    )


# ── regenerate_surface Tests ──────────────────────────


class TestRegenerateSurface:
    """Tests for the regenerate_surface function."""

    @pytest.mark.asyncio
    async def test_regenerate_produces_valid_surface(self, tmp_path: Path) -> None:
        """regenerate_surface generates, trims, and writes to disk."""
        surface = _make_surface()

        with (
            patch("neural_memory.surface.lifecycle.SurfaceGenerator") as mock_gen_cls,
            patch("neural_memory.surface.lifecycle.save_surface_text") as mock_save,
        ):
            mock_gen = AsyncMock()
            mock_gen.generate.return_value = surface
            mock_gen_cls.return_value = mock_gen
            mock_save.return_value = tmp_path / "surface.nm"

            result = await regenerate_surface(
                storage=AsyncMock(),
                brain_name="testbrain",
                token_budget=1200,
            )

        assert result.frontmatter.brain == "testbrain"
        assert len(result.graph) == 1
        mock_save.assert_called_once()
        # Verify serialize was passed to save
        saved_text = mock_save.call_args[0][0]
        assert "Chose Redis" in saved_text

    @pytest.mark.asyncio
    async def test_regenerate_respects_token_budget(self, tmp_path: Path) -> None:
        """regenerate_surface trims output to token budget."""
        surface = _make_surface()

        with (
            patch("neural_memory.surface.lifecycle.SurfaceGenerator") as mock_gen_cls,
            patch("neural_memory.surface.lifecycle.save_surface_text") as mock_save,
            patch("neural_memory.surface.lifecycle.trim_to_budget") as mock_trim,
        ):
            mock_gen = AsyncMock()
            mock_gen.generate.return_value = surface
            mock_gen_cls.return_value = mock_gen
            mock_trim.return_value = surface
            mock_save.return_value = tmp_path / "surface.nm"

            await regenerate_surface(
                storage=AsyncMock(),
                brain_name="testbrain",
                token_budget=800,
            )

        mock_trim.assert_called_once_with(surface, 800)


# ── show_surface Tests ─────────────────────────────────


class TestShowSurface:
    """Tests for the show_surface function."""

    @pytest.mark.asyncio
    async def test_show_missing_surface(self) -> None:
        """show_surface returns helpful message when no file exists."""
        with patch("neural_memory.surface.resolver.load_surface_text", return_value=None):
            result = await show_surface("testbrain")

        assert result["exists"] is False
        assert "No surface.nm found" in result["message"]

    @pytest.mark.asyncio
    async def test_show_valid_surface(self) -> None:
        """show_surface returns structured info for valid surface."""
        surface = _make_surface()
        text = serialize(surface)

        with patch("neural_memory.surface.resolver.load_surface_text", return_value=text):
            result = await show_surface("testbrain")

        assert result["exists"] is True
        assert result["valid"] is True
        assert result["brain"] == "testbrain"
        assert result["graph_nodes"] == 1
        assert result["clusters"] == 1
        assert result["signals"] == 1
        assert result["surface_text"] == text

    @pytest.mark.asyncio
    async def test_show_corrupt_surface(self) -> None:
        """show_surface handles corrupt .nm files gracefully."""
        with patch(
            "neural_memory.surface.resolver.load_surface_text",
            return_value="this is not valid .nm",
        ):
            result = await show_surface("testbrain")

        assert result["exists"] is True
        assert result["valid"] is False
        assert "corrupt" in result["error"]


# ── SurfaceHandler MCP Tests ──────────────────────────


class TestSurfaceHandler:
    """Tests for the SurfaceHandler MCP mixin."""

    def _make_server(self) -> MagicMock:
        """Create a mock MCPServer with SurfaceHandler."""
        from neural_memory.mcp.surface_handler import SurfaceHandler

        server = MagicMock()
        server._surface_text = ""
        server._surface_brain = ""

        # Bind handler methods
        server._surface = SurfaceHandler._surface.__get__(server)
        server._surface_generate = SurfaceHandler._surface_generate.__get__(server)
        server._surface_show = SurfaceHandler._surface_show.__get__(server)

        return server

    @pytest.mark.asyncio
    async def test_surface_show_action(self) -> None:
        """nmem_surface(action='show') returns surface info."""
        server = self._make_server()

        surface = _make_surface()
        text = serialize(surface)

        with patch("neural_memory.surface.resolver.load_surface_text", return_value=text):
            mock_storage = AsyncMock()
            mock_storage.brain_id = "brain123"
            mock_brain = MagicMock()
            mock_brain.name = "testbrain"
            mock_storage.get_brain = AsyncMock(return_value=mock_brain)
            server.get_storage = AsyncMock(return_value=mock_storage)

            result = await server._surface({"action": "show"})

        assert result["exists"] is True
        assert result["brain"] == "testbrain"

    @pytest.mark.asyncio
    async def test_surface_generate_action(self) -> None:
        """nmem_surface(action='generate') regenerates and returns info."""
        server = self._make_server()
        surface = _make_surface()

        mock_storage = AsyncMock()
        mock_storage.brain_id = "brain123"
        mock_brain = MagicMock()
        mock_brain.name = "testbrain"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        server.get_storage = AsyncMock(return_value=mock_storage)
        server.load_surface = MagicMock(return_value="")

        with patch(
            "neural_memory.surface.lifecycle.regenerate_surface",
            new_callable=AsyncMock,
            return_value=surface,
        ):
            result = await server._surface({"action": "generate"})

        assert result["action"] == "generate"
        assert result["brain"] == "testbrain"
        assert result["graph_nodes"] == 1

    @pytest.mark.asyncio
    async def test_surface_unknown_action(self) -> None:
        """nmem_surface with unknown action returns error."""
        server = self._make_server()
        result = await server._surface({"action": "invalid"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_surface_default_action_is_show(self) -> None:
        """nmem_surface with no action defaults to show."""
        server = self._make_server()

        mock_storage = AsyncMock()
        mock_storage.brain_id = "brain123"
        mock_brain = MagicMock()
        mock_brain.name = "testbrain"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        server.get_storage = AsyncMock(return_value=mock_storage)

        with patch("neural_memory.surface.resolver.load_surface_text", return_value=None):
            result = await server._surface({})

        assert result["exists"] is False

    @pytest.mark.asyncio
    async def test_surface_generate_no_brain(self) -> None:
        """nmem_surface generate with no brain returns error."""
        server = self._make_server()

        mock_storage = AsyncMock()
        mock_storage.brain_id = None
        server.get_storage = AsyncMock(return_value=mock_storage)

        result = await server._surface_generate({})
        assert "error" in result

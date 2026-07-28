"""MCP handler for Knowledge Surface operations.

Provides smem_surface tool for generating and showing the .nm surface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


def _require_brain_id(storage: Any) -> str:
    """Get current brain_id from storage, raise ValueError if not set."""
    brain_id: str | None = getattr(storage, "brain_id", None)
    if not brain_id:
        raise ValueError("No brain configured")
    return brain_id


def _surface_brain_name(config: Any, brain: Any) -> str:
    """Pick the brain key the surface file is stored under.

    ``config.current_brain`` first, because that is exactly what
    ``McpServer.load_surface`` reads. Generating under ``brain.name`` while the
    reader looks under ``current_brain`` writes a file nobody ever loads, and
    doctor then reports the surface as never generated.

    ``brain.name`` is only a fallback, and only after validation: a live install
    ended up with a surface whose brain key was a whole
    ``CLIConfig(data_dir=PosixPath(...), ...)`` repr, because an object was
    passed where a name was expected. The name reaches a file path, so anything
    that is not a plain name is rejected rather than written.
    """
    from surreal_memory.surface.resolver import validate_brain_name

    for candidate in (getattr(config, "current_brain", None), getattr(brain, "name", None)):
        if not isinstance(candidate, str):
            continue
        try:
            return validate_brain_name(candidate)
        except ValueError:
            logger.warning("Ignoring unusable brain name for surface path: %.60r", candidate)
    return "default"


class SurfaceHandler:
    """Mixin providing the smem_surface tool."""

    if TYPE_CHECKING:
        config: UnifiedConfig
        _surface_text: str
        _surface_brain: str

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

        def load_surface(self, brain_name: str = "") -> str:
            raise NotImplementedError

    async def _surface(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle smem_surface tool calls.

        Actions:
        - generate: Regenerate surface from brain.db
        - show: Show current surface info
        """
        action = args.get("action", "show")

        if action == "generate":
            return await self._surface_generate(args)
        elif action == "show":
            return await self._surface_show()
        else:
            return {"error": f"Unknown action: {action}. Use 'generate' or 'show'."}

    async def _surface_generate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Regenerate the Knowledge Surface from brain.db."""
        from surreal_memory.surface.lifecycle import regenerate_surface

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for surface generate")
            return {"error": "No brain configured"}

        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        brain_name = _surface_brain_name(self.config, brain)
        token_budget = args.get("token_budget", 1200)
        max_graph_nodes = args.get("max_graph_nodes", 30)

        try:
            surface = await regenerate_surface(
                storage=storage,
                brain_name=brain_name,
                token_budget=token_budget,
                max_graph_nodes=max_graph_nodes,
            )
        except Exception as e:
            logger.error("Surface generation failed: %s", e, exc_info=True)
            return {"error": "Surface generation failed"}

        # Reload cached surface
        self._surface_text = ""
        self._surface_brain = ""
        self.load_surface(brain_name)

        return {
            "action": "generate",
            "brain": brain_name,
            "graph_nodes": len(surface.graph),
            "clusters": len(surface.clusters),
            "signals": len(surface.signals),
            "depth_hints": len(surface.depth_map),
            "token_estimate": surface.token_estimate(),
            "message": f"Knowledge surface regenerated for brain '{brain_name}'",
        }

    async def _surface_show(self) -> dict[str, Any]:
        """Show current surface information."""
        from surreal_memory.surface.lifecycle import show_surface

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for surface show")
            return {"error": "No brain configured"}

        brain = await storage.get_brain(brain_id)
        brain_name = _surface_brain_name(self.config, brain)

        return await show_surface(brain_name)

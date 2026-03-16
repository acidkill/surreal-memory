"""Surface lifecycle management — regeneration and updates.

Handles the full lifecycle: generate surface from brain.db,
write to disk, and integrate with session-end hooks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.surface.generator import SurfaceGenerator
from neural_memory.surface.resolver import save_surface_text
from neural_memory.surface.serializer import serialize
from neural_memory.surface.token_budget import trim_to_budget

if TYPE_CHECKING:
    from neural_memory.surface.models import KnowledgeSurface

logger = logging.getLogger(__name__)


async def regenerate_surface(
    storage: Any,
    brain_name: str = "default",
    token_budget: int = 1200,
    max_graph_nodes: int = 30,
    max_signals: int = 10,
) -> KnowledgeSurface:
    """Regenerate the Knowledge Surface from brain.db.

    Generates a new surface, trims to budget, writes to disk atomically.

    Args:
        storage: NeuralStorage instance.
        brain_name: Brain to generate surface for.
        token_budget: Max token budget for the surface.
        max_graph_nodes: Maximum graph nodes to include.
        max_signals: Maximum signals to include.

    Returns:
        The generated KnowledgeSurface object.
    """
    generator = SurfaceGenerator(
        storage=storage,
        brain_name=brain_name,
        token_budget=token_budget,
        max_graph_nodes=max_graph_nodes,
        max_signals=max_signals,
    )

    surface = await generator.generate()

    # Trim to budget
    trimmed = trim_to_budget(surface, token_budget)

    # Serialize and write
    text = serialize(trimmed)
    path = save_surface_text(text, brain_name)
    logger.info("Knowledge surface regenerated: %s (%d chars)", path, len(text))

    return trimmed


async def show_surface(brain_name: str = "default") -> dict[str, Any]:
    """Load and return the current surface as structured data.

    Args:
        brain_name: Brain to show surface for.

    Returns:
        Dict with surface info or error message.
    """
    from neural_memory.surface.parser import parse
    from neural_memory.surface.resolver import load_surface_text

    text = load_surface_text(brain_name)
    if not text:
        return {
            "exists": False,
            "message": f"No surface.nm found for brain '{brain_name}'",
            "hint": "Run nmem_surface(action='generate') to create one",
        }

    try:
        surface = parse(text)
    except Exception as e:
        return {
            "exists": True,
            "valid": False,
            "error": f"Surface file is corrupt: {e}",
            "raw_length": len(text),
        }

    return {
        "exists": True,
        "valid": True,
        "brain": surface.frontmatter.brain,
        "updated": surface.frontmatter.updated,
        "neurons": surface.frontmatter.neurons,
        "synapses": surface.frontmatter.synapses,
        "graph_nodes": len(surface.graph),
        "clusters": len(surface.clusters),
        "signals": len(surface.signals),
        "depth_hints": len(surface.depth_map),
        "token_estimate": surface.token_estimate(),
        "token_budget": surface.frontmatter.token_budget,
        "coverage": surface.meta.coverage,
        "staleness": surface.meta.staleness,
        "top_entities": list(surface.meta.top_entities),
        "surface_text": text,
    }

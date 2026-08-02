"""Setup function for Surreal-Memory-Nanobot integration.

One-line initialization that creates storage, registers tools, and
returns a drop-in MemoryStore replacement::

    nm_store = await setup_surreal_memory(registry, workspace)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from surreal_memory.integrations.nanobot.context import NMContext
from surreal_memory.integrations.nanobot.memory_store import NMMemoryStore

logger = logging.getLogger(__name__)


async def setup_surreal_memory(
    registry: Any,
    workspace: Path,
    brain_id: str = "nanobot",
    db_filename: str | None = None,
) -> NMMemoryStore:
    """Initialize Surreal-Memory and register tools with Nanobot.

    Args:
        registry: Nanobot's ToolRegistry (any object with ``register(tool)``).
        workspace: Nanobot workspace directory.
        brain_id: Brain identifier (default: ``"nanobot"``).
        db_filename: JSON storage filename (default: ``f"{brain_id}.json"``,
            so PersistentStorage's own file-stem-as-brain-name creates the
            requested brain rather than a mismatched default one).

    Returns:
        NMMemoryStore that can replace Nanobot's ``MemoryStore``.
    """
    from surreal_memory.cli.storage import PersistentStorage
    from surreal_memory.core.brain import Brain

    db_dir = workspace / "memory"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / (db_filename or f"{brain_id}.json")

    storage = await PersistentStorage.load(db_path)

    brain = await storage.get_brain(brain_id)
    if brain is None:
        brain = Brain.create(name=brain_id, brain_id=brain_id)
        await storage.save_brain(brain)

    storage.set_brain(brain.id)

    ctx = NMContext(storage=storage, brain=brain, config=brain.config)

    from surreal_memory.integrations.nanobot.tools import (
        NMContextTool,
        NMHealthTool,
        NMRecallTool,
        NMRememberTool,
    )

    tools = [
        NMRememberTool(ctx),
        NMRecallTool(ctx),
        NMContextTool(ctx),
        NMHealthTool(ctx),
    ]
    for tool in tools:
        registry.register(tool)

    logger.info(
        "Surreal-Memory: registered %d tools for brain '%s' at %s",
        len(tools),
        brain_id,
        db_path,
    )

    return NMMemoryStore(ctx, workspace)

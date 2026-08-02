"""Shared context for Surreal-Memory-Nanobot integration.

Holds the initialized storage, brain, and config. All tools and the
NMMemoryStore receive this context at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surreal_memory.core.brain import Brain, BrainConfig


@dataclass
class NMContext:
    """Shared context for all Nanobot integration components."""

    storage: Any  # PersistentStorage | InMemoryStorage
    brain: Brain
    config: BrainConfig

    async def close(self) -> None:
        """Close the underlying storage connection."""
        if hasattr(self.storage, "close"):
            await self.storage.close()

    async def __aenter__(self) -> NMContext:
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        await self.close()

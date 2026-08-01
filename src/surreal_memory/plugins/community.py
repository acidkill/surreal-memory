"""Community plugin — Pro-tier capabilities without a commercial license.

Provides:
  - Directional compression (multi-axis semantic preservation), consumed by
    `engine/compression.py` via `plugins.get_compression_fn()`
  - A registered plugin, so `plugins.has_pro()` is True — which unlocks
    auto-tier during consolidation and is reported by `smem doctor`/`smem_stats`

This plugin is discovered by `plugins.discover()` and also registered
explicitly in `server/app.py` startup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from surreal_memory.plugins.base import ProPlugin

logger = logging.getLogger(__name__)


# ── Directional Compression ──────────────────────────────────────────


async def directional_compress(
    content: str,
    level: str,
    embed_fn: Any | None = None,
) -> str:
    """Multi-axis semantic compression preserving entity relationships.

    Levels:
    - 'summary': Keep top sentences by entity density (extractive)
    - 'essence': Keep only entity-containing sentences (entity-only)
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    if not sentences:
        return content

    # Entity patterns: capitalized words, quoted phrases, code-like tokens
    entity_pattern = re.compile(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|"
        r"`[^`]+`|"
        r"['\"].*?['\"]|"
        r"\b\w+\(\)|"
        r"https?://\S+"
    )

    scored = []
    for sent in sentences:
        entities = entity_pattern.findall(sent)
        score = len(entities) * 2 + (
            1.0
            if any(
                kw in sent.lower()
                for kw in (
                    "important",
                    "key",
                    "must",
                    "always",
                    "never",
                    "critical",
                    "note",
                )
            )
            else 0.0
        )
        scored.append((score, sent, entities))

    scored.sort(key=lambda x: x[0], reverse=True)

    if level == "essence":
        # Keep only sentences with entities
        kept = [s for sc, s, ents in scored if ents]
        return " ".join(kept) if kept else sentences[0]
    else:
        # 'summary' — keep top 40% of sentences, min 1
        keep_count = max(1, len(scored) // 3)
        kept = [s for _, s, _ in scored[:keep_count]]
        return " ".join(kept)


# ── Community Plugin ──────────────────────────────────────────────────


class CommunityPlugin(ProPlugin):
    """Built-in plugin providing full Pro features without a license."""

    @property
    def name(self) -> str:
        return "surreal-memory-community"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_compression_fn(self) -> Callable[..., Any] | None:
        return directional_compress


def auto_register() -> None:
    """Entry point for auto-discovery via importlib.metadata."""
    from surreal_memory.plugins import register

    register(CommunityPlugin())

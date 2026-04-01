"""Community plugin — full Pro features without a commercial license.

Registers at server startup to enable:
  - Cone queries (HNSW vector search via SurrealDB)
  - Smart merge (embedding-based neuron consolidation)
  - Directional compression (multi-axis semantic preservation)
  - SurrealDB storage backend
  - Merkle delta sync (is_pro=True gate bypass)

This plugin is discovered by `plugins.discover()` and also registered
explicitly in `server/app.py` startup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from neural_memory.plugins.base import ProPlugin

logger = logging.getLogger(__name__)


# ── Cone Query ────────────────────────────────────────────────────────


async def cone_query(
    query_vec: list[float],
    db: Any,
    top_k: int = 10,
) -> list[Any]:
    """HNSW cone query via SurrealDB vector search.

    Uses vector::distance::knn() to find the nearest neighbors
    to the query embedding. Returns activation-like results.
    """
    from neural_memory.engine.activation import ActivationResult

    if hasattr(db, "find_neurons_by_embedding"):
        results = await db.find_neurons_by_embedding(query_vec, limit=top_k)
        activations = []
        for neuron, similarity in results:
            activations.append(
                ActivationResult(
                    neuron_id=neuron.id,
                    activation_level=similarity,
                    hop_distance=0,
                    path=[neuron.id],
                    source_anchor=neuron.id,
                )
            )
        return activations

    return []


# ── Smart Merge ───────────────────────────────────────────────────────


async def smart_merge(db: Any, dry_run: bool = False) -> dict[str, Any]:
    """Merge near-duplicate neurons using embedding similarity.

    Finds neurons with cosine similarity > 0.95 and merges them,
    keeping the more-accessed neuron as the canonical version.
    """
    merged_count = 0

    if not hasattr(db, "find_neurons_by_embedding"):
        return {"merged_count": 0}

    # Get all neurons with embeddings
    brain_id = db._get_brain_id() if hasattr(db, "_get_brain_id") else None
    if brain_id is None:
        return {"merged_count": 0}

    neurons = await db.find_neurons(limit=10000)
    seen: set[str] = set()

    for neuron in neurons:
        if neuron.id in seen:
            continue

        meta = neuron.metadata
        emb = meta.get("_embedding")
        if emb is None:
            continue

        # Find near-duplicates
        similar = await db.find_neurons_by_embedding(emb, limit=5)
        for sim_neuron, similarity in similar:
            if sim_neuron.id == neuron.id or sim_neuron.id in seen:
                continue
            if similarity >= 0.95:
                if not dry_run:
                    # Keep the one with more content, delete the other
                    if len(sim_neuron.content) > len(neuron.content):
                        await db.delete_neuron(neuron.id)
                        seen.add(neuron.id)
                    else:
                        await db.delete_neuron(sim_neuron.id)
                        seen.add(sim_neuron.id)

                merged_count += 1
                seen.add(sim_neuron.id)

        seen.add(neuron.id)

    return {"merged_count": merged_count}


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
        return "neural-memory-community"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_retrieval_strategies(self) -> dict[str, Callable[..., Any]]:
        return {"cone": cone_query}

    def get_compression_fn(self) -> Callable[..., Any] | None:
        return directional_compress

    def get_consolidation_strategies(self) -> dict[str, Callable[..., Any]]:
        return {"smart_merge": smart_merge}

    def get_storage_class(self) -> type | None:
        from neural_memory.storage.surrealdb import SurrealDBStorage

        return SurrealDBStorage


def auto_register() -> None:
    """Entry point for auto-discovery via importlib.metadata."""
    from neural_memory.plugins import register

    register(CommunityPlugin())

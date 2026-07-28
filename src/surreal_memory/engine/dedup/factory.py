"""One place that turns configuration into a :class:`DedupPipeline`.

Write-time dedup used to be constructed in exactly one write path -- the MCP
``remember`` handler. Every other entry point built its ``MemoryEncoder``
without a pipeline, and ``DedupCheckStep`` opens with::

    if self.dedup_pipeline is None:
        return ctx

so dedup was silently skipped for the CLI, all three auto-capture hooks, the
HTTP API, the cognitive and session handlers, the nanobot integrations, the
langchain adapter and both trainers. Measured on the live brain: four CLI
writes of byte-identical content produced four separate anchors, while the same
content through MCP deduped correctly. The hooks are the highest-volume writers,
which is why the duplicate census only ever grew.

Constructing the pipeline in one helper means a new write path gets dedup by
using it, rather than by remembering to copy thirty lines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surreal_memory.engine.dedup.pipeline import DedupPipeline
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

__all__ = ["build_dedup_pipeline"]


def build_dedup_pipeline(
    storage: NeuralStorage,
    config: Any | None = None,
) -> DedupPipeline | None:
    """Return a configured dedup pipeline, or ``None`` when dedup is off.

    Args:
        storage: Brain-scoped storage the pipeline will search for candidates.
        config: A ``UnifiedConfig``. Defaults to the process-wide one, which is
            what every caller outside the MCP handlers wants.

    Returns:
        A ``DedupPipeline``, or ``None`` if dedup is disabled or the settings
        cannot be read. Returning ``None`` degrades to the previous behaviour
        (no dedup) rather than failing a write -- storing the memory matters
        more than deduplicating it.
    """
    try:
        if config is None:
            from surreal_memory.unified_config import get_config

            config = get_config()

        settings = config.dedup
        if not (isinstance(settings.enabled, bool) and settings.enabled):
            return None

        from surreal_memory.engine.dedup.config import DedupConfig
        from surreal_memory.engine.dedup.pipeline import DedupPipeline

        dedup_cfg = DedupConfig(
            enabled=True,
            simhash_threshold=int(settings.simhash_threshold),
            embedding_threshold=float(settings.embedding_threshold),
            embedding_ambiguous_low=float(settings.embedding_ambiguous_low),
            llm_enabled=bool(settings.llm_enabled),
            llm_provider=str(settings.llm_provider),
            llm_model=str(settings.llm_model),
            llm_max_pairs_per_encode=int(settings.llm_max_pairs_per_encode),
            merge_strategy=str(settings.merge_strategy),
            max_candidates=int(settings.max_candidates),
        )

        llm_judge = None
        if dedup_cfg.llm_enabled and dedup_cfg.llm_provider != "none":
            from surreal_memory.engine.dedup.llm_judge import create_judge

            llm_judge = create_judge(dedup_cfg.llm_provider, dedup_cfg.llm_model)

        return DedupPipeline(config=dedup_cfg, storage=storage, llm_judge=llm_judge)
    except (AttributeError, TypeError, ValueError):
        # Malformed or absent dedup settings: same fallback the MCP handler has
        # always used. Logged rather than swallowed silently, because "dedup
        # quietly not running" is the bug this module exists to end.
        logger.debug("dedup settings unusable; continuing without dedup", exc_info=True)
        return None

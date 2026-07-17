"""MCP handler mixin for reasoning-training (status / mine / patterns / config).

Thin wrapper over the reasoning engine + storage, mirroring the smem_reasoning
dashboard router. Handlers return plain dicts — the server wraps them as MCP
TextContent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.mcp.tool_handler_utils import _get_brain_or_error

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

_PATTERN_FETCH_LIMIT = 5000


class ReasoningHandler:
    """Mixin providing the ``smem_reasoning`` tool (status/mine/patterns/config)."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _reasoning(self, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the reasoning tool by ``action``."""
        action = str(args.get("action", "status"))

        # config is a pure config read — no storage/brain needed.
        if action == "config":
            return {"config": self.config.reasoning_training.to_dict()}

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err
        brain_id = brain.id

        if action == "status":
            return await self._reasoning_status(storage, brain_id)
        if action == "patterns":
            return await self._reasoning_patterns(storage, args)
        if action == "mine":
            return await self._reasoning_mine(brain_id, args)
        return {"error": f"Unknown action: {action}"}

    async def _reasoning_status(self, storage: NeuralStorage, brain_id: str) -> dict[str, Any]:
        from surreal_memory.engine.reasoning_distiller import reasoning_coverage

        stats = await storage.get_reasoning_stats(brain_id)
        fibers = await storage.find_fibers(
            metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
        )
        pattern_counts: dict[str, int] = {}
        for f in fibers:
            m = str(f.metadata.get("_source_model", ""))
            if m:
                pattern_counts[m] = pattern_counts.get(m, 0) + 1

        models = sorted(set(stats.get("by_model", {})) | set(pattern_counts))
        per_model: list[dict[str, Any]] = []
        for m in models:
            cov = await reasoning_coverage(storage, m, self.config)
            tstats = stats.get("by_model", {}).get(m, {})
            per_model.append(
                {
                    "model": m,
                    "trace_count": tstats.get("trace_count", 0),
                    "unprocessed": tstats.get("unprocessed", 0),
                    "pattern_count": pattern_counts.get(m, 0),
                    "coverage_percent": cov["coverage_percent"],
                }
            )

        return {
            "total_traces": stats.get("total", 0),
            "unprocessed_traces": stats.get("unprocessed", 0),
            "total_patterns": len(fibers),
            "mining_enabled": self.config.reasoning_training.mining_enabled,
            "injection_enabled": self.config.reasoning_training.injection_enabled,
            "models": per_model,
        }

    async def _reasoning_patterns(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        model = args.get("model")
        category = args.get("category")
        limit = max(1, min(int(args.get("limit", 50) or 50), 100))

        fibers = await storage.find_fibers(
            metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
        )
        rows: list[dict[str, Any]] = []
        for f in fibers:
            md = f.metadata
            if model and md.get("_source_model") != model:
                continue
            if category and md.get("_reasoning_category") != category:
                continue
            rows.append(
                {
                    "id": f.id,
                    "source_model": md.get("_source_model", ""),
                    "category": md.get("_reasoning_category", ""),
                    "title": md.get("_reasoning_title", ""),
                    "confidence": md.get("_reasoning_confidence", 0.0),
                    "frequency": md.get("_reasoning_frequency", 0),
                }
            )
        rows.sort(
            key=lambda r: float(r["confidence"] or 0.0) * float(r["frequency"] or 0), reverse=True
        )
        return {"patterns": rows[:limit], "count": len(rows)}

    async def _reasoning_mine(self, brain_id: str, args: dict[str, Any]) -> dict[str, Any]:
        from dataclasses import replace as dc_replace

        from surreal_memory.engine.reasoning_distiller import distill_reasoning_patterns
        from surreal_memory.engine.reasoning_miner import ingest_reasoning_traces
        from surreal_memory.unified_config import create_isolated_storage

        rt = self.config.reasoning_training
        if not rt.mining_enabled:
            # Privacy: don't scan ~/.claude transcripts unless mining is enabled.
            return {"error": "Mining is disabled; enable reasoning_training.mining_enabled first"}
        if args.get("dry_run"):
            return {"traces_ingested": 0, "patterns_learned": 0, "dry_run": True}

        overrides: dict[str, Any] = {}
        if args.get("backfill"):
            overrides["scan_lookback_days"] = 0
        models = args.get("models")
        if isinstance(models, list) and models:
            overrides["mining_models"] = tuple(str(m) for m in models)
        elif isinstance(models, str) and models.strip():
            overrides["mining_models"] = tuple(m.strip() for m in models.split(",") if m.strip())
        run_cfg = dc_replace(self.config, reasoning_training=dc_replace(rt, **overrides))

        # Isolated (non-cached) storage so a concurrently-served MCP call's
        # set_brain() on the shared SurrealDB singleton can't redirect this job's
        # graph writes into the wrong brain (the MCP HTTP transport is not serialized).
        storage = await create_isolated_storage(brain_id)
        owns_storage = self.config.storage_backend == "surrealdb"
        try:
            ingest = await ingest_reasoning_traces(storage, brain_id, run_cfg)
            distill = await distill_reasoning_patterns(storage, brain_id, run_cfg)
            return {
                "traces_ingested": ingest.traces_ingested,
                "patterns_learned": distill.patterns_learned,
            }
        finally:
            if owns_storage:
                await storage.close()

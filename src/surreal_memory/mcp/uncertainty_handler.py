"""smem_uncertainty MCP tool — brain-wide "what can't I trust?" diagnostics (U5/U6).

Separate from smem_conflicts (which stays a CRUD tool). Thin wrapper: the brain-wide
aggregation lives in ``engine.uncertainty_report`` (shared with the dashboard route,
which must not import ``mcp``). Actions:
- overview (default): counts + contradiction_rate + samples (engine.build_brain_uncertainty).
- contradictions: delegates to the ConflictHandler list (mixins share ``self``).
- drift: detected drift clusters (not yet implemented on this backend; always empty).
- expiring: memories expiring within N days, brain-wide.
- low_evidence: memories with trust_score <= 0.4 (bounded scan; surfaces truncation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.engine.uncertainty_report import (
    build_brain_uncertainty,
    get_detected_drift,
    scan_low_evidence_and_superseded,
)

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_MAX_LIMIT = 200
_SAMPLE_N = 10


class UncertaintyHandler:
    """Mixin providing the smem_uncertainty tool."""

    async def get_storage(self) -> NeuralStorage:  # pragma: no cover - provided by MCPServer
        raise NotImplementedError

    async def _uncertainty(self, args: dict[str, Any]) -> dict[str, Any]:
        from surreal_memory.mcp.tool_handler_utils import _get_brain_or_error

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "overview")
        within_days = _clamp_int(args.get("within_days", 14), 1, 365, 14)
        limit = _clamp_int(args.get("limit", _SAMPLE_N), 1, _MAX_LIMIT, _SAMPLE_N)

        if action == "overview":
            return {"brain": brain.id, **await build_brain_uncertainty(storage, within_days)}
        if action == "contradictions":
            # Delegate to the existing conflicts CRUD listing (shared mixin self).
            result: dict[str, Any] = await self._conflicts_list(args)  # type: ignore[attr-defined]
            return result
        if action == "drift":
            drift = await get_detected_drift(storage, limit)
            return {"drift_clusters": drift, "count": len(drift)}
        if action == "expiring":
            return await self._uncertainty_expiring(storage, within_days, limit)
        if action == "low_evidence":
            low_evidence, _superseded, scanned, truncated = await scan_low_evidence_and_superseded(
                storage
            )
            return {
                "low_evidence": low_evidence[:limit],
                "count": len(low_evidence),
                "scanned": scanned,
                "truncated": truncated,
            }
        return {
            "error": (
                f"Unknown action: {action}. "
                "Use overview, contradictions, drift, expiring, or low_evidence."
            )
        }

    async def _uncertainty_expiring(
        self, storage: NeuralStorage, within_days: int, limit: int
    ) -> dict[str, Any]:
        try:
            rows = await storage.get_expiring_memories(within_days=within_days, limit=limit)
        except Exception:
            logger.debug("get_expiring_memories failed", exc_info=True)
            rows = []
        out = [
            {
                "fiber_id": getattr(tm, "fiber_id", None),
                "memory_type": getattr(getattr(tm, "memory_type", None), "value", None),
                "expires_at": _iso(getattr(tm, "expires_at", None)),
            }
            for tm in rows
        ]
        return {"expiring": out, "count": len(out), "within_days": within_days}


def _iso(dt: Any) -> str | None:
    return dt.isoformat() if dt else None


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default

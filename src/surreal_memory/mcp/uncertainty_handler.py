"""smem_uncertainty MCP tool — brain-wide "what can't I trust?" diagnostics (U5).

Separate from smem_conflicts (which stays a CRUD tool). Read-only aggregation over
cheap, bounded queries. Actions:
- overview (default): counts + contradiction_rate + a small sample of each signal.
- contradictions: delegates to the ConflictHandler list (mixins share ``self``).
- drift: detected drift clusters (SQLite-only; empty on backends without it).
- expiring: memories expiring within N days, brain-wide.
- low_evidence: memories with trust_score <= threshold (default 0.4).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.core.synapse import SynapseType
from surreal_memory.mcp.tool_handler_utils import _get_brain_or_error

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_MAX_LIMIT = 200
_SAMPLE_N = 10
_LOW_TRUST_THRESHOLD = 0.4
# Bound the CONTRADICTS synapse scan so overview never does a full-table scan on
# SurrealDB (get_synapses with limit=None emits no LIMIT there). A brain with more
# active conflicts than this is already pathological; the count is reported capped.
_CONTRADICTION_SCAN_CAP = 2000


def _level(
    contradictions: int, low_evidence: int, superseded: int, expiring: int, drift: int
) -> str:
    if contradictions > 0 or low_evidence > 0:
        return "high"
    if superseded > 0 or expiring > 0 or drift > 0:
        return "medium"
    return "low"


class UncertaintyHandler:
    """Mixin providing the smem_uncertainty tool."""

    async def get_storage(self) -> NeuralStorage:  # pragma: no cover - provided by MCPServer
        raise NotImplementedError

    async def _uncertainty(self, args: dict[str, Any]) -> dict[str, Any]:
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "overview")
        within_days = _clamp_int(args.get("within_days", 14), 1, 365, 14)
        limit = _clamp_int(args.get("limit", _SAMPLE_N), 1, _MAX_LIMIT, _SAMPLE_N)

        if action == "overview":
            return await self._uncertainty_overview(storage, brain.id, within_days)
        if action == "contradictions":
            # Delegate to the existing conflicts CRUD listing (shared mixin self).
            result: dict[str, Any] = await self._conflicts_list(args)  # type: ignore[attr-defined]
            return result
        if action == "drift":
            return await self._uncertainty_drift(storage, limit)
        if action == "expiring":
            return await self._uncertainty_expiring(storage, within_days, limit)
        if action == "low_evidence":
            return await self._uncertainty_low_evidence(storage, limit)
        return {
            "error": (
                f"Unknown action: {action}. "
                "Use overview, contradictions, drift, expiring, or low_evidence."
            )
        }

    async def _uncertainty_overview(
        self, storage: NeuralStorage, brain_id: str, within_days: int
    ) -> dict[str, Any]:
        conflicts_active, contradictions_capped = await _count_active_contradictions(storage)
        expiring_count = await _safe_int(storage.get_expiring_memory_count(within_days))
        drift = await _get_drift(storage, _SAMPLE_N)
        low_evidence, superseded, scanned, scan_truncated = await _scan_typed(storage)

        try:
            total = await storage.count_typed_memories()
        except Exception:
            total = 0
        contradiction_rate = round(conflicts_active / total, 4) if total > 0 else 0.0

        counts = {
            "contradictions": conflicts_active,
            "low_evidence": len(low_evidence),
            "superseded": len(superseded),
            "expiring": expiring_count,
            "drift_clusters": len(drift),
        }
        return {
            "brain": brain_id,
            "level": _level(
                conflicts_active, len(low_evidence), len(superseded), expiring_count, len(drift)
            ),
            "counts": counts,
            "contradiction_rate": contradiction_rate,
            "total_memories": total,
            # Honesty about coverage: low_evidence/superseded reflect only the most-recent
            # `scanned` typed memories (recency-ordered); when truncated, older facts are
            # NOT counted. contradiction count is capped at _CONTRADICTION_SCAN_CAP.
            "scan": {
                "typed_scanned": scanned,
                "typed_scan_truncated": scan_truncated,
                "contradictions_capped": contradictions_capped,
            },
            "samples": {
                "low_evidence": low_evidence[:_SAMPLE_N],
                "superseded": superseded[:_SAMPLE_N],
                "drift_clusters": drift[:_SAMPLE_N],
            },
        }

    async def _uncertainty_drift(self, storage: NeuralStorage, limit: int) -> dict[str, Any]:
        drift = await _get_drift(storage, limit)
        return {"drift_clusters": drift, "count": len(drift)}

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

    async def _uncertainty_low_evidence(self, storage: NeuralStorage, limit: int) -> dict[str, Any]:
        low_evidence, _superseded, scanned, truncated = await _scan_typed(storage)
        return {
            "low_evidence": low_evidence[:limit],
            "count": len(low_evidence),
            "scanned": scanned,
            "truncated": truncated,
        }


# ── module helpers ──


def _iso(dt: Any) -> str | None:
    """ISO-format a datetime-or-None (defensive: returns None on anything falsy)."""
    return dt.isoformat() if dt else None


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


async def _safe_int(awaitable: Any) -> int:
    try:
        return int(await awaitable)
    except Exception:
        return 0


async def _count_active_contradictions(storage: NeuralStorage) -> tuple[int, bool]:
    """(count of unresolved CONTRADICTS, whether the scan hit the cap)."""
    try:
        contradicts = await storage.get_synapses(
            type=SynapseType.CONTRADICTS, limit=_CONTRADICTION_SCAN_CAP
        )
    except Exception:
        return 0, False
    active = sum(1 for s in contradicts if not s.metadata.get("_resolved"))
    return active, len(contradicts) >= _CONTRADICTION_SCAN_CAP


async def _get_drift(storage: NeuralStorage, limit: int) -> list[dict[str, Any]]:
    getter = getattr(storage, "get_drift_clusters", None)
    if getter is None:
        return []
    try:
        clusters = await getter(status="detected", limit=min(limit, _MAX_LIMIT))
    except Exception:
        return []
    return [
        {
            "id": c.get("id"),
            "canonical": c.get("canonical"),
            "confidence": c.get("confidence"),
        }
        for c in (clusters or [])
    ]


async def _scan_typed(
    storage: NeuralStorage,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, bool]:
    """(low-evidence list, superseded list, rows_scanned, truncated).

    No dedicated trust/superseded index method exists on the ABC, so this scans a
    bounded window (<=200) and filters in Python. Both backends order this by
    recency, so on a brain with >200 typed memories the result reflects only the
    most-recent rows — the caller MUST surface ``truncated`` so the counts are not
    read as exhaustive. Uses ``include_expired=False`` to match the scope of
    ``count_typed_memories`` (total_memories).
    """
    try:
        rows = await storage.find_typed_memories(limit=_MAX_LIMIT, include_expired=False)
    except Exception:
        return [], [], 0, False
    low_evidence: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    for tm in rows:
        trust = getattr(tm, "trust_score", None)
        if trust is not None and trust <= _LOW_TRUST_THRESHOLD:
            low_evidence.append({"fiber_id": tm.fiber_id, "trust_score": trust})
        if getattr(tm, "superseded_by", None):
            superseded.append({"fiber_id": tm.fiber_id, "superseded_by": tm.superseded_by})
    return low_evidence, superseded, len(rows), len(rows) >= _MAX_LIMIT

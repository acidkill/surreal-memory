"""Uncertainty surfacing (U5 — cheap aggregation over a finished recall).

``build_uncertainty_block`` assembles a compact "how much should you trust this
answer?" block from signals ALREADY at hand after a recall — disputed neurons,
superseded facts among the returned fibers, low answer confidence, soon-expiring
memories, and (SQLite-only) detected drift clusters intersecting the returned
fibers' tags. Read-only + defensive: every source is guarded so a missing method
or backend gap degrades to empty rather than raising. Returns None when there is
no uncertainty signal, so recall only attaches the block when it means something.

Neutral by construction: nothing here changes ranking, the answer, or what recall
returns — it is pure, opt-in reporting. This module lives in ``engine`` (not
``mcp``) so both the recall handler and the dashboard route can import it.
"""

from __future__ import annotations

import asyncio
from typing import Any

_TOP_N = 10
_DEFAULT_SUFFICIENCY = 0.7


def _level(counts: dict[str, int]) -> str:
    """Deterministic severity from the aggregated counts."""
    if counts["contradictions"] > 0 or counts["low_confidence"] > 0:
        return "high"
    if counts["superseded"] > 0 or counts["expiring"] > 0 or counts["drift_clusters"] > 0:
        return "medium"
    return "low"


async def _contradictions(storage: Any, disputed_ids: list[str]) -> list[dict[str, Any]]:
    if not disputed_ids:
        return []
    top = disputed_ids[:_TOP_N]
    neurons: dict[str, Any] = {}
    try:
        neurons = await storage.get_neurons_batch(top)
    except Exception:
        neurons = {}
    out: list[dict[str, Any]] = []
    for nid in top:
        neuron = neurons.get(nid) if isinstance(neurons, dict) else None
        content = (getattr(neuron, "content", "") or "") if neuron is not None else ""
        out.append({"neuron_id": nid, "content": content[:200]})
    return out


async def _superseded_and_tags(
    storage: Any, fiber_ids: list[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return superseded entries among the fibers + the union of their tags.

    One batched typed-memory read serves both the superseded list and the tag set
    used to scope drift clusters, keeping this cheap.
    """
    if not fiber_ids:
        return [], set()
    try:
        tms = await storage.get_typed_memories_batch(list(fiber_ids))
    except Exception:
        return [], set()
    if not isinstance(tms, dict):
        return [], set()

    superseded: list[dict[str, Any]] = []
    tags: set[str] = set()
    for fid, tm in tms.items():
        if tm is None:
            continue
        for t in getattr(tm, "tags", ()) or ():
            tags.add(str(t).lower())
        if getattr(tm, "superseded_by", None):
            if len(superseded) < _TOP_N:
                superseded.append({"fiber_id": fid, "superseded_by": tm.superseded_by})
    return superseded, tags


async def _expiring(storage: Any, fiber_ids: list[str], within_days: int) -> list[dict[str, Any]]:
    if not fiber_ids:
        return []
    try:
        rows = await storage.get_expiring_memories_for_fibers(
            list(fiber_ids), within_days=within_days
        )
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for tm in rows[:_TOP_N]:
        expires_at = getattr(tm, "expires_at", None)
        out.append(
            {
                "fiber_id": getattr(tm, "fiber_id", None),
                "expires_at": expires_at.isoformat() if expires_at else None,
            }
        )
    return out


async def _drift(storage: Any, tags: set[str]) -> list[dict[str, Any]]:
    """Detected drift clusters intersecting the returned fibers' tags (SQLite-only)."""
    getter = getattr(storage, "get_drift_clusters", None)
    if getter is None:
        return []
    try:
        clusters = await getter(status="detected", limit=50)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for cluster in clusters or []:
        canonical = str(cluster.get("canonical", "")).lower()
        members = cluster.get("members") or []
        member_set = {str(m).lower() for m in members} if isinstance(members, list) else set()
        # Only surface a cluster when it touches something this recall returned.
        if not tags or (canonical in tags or (member_set & tags)):
            out.append(
                {
                    "id": cluster.get("id"),
                    "canonical": cluster.get("canonical"),
                    "confidence": cluster.get("confidence"),
                }
            )
        if len(out) >= _TOP_N:
            break
    return out


async def build_uncertainty_block(
    storage: Any,
    result: Any,
    config: Any,
    within_days: int = 14,
) -> dict[str, Any] | None:
    """Assemble the uncertainty block for a recall result, or None if nothing uncertain.

    Cheap sources only; never raises. ``config`` is the brain config (read for
    ``sufficiency_threshold``).
    """
    metadata = getattr(result, "metadata", None) or {}
    disputed_ids = list(metadata.get("disputed_ids", []) or [])
    fiber_ids = getattr(result, "fibers_matched", []) or []
    if not isinstance(fiber_ids, list):
        fiber_ids = []

    # Independent reads run concurrently (each guards its own errors); drift depends
    # on the tag set from the superseded scan, so it runs after.
    contradictions, (superseded, tags), expiring = await asyncio.gather(
        _contradictions(storage, disputed_ids),
        _superseded_and_tags(storage, fiber_ids),
        _expiring(storage, fiber_ids, within_days),
    )
    drift_clusters = await _drift(storage, tags)

    try:
        threshold = float(getattr(config, "sufficiency_threshold", _DEFAULT_SUFFICIENCY))
    except (TypeError, ValueError):
        threshold = _DEFAULT_SUFFICIENCY
    try:
        confidence = float(getattr(result, "confidence", 1.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 1.0
    low_confidence: dict[str, Any] | None = None
    if confidence < threshold:
        low_confidence = {"confidence": round(confidence, 4), "threshold": threshold}

    counts = {
        "contradictions": len(contradictions),
        "superseded": len(superseded),
        "low_confidence": 1 if low_confidence else 0,
        "expiring": len(expiring),
        "drift_clusters": len(drift_clusters),
    }
    if not any(counts.values()):
        return None

    return {
        "level": _level(counts),
        "counts": counts,
        "contradictions": contradictions,
        "superseded": superseded,
        "low_confidence": low_confidence,
        "expiring": expiring,
        "drift_clusters": drift_clusters,
    }

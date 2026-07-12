"""Build a compact RetrievalTrace from a finished recall (U4 — telemetry).

Pure + defensive: extracts a bounded, queryable summary of one recall from the
``RetrievalResult`` and the recall arguments. Never raises — a telemetry builder
must not be able to break recall — and never dumps subgraphs or context (the
RetrievalTrace model bounds query/id lengths by construction).
"""

from __future__ import annotations

from typing import Any

from surreal_memory.core.retrieval_trace import RetrievalTrace

# Recall args that describe *what was filtered* — surfaced in the trace so a later
# query can answer "which recalls applied filter X". Key names MUST match the actual
# smem_recall argument names (e.g. "tier", not "recall_tier").
_FILTER_KEYS = (
    "tags",
    "valid_at",
    "near",
    "min_trust",
    "min_confidence",
    "include_superseded",
    "tier",
    "brains",
    "prefer_recent",
)

# Defensive bounds on filter values — smem_recall args are NOT jsonschema-validated at
# the MCP boundary, so a caller could pass an arbitrarily large tags list. Keep the
# persisted trace small (the RetrievalTrace <2 KB contract) by capping here.
_MAX_FILTER_LIST = 20
_MAX_FILTER_STR = 200


def _bound_filter_value(value: Any) -> Any:
    """Bound a filter value so it can't blow the trace size budget."""
    if isinstance(value, (list, set, tuple)):
        return [str(v)[:_MAX_FILTER_STR] for v in list(value)[:_MAX_FILTER_LIST]]
    if isinstance(value, str):
        return value[:_MAX_FILTER_STR]
    return value  # bool / int / float scalars are already small


def _as_str_tuple(value: Any, cap: int) -> tuple[str, ...]:
    """Best-effort coerce an iterable of ids to a capped tuple of strings."""
    if not value:
        return ()
    try:
        return tuple(str(v) for v in list(value)[:cap])
    except TypeError:
        return ()


def _extract_filters(args: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    filters: dict[str, Any] = {"mode": mode}
    if not args:
        return filters
    for key in _FILTER_KEYS:
        if key in args and args[key] not in (None, "", [], {}):
            filters[key] = _bound_filter_value(args[key])
    return filters


def build_retrieval_trace(
    result: Any,
    *,
    query: str,
    brain_id: str,
    mode: str,
    args: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> RetrievalTrace:
    """Assemble a RetrievalTrace from a recall ``result`` + its arguments.

    All field access is ``getattr``-guarded so a partially-populated or mock
    result never raises. The returned trace is size-bounded by RetrievalTrace's
    own __post_init__ (query<=500, ids<=10).
    """
    depth_raw = getattr(result, "depth_used", 0)
    try:
        depth_used = int(getattr(depth_raw, "value", depth_raw))
    except (TypeError, ValueError):
        depth_used = 0

    subgraph = getattr(result, "subgraph", None)
    anchor_ids = _as_str_tuple(getattr(subgraph, "anchor_ids", ()), 10)

    synthesis = getattr(result, "synthesis_method", "") or ""
    retrievers = (str(synthesis),) if synthesis else ()

    try:
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        latency_ms = float(getattr(result, "latency_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        latency_ms = 0.0

    return RetrievalTrace(
        brain_id=brain_id,
        session_id=session_id,
        query=query or "",
        depth_used=depth_used,
        mode=mode,
        confidence=confidence,
        latency_ms=latency_ms,
        anchor_ids=anchor_ids,
        retrievers=retrievers,
        fiber_ids=_as_str_tuple(getattr(result, "fibers_matched", ()), 10),
        fiber_scores=(),  # no per-fiber score vector on RetrievalResult; telemetry-optional
        filters=_extract_filters(args, mode),
        config_snapshot=dict(config_snapshot or {}),
    )

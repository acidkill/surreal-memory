"""RetrievalTrace — a compact, queryable record of one recall (schema v9).

Telemetry only: captures what fed a recall answer (fiber ids/scores, anchors,
depth, mode, confidence, latency, config snapshot) without dumping subgraphs or
full context. Frozen + size-bounded (<2 KB) by construction so persisting a
trace never becomes a memory-shaped payload of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from surreal_memory.utils.timeutils import utcnow

_MAX_QUERY_LEN = 500
_MAX_IDS = 10


def _parse_dt(value: Any) -> datetime | None:
    """Best-effort parse of a datetime or ISO-8601 string (None on failure)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class RetrievalTrace:
    """One recall's provenance record — see module docstring for the contract.

    Attributes:
        id: Unique identifier
        brain_id: Brain this trace belongs to
        session_id: Optional session grouping
        query: The recall query text (truncated to 500 chars)
        depth_used: Spreading-activation depth actually used
        mode: Retrieval mode (e.g. "fast", "deep")
        confidence: Final answer confidence 0.0-1.0
        latency_ms: Wall-clock latency of the recall
        anchor_ids: Top anchor fiber ids (capped at 10)
        retrievers: Names of retrievers that contributed
        fiber_ids: Top matched fiber ids (capped at 10)
        fiber_scores: Scores parallel to fiber_ids (capped at 10)
        filters: Recall filters applied (tags, valid_at, near, ...)
        config_snapshot: A few scalar config values in effect
        trace_version: Schema version of this trace record
        created_at: When the recall happened
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    brain_id: str = ""
    session_id: str | None = None
    query: str = ""
    depth_used: int = 0
    mode: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    anchor_ids: tuple[str, ...] = ()
    retrievers: tuple[str, ...] = ()
    fiber_ids: tuple[str, ...] = ()
    fiber_scores: tuple[float, ...] = ()
    filters: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    trace_version: int = 1
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        # frozen dataclass: normalise/bound oversized fields via object.__setattr__.
        if len(self.query) > _MAX_QUERY_LEN:
            object.__setattr__(self, "query", self.query[:_MAX_QUERY_LEN])
        if len(self.anchor_ids) > _MAX_IDS:
            object.__setattr__(self, "anchor_ids", tuple(self.anchor_ids[:_MAX_IDS]))
        if len(self.fiber_ids) > _MAX_IDS:
            object.__setattr__(self, "fiber_ids", tuple(self.fiber_ids[:_MAX_IDS]))
        if len(self.fiber_scores) > _MAX_IDS:
            object.__setattr__(self, "fiber_scores", tuple(self.fiber_scores[:_MAX_IDS]))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (lists, ISO datetime)."""
        return {
            "id": self.id,
            "brain_id": self.brain_id,
            "session_id": self.session_id,
            "query": self.query,
            "depth_used": self.depth_used,
            "mode": self.mode,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "anchor_ids": list(self.anchor_ids),
            "retrievers": list(self.retrievers),
            "fiber_ids": list(self.fiber_ids),
            "fiber_scores": list(self.fiber_scores),
            "filters": dict(self.filters),
            "config_snapshot": dict(self.config_snapshot),
            "trace_version": self.trace_version,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalTrace:
        """Rebuild a RetrievalTrace from a dict, tolerating missing keys."""
        kwargs: dict[str, Any] = {
            "brain_id": str(data.get("brain_id", "")),
            "session_id": data.get("session_id"),
            "query": str(data.get("query", "")),
            "depth_used": int(data.get("depth_used", 0) or 0),
            "mode": str(data.get("mode", "")),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "latency_ms": float(data.get("latency_ms", 0.0) or 0.0),
            "anchor_ids": tuple(data.get("anchor_ids") or ()),
            "retrievers": tuple(data.get("retrievers") or ()),
            "fiber_ids": tuple(data.get("fiber_ids") or ()),
            "fiber_scores": tuple(float(s) for s in (data.get("fiber_scores") or ())),
            "filters": dict(data.get("filters") or {}),
            "config_snapshot": dict(data.get("config_snapshot") or {}),
            "trace_version": int(data.get("trace_version", 1) or 1),
        }
        if data.get("id"):
            kwargs["id"] = str(data["id"])
        created_at = _parse_dt(data.get("created_at"))
        if created_at is not None:
            kwargs["created_at"] = created_at
        return cls(**kwargs)

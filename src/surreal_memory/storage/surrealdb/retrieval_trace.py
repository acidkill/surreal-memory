"""SurrealDB storage mixin for retrieval traces (schema v9).

Stores first-class scalar columns + a ``fiber_ids`` array + a FLEXIBLE ``payload``
object holding the remaining RetrievalTrace fields. brain_id is inlined as a
validated literal (SurrealDB 3.2.0 only uses the brain_id index for inline
literals; ``$bind`` = full scan) via the single-sourced ``_safe_brain_id`` guard.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.storage.surrealdb._ids import _safe_brain_id, _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

# RetrievalTrace fields NOT stored as first-class columns — they live in `payload`.
_PAYLOAD_KEYS = (
    "anchor_ids",
    "retrievers",
    "fiber_scores",
    "filters",
    "config_snapshot",
    "trace_version",
)


def _brain_literal(brain_id: str) -> str:
    """Return a validated, injection-safe inline SurQL string literal for brain_id."""
    return f'"{_safe_brain_id(brain_id)}"'


def _row_to_retrieval_trace(row: dict[str, Any]) -> RetrievalTrace:
    """Convert a SurrealDB retrieval_trace record back into a RetrievalTrace."""
    payload = dict(row.get("payload") or {})
    raw_id = row.get("id")
    trace_id = str(raw_id).rsplit(":", 1)[-1] if raw_id is not None else ""
    data: dict[str, Any] = {
        **payload,
        "id": trace_id,
        "brain_id": row.get("brain_id", ""),
        "session_id": row.get("session_id"),
        "query": row.get("query", ""),
        "depth_used": row.get("depth_used", 0),
        "mode": row.get("mode", ""),
        "confidence": row.get("confidence", 0.0),
        "latency_ms": row.get("latency_ms", 0.0),
        "fiber_ids": row.get("fiber_ids") or (),
        "created_at": row.get("created_at"),
    }
    return RetrievalTrace.from_dict(data)


class SurrealDBRetrievalTraceMixin:
    """Mixin providing retrieval-trace CRUD/find/prune for SurrealDBStorage."""

    # ------------------------------------------------------------------
    # Protocol stubs — satisfied by SurrealDBStorage at runtime
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def add_retrieval_trace(self, trace: RetrievalTrace) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(trace.id)

        full = trace.to_dict()
        payload = {k: full[k] for k in _PAYLOAD_KEYS}
        record_data: dict[str, Any] = {
            "brain_id": brain_id,
            "session_id": trace.session_id,
            "query": trace.query,
            "depth_used": trace.depth_used,
            "mode": trace.mode,
            "confidence": trace.confidence,
            "latency_ms": trace.latency_ms,
            "fiber_ids": list(trace.fiber_ids),
            "payload": payload,
            "created_at": trace.created_at,
        }

        try:
            await conn.query(
                f"UPSERT retrieval_trace:{sid} CONTENT $data",
                {"data": record_data},
            )
        except Exception:
            try:
                await conn.delete(f"retrieval_trace:{sid}")
            except Exception:
                pass
            insert_data = dict(record_data)
            insert_data["id"] = sid
            await conn.insert("retrieval_trace", insert_data)

        return trace.id

    async def get_retrieval_trace(self, trace_id: str) -> RetrievalTrace | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(trace_id)
        try:
            result = await conn.select(f"retrieval_trace:{sid}")
        except Exception:
            return None
        if not result:
            return None
        row = result[0] if isinstance(result, list) else result
        if str(row.get("brain_id", "")) != brain_id:
            return None
        return _row_to_retrieval_trace(row)

    async def find_retrieval_traces(
        self,
        fiber_id: str | None = None,
        query_contains: str | None = None,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list[RetrievalTrace]:
        brain_id = self._get_brain_id()
        cap = min(int(limit), 1000)
        # Over-fetch when substring-filtering so the Python post-filter has candidates.
        fetch = cap if query_contains is None else min(max(cap * 5, 100), 1000)

        parts = [f"SELECT * FROM retrieval_trace WHERE brain_id = {_brain_literal(brain_id)}"]
        params: dict[str, Any] = {}
        if fiber_id is not None:
            parts.append("AND $fiber_id IN fiber_ids")
            params["fiber_id"] = fiber_id
        if since is not None:
            parts.append("AND created_at >= $since")
            params["since"] = since
        parts.append(f"ORDER BY created_at DESC LIMIT {fetch}")

        rows = await self._query(" ".join(parts), **params)
        traces = [_row_to_retrieval_trace(r) for r in rows]
        if query_contains is not None:
            needle = query_contains.lower()
            traces = [t for t in traces if needle in t.query.lower()]
        return traces[:cap]

    async def prune_retrieval_traces(
        self,
        retention_days: int | None = None,
        max_traces: int | None = None,
    ) -> int:
        brain_id = self._get_brain_id()
        lit = _brain_literal(brain_id)
        before = await self._count_retrieval_traces(lit)

        if retention_days is not None:
            cutoff = utcnow() - timedelta(days=retention_days)
            await self._query(
                f"DELETE retrieval_trace WHERE brain_id = {lit} AND created_at < $cutoff",
                cutoff=cutoff,
            )
        if max_traces is not None:
            await self._query(
                f"DELETE retrieval_trace WHERE brain_id = {lit} AND id NOT IN "
                f"(SELECT VALUE id FROM retrieval_trace WHERE brain_id = {lit} "
                f"ORDER BY created_at DESC LIMIT {int(max_traces)})"
            )

        after = await self._count_retrieval_traces(lit)
        return before - after

    async def _count_retrieval_traces(self, brain_literal: str) -> int:
        rows = await self._query(
            f"SELECT count() AS c FROM retrieval_trace WHERE brain_id = {brain_literal} GROUP ALL"
        )
        return int(rows[0]["c"]) if rows and rows[0].get("c") is not None else 0

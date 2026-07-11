"""SQLite storage mixin for retrieval traces (schema v9).

Scalar columns + JSON ``fiber_ids`` array + JSON ``payload`` object holding the
remaining RetrievalTrace fields. Mirrors the SurrealDB retrieval_trace mixin.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

# RetrievalTrace fields NOT stored as first-class columns — they live in `payload`.
_PAYLOAD_KEYS = (
    "anchor_ids",
    "retrievers",
    "fiber_scores",
    "filters",
    "config_snapshot",
    "trace_version",
)


def _row_to_retrieval_trace(row: aiosqlite.Row) -> RetrievalTrace:
    """Convert a retrieval_traces row into a RetrievalTrace."""
    payload = json.loads(row["payload"]) if row["payload"] else {}
    fiber_ids = json.loads(row["fiber_ids"]) if row["fiber_ids"] else []
    data: dict[str, Any] = {
        **payload,
        "id": row["id"],
        "brain_id": row["brain_id"],
        "session_id": row["session_id"],
        "query": row["query"],
        "depth_used": row["depth_used"],
        "mode": row["mode"],
        "confidence": row["confidence"],
        "latency_ms": row["latency_ms"],
        "fiber_ids": fiber_ids,
        "created_at": row["created_at"],
    }
    return RetrievalTrace.from_dict(data)


class SQLiteRetrievalTraceMixin:
    """Mixin providing retrieval-trace CRUD/find/prune for SQLiteStorage."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_retrieval_trace(self, trace: RetrievalTrace) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        full = trace.to_dict()
        payload = {k: full[k] for k in _PAYLOAD_KEYS}
        await conn.execute(
            """INSERT OR REPLACE INTO retrieval_traces
               (id, brain_id, session_id, query, depth_used, mode, confidence,
                latency_ms, fiber_ids, payload, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace.id,
                brain_id,
                trace.session_id,
                trace.query,
                trace.depth_used,
                trace.mode,
                trace.confidence,
                trace.latency_ms,
                json.dumps(list(trace.fiber_ids)),
                json.dumps(payload),
                trace.created_at.isoformat(),
            ),
        )
        await conn.commit()
        return trace.id

    async def get_retrieval_trace(self, trace_id: str) -> RetrievalTrace | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        async with conn.execute(
            "SELECT * FROM retrieval_traces WHERE id = ? AND brain_id = ?",
            (trace_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_retrieval_trace(row)

    async def find_retrieval_traces(
        self,
        fiber_id: str | None = None,
        query_contains: str | None = None,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list[RetrievalTrace]:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        cap = min(int(limit), 1000)
        query = "SELECT * FROM retrieval_traces WHERE brain_id = ?"
        params: list[Any] = [brain_id]
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY created_at DESC LIMIT ?"
        # Over-fetch when a Python post-filter (fiber_id / substring) still has to run.
        needs_pf = fiber_id is not None or query_contains is not None
        params.append(cap if not needs_pf else min(max(cap * 5, 100), 1000))

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        traces = [_row_to_retrieval_trace(r) for r in rows]
        if fiber_id is not None:
            traces = [t for t in traces if fiber_id in t.fiber_ids]
        if query_contains is not None:
            needle = query_contains.lower()
            traces = [t for t in traces if needle in t.query.lower()]
        return traces[:cap]

    async def prune_retrieval_traces(
        self,
        retention_days: int | None = None,
        max_traces: int | None = None,
    ) -> int:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        removed = 0
        if retention_days is not None:
            cutoff = (utcnow() - timedelta(days=retention_days)).isoformat()
            cursor = await conn.execute(
                "DELETE FROM retrieval_traces WHERE brain_id = ? AND created_at < ?",
                (brain_id, cutoff),
            )
            removed += cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        if max_traces is not None:
            cursor = await conn.execute(
                """DELETE FROM retrieval_traces
                   WHERE brain_id = ? AND id NOT IN (
                       SELECT id FROM retrieval_traces WHERE brain_id = ?
                       ORDER BY created_at DESC LIMIT ?
                   )""",
                (brain_id, brain_id, int(max_traces)),
            )
            removed += cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        await conn.commit()
        return removed

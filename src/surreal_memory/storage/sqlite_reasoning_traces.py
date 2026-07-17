"""SQLite mixin for reasoning-trace storage (staging buffer for reasoning mining).

Parity with the SQLite ``tool_events`` staging table, but for reasoning traces
mined from model ``thinking`` blocks. Rows are brain-scoped and deduplicated by
``trace_hash`` via UNIQUE(brain_id, trace_hash) + ``INSERT OR IGNORE``, so
re-ingesting an already-seen trace is a no-op. The consolidation
``PROCESS_REASONING_TRACES`` strategy ingests transcripts into this table;
``LEARN_REASONING`` later distills unprocessed rows into reasoning patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Cap per brain to prevent unbounded growth (config default max_traces_total).
_MAX_TRACES_PER_BRAIN = 20_000

# task_context is truncated to keep the staging table compact (plan: <=500).
_MAX_TASK_CONTEXT = 500


class SQLiteReasoningTracesMixin:
    """Mixin providing CRUD for the ``reasoning_traces`` table."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def insert_reasoning_traces(
        self,
        brain_id: str,
        traces: list[dict[str, Any]],
    ) -> int:
        """Insert reasoning traces into the staging table (dedup by trace_hash).

        Uses ``INSERT OR IGNORE`` against UNIQUE(brain_id, trace_hash), so
        re-ingesting the same trace is a no-op.

        Args:
            brain_id: Brain context.
            traces: Dicts with keys: trace_hash, model, session_id, project,
                task_context, content, content_chars (optional), category
                (optional), created_at (optional ISO string).

        Returns:
            Number of rows actually inserted (excludes ignored duplicates).
        """
        if not traces:
            return 0
        conn = self._ensure_conn()
        now = utcnow().isoformat()
        inserted = 0
        for tr in traces:
            trace_hash = str(tr.get("trace_hash", ""))
            if not trace_hash:
                # A hashless trace cannot be deduped — skip (parity with the
                # SurrealDB / in-memory backends, which also drop empty hashes).
                continue
            content = str(tr.get("content", ""))
            raw_chars = tr.get("content_chars")
            content_chars = int(raw_chars) if raw_chars is not None else len(content)
            cursor = await conn.execute(
                """INSERT OR IGNORE INTO reasoning_traces
                   (brain_id, trace_hash, model, session_id, project,
                    task_context, content, content_chars, category,
                    processed, created_at, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (
                    brain_id,
                    trace_hash,
                    tr.get("model", ""),
                    tr.get("session_id", ""),
                    tr.get("project", ""),
                    str(tr.get("task_context", ""))[:_MAX_TASK_CONTEXT],
                    content,
                    content_chars,
                    tr.get("category", ""),
                    tr.get("created_at", now),
                    now,
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                inserted += 1
        await conn.commit()
        return inserted

    async def get_unprocessed_reasoning_traces(
        self,
        brain_id: str,
        limit: int = 200,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get unprocessed reasoning traces (oldest first) for distillation.

        Optionally filter by ``model`` so the distiller can batch per model.
        """
        conn = self._ensure_read_conn()
        safe_limit = min(int(limit), 10000)
        query = (
            "SELECT id, trace_hash, model, session_id, project, task_context,"
            " content, content_chars, category, created_at"
            " FROM reasoning_traces WHERE brain_id = ? AND processed = 0"
        )
        params: list[Any] = [brain_id]
        if model:
            query += " AND model = ?"
            params.append(model)
        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(safe_limit)
        results: list[dict[str, Any]] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                results.append(
                    {
                        "id": row["id"],
                        "trace_hash": row["trace_hash"],
                        "model": row["model"],
                        "session_id": row["session_id"],
                        "project": row["project"],
                        "task_context": row["task_context"],
                        "content": row["content"],
                        "content_chars": row["content_chars"],
                        "category": row["category"],
                        "created_at": row["created_at"],
                    }
                )
        return results

    async def mark_reasoning_traces_processed(
        self,
        brain_id: str,
        trace_ids: list[Any],
    ) -> None:
        """Mark reasoning traces as processed by row id."""
        if not trace_ids:
            return
        conn = self._ensure_conn()
        placeholders = ",".join("?" for _ in trace_ids)
        # Table/column names are hardcoded — only placeholders are interpolated.
        await conn.execute(
            "UPDATE reasoning_traces SET processed = 1 "
            f"WHERE brain_id = ? AND id IN ({placeholders})",
            [brain_id, *trace_ids],
        )
        await conn.commit()

    async def set_trace_categories(
        self,
        brain_id: str,
        categories: dict[Any, str],
    ) -> None:
        """Set the category label on reasoning traces (id -> category)."""
        if not categories:
            return
        conn = self._ensure_conn()
        for trace_id, category in categories.items():
            await conn.execute(
                "UPDATE reasoning_traces SET category = ? WHERE brain_id = ? AND id = ?",
                (category, brain_id, trace_id),
            )
        await conn.commit()

    async def prune_reasoning_traces(
        self,
        brain_id: str,
        keep_days: int = 90,
    ) -> int:
        """Delete processed traces older than keep_days. Returns rows deleted."""
        from datetime import timedelta

        conn = self._ensure_conn()
        cutoff = (utcnow() - timedelta(days=keep_days)).isoformat()
        cursor = await conn.execute(
            "DELETE FROM reasoning_traces WHERE brain_id = ? AND processed = 1 AND created_at < ?",
            (brain_id, cutoff),
        )
        deleted = cursor.rowcount
        await conn.commit()
        return deleted

    async def cap_reasoning_traces(
        self,
        brain_id: str,
        max_total: int = _MAX_TRACES_PER_BRAIN,
    ) -> int:
        """Enforce max traces per brain by deleting oldest processed rows."""
        conn = self._ensure_conn()
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM reasoning_traces WHERE brain_id = ?",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            total = row["cnt"] if row else 0

        if total <= max_total:
            return 0

        excess = total - max_total
        cursor = await conn.execute(
            """DELETE FROM reasoning_traces WHERE brain_id = ? AND id IN (
                SELECT id FROM reasoning_traces WHERE brain_id = ? AND processed = 1
                ORDER BY created_at ASC LIMIT ?
            )""",
            (brain_id, brain_id, excess),
        )
        deleted = cursor.rowcount
        await conn.commit()
        return deleted

    async def get_reasoning_stats(self, brain_id: str) -> dict[str, Any]:
        """Aggregate reasoning-trace stats: per model, per category, totals."""
        conn = self._ensure_read_conn()

        by_model: dict[str, dict[str, Any]] = {}
        async with conn.execute(
            """SELECT model, COUNT(*) as cnt,
                      SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) as unproc,
                      MAX(created_at) as last_at
               FROM reasoning_traces WHERE brain_id = ?
               GROUP BY model""",
            (brain_id,),
        ) as cursor:
            async for row in cursor:
                by_model[row["model"]] = {
                    "trace_count": row["cnt"],
                    "unprocessed": row["unproc"],
                    "last_trace_at": row["last_at"],
                }

        by_category: dict[str, int] = {}
        async with conn.execute(
            "SELECT category, COUNT(*) as cnt FROM reasoning_traces "
            "WHERE brain_id = ? GROUP BY category",
            (brain_id,),
        ) as cursor:
            async for row in cursor:
                by_category[row["category"]] = row["cnt"]

        total = sum(m["trace_count"] for m in by_model.values())
        unprocessed = sum(m["unprocessed"] for m in by_model.values())
        return {
            "by_model": by_model,
            "by_category": by_category,
            "total": total,
            "unprocessed": unprocessed,
        }

    async def get_reasoning_trace_models(self, brain_id: str) -> list[str]:
        """Return DISTINCT model names present in reasoning_traces (sorted)."""
        conn = self._ensure_read_conn()
        results: list[str] = []
        async with conn.execute(
            "SELECT DISTINCT model FROM reasoning_traces WHERE brain_id = ? ORDER BY model ASC",
            (brain_id,),
        ) as cursor:
            async for row in cursor:
                if row["model"]:
                    results.append(row["model"])
        return results

    async def delete_reasoning_traces_by_model(self, brain_id: str, model: str) -> int:
        """Delete all reasoning traces for a model (privacy wipe). Returns rows deleted."""
        if not model:
            return 0
        conn = self._ensure_conn()
        cursor = await conn.execute(
            "DELETE FROM reasoning_traces WHERE brain_id = ? AND model = ?",
            (brain_id, model),
        )
        deleted = cursor.rowcount
        await conn.commit()
        return deleted

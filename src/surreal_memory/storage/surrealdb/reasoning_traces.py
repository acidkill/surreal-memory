"""SurrealDB mixin for reasoning-trace storage.

Parity with the SQLite ``reasoning_traces`` staging table. Rows are brain-scoped
and deduplicated by ``trace_hash`` (globally unique = sha256(session:uuid:index)).
SurrealDB has no cheap UNIQUE-on-insert story here, so dedup is done by a
pre-filter SELECT (consolidation is single-threaded per brain), mirroring the
SQLite ``INSERT OR IGNORE`` semantics. ``trace_hash`` doubles as the natural key
returned as ``id`` so ``mark_reasoning_traces_processed`` / ``set_trace_categories``
filter by a plain string field instead of juggling SurrealDB record ids (mirrors
the ``event_id`` approach in the tool-events mixin).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from surreal_memory.utils.timeutils import utcnow

# Cap per brain to prevent unbounded growth (config default max_traces_total).
_MAX_TRACES_PER_BRAIN = 20_000

# task_context is truncated to keep the staging table compact (plan: <=500).
_MAX_TASK_CONTEXT = 500


def _iso(val: Any) -> str:
    """Return an ISO-8601 string; callers parse created_at via fromisoformat."""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val) if val is not None else ""


def _as_datetime(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str) and val:
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            return utcnow()
    return utcnow()


class SurrealDBReasoningTracesMixin:
    """Reasoning-trace CRUD backed by the ``reasoning_traces`` table."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def insert_reasoning_traces(
        self,
        brain_id: str,
        traces: list[dict[str, Any]],
    ) -> int:
        """Insert reasoning traces (dedup by trace_hash via pre-filter SELECT)."""
        if not traces:
            return 0
        # Deduplicate the incoming batch by trace_hash (keep first).
        by_hash: dict[str, dict[str, Any]] = {}
        for tr in traces:
            h = str(tr.get("trace_hash", ""))
            if h and h not in by_hash:
                by_hash[h] = tr
        if not by_hash:
            return 0
        existing = await self._query(
            "SELECT trace_hash FROM reasoning_traces "
            "WHERE brain_id = $bid AND trace_hash IN $hashes",
            bid=brain_id,
            hashes=list(by_hash.keys()),
        )
        seen = {r.get("trace_hash") for r in existing}
        conn = self._ensure_conn()
        now = utcnow()
        inserted = 0
        for h, tr in by_hash.items():
            if h in seen:
                continue
            content = str(tr.get("content", ""))
            raw_chars = tr.get("content_chars")
            content_chars = int(raw_chars) if raw_chars is not None else len(content)
            await conn.insert(
                "reasoning_traces",
                {
                    "id": str(uuid.uuid4()).replace("-", "_"),
                    "trace_hash": h,
                    "brain_id": brain_id,
                    "model": tr.get("model", ""),
                    "session_id": tr.get("session_id", ""),
                    "project": tr.get("project", ""),
                    "task_context": str(tr.get("task_context", ""))[:_MAX_TASK_CONTEXT],
                    "content": content,
                    "content_chars": content_chars,
                    "category": tr.get("category", ""),
                    "processed": False,
                    "created_at": _as_datetime(tr.get("created_at")),
                    "ingested_at": now,
                },
            )
            inserted += 1
        return inserted

    async def get_unprocessed_reasoning_traces(
        self,
        brain_id: str,
        limit: int = 200,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get unprocessed reasoning traces (oldest first), optional model filter."""
        safe_limit = min(int(limit), 10000)
        sql = (
            "SELECT trace_hash, model, session_id, project, task_context,"
            " content, content_chars, category, created_at"
            " FROM reasoning_traces WHERE brain_id = $bid AND processed = false"
        )
        params: dict[str, Any] = {"bid": brain_id}
        if model:
            sql += " AND model = $model"
            params["model"] = model
        sql += f" ORDER BY created_at ASC LIMIT {safe_limit}"
        rows = await self._query(sql, **params)
        return [
            {
                "id": r.get("trace_hash"),
                "trace_hash": r.get("trace_hash"),
                "model": r.get("model", ""),
                "session_id": r.get("session_id", ""),
                "project": r.get("project", ""),
                "task_context": r.get("task_context", ""),
                "content": r.get("content", ""),
                "content_chars": r.get("content_chars", 0),
                "category": r.get("category", ""),
                "created_at": _iso(r.get("created_at")),
            }
            for r in rows
        ]

    async def mark_reasoning_traces_processed(
        self,
        brain_id: str,
        trace_ids: list[Any],
    ) -> None:
        """Mark reasoning traces as processed by their trace_hash."""
        if not trace_ids:
            return
        await self._query(
            "UPDATE reasoning_traces SET processed = true "
            "WHERE brain_id = $bid AND trace_hash IN $ids",
            bid=brain_id,
            ids=[str(t) for t in trace_ids],
        )

    async def reset_reasoning_traces_processed(
        self,
        brain_id: str,
        models: list[str] | None = None,
    ) -> int:
        """Re-mark processed traces unprocessed; return rows flipped.

        ``models=None`` resets every model; an empty list is a no-op.
        """
        if models is not None and not models:
            return 0
        where = "brain_id = $bid AND processed = true"
        params: dict[str, Any] = {"bid": brain_id}
        if models:
            where += " AND model IN $models"
            params["models"] = [str(m) for m in models]
        rows = await self._query(
            f"SELECT count() AS c FROM reasoning_traces WHERE {where} GROUP ALL",
            **params,
        )
        reset = int(rows[0]["c"]) if rows else 0
        await self._query(
            f"UPDATE reasoning_traces SET processed = false WHERE {where}",
            **params,
        )
        return reset

    async def set_trace_categories(
        self,
        brain_id: str,
        categories: dict[Any, str],
    ) -> None:
        """Set the category label on reasoning traces (trace_hash -> category)."""
        if not categories:
            return
        for trace_id, category in categories.items():
            await self._query(
                "UPDATE reasoning_traces SET category = $cat "
                "WHERE brain_id = $bid AND trace_hash = $id",
                bid=brain_id,
                cat=category,
                id=str(trace_id),
            )

    async def prune_reasoning_traces(
        self,
        brain_id: str,
        keep_days: int = 90,
    ) -> int:
        """Delete processed traces older than keep_days; return rows deleted."""
        cutoff = utcnow() - timedelta(days=keep_days)
        rows = await self._query(
            "SELECT count() AS c FROM reasoning_traces"
            " WHERE brain_id = $bid AND processed = true AND created_at < $cutoff GROUP ALL",
            bid=brain_id,
            cutoff=cutoff,
        )
        deleted = int(rows[0]["c"]) if rows else 0
        await self._query(
            "DELETE reasoning_traces"
            " WHERE brain_id = $bid AND processed = true AND created_at < $cutoff",
            bid=brain_id,
            cutoff=cutoff,
        )
        return deleted

    async def cap_reasoning_traces(
        self,
        brain_id: str,
        max_total: int = _MAX_TRACES_PER_BRAIN,
    ) -> int:
        """Enforce max traces per brain by deleting oldest processed rows."""
        rows = await self._query(
            "SELECT count() AS c FROM reasoning_traces WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )
        total = int(rows[0]["c"]) if rows else 0
        if total <= max_total:
            return 0
        excess = total - max_total
        # NB: SurrealDB v3.2.0 requires every ORDER BY field to appear in the
        # SELECT projection, so created_at is projected even though only
        # trace_hash is consumed below.
        victims = await self._query(
            "SELECT trace_hash, created_at FROM reasoning_traces"
            " WHERE brain_id = $bid AND processed = true"
            f" ORDER BY created_at ASC LIMIT {int(excess)}",
            bid=brain_id,
        )
        ids = [r.get("trace_hash") for r in victims if r.get("trace_hash")]
        if not ids:
            return 0
        await self._query(
            "DELETE reasoning_traces WHERE brain_id = $bid AND trace_hash IN $ids",
            bid=brain_id,
            ids=ids,
        )
        return len(ids)

    async def get_reasoning_stats(self, brain_id: str) -> dict[str, Any]:
        """Aggregate reasoning-trace stats: per model, per category, totals.

        ``last_trace_at`` is fetched per model with an ORDER BY ... LIMIT 1
        rather than a datetime aggregate, keeping the SurrealQL unambiguous.
        That per-model query MUST keep its ``WITH INDEX idx_rtr_model`` hint —
        see the comment on the query below; without it this whole endpoint
        exceeds the SDK's 30s client timeout and the caller 500s.
        """
        model_rows = await self._query(
            "SELECT model, count() AS cnt FROM reasoning_traces"
            " WHERE brain_id = $bid GROUP BY model",
            bid=brain_id,
        )
        unproc_rows = await self._query(
            "SELECT model, count() AS cnt FROM reasoning_traces"
            " WHERE brain_id = $bid AND processed = false GROUP BY model",
            bid=brain_id,
        )
        unproc_by_model = {r.get("model", ""): int(r.get("cnt", 0) or 0) for r in unproc_rows}

        by_model: dict[str, dict[str, Any]] = {}
        for r in model_rows:
            name = r.get("model", "")
            # WITH INDEX idx_rtr_model is load-bearing, not a micro-optimisation.
            # Left to its own devices the 3.2.x planner satisfies the ORDER BY by
            # walking idx_rtr_time (brain_id, created_at) BACKWARD across the
            # entire brain scope and applying `model` as a post-index Filter, so
            # cost scales with the brain's total trace count, not the model's.
            # Measured on a 10.5k-row brain: EXPLAIN FULL showed IndexScan
            # idx_rtr_time direction=Backward output_rows=10496 taking 92.6s to
            # answer for a model owning 57 rows — past the SDK's 30s client
            # timeout, so /reasoning status raised TimeoutError and 500'd. The
            # hint pins the composite (brain_id, model) index, so the scan reads
            # only that model's rows and sorts them: 92.6s -> 2.2ms.
            # Safe on installs predating idx_rtr_model: SurrealDB 3.2.3 does not
            # error on an unknown index name in the hint, it just declines the
            # backward-idx_rtr_time plan (measured 13-16ms, correct rows), so a
            # missing index degrades to "merely fast" rather than hard-failing.
            # A single grouped `time::max(created_at) ... GROUP BY model` would
            # also collapse this N+1 (measured ~98ms for all models, identical
            # values) — viable if the loop ever needs to go, but the hint alone
            # removes the pathology.
            latest = await self._query(
                "SELECT created_at FROM reasoning_traces WITH INDEX idx_rtr_model"
                " WHERE brain_id = $bid AND model = $model"
                " ORDER BY created_at DESC LIMIT 1",
                bid=brain_id,
                model=name,
            )
            by_model[name] = {
                "trace_count": int(r.get("cnt", 0) or 0),
                "unprocessed": unproc_by_model.get(name, 0),
                "last_trace_at": _iso(latest[0].get("created_at")) if latest else "",
            }

        cat_rows = await self._query(
            "SELECT category, count() AS cnt FROM reasoning_traces"
            " WHERE brain_id = $bid GROUP BY category",
            bid=brain_id,
        )
        by_category = {r.get("category", ""): int(r.get("cnt", 0) or 0) for r in cat_rows}
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
        rows = await self._query(
            "SELECT model FROM reasoning_traces WHERE brain_id = $bid GROUP BY model",
            bid=brain_id,
        )
        return sorted({r.get("model", "") for r in rows if r.get("model")})

    async def delete_reasoning_traces_by_model(self, brain_id: str, model: str) -> int:
        """Delete all reasoning traces for a model (privacy wipe); return count."""
        if not model:
            return 0
        rows = await self._query(
            "SELECT count() AS c FROM reasoning_traces"
            " WHERE brain_id = $bid AND model = $model GROUP ALL",
            bid=brain_id,
            model=model,
        )
        deleted = int(rows[0]["c"]) if rows else 0
        await self._query(
            "DELETE reasoning_traces WHERE brain_id = $bid AND model = $model",
            bid=brain_id,
            model=model,
        )
        return deleted

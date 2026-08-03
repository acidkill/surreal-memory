"""SurrealDB mixin for tool-event storage and statistics.

Parity with the SQLite ``tool_events`` table. Without this, the SurrealDB
backend has no tool-event surface, so consolidation's process_tool_events
strategy (which calls ``get_unprocessed_events``) and the dashboard tool-stats
endpoint raise AttributeError on SurrealDB.

Rows carry a plain ``event_id`` (uuid string) so ``mark_events_processed`` can
filter by a simple string field instead of juggling SurrealDB record ids.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

# Cap per brain to prevent unbounded growth (mirrors the SQLite mixin).
_MAX_EVENTS_PER_BRAIN = 100_000


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


class SurrealDBToolEventsMixin:
    """Tool-event CRUD + stats backed by the ``tool_events`` table."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def insert_tool_events(
        self,
        brain_id: str,
        events: list[dict[str, Any]],
    ) -> int:
        """Insert raw tool events into the staging table."""
        if not events:
            return 0
        conn = self._ensure_conn()
        inserted = 0
        for ev in events:
            event_id = str(uuid.uuid4())
            await conn.insert(
                "tool_events",
                {
                    "id": _to_surreal_id(event_id),
                    "event_id": event_id,
                    "brain_id": brain_id,
                    "tool_name": ev.get("tool_name", ""),
                    "server_name": ev.get("server_name", ""),
                    "args_summary": str(ev.get("args_summary", ""))[:200],
                    "success": bool(ev.get("success", True)),
                    "duration_ms": int(ev.get("duration_ms", 0) or 0),
                    "session_id": ev.get("session_id", ""),
                    "task_context": ev.get("task_context", ""),
                    "processed": False,
                    "created_at": _as_datetime(ev.get("created_at")),
                },
            )
            inserted += 1
        return inserted

    async def get_unprocessed_events(
        self,
        brain_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get unprocessed tool events (oldest first) for pattern detection."""
        safe_limit = min(int(limit), 10000)
        rows = await self._query(
            "SELECT event_id, tool_name, server_name, args_summary, success,"
            " duration_ms, session_id, task_context, created_at"
            " FROM tool_events WHERE brain_id = $bid AND processed = false"
            f" ORDER BY created_at ASC LIMIT {safe_limit}",
            bid=brain_id,
        )
        return [
            {
                "id": r.get("event_id"),
                "tool_name": r.get("tool_name", ""),
                "server_name": r.get("server_name", ""),
                "args_summary": r.get("args_summary", ""),
                "success": bool(r.get("success", True)),
                "duration_ms": r.get("duration_ms", 0),
                "session_id": r.get("session_id", ""),
                "task_context": r.get("task_context", ""),
                "created_at": _iso(r.get("created_at")),
            }
            for r in rows
        ]

    async def get_tool_events_for_mining(
        self,
        brain_id: str,
        since: datetime | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Return tool events (tool_name + created_at) oldest-first for habit mining.

        Unlike ``get_unprocessed_events`` this ignores the ``processed`` flag —
        habit mining observes the full accumulated history and never mutates it
        (tool events carry no session_id, so mining treats them as one stream).
        """
        safe_limit = min(int(limit), 20000)
        sql = "SELECT tool_name, session_id, created_at FROM tool_events WHERE brain_id = $bid"
        params: dict[str, Any] = {"bid": brain_id}
        if since is not None:
            sql += " AND created_at > $since"
            params["since"] = since
        sql += f" ORDER BY created_at ASC LIMIT {safe_limit}"
        rows = await self._query(sql, **params)
        return [
            {
                "tool_name": r.get("tool_name", ""),
                "session_id": r.get("session_id", ""),
                "created_at": _iso(r.get("created_at")),
            }
            for r in rows
        ]

    async def mark_events_processed(
        self,
        brain_id: str,
        event_ids: list[Any],
    ) -> None:
        """Mark tool events as processed by their event_id."""
        if not event_ids:
            return
        await self._query(
            "UPDATE tool_events SET processed = true WHERE brain_id = $bid AND event_id IN $ids",
            bid=brain_id,
            ids=[str(e) for e in event_ids],
        )

    async def prune_old_events(
        self,
        brain_id: str,
        keep_days: int = 90,
    ) -> int:
        """Delete processed events older than keep_days; return rows deleted."""
        cutoff = utcnow() - timedelta(days=keep_days)
        rows = await self._query(
            "SELECT count() AS c FROM tool_events"
            " WHERE brain_id = $bid AND processed = true AND created_at < $cutoff GROUP ALL",
            bid=brain_id,
            cutoff=cutoff,
        )
        deleted = int(rows[0]["c"]) if rows else 0
        await self._query(
            "DELETE tool_events WHERE brain_id = $bid AND processed = true AND created_at < $cutoff",
            bid=brain_id,
            cutoff=cutoff,
        )
        return deleted

    async def cap_tool_events(self, brain_id: str) -> int:
        """Enforce max events per brain by deleting oldest processed rows."""
        rows = await self._query(
            "SELECT count() AS c FROM tool_events WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )
        total = int(rows[0]["c"]) if rows else 0
        if total <= _MAX_EVENTS_PER_BRAIN:
            return 0
        excess = total - _MAX_EVENTS_PER_BRAIN
        victims = await self._query(
            "SELECT event_id FROM tool_events WHERE brain_id = $bid AND processed = true"
            f" ORDER BY created_at ASC LIMIT {int(excess)}",
            bid=brain_id,
        )
        ids = [r.get("event_id") for r in victims if r.get("event_id")]
        if not ids:
            return 0
        await self._query(
            "DELETE tool_events WHERE brain_id = $bid AND event_id IN $ids",
            bid=brain_id,
            ids=ids,
        )
        return len(ids)

    async def get_tool_stats(self, brain_id: str, days: int = 30) -> dict[str, Any]:
        """Tool usage statistics: total_events, success_rate, top_tools.

        ``days`` filters the whole summary, not just the daily series — before
        this, the caller-facing filter matched the per-day chart but the
        summary above it was always all-time, so three different date ranges
        rendered a byte-identical summary.
        """
        safe_days = min(max(int(days), 1), 365)
        cutoff = utcnow() - timedelta(days=safe_days)
        total_rows = await self._query(
            "SELECT count() AS c FROM tool_events"
            " WHERE brain_id = $bid AND created_at >= $cutoff GROUP ALL",
            bid=brain_id,
            cutoff=cutoff,
        )
        total = int(total_rows[0]["c"]) if total_rows else 0
        ok_rows = await self._query(
            "SELECT count() AS c FROM tool_events"
            " WHERE brain_id = $bid AND created_at >= $cutoff AND success = true GROUP ALL",
            bid=brain_id,
            cutoff=cutoff,
        )
        successes = int(ok_rows[0]["c"]) if ok_rows else 0
        grouped = await self._query(
            "SELECT tool_name, server_name, count() AS cnt,"
            " math::mean(duration_ms) AS avg_ms FROM tool_events"
            " WHERE brain_id = $bid AND created_at >= $cutoff"
            " GROUP BY tool_name, server_name",
            bid=brain_id,
            cutoff=cutoff,
        )
        # Per-tool success counts. `math::sum(success)` does NOT coerce a bool to
        # 1/0 on SurrealDB (it returns 0), so count the success=true rows per
        # group separately instead. Without success_rate/avg_duration_ms here the
        # web UI renders "NaN%" / "NaNs" for every tool row.
        ok_grouped = await self._query(
            "SELECT tool_name, server_name, count() AS ok FROM tool_events"
            " WHERE brain_id = $bid AND created_at >= $cutoff AND success = true"
            " GROUP BY tool_name, server_name",
            bid=brain_id,
            cutoff=cutoff,
        )
        ok_by_key = {
            (r.get("tool_name", ""), r.get("server_name", "")): int(r.get("ok", 0) or 0)
            for r in ok_grouped
        }
        grouped.sort(key=lambda r: int(r.get("cnt", 0) or 0), reverse=True)
        top_tools = []
        for r in grouped[:20]:
            name = r.get("tool_name", "")
            server = r.get("server_name", "")
            cnt = int(r.get("cnt", 0) or 0)
            ok = ok_by_key.get((name, server), 0)
            top_tools.append(
                {
                    "tool_name": name,
                    "server_name": server,
                    "count": cnt,
                    "success_rate": round(ok / cnt, 2) if cnt > 0 else 0.0,
                    "avg_duration_ms": round(float(r.get("avg_ms", 0) or 0)),
                }
            )
        return {
            "total_events": total,
            "success_rate": round(successes / total, 2) if total > 0 else 0,
            "top_tools": top_tools,
        }

    async def get_tool_stats_by_period(
        self,
        brain_id: str,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Tool usage stats aggregated by day (best-effort on SurrealDB)."""
        safe_days = min(max(int(days), 1), 365)
        cutoff = utcnow() - timedelta(days=safe_days)
        rows = await self._query(
            "SELECT time::format(created_at, '%Y-%m-%d') AS day, tool_name,"
            " count() AS cnt, math::mean(duration_ms) AS avg_ms FROM tool_events"
            " WHERE brain_id = $bid AND created_at >= $cutoff"
            " GROUP BY day, tool_name",
            bid=brain_id,
            cutoff=cutoff,
        )
        # Per (day, tool) success counts so the trend chart's success line has a
        # real success_rate instead of NaN.
        ok_rows = await self._query(
            "SELECT time::format(created_at, '%Y-%m-%d') AS day, tool_name,"
            " count() AS ok FROM tool_events"
            " WHERE brain_id = $bid AND created_at >= $cutoff AND success = true"
            " GROUP BY day, tool_name",
            bid=brain_id,
            cutoff=cutoff,
        )
        ok_by_key = {
            (r.get("day", ""), r.get("tool_name", "")): int(r.get("ok", 0) or 0) for r in ok_rows
        }
        rows.sort(
            key=lambda r: (str(r.get("day", "")), int(r.get("cnt", 0) or 0)),
            reverse=True,
        )
        result: list[dict[str, Any]] = []
        for r in rows[: min(int(limit), 50)]:
            day = r.get("day", "")
            name = r.get("tool_name", "")
            cnt = int(r.get("cnt", 0) or 0)
            ok = ok_by_key.get((day, name), 0)
            result.append(
                {
                    "date": day,
                    "tool_name": name,
                    "count": cnt,
                    "success_rate": round(ok / cnt, 2) if cnt > 0 else 0.0,
                    "avg_duration_ms": round(float(r.get("avg_ms", 0) or 0)),
                }
            )
        return result

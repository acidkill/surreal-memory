"""Unit tests for the SurrealDB tool-events mixin (no live DB required).

Regression coverage: the SurrealDB backend lacked any tool-event surface, so
consolidation's process_tool_events strategy raised
``AttributeError: 'SurrealDBStorage' object has no attribute 'get_unprocessed_events'``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from surreal_memory.storage.surrealdb.tool_events import SurrealDBToolEventsMixin


class _ToolEventsStore(SurrealDBToolEventsMixin):
    """Routes _query by SQL fragment; records UPDATE/insert calls."""

    def __init__(self, unprocessed=None, total=0, ok=0, grouped=None, ok_grouped=None) -> None:
        self._unprocessed = unprocessed or []
        self._total = total
        self._ok = ok
        self._grouped = grouped or []
        self._ok_grouped = ok_grouped or []
        self.updates: list[dict[str, Any]] = []
        self.inserts: list[dict[str, Any]] = []
        self.captured_cutoffs: list[Any] = []

    def _ensure_conn(self) -> Any:
        store = self

        class _Conn:
            async def insert(self, table: str, data: dict[str, Any]) -> None:
                store.inserts.append({"table": table, "data": data})

        return _Conn()

    def _get_brain_id(self) -> str:
        return "default"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        if "cutoff" in params:
            self.captured_cutoffs.append(params["cutoff"])
        if sql.startswith("UPDATE tool_events SET processed = true"):
            self.updates.append(params)
            return []
        if "processed = false" in sql:
            return self._unprocessed
        if "AND success = true GROUP ALL" in sql:
            return [{"c": self._ok}]
        if "count() AS c FROM tool_events" in sql and "success" not in sql:
            return [{"c": self._total}]
        # Per-tool success counts (check before the generic grouped route).
        if "AND success = true" in sql and "GROUP BY tool_name, server_name" in sql:
            return self._ok_grouped
        if "GROUP BY tool_name, server_name" in sql:
            return self._grouped
        return []


async def test_get_unprocessed_events_maps_rows() -> None:
    store = _ToolEventsStore(
        unprocessed=[
            {
                "event_id": "abc",
                "tool_name": "Read",
                "server_name": "",
                "args_summary": "x",
                "success": True,
                "duration_ms": 12,
                "session_id": "s1",
                "task_context": "t",
                "created_at": datetime(2026, 5, 29, 7, 0, 0),
            }
        ]
    )
    events = await store.get_unprocessed_events("default", 200)
    assert len(events) == 1
    ev = events[0]
    assert ev["id"] == "abc"
    assert ev["tool_name"] == "Read"
    assert ev["success"] is True
    # created_at must be an ISO string (process_events calls fromisoformat on it)
    assert isinstance(ev["created_at"], str)
    datetime.fromisoformat(ev["created_at"])


async def test_mark_events_processed_issues_update_with_ids() -> None:
    store = _ToolEventsStore()
    await store.mark_events_processed("default", ["abc", "def"])
    assert len(store.updates) == 1
    assert store.updates[0]["ids"] == ["abc", "def"]


async def test_mark_events_processed_noop_on_empty() -> None:
    store = _ToolEventsStore()
    await store.mark_events_processed("default", [])
    assert store.updates == []


async def test_get_tool_stats_computes_rate_and_top_tools() -> None:
    store = _ToolEventsStore(
        total=4,
        ok=3,
        grouped=[
            {"tool_name": "Read", "server_name": "", "cnt": 3, "avg_ms": 12.5},
            {"tool_name": "Bash", "server_name": "", "cnt": 1, "avg_ms": 800.0},
        ],
        ok_grouped=[
            {"tool_name": "Read", "server_name": "", "ok": 2},
            {"tool_name": "Bash", "server_name": "", "ok": 1},
        ],
    )
    stats = await store.get_tool_stats("default")
    assert stats["total_events"] == 4
    assert stats["success_rate"] == 0.75
    read = stats["top_tools"][0]
    assert read["tool_name"] == "Read"
    assert read["count"] == 3
    # Per-tool fields must be present and numeric — the web UI renders "NaN%" /
    # "NaNs" when success_rate / avg_duration_ms are missing.
    assert read["success_rate"] == round(2 / 3, 2)
    assert read["avg_duration_ms"] == round(12.5)  # 12 — Python banker's rounding
    bash = stats["top_tools"][1]
    assert bash["success_rate"] == 1.0
    assert bash["avg_duration_ms"] == 800


async def test_get_tool_stats_no_success_rows_is_zero_not_nan() -> None:
    """A tool with zero successes must yield success_rate 0.0, never NaN."""
    store = _ToolEventsStore(
        total=2,
        ok=0,
        grouped=[{"tool_name": "Bash", "server_name": "", "cnt": 2, "avg_ms": 0.0}],
        ok_grouped=[],  # no success=true rows for any tool
    )
    stats = await store.get_tool_stats("default")
    tool = stats["top_tools"][0]
    assert tool["success_rate"] == 0.0
    assert tool["avg_duration_ms"] == 0
    assert isinstance(tool["success_rate"], float)


async def test_get_tool_stats_passes_days_as_a_cutoff_to_every_query() -> None:
    """`days` must filter the summary, not just the daily series.

    Before this, `get_tool_stats` took no `days` at all, so the dashboard's
    days=7/30/90 filter changed the per-day chart but left the summary above
    it byte-identical -- a working-looking filter that filtered nothing.
    """
    store = _ToolEventsStore(total=1, ok=1)

    await store.get_tool_stats("default", days=7)

    # All 4 queries (total, ok, grouped, ok_grouped) must carry the same cutoff.
    assert len(store.captured_cutoffs) == 4
    assert len(set(store.captured_cutoffs)) == 1


async def test_get_tool_stats_default_days_is_30() -> None:
    store = _ToolEventsStore(total=1, ok=1)

    await store.get_tool_stats("default")

    assert len(store.captured_cutoffs) == 4


async def test_get_tool_stats_clamps_days_to_one_year() -> None:
    """Matches get_tool_stats_by_period's existing clamp -- an operator-supplied
    `days` should not silently become an unbounded full-table scan."""
    store_small = _ToolEventsStore(total=1, ok=1)
    store_large = _ToolEventsStore(total=1, ok=1)

    await store_small.get_tool_stats("default", days=1)
    await store_large.get_tool_stats("default", days=999_999)

    # The clamp caps at 365 days, so an absurd `days` produces the same
    # earliest-allowed cutoff as 365 would -- not an even-earlier one.
    assert store_small.captured_cutoffs[0] > store_large.captured_cutoffs[0]

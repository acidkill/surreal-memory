from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.activity import SurrealDBActivityMixin, _parse_datetime
from surreal_memory.storage.surrealdb.tool_events import SurrealDBToolEventsMixin


class _ActionReader(SurrealDBActivityMixin):
    def __init__(self) -> None:
        self.sql = ""
        self.params: dict[str, Any] = {}

    def _get_brain_id(self) -> str:
        return "brain-a"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.sql, self.params = sql, params
        return [
            {
                "id": "action_log:physical_action",
                "action_type": "recall",
                "action_context": "alpha beta",
                "session_id": "session-a",
                "created_at": params["until"],
            }
        ]

    def _ensure_conn(self) -> Any:
        raise AssertionError("connection should not be needed")


class _ToolReader(SurrealDBToolEventsMixin):
    def __init__(self) -> None:
        self.sql = ""
        self.params: dict[str, Any] = {}

    def _get_brain_id(self) -> str:
        return "brain-a"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.sql, self.params = sql, params
        return [
            {
                "id": "tool_events:physical_tool_id",
                "event_id": "different_logical_id",
                "tool_name": "Read",
                "session_id": "session-a",
                "created_at": params["until"],
            }
        ]

    def _ensure_conn(self) -> Any:
        raise AssertionError("connection should not be needed")


@pytest.mark.parametrize("value", [None, "not-a-timestamp"])
def test_action_timestamp_parser_rejects_missing_and_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        _parse_datetime(value)


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-24T14:00:00+02:00",
        datetime(2026, 10, 24, 14, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_action_timestamp_parser_normalizes_offset_to_utc(value: Any) -> None:
    assert _parse_datetime(value) == datetime(2026, 10, 24, 12)


@pytest.mark.asyncio
async def test_action_reader_binds_frozen_window_and_physical_cursor() -> None:
    storage = _ActionReader()
    since = datetime(2026, 10, 1)
    until = datetime(2026, 10, 24)
    cursor_time = until - timedelta(seconds=1)

    rows = await storage.get_action_sequences(
        since=since, until=until, limit=2, after=(cursor_time, "cursor-id")
    )

    assert rows[0].id == "physical_action"
    assert "created_at <= $until" in storage.sql
    assert "id > type::record('action_log', $cursor_id)" in storage.sql
    assert "ORDER BY created_at ASC, id ASC LIMIT $limit" in storage.sql
    assert storage.params["since"] == since
    assert storage.params["until"] == until
    assert storage.params["cursor_id"] == "cursor_id"


@pytest.mark.asyncio
async def test_tool_reader_projects_and_returns_physical_record_id() -> None:
    storage = _ToolReader()
    since = datetime(2026, 10, 1)
    until = datetime(2026, 10, 24)
    cursor_time = until - timedelta(seconds=1)

    rows = await storage.get_tool_events_for_mining(
        "brain-a", since, 2, (cursor_time, "cursor-id"), until=until
    )

    assert rows[0]["id"] == "physical_tool_id"
    assert rows[0]["id"] != "different_logical_id"
    assert "SELECT id, event_id, tool_name" in storage.sql
    assert "created_at <= $until" in storage.sql
    assert "id > type::record('tool_events', $cursor_id)" in storage.sql
    assert "ORDER BY created_at ASC, id ASC LIMIT 2" in storage.sql
    assert storage.params["since"] == since
    assert storage.params["until"] == until
    assert storage.params["cursor_id"] == "cursor_id"

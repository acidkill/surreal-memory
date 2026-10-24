from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

import surreal_memory.core.action_event as action_event_module
from surreal_memory.core.action_event import ActionEvent
from surreal_memory.engine.habit_input import action_events, tool_events
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow


class _ActionPages:
    def __init__(self, events: list[ActionEvent], *, repeat: bool = False) -> None:
        self.events = events
        self.repeat = repeat
        self.calls = 0

    async def get_action_sequences(
        self,
        *,
        since: datetime,
        until: datetime,
        limit: int,
        after: tuple[datetime, str] | None,
    ) -> list[ActionEvent]:
        self.calls += 1
        rows = (
            self.events
            if self.repeat and after
            else [
                event
                for event in self.events
                if ensure_naive_utc(event.created_at) <= ensure_naive_utc(until)
                and (after is None or (ensure_naive_utc(event.created_at), event.id) > after)
            ]
        )
        return rows[:limit]


class _ToolPages:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def get_tool_events_for_mining(
        self,
        brain_id: str,
        *,
        since: datetime,
        until: datetime,
        limit: int,
        after: tuple[datetime, str] | None,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.rows
            if row["brain_id"] == brain_id
            and ensure_naive_utc(row["created_at"]) <= ensure_naive_utc(until)
            and (after is None or (ensure_naive_utc(row["created_at"]), row["id"]) > after)
        ]
        return rows[:limit]


@pytest.mark.asyncio
async def test_action_pages_keep_timestamp_ties_and_exclude_future_rows() -> None:
    at = utcnow()
    events = [
        ActionEvent(id=f"event-{index}", created_at=at, action_type="recall") for index in range(3)
    ]
    events.append(ActionEvent(id="future", created_at=at + timedelta(microseconds=1)))
    storage = _ActionPages(events)

    result = [
        event
        async for event in action_events(
            storage, since=at - timedelta(days=1), until=at, page_size=2
        )
    ]

    assert [event.id for event in result] == ["event-0", "event-1", "event-2"]


@pytest.mark.asyncio
async def test_action_pages_normalize_mixed_timestamp_offsets_before_ordering() -> None:
    instant = datetime(2026, 10, 24, 12, tzinfo=UTC)
    events = [
        ActionEvent(
            id="a",
            created_at=datetime(2026, 10, 24, 14, tzinfo=timezone(timedelta(hours=2))),
        ),
        ActionEvent(id="b", created_at=datetime(2026, 10, 24, 12)),
    ]

    result = [
        event
        async for event in action_events(
            _ActionPages(events),
            since=instant - timedelta(days=1),
            until=instant,
            page_size=1,
        )
    ]

    assert [(event.id, event.created_at) for event in result] == [
        ("a", datetime(2026, 10, 24, 12)),
        ("b", datetime(2026, 10, 24, 12)),
    ]


@pytest.mark.asyncio
async def test_tool_pages_normalize_mixed_timestamp_offsets_before_ordering() -> None:
    instant = datetime(2026, 10, 24, 12, tzinfo=UTC)
    rows = [
        {
            "brain_id": "brain-a",
            "id": "a",
            "created_at": datetime(2026, 10, 24, 14, tzinfo=timezone(timedelta(hours=2))),
            "tool_name": "Read",
        },
        {
            "brain_id": "brain-a",
            "id": "b",
            "created_at": datetime(2026, 10, 24, 12),
            "tool_name": "Write",
        },
    ]

    result = [
        row
        async for row in tool_events(
            _ToolPages(rows),
            "brain-a",
            since=instant - timedelta(days=1),
            until=instant,
            page_size=1,
        )
    ]

    assert [(row["id"], row["created_at"]) for row in result] == [
        ("a", datetime(2026, 10, 24, 12)),
        ("b", datetime(2026, 10, 24, 12)),
    ]


@pytest.mark.asyncio
async def test_action_pages_fail_on_nonadvancing_cursor() -> None:
    at = utcnow()
    event = ActionEvent(id="same", created_at=at)
    storage = _ActionPages([event], repeat=True)

    with pytest.raises(RuntimeError, match="cursor did not advance"):
        _ = [
            row
            async for row in action_events(
                storage, since=at - timedelta(days=1), until=at, page_size=1
            )
        ]


@pytest.mark.asyncio
async def test_action_pages_fail_on_missing_stable_id() -> None:
    at = utcnow()
    storage = _ActionPages([ActionEvent(id="", created_at=at)])

    with pytest.raises(RuntimeError, match="missing stable id"):
        _ = [
            row
            async for row in action_events(
                storage, since=at - timedelta(days=1), until=at, page_size=1
            )
        ]


@pytest.mark.asyncio
async def test_tool_pages_fail_on_missing_stable_id() -> None:
    at = utcnow()
    storage = _ToolPages([{"brain_id": "brain-a", "id": "", "created_at": at, "tool_name": "Read"}])

    with pytest.raises(RuntimeError, match="missing stable id"):
        _ = [
            row
            async for row in tool_events(
                storage,
                "brain-a",
                since=at - timedelta(days=1),
                until=at,
                page_size=1,
            )
        ]


@pytest.mark.asyncio
async def test_in_memory_habit_readers_keep_brains_separate_and_freeze_time() -> None:
    from surreal_memory.core.brain import Brain

    storage = InMemoryStorage()
    brain_a = Brain.create(name="brain-a", owner_id="test")
    brain_b = Brain.create(name="brain-b", owner_id="test")
    await storage.save_brain(brain_a)
    await storage.save_brain(brain_b)
    cutoff = utcnow()

    storage.set_brain(brain_a.id)
    await storage.record_action("recall", "alpha beta", session_id="a")
    storage._action_events[brain_a.id][-1]["created_at"] = cutoff
    await storage.insert_tool_events(
        brain_a.id,
        [
            {"id": "tool-a", "tool_name": "Read", "created_at": cutoff},
            {
                "id": "tool-a-future",
                "tool_name": "FutureRead",
                "created_at": cutoff + timedelta(microseconds=1),
            },
        ],
    )
    storage.set_brain(brain_b.id)
    await storage.record_action("recall", "foreign topic", session_id="b")
    storage._action_events[brain_b.id][-1]["created_at"] = cutoff
    await storage.record_action("future", session_id="b")
    storage._action_events[brain_b.id][-1]["created_at"] = cutoff + timedelta(microseconds=1)
    await storage.insert_tool_events(
        brain_b.id,
        [{"id": "tool-b", "tool_name": "Write", "created_at": cutoff}],
    )

    actions = await storage.get_action_sequences(until=cutoff)
    tools_a = await storage.get_tool_events_for_mining(
        brain_a.id, since=cutoff - timedelta(days=1), until=cutoff
    )

    assert [event.session_id for event in actions] == ["b"]
    assert [row["tool_name"] for row in tools_a] == ["Read"]


@pytest.mark.asyncio
async def test_in_memory_action_page_constructs_only_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryStorage()
    storage.set_brain("brain-a")
    at = datetime(2026, 10, 24)
    storage._action_events["brain-a"] = [
        {
            "id": f"event-{index:04}",
            "brain_id": "brain-a",
            "session_id": "session-a",
            "action_type": "recall",
            "action_context": "",
            "tags": (),
            "fiber_id": None,
            "created_at": at,
        }
        for index in range(100)
    ]
    original_action_event = action_event_module.ActionEvent
    constructed: list[str] = []

    def track_construction(**kwargs: Any) -> ActionEvent:
        constructed.append(kwargs["id"])
        return original_action_event(**kwargs)

    monkeypatch.setattr(action_event_module, "ActionEvent", track_construction)

    page = await storage.get_action_sequences(limit=3)

    assert [event.id for event in page] == ["event-0000", "event-0001", "event-0002"]
    assert constructed == ["event-0000", "event-0001", "event-0002"]


@pytest.mark.asyncio
async def test_in_memory_tool_page_builds_only_selected_rows() -> None:
    class TrackedEvent(dict[str, Any]):
        session_id_reads = 0

        def get(self, key: str, default: Any = None) -> Any:
            if key == "session_id":
                self.session_id_reads += 1
            return super().get(key, default)

    storage = InMemoryStorage()
    storage.set_brain("brain-a")
    at = datetime(2026, 10, 24)
    events = [
        TrackedEvent(
            id=f"event-{index:04}",
            tool_name="Read",
            session_id="session-a",
            created_at=at,
        )
        for index in range(100)
    ]
    storage._action_events["brain-a"] = events

    page = await storage.get_tool_events_for_mining("brain-a", limit=3)

    assert [row["id"] for row in page] == ["event-0000", "event-0001", "event-0002"]
    assert sum(event.session_id_reads for event in events) == 3

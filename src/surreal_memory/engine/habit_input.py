"""Bounded, frozen-window readers for habitual input event streams."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from typing import Any

from surreal_memory.utils.timeutils import ensure_naive_utc


def _key(created_at: Any, event_id: Any) -> tuple[datetime, str]:
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError("habit input page has an invalid timestamp") from exc
    if not isinstance(created_at, datetime):
        raise RuntimeError("habit input page has a missing timestamp")
    if not isinstance(event_id, str) or not event_id.strip():
        raise RuntimeError("habit input page has a missing stable id")
    return ensure_naive_utc(created_at), event_id


async def action_events(
    storage: Any,
    *,
    since: datetime,
    until: datetime,
    page_size: int,
) -> AsyncIterator[Any]:
    """Yield action rows in strict key order without retaining prior pages."""
    cursor: tuple[datetime, str] | None = None
    while True:
        page = await storage.get_action_sequences(
            since=since, until=until, limit=page_size, after=cursor
        )
        if len(page) > page_size:
            raise RuntimeError("habit input page exceeded its requested size")
        previous = cursor
        for event in page:
            key = _key(event.created_at, event.id)
            if previous is not None and key <= previous:
                raise RuntimeError("habit input action cursor did not advance")
            if key[0] > ensure_naive_utc(until):
                raise RuntimeError("habit input action exceeded its frozen window")
            yield replace(event, created_at=key[0])
            previous = key
        if len(page) < page_size:
            return
        if previous is None or previous == cursor:
            raise RuntimeError("habit input action cursor did not advance")
        cursor = previous


async def tool_events(
    storage: Any,
    brain_id: str,
    *,
    since: datetime,
    until: datetime,
    page_size: int,
) -> AsyncIterator[dict[str, Any]]:
    """Yield tool rows in strict key order without retaining prior pages."""
    cursor: tuple[datetime, str] | None = None
    while True:
        page = await storage.get_tool_events_for_mining(
            brain_id, since=since, until=until, limit=page_size, after=cursor
        )
        if len(page) > page_size:
            raise RuntimeError("habit input page exceeded its requested size")
        previous = cursor
        for row in page:
            key = _key(row.get("created_at"), row.get("id"))
            if previous is not None and key <= previous:
                raise RuntimeError("habit input tool cursor did not advance")
            if key[0] > ensure_naive_utc(until):
                raise RuntimeError("habit input tool exceeded its frozen window")
            normalized = dict(row)
            normalized["created_at"] = key[0]
            yield normalized
            previous = key
        if len(page) < page_size:
            return
        if previous is None or previous == cursor:
            raise RuntimeError("habit input tool cursor did not advance")
        cursor = previous

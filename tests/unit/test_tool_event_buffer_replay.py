"""Crash and concurrent-append coverage for JSONL tool-event ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from surreal_memory.engine.tool_memory import ingest_buffer


class _BufferStorage:
    """Durable-ID fake that can fail after a partial batch commit."""

    def __init__(self) -> None:
        self.events: dict[str, dict[str, Any]] = {}
        self.fail_after_first = False
        self.append_path: Path | None = None
        self.append_line: str | None = None

    async def insert_tool_events(self, brain_id: str, events: list[dict[str, Any]]) -> int:
        del brain_id
        inserted = 0
        for index, event in enumerate(events):
            event_id = event["event_id"]
            if event_id not in self.events:
                self.events[event_id] = {**event, "processed": False}
                inserted += 1
            if self.fail_after_first and index == 0:
                self.fail_after_first = False
                raise RuntimeError("simulated crash after DB commit")
        if self.append_path is not None and self.append_line is not None:
            with self.append_path.open("a", encoding="utf-8") as buffer:
                buffer.write(self.append_line)
            self.append_path = None
        return inserted


def _line(tool_name: str, event_id: str | None = None) -> str:
    event: dict[str, Any] = {"tool_name": tool_name, "created_at": "2026-09-23T10:00:00"}
    if event_id is not None:
        event["event_id"] = event_id
    return json.dumps(event) + "\n"


@pytest.mark.asyncio
async def test_partial_db_commit_replays_with_same_legacy_ids(tmp_path: Path) -> None:
    path = tmp_path / "tool_events.jsonl"
    original = _line("Read") + _line("Grep")
    path.write_text(original, encoding="utf-8")
    storage = _BufferStorage()
    storage.fail_after_first = True

    with pytest.raises(RuntimeError, match="after DB commit"):
        await ingest_buffer(storage, "default", path, max_lines=10)

    first_id = next(iter(storage.events))
    assert path.read_text(encoding="utf-8") == original

    resumed = await ingest_buffer(storage, "default", path, max_lines=10)

    assert resumed.events_ingested == 1
    assert len(storage.events) == 2
    assert first_id in storage.events
    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_max_lines_acknowledges_only_oldest_complete_prefix(tmp_path: Path) -> None:
    path = tmp_path / "tool_events.jsonl"
    lines = [_line("Read", "event-a"), _line("Grep", "event-b"), _line("Edit", "event-c")]
    path.write_text("".join(lines), encoding="utf-8")
    storage = _BufferStorage()

    result = await ingest_buffer(storage, "default", path, max_lines=2)

    assert result.events_ingested == 2
    assert set(storage.events) == {"event-a", "event-b"}
    assert path.read_text(encoding="utf-8") == lines[2]


@pytest.mark.asyncio
async def test_concurrent_append_is_preserved_after_prefix_ack(tmp_path: Path) -> None:
    path = tmp_path / "tool_events.jsonl"
    first = _line("Read", "event-a")
    appended = _line("Grep", "event-b")
    path.write_text(first, encoding="utf-8")
    storage = _BufferStorage()
    storage.append_path = path
    storage.append_line = appended

    result = await ingest_buffer(storage, "default", path)

    assert result.events_ingested == 1
    assert set(storage.events) == {"event-a"}
    assert path.read_text(encoding="utf-8") == appended


@pytest.mark.asyncio
async def test_incomplete_tail_is_not_acknowledged(tmp_path: Path) -> None:
    path = tmp_path / "tool_events.jsonl"
    incomplete = '{"tool_name":"Read","event_id":"partial"'
    path.write_text(incomplete, encoding="utf-8")
    storage = _BufferStorage()

    result = await ingest_buffer(storage, "default", path)

    assert result.events_ingested == 0
    assert storage.events == {}
    assert path.read_text(encoding="utf-8") == incomplete

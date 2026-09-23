"""Focused tests for stable tool-event identity and sidecar locking."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from surreal_memory.hooks.post_tool_use import (
    _append_to_buffer,
    _check_buffer_rotation,
    _format_event,
)


def test_formatted_events_have_distinct_stable_uuid_identities() -> None:
    first = _format_event({"tool_name": "Read"})
    second = _format_event({"tool_name": "Read"})

    first_id = uuid.UUID(first["event_id"])
    second_id = uuid.UUID(second["event_id"])
    assert first_id.version == 4
    assert second_id.version == 4
    assert first_id != second_id


def test_append_and_rotation_share_stable_sidecar_lock(tmp_path: Path) -> None:
    fcntl = pytest.importorskip("fcntl")
    buffer_path = tmp_path / "events.jsonl"
    lock_path = Path(f"{buffer_path}.lock")
    buffer_path.write_text(
        "".join(json.dumps({"seq": i}) + "\n" for i in range(6)),
        encoding="utf-8",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        for operation in (
            lambda: _append_to_buffer({"seq": 6}, buffer_path),
            lambda: _check_buffer_rotation(buffer_path, max_lines=4),
        ):
            started = threading.Event()

            def run_operation(operation=operation, started=started) -> object:
                started.set()
                return operation()

            with open(lock_path, "a", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                future = pool.submit(run_operation)
                assert started.wait(timeout=1)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.05)
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            assert future.result(timeout=1) in (True, None)

    assert lock_path.exists()
    lines = buffer_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 4
    assert all(isinstance(json.loads(line), dict) for line in lines)

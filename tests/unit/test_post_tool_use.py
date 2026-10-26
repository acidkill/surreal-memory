"""Focused tests for stable tool-event identity and sidecar locking."""

from __future__ import annotations

import io
import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from surreal_memory.hooks.post_tool_use import (
    _append_to_buffer,
    _check_buffer_rotation,
    _format_event,
    _truncate_args,
    main,
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


def test_arguments_are_redacted_before_preview_truncation() -> None:
    args = {
        "access_token": "do-not-store",
        "cmd": "SURREALDB_PASS='very secret' curl -H 'Authorization: Bearer other-secret' -H 'Authorization: Basic encoded-secret' https://user:pass@example.test",
    }
    preview = _truncate_args(args)
    for secret in ("do-not-store", "very secret", "other-secret", "encoded-secret", "user:pass"):
        assert secret not in preview
    assert "REDACTED" in preview


def test_failed_noise_tool_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("SMEM_HOOK_SOURCE", "codex")
    monkeypatch.setattr(sys, "argv", ["smem-hook-post-tool-use", "--json"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "tool_name": "mcp__context_mode__ctx_search",
                    "session_id": "codex-a",
                    "tool_error": "unavailable",
                }
            )
        ),
    )
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "saved"
    event = json.loads((tmp_path / "tool_events.jsonl").read_text())
    assert event["session_id"] == "codex-a"
    assert event["capture_client"] == "codex"
    assert event["success"] is False
    assert (tmp_path / "tool_events.jsonl").stat().st_mode & 0o077 == 0


def test_buffer_failure_is_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import surreal_memory.hooks.post_tool_use as hook

    monkeypatch.setattr(sys, "argv", ["smem-hook-post-tool-use", "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"tool_name":"Edit"}'))
    monkeypatch.setattr(hook, "_append_to_buffer", lambda *_: False)
    with pytest.raises(SystemExit, match="1"):
        main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "failed"
    assert receipt["saved"] == 0


def test_noise_fast_path_does_not_read_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import surreal_memory.hooks.post_tool_use as hook

    monkeypatch.setattr(sys, "argv", ["smem-hook-post-tool-use", "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"tool_name":"mcp__context_mode__ctx_search"}'))

    def unexpected_config() -> dict:
        raise AssertionError("noise must be filtered before config I/O")

    monkeypatch.setattr(hook, "_read_tool_memory_config", unexpected_config)
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"

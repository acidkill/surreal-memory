"""Tests for the PreCompact Claude Code hook — pure-function coverage."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.hooks.pre_compact import (
    MAX_FLUSH_CHARS,
    _extract_text,
    read_hook_input,
    read_transcript_tail,
)

# ---------------------------------------------------------------------------
# read_hook_input
# ---------------------------------------------------------------------------


def test_read_hook_input_empty_stdin() -> None:
    """Empty stdin returns an empty dict without raising."""
    with patch("sys.stdin", io.StringIO("")):
        result = read_hook_input()
    assert result == {}


def test_read_hook_input_whitespace_only() -> None:
    """Whitespace-only stdin is treated as empty and returns {}."""
    with patch("sys.stdin", io.StringIO("   \n\t  ")):
        result = read_hook_input()
    assert result == {}


def test_read_hook_input_valid_json() -> None:
    """Valid JSON on stdin is parsed and returned."""
    payload = {"transcript_path": "/tmp/session.jsonl", "turn": 5}
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        result = read_hook_input()
    assert result == payload


def test_read_hook_input_malformed_json() -> None:
    """Malformed JSON returns an empty dict without raising."""
    with patch("sys.stdin", io.StringIO("{not: valid json}")):
        result = read_hook_input()
    assert result == {}


@pytest.mark.asyncio
async def test_auto_capture_off_skips_precompact_before_opening_storage(monkeypatch) -> None:
    from surreal_memory.hooks.pre_compact import flush_text
    from surreal_memory.unified_config import WriteGateConfig

    storage_factory = AsyncMock()
    monkeypatch.setattr(
        "surreal_memory.safety.input_firewall.check_content",
        lambda _text: SimpleNamespace(blocked=False, sanitized=None),
    )
    monkeypatch.setattr(
        "surreal_memory.unified_config.get_config",
        lambda: SimpleNamespace(
            current_brain="test", write_gate=WriteGateConfig(auto_capture_mode="off")
        ),
    )
    monkeypatch.setattr("surreal_memory.unified_config.get_shared_storage", storage_factory)

    result = await flush_text("A long enough transcript for automatic capture.")

    assert result["saved"] == 0
    assert result["message"].startswith("Automatic capture disabled")
    storage_factory.assert_not_awaited()


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_string_content() -> None:
    """String content field is returned directly."""
    entry = {"role": "user", "content": "Hello world"}
    assert _extract_text(entry) == "Hello world"


def test_extract_text_list_content_with_text_items() -> None:
    """List content with text-typed items is joined."""
    entry = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Part one"},
            {"type": "tool_use", "id": "xyz"},
            {"type": "text", "text": "Part two"},
        ],
    }
    result = _extract_text(entry)
    assert "Part one" in result
    assert "Part two" in result


def test_extract_text_list_content_no_text_items() -> None:
    """List content with no text-typed items returns an empty string."""
    entry = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "abc"},
            {"type": "image", "source": {}},
        ],
    }
    assert _extract_text(entry) == ""


def test_extract_text_nested_message_format() -> None:
    """Nested message dict is recursed into."""
    entry = {
        "type": "assistant",
        "message": {
            "content": "Nested text here",
        },
    }
    assert _extract_text(entry) == "Nested text here"


def test_extract_text_direct_text_field() -> None:
    """Fallback to top-level 'text' field when no content or message."""
    entry = {"text": "Fallback text"}
    assert _extract_text(entry) == "Fallback text"


def test_extract_text_empty_entry() -> None:
    """Completely empty entry returns an empty string."""
    assert _extract_text({}) == ""


def test_extract_text_non_string_text_field() -> None:
    """Non-string 'text' field returns an empty string."""
    entry = {"text": 42}
    assert _extract_text(entry) == ""


def test_extract_text_empty_string_content() -> None:
    """Empty string content is returned as-is."""
    entry = {"content": ""}
    assert _extract_text(entry) == ""


# ---------------------------------------------------------------------------
# read_transcript_tail
# ---------------------------------------------------------------------------


def test_read_transcript_tail_nonexistent_file() -> None:
    """Missing transcript file returns an empty string without raising."""
    result = read_transcript_tail("/tmp/does_not_exist_xyz_12345.jsonl")
    assert result == ""


def test_read_transcript_tail_empty_file(tmp_path: Path) -> None:
    """Empty transcript file returns an empty string."""
    f = tmp_path / "session.jsonl"
    f.write_text("", encoding="utf-8")
    assert read_transcript_tail(str(f)) == ""


def test_read_transcript_tail_valid_jsonl(tmp_path: Path) -> None:
    """Valid JSONL with long-enough text entries are included in output."""
    entries = [
        {"role": "user", "content": "This is a meaningful message that has enough characters"},
        {"role": "assistant", "content": "And this is the assistant response with enough text too"},
    ]
    f = tmp_path / "session.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    result = read_transcript_tail(str(f))
    assert "meaningful message" in result
    assert "assistant response" in result


def test_read_transcript_tail_short_entries_skipped(tmp_path: Path) -> None:
    """Entries with content <= 20 characters are skipped."""
    entries = [
        {"role": "user", "content": "ok"},  # too short
        {"role": "assistant", "content": "This entry is long enough to be included in the output"},
    ]
    f = tmp_path / "session.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    result = read_transcript_tail(str(f))
    assert "ok" not in result
    assert "long enough" in result


def test_read_transcript_tail_malformed_lines_skipped(tmp_path: Path) -> None:
    """Malformed JSON lines are skipped without raising."""
    f = tmp_path / "session.jsonl"
    f.write_text(
        'not valid json\n{"role": "user", "content": "This valid entry is long enough to include"}\n',
        encoding="utf-8",
    )

    result = read_transcript_tail(str(f))
    assert "valid entry" in result


def test_read_transcript_tail_max_lines_respected(tmp_path: Path) -> None:
    """Only the last max_lines entries are read."""
    entries = [
        {"role": "user", "content": f"Entry number {i} with enough characters to not be skipped"}
        for i in range(20)
    ]
    f = tmp_path / "session.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    # With max_lines=3, only last 3 entries should appear
    result = read_transcript_tail(str(f), max_lines=3)
    assert "Entry number 19" in result
    assert "Entry number 18" in result
    assert "Entry number 17" in result
    assert "Entry number 0" not in result


def test_read_transcript_tail_truncation(tmp_path: Path) -> None:
    """Output is truncated to MAX_FLUSH_CHARS when content exceeds it."""
    # Generate content that exceeds MAX_FLUSH_CHARS
    big_text = "x" * (MAX_FLUSH_CHARS + 500)
    entry = {"role": "user", "content": big_text}
    f = tmp_path / "session.jsonl"
    f.write_text(json.dumps(entry), encoding="utf-8")

    result = read_transcript_tail(str(f))
    assert len(result) <= MAX_FLUSH_CHARS


def test_read_transcript_tail_path_as_string_or_path_object(tmp_path: Path) -> None:
    """Function accepts string path and returns correct output."""
    entry = {"role": "user", "content": "This message is definitely long enough to be included"}
    f = tmp_path / "session.jsonl"
    f.write_text(json.dumps(entry), encoding="utf-8")

    result = read_transcript_tail(str(f))
    assert "long enough" in result

"""Bounded readers for Claude JSONL and Codex rollout messages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def tail_lines(path: str, max_lines: int, max_bytes: int = 512 * 1024) -> list[str]:
    """Read at most max_bytes from disk, never the whole transcript."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - max_bytes)
        stream.seek(start)
        if start:
            stream.readline(max_bytes)
        return stream.read(max_bytes).decode("utf-8", errors="replace").splitlines()[-max_lines:]


def message(entry: Any) -> tuple[str, str] | None:
    """Accept conversation text only; never reasoning, tools, or event summaries."""
    if not isinstance(entry, dict):
        return None
    if entry.get("type") == "response_item":
        entry = entry.get("payload")
        if not isinstance(entry, dict) or entry.get("type") != "message":
            return None
    elif entry.get("type") in {"event_msg", "compacted", "session_meta", "turn_context"}:
        return None
    nested = entry.get("message")
    if isinstance(nested, dict):
        entry = nested
    role = entry.get("role") or entry.get("type")
    role = "user" if role == "human" else role
    if role not in {"user", "assistant"}:
        return None
    content = entry.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "input_text", "output_text"}
            and isinstance(item.get("text"), str)
        )
    return (role, content) if isinstance(content, str) and content.strip() else None


def read_incremental(
    path: Path, cursor: dict[str, Any], max_bytes: int = 512 * 1024
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Return complete new messages and an uncommitted cursor; detect rotation/truncation."""
    with path.open("rb") as stream:
        stat = path.stat()
        identity = [stat.st_dev, stat.st_ino]
        offset = int(cursor.get("offset", 0)) if cursor.get("identity") == identity else 0
        if offset > stat.st_size:
            offset = 0
        clipped = stat.st_size - offset > max_bytes
        if clipped:
            offset = stat.st_size - max_bytes
        stream.seek(offset)
        if clipped:
            stream.readline(max_bytes)
        start = stream.tell()
        raw = stream.read(max_bytes)
        end = raw.rfind(b"\n") + 1
        rows: list[tuple[str, str]] = []
        for line in raw[:end].splitlines():
            try:
                row = message(json.loads(line))
                if row:
                    rows.append(row)
            except (ValueError, UnicodeError):
                continue
        return rows, {"identity": identity, "offset": start + end, "clipped": clipped}

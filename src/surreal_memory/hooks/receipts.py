"""Versioned JSON receipts shared by durable hooks."""

from __future__ import annotations

from typing import Any

RECEIPT_SCHEMA_VERSION = 1


def receipt(
    status: str,
    saved: int = 0,
    session_id: str = "",
    memory_ids: list[str] | None = None,
    error: str | None = None,
    scope_id: str | None = None,
    revision: int | None = None,
    content_hash: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build the stable machine-readable hook result contract."""
    if status not in {"saved", "skipped", "failed", "pending"}:
        raise ValueError(f"unsupported receipt status: {status}")
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
        "saved": saved,
        "session_id": session_id,
        "memory_ids": memory_ids or [],
        "error": error,
    }
    if scope_id is not None and revision is not None and content_hash is not None:
        result.update(scope_id=scope_id, revision=revision, content_hash=content_hash)
    if event_id is not None:
        result["event_id"] = event_id
    return result


def emit_receipt(value: dict[str, Any]) -> None:
    """Write a receipt as a single stdout JSON line."""
    import json
    import sys

    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), file=sys.stdout)  # noqa: T201


def capture_receipt(result: dict[str, Any], session_id: str = "") -> dict[str, Any]:
    """Translate a legacy hook result into the version-1 receipt contract."""
    saved = int(result.get("saved", 0) or 0)
    failed = int(result.get("failed_writes", 0) or 0)
    error = "capture_failed" if result.get("error") or failed else None
    status = "failed" if error else ("saved" if saved else "skipped")
    import os

    event_id = os.environ.get("SMEM_EVENT_ID", "").strip() or None
    return receipt(status, saved, session_id, result.get("memory_ids"), error, event_id=event_id)


def runtime_session_id() -> str:
    """Return the active runtime session identifier, preferring Codex."""
    import os

    return (
        os.environ.get("CODEX_SESSION_ID", "").strip()
        or os.environ.get("CLAUDE_SESSION_ID", "").strip()
    )

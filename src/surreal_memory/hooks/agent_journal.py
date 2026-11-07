"""Private, bounded local retry journal. Pending is never a durability receipt."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from surreal_memory.hooks.capture_state import local_scope_lock

logger = logging.getLogger(__name__)


def data_dir() -> Path:
    """Return the configured writable smem state directory."""
    return Path(os.environ.get("SURREAL_MEMORY_DIR") or Path.home() / ".surrealmemory")


def read_state(name: str) -> dict[str, Any]:
    """Read a private local JSON object; reject malformed state explicitly."""
    path = data_dir() / name
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid coding-memory journal")
    return value


def write_state(name: str, value: dict[str, Any]) -> None:
    """Atomically replace a mode-0600 local state file."""
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".coding-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / name)
    finally:
        Path(temporary).unlink(missing_ok=True)


def event_key(kind: str, payload: dict[str, Any]) -> str:
    """Stable key permits hook/MCP retries without duplicate queue entries."""
    return hashlib.sha256(
        json.dumps([kind, payload], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def pending() -> dict[str, Any]:
    """Read queued writes, reporting retention expiry rather than hiding it."""
    with local_scope_lock("coding-outbox"):
        items = read_state("coding-outbox.json")
        cutoff = time.time() - float(os.environ.get("SMEM_RETRY_MAX_AGE_DAYS", "7")) * 86400
        expired = [key for key, value in items.items() if value["created_at"] < cutoff]
        if expired:
            for key in expired:
                del items[key]
            write_state("coding-outbox.json", items)
            logger.warning(
                "Coding memory: %d pending writes expired without persistence", len(expired)
            )
        return items


def enqueue(kind: str, payload: dict[str, Any]) -> str:
    """Queue only previously validated, redacted payloads; never evict on overflow."""
    from surreal_memory.safety.sensitive import auto_redact_content

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    redacted, matches, _ = auto_redact_content(encoded, min_severity=2)
    if matches or redacted != encoded:
        raise ValueError("sensitive retry payload rejected")
    key = event_key(kind, payload)
    with local_scope_lock("coding-outbox"):
        items = read_state("coding-outbox.json")
        items.setdefault(
            key, {"schema_version": 1, "kind": kind, "payload": payload, "created_at": time.time()}
        )
        size = len(json.dumps(items, ensure_ascii=False).encode())
        if size > int(os.environ.get("SMEM_RETRY_MAX_BYTES", str(5 * 1024 * 1024))):
            logger.error("Coding memory retry queue full; new write was not queued")
            raise OSError("coding_retry_queue_full")
        write_state("coding-outbox.json", items)
    return key


def acknowledge(key: str) -> None:
    """Remove a write only after the caller confirms durable persistence."""
    with local_scope_lock("coding-outbox"):
        items = read_state("coding-outbox.json")
        if key in items:
            del items[key]
            write_state("coding-outbox.json", items)


@asynccontextmanager
async def scope_lock(scope_id: str) -> AsyncIterator[None]:
    """Cross-process lock without blocking the asyncio event loop or cancellation."""
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ("coding-" + scope_id + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    file = os.fdopen(fd, "a+b")
    acquired = False
    try:
        while not acquired:
            try:
                if os.name == "nt":
                    msvcrt: Any = __import__("msvcrt")

                    file.seek(0)
                    if not file.read(1):
                        file.write(b"0")
                        file.flush()
                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                await asyncio.sleep(0.02)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                msvcrt = __import__("msvcrt")

                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        file.close()

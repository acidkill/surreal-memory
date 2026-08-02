"""Record type for `smem watch`'s file-ingestion tracking, independent of any backend.

Lived only inside the SQLite-backed ``WatchStateTracker`` because that backend
defined it first — every storage backend and the file watcher should speak
this same shape (see issue #138).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatchedFile:
    """State of a watched file."""

    file_path: str
    mtime: float
    simhash: int
    neuron_count: int
    last_ingested: str
    status: str = "active"

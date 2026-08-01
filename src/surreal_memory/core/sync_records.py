"""Records exchanged by multi-device sync, independent of any backend.

These describe the sync contract itself — what a registered device is, and what
a single logged change looks like — so every storage backend and the sync
engine speak the same shapes. They lived in the SQLite mixins only because that
backend defined them first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DeviceRecord:
    """A registered device for a brain."""

    device_id: str
    brain_id: str
    device_name: str
    last_sync_at: datetime | None
    last_sync_sequence: int
    registered_at: datetime


@dataclass(frozen=True)
class ChangeEntry:
    """A single change log entry."""

    id: int  # Auto-incremented sequence number
    brain_id: str
    entity_type: str  # "neuron", "synapse", "fiber"
    entity_id: str
    operation: str  # "insert", "update", "delete"
    device_id: str
    changed_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    synced: bool = False

"""Sync protocol data structures for incremental multi-device sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SyncStatus(StrEnum):
    """Sync operation status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    ERROR = "error"


class ConflictStrategy(StrEnum):
    """How to resolve sync conflicts."""

    PREFER_RECENT = "prefer_recent"
    PREFER_LOCAL = "prefer_local"
    PREFER_REMOTE = "prefer_remote"
    PREFER_STRONGER = "prefer_stronger"


@dataclass(frozen=True)
class SyncChange:
    """A single change to be synced."""

    sequence: int
    entity_type: str  # "neuron", "synapse", "fiber"
    entity_id: str
    operation: str  # "insert", "update", "delete"
    device_id: str
    changed_at: str  # ISO format
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncRequest:
    """Request from a device to sync changes."""

    device_id: str
    brain_id: str
    last_sequence: int  # Last known sequence from hub
    changes: list[SyncChange] = field(default_factory=list)
    strategy: ConflictStrategy = ConflictStrategy.PREFER_RECENT


@dataclass(frozen=True)
class SyncConflict:
    """A conflict detected during sync."""

    entity_type: str
    entity_id: str
    local_device: str
    remote_device: str
    resolution: str  # which side won
    details: str = ""


@dataclass(frozen=True)
class SyncResponse:
    """Response from hub after sync."""

    hub_sequence: int  # Current hub sequence after sync
    changes: list[SyncChange] = field(default_factory=list)  # Changes for the requesting device
    conflicts: list[SyncConflict] = field(default_factory=list)
    status: SyncStatus = SyncStatus.SUCCESS
    message: str = ""


# ── Merkle delta sync protocol ────────────────────────────────────


@dataclass(frozen=True)
class MerkleSyncRequest:
    """Single-round Merkle sync request.

    Device sends all bucket hashes (~49KB) so the hub can compare in one pass.
    """

    device_id: str
    brain_id: str
    root_hash: str
    buckets: dict[str, dict[str, str]]  # {entity_type: {prefix: hash}}
    strategy: ConflictStrategy = ConflictStrategy.PREFER_RECENT


@dataclass(frozen=True)
class MerkleBucketDiff:
    """Diff payload for a single bucket that differs between device and hub."""

    entity_type: str
    prefix: str
    entity_ids: list[str] = field(default_factory=list)  # Full ID list for delete detection
    entities: list[dict[str, Any]] = field(default_factory=list)  # Changed entity payloads


@dataclass(frozen=True)
class MerkleSyncResponse:
    """Hub response to a Merkle sync request.

    If root hashes match: status="in_sync", no diffs.
    Otherwise: status="diff" with changed bucket payloads.
    """

    status: str  # "in_sync" or "diff"
    hub_root_hash: str = ""
    changed_prefixes: list[str] = field(default_factory=list)
    diffs: list[MerkleBucketDiff] = field(default_factory=list)
    hub_sequence: int = 0
    message: str = ""

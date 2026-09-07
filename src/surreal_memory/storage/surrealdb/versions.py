"""SurrealDB versions storage mixin (compressed brain snapshots).

Single-version lookups rebuild the record id in SurrealQL with
``type::record('brain_versions', $sid)``. Comparing ``id`` with a
``"brain_versions:<sid>"`` *string* is unconditionally false — ``id`` holds a
record id — so ``get_version`` and ``delete_version`` could never find a row,
and every restore/diff path in ``engine/brain_versioning.py`` saw an empty
result. Same trap as the ``typed_memory`` / ``alerts`` lookups in this package.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import zlib
from datetime import datetime
from typing import Any

from surreal_memory.engine.brain_versioning import BrainVersion
from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

_MAX_LIST_LIMIT = 100


def _parse_datetime(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo is not None else val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except (ValueError, AttributeError):
            return None
    return None


def _compress_snapshot(snapshot_json: str) -> str:
    """Compress snapshot JSON with zlib and base64-encode for safe text storage."""
    raw = snapshot_json.encode("utf-8")
    return base64.b64encode(zlib.compress(raw, level=6)).decode("ascii")


def _decompress_snapshot(raw_data: str) -> str:
    """Decompress snapshot data; fall back to raw text for legacy uncompressed data."""
    if not raw_data:
        return ""
    try:
        compressed_bytes = base64.b64decode(raw_data)
        return zlib.decompress(compressed_bytes).decode("utf-8")
    except (zlib.error, binascii.Error):
        return raw_data


def _row_to_version(row: dict[str, Any]) -> BrainVersion:
    """Convert a SurrealDB record to a BrainVersion."""
    metadata_raw = row.get("metadata")
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    else:
        metadata = dict(metadata_raw or {})

    raw_id = str(row.get("id", ""))
    vid = _record_id_part(raw_id)

    return BrainVersion(
        id=vid,
        brain_id=str(row["brain_id"]),
        version_name=str(row.get("version_name", "")),
        version_number=int(row.get("version_number", 1)),
        description=str(row.get("description", "")),
        neuron_count=int(row.get("neuron_count", 0)),
        synapse_count=int(row.get("synapse_count", 0)),
        fiber_count=int(row.get("fiber_count", 0)),
        snapshot_hash=str(row.get("snapshot_hash", "")),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        metadata=metadata,
    )


class SurrealDBVersionsMixin:
    """Mixin providing brain version CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def save_version(
        self,
        brain_id: str,
        version: BrainVersion,
        snapshot_json: str,
    ) -> None:
        """Persist a brain version with its compressed snapshot data."""
        conn = self._ensure_conn()
        sid = _to_surreal_id(version.id)
        compressed = _compress_snapshot(snapshot_json)

        record_data: dict[str, Any] = {
            "id": sid,
            "brain_id": brain_id,
            "version_name": version.version_name,
            "version_number": int(version.version_number),
            "description": version.description or "",
            "neuron_count": int(version.neuron_count),
            "synapse_count": int(version.synapse_count),
            "fiber_count": int(version.fiber_count),
            "snapshot_hash": version.snapshot_hash,
            "snapshot_data": compressed,
            "created_at": version.created_at,
            "metadata": dict(version.metadata or {}),
        }

        try:
            await conn.insert("brain_versions", record_data)
        except Exception:
            try:
                # Rebuild the id in SurrealQL rather than handing the SDK a
                # "brain_versions:<sid>" string: a letter-free sid comes back
                # as a *numeric* record id there, so the delete would clear a
                # different, absent record without raising, and the retry below
                # would hit the same collision. See _ids._record_id_part.
                await self._query("DELETE type::record('brain_versions', $sid)", sid=sid)
            except Exception:
                logger.debug("version insert retry: delete of the clashing row failed")
            await conn.insert("brain_versions", record_data)

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> tuple[BrainVersion, str] | None:
        """Get a version and its decompressed snapshot JSON by ID."""
        sid = _to_surreal_id(version_id)
        rows = await self._query(
            "SELECT * FROM brain_versions WHERE brain_id = $brain_id"
            " AND id = type::record('brain_versions', $sid) LIMIT 1",
            brain_id=brain_id,
            sid=sid,
        )
        if not rows:
            return None

        row = rows[0]
        version = _row_to_version(row)
        snapshot_json = _decompress_snapshot(str(row.get("snapshot_data", "")))
        return version, snapshot_json

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        """List versions for a brain, most recent first (by version_number DESC)."""
        capped = min(limit, _MAX_LIST_LIMIT)
        rows = await self._query(
            "SELECT id, brain_id, version_name, version_number, description,"
            " neuron_count, synapse_count, fiber_count, snapshot_hash,"
            " created_at, metadata"
            " FROM brain_versions"
            " WHERE brain_id = $brain_id"
            " ORDER BY version_number DESC LIMIT $limit",
            brain_id=brain_id,
            limit=capped,
        )
        return [_row_to_version(r) for r in rows]

    async def get_next_version_number(self, brain_id: str) -> int:
        """Get the next auto-incrementing version number for a brain."""
        rows = await self._query(
            "SELECT version_number FROM brain_versions"
            " WHERE brain_id = $brain_id"
            " ORDER BY version_number DESC LIMIT 1",
            brain_id=brain_id,
        )
        if not rows:
            return 1
        return int(rows[0].get("version_number", 0)) + 1

    async def delete_version(self, brain_id: str, version_id: str) -> bool:
        """Delete a specific version. Returns True if a row was deleted."""
        sid = _to_surreal_id(version_id)
        existing = await self._query(
            "SELECT id FROM brain_versions WHERE brain_id = $brain_id"
            " AND id = type::record('brain_versions', $sid) LIMIT 1",
            brain_id=brain_id,
            sid=sid,
        )
        if not existing:
            return False

        conn = self._ensure_conn()
        rid = existing[0].get("id")
        if not rid:
            return False
        # The id object the query returned, rather than an id rebuilt from
        # ``sid``: a letter-free sid rebuilt as ``f"brain_versions:{sid}"``
        # parses as a *numeric* record id and addresses a different, absent
        # row -- the delete then reports success having removed nothing.
        await conn.delete(rid)
        return True

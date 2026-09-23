"""SurrealDB compression backups and neuron snapshots mixin."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


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


class SurrealDBCompressionMixin:
    """Mixin providing compression backups and neuron snapshots for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ---------------- compression_backups ----------------

    async def save_compression_backup(
        self,
        fiber_id: str,
        original_content: str,
        compression_tier: int,
        original_token_count: int,
        compressed_token_count: int,
    ) -> None:
        """Upsert a compression backup for a fiber (most recent snapshot wins)."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        sid = f"{_to_surreal_id(brain_id)}_{_to_surreal_id(fiber_id)}"
        now = utcnow()

        record_data: dict[str, Any] = {
            "fiber_id": fiber_id,
            "brain_id": brain_id,
            "original_content": original_content,
            "compression_tier": int(compression_tier),
            "compressed_at": now,
            "original_token_count": int(original_token_count),
            "compressed_token_count": int(compressed_token_count),
        }

        existing = await self._query(
            "SELECT id FROM compression_backups"
            " WHERE brain_id = $brain_id AND fiber_id IN $fiber_ids LIMIT 1",
            brain_id=brain_id,
            fiber_ids=[fiber_id, _to_surreal_id(fiber_id)],
        )
        if existing:
            # Merge by existing record id (not recomputed sid) — survives a brain
            # rename where the id keeps the old brain prefix (else silent no-op).
            await conn.merge(existing[0]["id"], record_data)
        else:
            insert_data = dict(record_data)
            insert_data["id"] = sid
            await conn.insert("compression_backups", insert_data)

    async def get_compression_backup(self, fiber_id: str) -> dict[str, Any] | None:
        """Retrieve the compression backup for fiber_id, if any."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT fiber_id, brain_id, original_content, compression_tier,"
            " compressed_at, original_token_count, compressed_token_count"
            " FROM compression_backups"
            " WHERE brain_id = $brain_id AND fiber_id IN $fiber_ids LIMIT 1",
            brain_id=brain_id,
            fiber_ids=[fiber_id, _to_surreal_id(fiber_id)],
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "fiber_id": str(r.get("fiber_id", "")),
            "brain_id": str(r.get("brain_id", "")),
            "original_content": str(r.get("original_content", "")),
            "compression_tier": int(r.get("compression_tier", 0)),
            "compressed_at": _parse_datetime(r.get("compressed_at")),
            "original_token_count": int(r.get("original_token_count", 0)),
            "compressed_token_count": int(r.get("compressed_token_count", 0)),
        }

    async def delete_compression_backup(self, fiber_id: str) -> bool:
        """Delete the compression backup for fiber_id. Returns True if deleted."""
        brain_id = self._get_brain_id()
        existing = await self._query(
            "SELECT id FROM compression_backups"
            " WHERE brain_id = $brain_id AND fiber_id IN $fiber_ids LIMIT 1",
            brain_id=brain_id,
            fiber_ids=[fiber_id, _to_surreal_id(fiber_id)],
        )
        if not existing:
            return False

        conn = self._ensure_conn()
        rid = str(existing[0].get("id", ""))
        if not rid:
            return False
        await conn.delete(rid)
        return True

    async def get_compression_stats(self) -> dict[str, Any]:
        """Aggregate compression stats: total_backups, by_tier, total_tokens_saved."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT compression_tier,"
            " count() AS backup_count,"
            " math::sum(original_token_count) AS total_original,"
            " math::sum(compressed_token_count) AS total_compressed"
            " FROM compression_backups"
            " WHERE brain_id = $brain_id"
            " GROUP BY compression_tier",
            brain_id=brain_id,
        )

        total_backups = 0
        total_tokens_saved = 0
        by_tier: dict[int, int] = {}

        for r in rows:
            tier = int(r.get("compression_tier", 0))
            count = int(r.get("backup_count", 0))
            original = int(r.get("total_original") or 0)
            compressed = int(r.get("total_compressed") or 0)

            by_tier[tier] = count
            total_backups += count
            total_tokens_saved += max(0, original - compressed)

        return {
            "total_backups": total_backups,
            "by_tier": by_tier,
            "total_tokens_saved": total_tokens_saved,
        }

    # ---------------- neuron_snapshots ----------------

    async def save_neuron_snapshot(
        self,
        neuron_id: str,
        brain_id: str,
        original_content: str,
        compressed_at: str,
        tier: int,
    ) -> None:
        """Upsert a pre-compression content snapshot for a neuron."""
        conn = self._ensure_conn()
        sid = f"{_to_surreal_id(brain_id)}_{_to_surreal_id(neuron_id)}"

        record_data: dict[str, Any] = {
            "neuron_id": neuron_id,
            "brain_id": brain_id,
            "original_content": original_content,
            # neuron_snapshots.compressed_at is declared TYPE string, so send a
            # string: a datetime fails coercion on a SCHEMAFULL table and the
            # caller's fail-soft except swallows it, leaving no snapshot to
            # recover from. get_neuron_snapshot parses it back on read.
            "compressed_at": (_parse_datetime(compressed_at) or utcnow()).isoformat(),
            "tier": int(tier),
        }

        existing = await self._query(
            "SELECT id FROM neuron_snapshots"
            " WHERE brain_id = $brain_id AND neuron_id IN $neuron_ids LIMIT 1",
            brain_id=brain_id,
            neuron_ids=[neuron_id, _to_surreal_id(neuron_id)],
        )
        if existing:
            # Merge by existing record id (not recomputed sid) — survives a brain
            # rename where the id keeps the old brain prefix (else silent no-op).
            await conn.merge(existing[0]["id"], record_data)
        else:
            insert_data = dict(record_data)
            insert_data["id"] = sid
            await conn.insert("neuron_snapshots", insert_data)

    async def get_neuron_snapshot(self, neuron_id: str) -> dict[str, Any] | None:
        """Retrieve the snapshot for a neuron, if any."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT neuron_id, brain_id, original_content, compressed_at, tier"
            " FROM neuron_snapshots"
            " WHERE brain_id = $brain_id AND neuron_id IN $neuron_ids LIMIT 1",
            brain_id=brain_id,
            neuron_ids=[neuron_id, _to_surreal_id(neuron_id)],
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "neuron_id": str(r.get("neuron_id", "")),
            "brain_id": str(r.get("brain_id", "")),
            "original_content": str(r.get("original_content", "")),
            "compressed_at": _parse_datetime(r.get("compressed_at")),
            "tier": int(r.get("tier", 0)),
        }

    async def delete_neuron_snapshot(self, neuron_id: str) -> bool:
        """Delete the snapshot for a neuron. Returns True if deleted."""
        brain_id = self._get_brain_id()
        existing = await self._query(
            "SELECT id FROM neuron_snapshots"
            " WHERE brain_id = $brain_id AND neuron_id IN $neuron_ids LIMIT 1",
            brain_id=brain_id,
            neuron_ids=[neuron_id, _to_surreal_id(neuron_id)],
        )
        if not existing:
            return False

        conn = self._ensure_conn()
        rid = str(existing[0].get("id", ""))
        if not rid:
            return False
        await conn.delete(rid)
        return True

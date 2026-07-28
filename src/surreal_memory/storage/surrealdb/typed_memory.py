"""SurrealDB typed memory operations mixin."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from surreal_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from surreal_memory.storage.sqlite_row_mappers import provenance_to_dict
from surreal_memory.storage.surrealdb._ids import _safe_brain_id, _to_surreal_id
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


_PRIORITY_LABEL_MAP = {
    "lowest": 0,
    "low": 2,
    "medium": 5,
    "normal": 5,
    "high": 7,
    "critical": 10,
}


def _row_to_typed_memory(row: dict[str, Any]) -> TypedMemory:
    """Convert a SurrealDB typed_memory record to TypedMemory."""
    # Provenance is embedded inside metadata under _provenance
    metadata_raw = dict(row.get("metadata") or {})
    prov_data = metadata_raw.pop("_provenance", None) or {}
    if not isinstance(prov_data, dict):
        prov_data = {}

    provenance = Provenance(
        source=str(prov_data.get("source", "unknown")),
        confidence=Confidence(prov_data.get("confidence", "medium")),
        verified=bool(prov_data.get("verified", False)),
        verified_at=_parse_datetime(prov_data.get("verified_at")),
        created_by=str(prov_data.get("created_by", "unknown")),
        last_confirmed=_parse_datetime(prov_data.get("last_confirmed")),
    )

    # priority stored as stringified int ("5") or legacy label ("medium")
    priority_raw = str(row.get("priority", "5"))
    try:
        priority = Priority.from_int(int(priority_raw))
    except (ValueError, TypeError):
        priority = Priority.from_int(_PRIORITY_LABEL_MAP.get(priority_raw.lower(), 5))

    tags_raw = row.get("tags") or []
    if isinstance(tags_raw, str):
        import json

        try:
            tags_raw = json.loads(tags_raw)
        except Exception:
            tags_raw = []

    trust_raw = row.get("trust_score")
    trust_score = float(trust_raw) if trust_raw is not None else None

    return TypedMemory(
        fiber_id=str(row["fiber_id"]),
        memory_type=MemoryType(str(row["memory_type"])),
        priority=priority,
        provenance=provenance,
        expires_at=_parse_datetime(row.get("expires_at")),
        project_id=row.get("project_id"),
        tags=frozenset(str(t) for t in tags_raw),
        metadata=metadata_raw,
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        trust_score=trust_score,
        source=row.get("source"),
        tier=str(row.get("tier") or "warm"),
        valid_from=_parse_datetime(row.get("valid_from")),
        valid_until=_parse_datetime(row.get("valid_until")),
        superseded_by=(str(row["superseded_by"]) if row.get("superseded_by") is not None else None),
    )


class SurrealDBTypedMemoryMixin:
    """Mixin providing typed memory CRUD for SurrealDBStorage."""

    # ------------------------------------------------------------------
    # Protocol stubs — satisfied by SurrealDBStorage at runtime
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(typed_memory.fiber_id)

        full_metadata = dict(typed_memory.metadata)
        full_metadata["_provenance"] = provenance_to_dict(typed_memory.provenance)

        record_data: dict[str, Any] = {
            "fiber_id": typed_memory.fiber_id,
            "brain_id": brain_id,
            "memory_type": typed_memory.memory_type.value,
            "priority": str(typed_memory.priority.value),
            "tags": list(typed_memory.tags),
            "trust_score": typed_memory.trust_score,
            "source": typed_memory.source,
            "project_id": typed_memory.project_id,
            "expires_at": typed_memory.expires_at,
            "tier": typed_memory.tier,
            "valid_from": typed_memory.valid_from,
            "valid_until": typed_memory.valid_until,
            "superseded_by": typed_memory.superseded_by,
            "metadata": full_metadata,
            "created_at": typed_memory.created_at,
            "updated_at": utcnow(),
        }

        try:
            await conn.query(
                f"UPSERT typed_memory:{sid} CONTENT $data",
                {"data": record_data},
            )
        except Exception:
            # Fallback: delete then insert
            try:
                await conn.delete(f"typed_memory:{sid}")
            except Exception:
                pass
            insert_data = dict(record_data)
            insert_data["id"] = sid
            await conn.insert("typed_memory", insert_data)

        return typed_memory.fiber_id

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        brain_id = self._get_brain_id()
        # Match on the sanitized record id so BOTH the original dash uuid and the
        # underscore record-id form resolve (a Fiber loaded from SurrealDB carries the
        # underscore form, while typed_memory.fiber_id is stored as the dash uuid).
        rows = await self._query(
            "SELECT * FROM typed_memory WHERE brain_id = $brain_id "
            "AND id = type::record('typed_memory', $sid) LIMIT 1",
            brain_id=brain_id,
            sid=_to_surreal_id(fiber_id),
        )
        if not rows:
            return None
        return _row_to_typed_memory(rows[0])

    async def find_typed_memories(
        self,
        memory_type: MemoryType | None = None,
        min_priority: Priority | None = None,
        include_expired: bool = False,
        project_id: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
        tier: str | None = None,
    ) -> list[TypedMemory]:
        limit = min(limit, 1000)
        brain_id = self._get_brain_id()

        parts = ["SELECT * FROM typed_memory WHERE brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if memory_type is not None:
            parts.append("AND memory_type = $memory_type")
            params["memory_type"] = memory_type.value

        if not include_expired:
            parts.append("AND (expires_at IS NONE OR expires_at > time::now())")

        if project_id is not None:
            parts.append("AND project_id = $project_id")
            params["project_id"] = project_id

        if tier is not None:
            parts.append("AND tier = $tier")
            params["tier"] = tier

        parts.append("ORDER BY created_at DESC LIMIT $limit")
        params["limit"] = limit

        rows = await self._query(" ".join(parts), **params)
        memories = [_row_to_typed_memory(r) for r in rows]

        if min_priority is not None:
            memories = [m for m in memories if m.priority >= min_priority]

        if tags is not None:
            memories = [m for m in memories if tags.issubset(m.tags)]

        return memories

    async def get_typed_memories_batch(self, fiber_ids: Sequence[str]) -> dict[str, TypedMemory]:
        ids = list(fiber_ids)
        if not ids:
            return {}
        brain_id = self._get_brain_id()
        # Normalise each query id to its sanitized record-id part so BOTH the dash
        # uuid and the underscore record-id form resolve (a Fiber loaded from
        # SurrealDB carries the underscore form; typed_memory.fiber_id is the dash
        # uuid). Inline the validated brain_id literal so the brain_id index is used.
        lit = f'"{_safe_brain_id(brain_id)}"'
        sid_to_fid: dict[str, str] = {}
        for fid in ids:
            sid_to_fid.setdefault(_to_surreal_id(fid), fid)
        sids = list(sid_to_fid.keys())
        rows = await self._query(
            f"SELECT * FROM typed_memory WHERE brain_id = {lit} AND meta::id(id) IN $sids",
            sids=sids,
        )
        result: dict[str, TypedMemory] = {}
        for r in rows:
            tm = _row_to_typed_memory(r)
            orig = sid_to_fid.get(_to_surreal_id(tm.fiber_id))
            if orig is not None:
                result[orig] = tm
        return result

    async def get_expiring_memories(
        self, within_days: int = 7, limit: int = 200
    ) -> list[TypedMemory]:
        brain_id = self._get_brain_id()
        capped = min(int(limit), 1000)
        lit = f'"{_safe_brain_id(brain_id)}"'
        deadline = utcnow() + timedelta(days=within_days)
        rows = await self._query(
            f"SELECT * FROM typed_memory WHERE brain_id = {lit} "
            "AND expires_at IS NOT NONE AND expires_at > time::now() "
            "AND expires_at <= $deadline "
            f"ORDER BY expires_at ASC LIMIT {capped}",
            deadline=deadline,
        )
        return [_row_to_typed_memory(r) for r in rows]

    async def count_typed_memories(
        self,
        tier: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        brain_id = self._get_brain_id()

        parts = [
            "SELECT count() AS cnt FROM typed_memory",
            "WHERE brain_id = $brain_id",
            "AND (expires_at IS NONE OR expires_at > time::now())",
        ]
        params: dict[str, Any] = {"brain_id": brain_id}

        if tier is not None:
            parts.append("AND tier = $tier")
            params["tier"] = tier

        if memory_type is not None:
            parts.append("AND memory_type = $memory_type")
            params["memory_type"] = memory_type.value

        parts.append("GROUP ALL")

        rows = await self._query(" ".join(parts), **params)
        return int(rows[0]["cnt"]) if rows else 0

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(typed_memory.fiber_id)

        full_metadata = dict(typed_memory.metadata)
        full_metadata["_provenance"] = provenance_to_dict(typed_memory.provenance)

        rows = await self._query(
            "SELECT id FROM typed_memory WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=typed_memory.fiber_id,
        )
        if not rows:
            raise ValueError(f"TypedMemory for fiber {typed_memory.fiber_id} does not exist")

        await conn.merge(
            f"typed_memory:{sid}",
            {
                "memory_type": typed_memory.memory_type.value,
                "priority": str(typed_memory.priority.value),
                "tags": list(typed_memory.tags),
                "trust_score": typed_memory.trust_score,
                "source": typed_memory.source,
                "project_id": typed_memory.project_id,
                "expires_at": typed_memory.expires_at,
                "tier": typed_memory.tier,
                "valid_from": typed_memory.valid_from,
                "valid_until": typed_memory.valid_until,
                "superseded_by": typed_memory.superseded_by,
                "metadata": full_metadata,
                "updated_at": utcnow(),
            },
        )

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        rows = await self._query(
            "SELECT id FROM typed_memory WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=fiber_id,
        )
        if not rows:
            return False

        sid = _to_surreal_id(fiber_id)
        await conn.merge(f"typed_memory:{sid}", {"source": source, "updated_at": utcnow()})
        return True

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()

        rows = await self._query(
            "SELECT id FROM typed_memory WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=fiber_id,
        )
        if not rows:
            return False

        conn = self._ensure_conn()
        sid = _to_surreal_id(fiber_id)
        await conn.delete(f"typed_memory:{sid}")
        return True

    async def get_project_memories(
        self,
        project_id: str,
        include_expired: bool = False,
    ) -> list[TypedMemory]:
        return await self.find_typed_memories(
            project_id=project_id,
            include_expired=include_expired,
        )

    async def get_expired_memories(self, limit: int = 100) -> list[TypedMemory]:
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        rows = await self._query(
            "SELECT * FROM typed_memory WHERE brain_id = $brain_id"
            " AND expires_at IS NOT NONE AND expires_at <= time::now()"
            " LIMIT $limit",
            brain_id=brain_id,
            limit=limit,
        )
        return [_row_to_typed_memory(r) for r in rows]

    async def get_expired_memory_count(self) -> int:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT count() AS cnt FROM typed_memory"
            " WHERE brain_id = $brain_id AND expires_at IS NOT NONE AND expires_at <= time::now()"
            " GROUP ALL",
            brain_id=brain_id,
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_expiring_memories_for_fibers(
        self,
        fiber_ids: list[str],
        within_days: int = 7,
    ) -> list[TypedMemory]:
        if not fiber_ids:
            return []
        brain_id = self._get_brain_id()
        deadline = utcnow() + timedelta(days=within_days)

        all_memories: list[TypedMemory] = []
        for fid in fiber_ids:
            rows = await self._query(
                "SELECT * FROM typed_memory"
                " WHERE brain_id = $brain_id AND fiber_id = $fiber_id"
                " AND expires_at IS NOT NONE"
                " AND expires_at > time::now() AND expires_at <= $deadline",
                brain_id=brain_id,
                fiber_id=fid,
                deadline=deadline,
            )
            all_memories.extend(_row_to_typed_memory(r) for r in rows)
        return all_memories

    async def get_expiring_memory_count(self, within_days: int = 7) -> int:
        brain_id = self._get_brain_id()
        deadline = utcnow() + timedelta(days=within_days)
        rows = await self._query(
            "SELECT count() AS cnt FROM typed_memory"
            " WHERE brain_id = $brain_id AND expires_at IS NOT NONE"
            " AND expires_at > time::now() AND expires_at <= $deadline"
            " GROUP ALL",
            brain_id=brain_id,
            deadline=deadline,
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_promotion_candidates(
        self,
        min_frequency: int = 5,
        source_type: str = "context",
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()

        tm_rows = await self._query(
            "SELECT fiber_id, memory_type, expires_at, metadata FROM typed_memory"
            " WHERE brain_id = $brain_id AND memory_type = $source_type LIMIT 200",
            brain_id=brain_id,
            source_type=source_type,
        )
        if not tm_rows:
            return []

        results: list[dict[str, Any]] = []
        for r in tm_rows:
            fid = str(r.get("fiber_id", ""))
            if not fid:
                continue
            fiber_rows = await self._query(
                # `id = $sid` with a "fiber:<raw>" *string* can never match a
                # record id, so this lookup returned zero rows for every fiber
                # and promotion was structurally impossible. Verified on the
                # live brain: the string form yields count 0, type::record
                # yields count 1 for the same fiber. Both halves are required:
                # the record id is built from a RAW id, so the bound value must
                # lose the "fiber:" prefix. Same shape as the typed_memory
                # lookup earlier in this module.
                "SELECT frequency, conductivity, pinned FROM fiber"
                " WHERE brain_id = $brain_id AND id = type::record('fiber', $sid) LIMIT 1",
                brain_id=brain_id,
                sid=_to_surreal_id(fid),
            )
            if not fiber_rows:
                continue
            fr = fiber_rows[0]
            freq = int(fr.get("frequency", 0))
            if freq < min_frequency or bool(fr.get("pinned", False)):
                continue
            meta_raw = dict(r.get("metadata") or {})
            meta_raw.pop("_provenance", None)
            results.append(
                {
                    "fiber_id": fid,
                    "memory_type": str(r.get("memory_type", "")),
                    "expires_at": _parse_datetime(r.get("expires_at")),
                    "metadata": meta_raw,
                    "frequency": freq,
                    "conductivity": float(fr.get("conductivity", 1.0)),
                }
            )
        return results

    async def promote_memory_type(
        self,
        fiber_id: str,
        new_type: MemoryType,
        new_expires_at: str | None = None,
    ) -> bool:
        brain_id = self._get_brain_id()

        rows = await self._query(
            "SELECT * FROM typed_memory WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=fiber_id,
        )
        if not rows:
            return False

        row = rows[0]
        old_type = str(row.get("memory_type", ""))
        if old_type == new_type.value:
            return False

        meta_raw = dict(row.get("metadata") or {})
        meta_raw["auto_promoted"] = True
        meta_raw["promoted_from"] = old_type
        meta_raw["promoted_at"] = utcnow().isoformat()

        conn = self._ensure_conn()
        sid = _to_surreal_id(fiber_id)
        await conn.merge(
            f"typed_memory:{sid}",
            {
                "memory_type": new_type.value,
                "expires_at": _parse_datetime(new_expires_at) if new_expires_at else None,
                "metadata": meta_raw,
                "updated_at": utcnow(),
            },
        )
        logger.info("Auto-promoted fiber %s from %s to %s", fiber_id, old_type, new_type.value)
        return True

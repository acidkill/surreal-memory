"""SurrealDB source registry operations mixin."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from surreal_memory.core.source import Source, SourceStatus, SourceType
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


def _row_to_source(row: dict[str, Any]) -> Source:
    """Convert a SurrealDB source record to a Source dataclass."""
    raw_source_type = str(row.get("source_type", "document"))
    try:
        source_type = SourceType(raw_source_type)
    except ValueError:
        logger.warning(
            "Unknown source_type %r for source %s, using DOCUMENT", raw_source_type, row.get("id")
        )
        source_type = SourceType.DOCUMENT

    raw_status = str(row.get("status", "active"))
    try:
        status = SourceStatus(raw_status)
    except ValueError:
        logger.warning("Unknown status %r for source %s, using ACTIVE", raw_status, row.get("id"))
        status = SourceStatus.ACTIVE

    metadata = dict(row.get("metadata") or {})

    return Source(
        id=str(row["id"]),
        brain_id=str(row["brain_id"]),
        name=str(row["name"]),
        source_type=source_type,
        version=str(row.get("version") or ""),
        effective_date=_parse_datetime(row.get("effective_date")),
        expires_at=_parse_datetime(row.get("expires_at")),
        status=status,
        file_hash=str(row.get("file_hash") or ""),
        metadata=metadata,
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        updated_at=_parse_datetime(row.get("updated_at")) or utcnow(),
        trust=(float(row["trust"]) if row.get("trust") is not None else None),
    )


class SurrealDBSourcesMixin:
    """Mixin providing source registry CRUD for SurrealDBStorage."""

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

    async def add_source(self, source: Source) -> str:
        conn = self._ensure_conn()
        sid = _to_surreal_id(source.id)

        record_data: dict[str, Any] = {
            "id": sid,
            "brain_id": source.brain_id,
            "name": source.name,
            "source_type": source.source_type.value,
            "version": source.version,
            "effective_date": source.effective_date,
            "expires_at": source.expires_at,
            "status": source.status.value,
            "file_hash": source.file_hash,
            "metadata": dict(source.metadata),
            "created_at": source.created_at,
            "updated_at": source.updated_at,
            "trust": source.trust,
        }

        content_data = {k: v for k, v in record_data.items() if k != "id"}
        try:
            await conn.query(
                f"UPSERT source:{sid} CONTENT $data",
                {"data": content_data},
            )
        except Exception:
            try:
                await conn.delete(f"source:{sid}")
            except Exception:
                pass
            await conn.insert("source", record_data)

        return source.id

    async def get_source(self, source_id: str) -> Source | None:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM source WHERE brain_id = $brain_id AND id = $sid LIMIT 1",
            brain_id=brain_id,
            sid=_to_surreal_id(source_id),
        )
        if not rows:
            return None
        return _row_to_source(rows[0])

    async def list_sources(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Source]:
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        parts = ["SELECT * FROM source WHERE brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if source_type is not None:
            parts.append("AND source_type = $source_type")
            params["source_type"] = source_type

        if status is not None:
            parts.append("AND status = $status")
            params["status"] = status

        parts.append("ORDER BY created_at DESC LIMIT $limit")
        params["limit"] = limit

        rows = await self._query(" ".join(parts), **params)
        return [_row_to_source(r) for r in rows]

    async def update_source(
        self,
        source_id: str,
        status: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        brain_id = self._get_brain_id()

        rows = await self._query(
            "SELECT id FROM source WHERE brain_id = $brain_id AND id = $sid LIMIT 1",
            brain_id=brain_id,
            sid=_to_surreal_id(source_id),
        )
        if not rows:
            return False

        if status is not None:
            try:
                SourceStatus(status)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {status!r}. Must be one of {[s.value for s in SourceStatus]}"
                )

        update: dict[str, Any] = {"updated_at": utcnow()}
        if status is not None:
            update["status"] = status
        if version is not None:
            update["version"] = version
        if metadata is not None:
            update["metadata"] = metadata

        conn = self._ensure_conn()
        sid = _to_surreal_id(source_id)
        await conn.merge(f"source:{sid}", update)
        return True

    async def delete_source(self, source_id: str) -> bool:
        brain_id = self._get_brain_id()

        rows = await self._query(
            "SELECT id FROM source WHERE brain_id = $brain_id AND id = $sid LIMIT 1",
            brain_id=brain_id,
            sid=_to_surreal_id(source_id),
        )
        if not rows:
            return False

        conn = self._ensure_conn()
        sid = _to_surreal_id(source_id)
        await conn.delete(f"source:{sid}")
        return True

    async def find_source_by_name(self, name: str) -> Source | None:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM source WHERE brain_id = $brain_id AND name = $name LIMIT 1",
            brain_id=brain_id,
            name=name,
        )
        if not rows:
            return None
        return _row_to_source(rows[0])

    async def count_neurons_for_source(self, source_id: str) -> int:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT count() AS cnt FROM synapse"
            " WHERE brain_id = $brain_id AND in = type::record('neuron', $source_id)"
            " AND type = 'source_of'"
            " GROUP ALL",
            brain_id=brain_id,
            source_id=_to_surreal_id(source_id),
        )
        return int(rows[0]["cnt"]) if rows else 0

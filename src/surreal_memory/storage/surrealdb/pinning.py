"""SurrealDB mixin for pinned (knowledge base) fibers.

A pinned fiber is permanent: decay, pruning and compression all skip it. The
``pinned`` flag itself has always been part of the SurrealDB fiber schema
(``DEFINE FIELD pinned ON fiber TYPE bool DEFAULT false``) and has always
round-tripped through ``add_fiber``/``update_fiber``, so compression, the tier
engine and the typed-memory TTL sweep already honoured it. What was missing were
the three operations below, which existed only on ``SQLiteStorage``:

* ``pin_fibers`` — without it ``smem_pin`` answered "Storage does not support
  pinning" for pin, unpin *and* list.
* ``get_pinned_neuron_ids`` — decay (``engine.lifecycle``) and prune
  (``engine.consolidation``) resolve a fiber's pinned state through this call.
  Both probed for it with ``hasattr`` and fell back to an empty set, so on
  SurrealDB they decayed pinned neurons to zero and then pruned them — deleting
  precisely the content ``smem_train`` documents as a permanent KB.
* ``list_pinned_fibers`` — backs ``smem_pin(action="list")``.

Two SurrealDB-specific traps are worked around here, both already documented
elsewhere in this package:

* Record-id sets go in the ``FROM`` clause as an interpolated record list, never
  through ``IN``. ``id`` holds a ``RecordID``, so ``id IN ['fiber:x']`` compares
  against plain strings and matches nothing — and ``get_neurons_batch`` notes
  that ``IN`` also loses index selection. Interpolated ids are filtered to
  ``[A-Za-z0-9_]`` first, exactly as that method does.
* ``fiber``'s record id carries the underscore form of the uuid while
  ``typed_memory.fiber_id`` stores the original dash form (see
  ``typed_memory.get_typed_memories_batch``). ``type`` and ``priority`` live on
  ``typed_memory``, so they are fetched by ``typed_memory:{underscore-id}``
  record id rather than by joining on the mismatched field.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from surreal_memory.storage.surrealdb._ids import _to_surreal_id

logger = logging.getLogger(__name__)

# Mirrors the SQLite cap so ``smem_pin(action="list")`` cannot be used to pull an
# unbounded result set through the MCP layer.
_MAX_LIST_LIMIT = 200

# Keeps an interpolated FROM record list to a sane query size, as in the store's
# batch neuron fetches.
_ID_CHUNK = 1000


def _safe_record_names(ids: list[str]) -> list[str]:
    """Map ids to SurrealDB record names, dropping anything unsafe to inline.

    These are interpolated into the FROM clause, so the charset filter is the
    injection guard — same contract as ``store.get_neurons_batch``.
    """
    return [
        sid
        for raw in ids
        for sid in (_to_surreal_id(raw),)
        if sid and all(c.isalnum() or c == "_" for c in sid)
    ]


class SurrealDBPinningMixin:
    """Mixin providing pinned-fiber operations for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def pin_fibers(self, fiber_ids: list[str], pinned: bool = True) -> int:
        """Pin or unpin fibers by ID. Returns the number of fibers updated.

        The count is how many of the requested ids exist in this brain, whether
        or not the flag actually changed — matching the SQLite rowcount, so
        re-pinning an already-pinned fiber still reports it as updated.

        Like SQLite's, this does not write a change-log entry: the pinned flag
        propagates to peers on the next ``update_fiber``, not on pin alone.
        """
        if not fiber_ids:
            return 0

        safe = _safe_record_names(fiber_ids)
        if not safe:
            return 0

        brain_id = self._get_brain_id()
        updated = 0
        for i in range(0, len(safe), _ID_CHUNK):
            things = ", ".join(f"fiber:{s}" for s in safe[i : i + _ID_CHUNK])
            # UPDATE (not UPSERT) never creates a missing record, and the
            # brain_id filter stops a caller pinning another brain's fiber by id.
            rows = await self._query(
                f"UPDATE {things} SET pinned = $pinned WHERE brain_id = $brain_id",
                brain_id=brain_id,
                pinned=pinned,
            )
            updated += len(rows)
        return updated

    async def get_pinned_neuron_ids(self) -> set[str]:
        """Get every neuron ID belonging to a pinned fiber in the current brain."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT VALUE neuron_ids FROM fiber WHERE brain_id = $brain_id AND pinned = true",
            brain_id=brain_id,
        )

        result: set[str] = set()
        for row in rows:
            # ``SELECT VALUE neuron_ids`` yields the arrays themselves; a fiber
            # written before the field existed can still surface as NONE.
            if row:
                result.update(str(nid) for nid in row)
        return result

    async def list_pinned_fibers(self, limit: int = 50) -> list[dict[str, Any]]:
        """List pinned fibers for the current brain, newest first."""
        brain_id = self._get_brain_id()
        safe_limit = min(max(int(limit), 0), _MAX_LIST_LIMIT)
        if safe_limit == 0:
            return []

        rows = await self._query(
            "SELECT id, summary, auto_tags, agent_tags, created_at FROM fiber "
            "WHERE brain_id = $brain_id AND pinned = true "
            f"ORDER BY created_at DESC LIMIT {safe_limit}",
            brain_id=brain_id,
        )
        if not rows:
            return []

        fiber_ids = [_strip_table_prefix(r.get("id")) for r in rows]
        by_fiber = await self._typed_memory_meta(brain_id, fiber_ids)

        results: list[dict[str, Any]] = []
        for row, fiber_id in zip(rows, fiber_ids, strict=True):
            meta = by_fiber.get(fiber_id, {})
            tags = meta.get("tags")
            if not tags:
                # No typed-memory row — fall back to the fiber's own tag union,
                # which is what the SQLite ``fibers.tags`` column holds.
                tags = list(row.get("auto_tags") or []) + list(row.get("agent_tags") or [])
            results.append(
                {
                    "fiber_id": fiber_id,
                    "summary": row.get("summary") or "",
                    "type": str(meta.get("memory_type") or "unknown"),
                    "priority": _coerce_priority(meta.get("priority")),
                    "tags": list(tags),
                    "created_at": _iso(row.get("created_at")),
                }
            )
        return results

    async def _typed_memory_meta(
        self, brain_id: str, fiber_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch type/priority/tags per fiber, keyed by the fiber's record name.

        ``typed_memory`` rows are stored at ``typed_memory:{_to_surreal_id(fiber_id)}``,
        so they are addressable by the same record name the fiber id already
        carries. Joining on ``typed_memory.fiber_id`` would silently match
        nothing — that field holds the original dash-form uuid.
        """
        safe = _safe_record_names(fiber_ids)
        if not safe:
            return {}

        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(safe), _ID_CHUNK):
            chunk = safe[i : i + _ID_CHUNK]
            things = ", ".join(f"typed_memory:{s}" for s in chunk)
            rows = await self._query(
                f"SELECT id, memory_type, priority, tags FROM {things} WHERE brain_id = $brain_id",
                brain_id=brain_id,
            )
            for row in rows:
                out[_strip_table_prefix(row.get("id"))] = row
        return out


def _iso(value: Any) -> str:
    """Render a SurrealDB datetime as an ISO string, matching SQLite's TEXT column."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else ""


def _strip_table_prefix(rid: Any) -> str:
    """Render a SurrealDB record id as the bare record name."""
    if rid is None:
        return ""
    text = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    return text.split(":", 1)[1] if ":" in text else text


def _coerce_priority(value: Any) -> int:
    """typed_memory.priority is a string field; the pin listing reports an int.

    Rows written by older versions hold a label ("high") rather than a
    stringified int, so fall back to the same map ``_row_to_typed_memory`` uses
    instead of flattening every legacy row to the default.
    """
    from surreal_memory.storage.surrealdb.typed_memory import _PRIORITY_LABEL_MAP

    raw = str(value if value is not None else "5")
    try:
        return int(raw)
    except ValueError:
        return _PRIORITY_LABEL_MAP.get(raw.lower(), 5)

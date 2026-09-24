"""SurrealDB co-activation and action-log mixin."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import uuid4

from surreal_memory.core.action_event import ActionEvent
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _parse_datetime(val: Any) -> datetime:
    if val is None:
        return utcnow()
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo is not None else val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except (ValueError, AttributeError):
            pass
    return utcnow()


def _row_to_action_event(row: dict[str, Any], brain_id: str) -> ActionEvent:
    raw_tags = row.get("tags", [])
    tags: tuple[str, ...] = tuple(str(t) for t in raw_tags) if raw_tags else ()
    raw_id = str(row.get("id", ""))
    eid = raw_id.split(":")[-1] if ":" in raw_id else raw_id
    return ActionEvent(
        id=eid or str(uuid4()),
        brain_id=brain_id,
        session_id=row.get("session_id"),
        action_type=str(row.get("action_type", "")),
        action_context=str(row.get("action_context", "")),
        tags=tags,
        fiber_id=row.get("fiber_id"),
        created_at=_parse_datetime(row.get("created_at")),
    )


class SurrealDBActivityMixin:
    """Mixin providing co-activation and action-log CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ─── Co-activations ─────────────────────────────────────────────────────

    async def record_co_activation(
        self,
        neuron_a: str,
        neuron_b: str,
        binding_strength: float,
        source_anchor: str | None = None,
    ) -> str:
        """Record a Hebbian co-activation; pairs stored in canonical order (a <= b)."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        # Canonical order prevents duplicate pair permutations
        a, b = (neuron_a, neuron_b) if neuron_a <= neuron_b else (neuron_b, neuron_a)
        eid = str(uuid4())
        sid = _to_surreal_id(eid)

        await conn.insert(
            "co_activations",
            {
                "id": sid,
                "brain_id": brain_id,
                "neuron_a": a,
                "neuron_b": b,
                "binding_strength": float(binding_strength),
                "source_anchor": source_anchor,
                "created_at": utcnow(),
            },
        )
        return eid

    async def get_co_activation_counts(
        self,
        since: datetime | None = None,
        min_count: int = 1,
    ) -> list[tuple[str, str, int, float]]:
        """Return aggregated (neuron_a, neuron_b, count, avg_binding_strength) tuples."""
        brain_id = self._get_brain_id()
        sql = (
            "SELECT neuron_a, neuron_b, binding_strength FROM co_activations"
            " WHERE brain_id = $brain_id"
        )
        params: dict[str, Any] = {"brain_id": brain_id}
        if since is not None:
            sql += " AND created_at > $since"
            params["since"] = since

        rows = await self._query(sql, **params)

        # Aggregate in Python — avoids SurrealQL HAVING compatibility concerns
        pairs: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            pair = (str(r.get("neuron_a", "")), str(r.get("neuron_b", "")))
            pairs.setdefault(pair, []).append(float(r.get("binding_strength", 0.0)))

        result: list[tuple[str, str, int, float]] = []
        for (a, b), strengths in pairs.items():
            cnt = len(strengths)
            if cnt >= min_count:
                avg = sum(strengths) / cnt
                result.append((a, b, cnt, avg))
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    async def iter_co_activation_counts(
        self,
        *,
        since: datetime,
        until: datetime,
        min_count: int = 1,
        after_pair: tuple[str, str] | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[tuple[str, str, int, float]]:
        """Yield stable, keyset-paged pair aggregates bounded to one DB page."""
        if page_size < 1:
            raise ValueError("page_size must be positive")
        brain_id = self._get_brain_id()
        cursor = after_pair
        while True:
            conditions = [
                "brain_id = $brain_id",
                "created_at > $since",
                "created_at <= $until",
            ]
            params: dict[str, Any] = {
                "brain_id": brain_id,
                "since": since,
                "until": until,
                "limit": min(int(page_size), 2000),
            }
            if cursor is not None:
                conditions.append(
                    "(neuron_a > $after_a OR (neuron_a = $after_a AND neuron_b > $after_b))"
                )
                params["after_a"], params["after_b"] = cursor
            rows = await self._query(
                "SELECT neuron_a, neuron_b, count() AS pair_count, "
                "math::mean(binding_strength) AS average_strength "
                "FROM co_activations WHERE "
                + " AND ".join(conditions)
                + " GROUP BY neuron_a, neuron_b ORDER BY neuron_a, neuron_b LIMIT $limit",
                **params,
            )
            if not rows:
                return
            previous = cursor
            for row in rows:
                neuron_a = str(row.get("neuron_a", ""))
                neuron_b = str(row.get("neuron_b", ""))
                pair = (neuron_a, neuron_b)
                if not neuron_a or not neuron_b or (previous is not None and pair <= previous):
                    raise RuntimeError("co-activation aggregate page did not advance")
                previous = pair
                count = int(row.get("pair_count", 0))
                strength = float(row.get("average_strength", 0.0))
                if count >= min_count:
                    yield neuron_a, neuron_b, count, strength
            cursor = previous
            if len(rows) < min(page_size, 2000):
                return

    async def prune_co_activations(self, older_than: datetime) -> int:
        """Delete co-activation events older than older_than. Returns count deleted."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM co_activations WHERE brain_id = $brain_id AND created_at < $older_than",
            brain_id=brain_id,
            older_than=older_than,
        )
        if not rows:
            return 0
        conn = self._ensure_conn()
        deleted = 0
        for r in rows:
            rid = str(r.get("id", ""))
            if rid:
                await conn.delete(rid)
                deleted += 1
        return deleted

    async def get_co_activation_prune_page(
        self, older_than: datetime, after_id: str | None = None, *, limit: int = 500
    ) -> list[str]:
        """Return an ID-keyset page without materializing the prune backlog."""
        page_limit = min(max(int(limit), 1), 2000)
        conditions = ["brain_id = $brain_id", "created_at < $older_than"]
        params: dict[str, Any] = {
            "brain_id": self._get_brain_id(),
            "older_than": older_than,
            "limit": page_limit,
        }
        if after_id is not None:
            conditions.append("id > type::record('co_activations', $after_id)")
            params["after_id"] = _to_surreal_id(after_id)
        rows = await self._query(
            "SELECT id FROM co_activations WHERE "
            + " AND ".join(conditions)
            + " ORDER BY id ASC LIMIT $limit",
            **params,
        )
        return [str(row.get("id", "")) for row in rows if row.get("id") is not None]

    async def prune_co_activation_ids(self, event_ids: list[str]) -> int:
        """Delete only still-live IDs from one bounded, replayable prune batch."""
        unique_ids = list(dict.fromkeys(event_ids))
        if not unique_ids:
            return 0
        deleted = 0
        for start in range(0, len(unique_ids), 200):
            batch = unique_ids[start : start + 200]
            params: dict[str, Any] = {"brain_id": self._get_brain_id()}
            record_refs: list[str] = []
            for index, event_id in enumerate(batch):
                key = f"event_id_{index}"
                params[key] = _to_surreal_id(event_id)
                record_refs.append(f"type::record('co_activations', ${key})")
            rows = await self._query(
                "SELECT id FROM co_activations WHERE brain_id = $brain_id "
                f"AND id IN [{', '.join(record_refs)}]",
                **params,
            )
            for row in rows:
                rid = row.get("id")
                if rid is not None:
                    await self._ensure_conn().delete(rid)
                    deleted += 1
        return deleted

    # ─── Action log ─────────────────────────────────────────────────────────

    async def record_action(
        self,
        action_type: str,
        action_context: str = "",
        tags: tuple[str, ...] | list[str] = (),
        session_id: str | None = None,
        fiber_id: str | None = None,
    ) -> str:
        """Record an action event in the hippocampal buffer. Returns event ID."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        eid = str(uuid4())
        sid = _to_surreal_id(eid)

        await conn.insert(
            "action_log",
            {
                "id": sid,
                "brain_id": brain_id,
                "action_type": action_type,
                "action_context": action_context or "",
                "tags": list(tags),
                "session_id": session_id,
                "fiber_id": fiber_id,
                "created_at": utcnow(),
            },
        )
        return eid

    async def get_action_sequences(
        self,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[ActionEvent]:
        """Return action events ordered by created_at ASC."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 5000)
        sql = "SELECT * FROM action_log WHERE brain_id = $brain_id"
        params: dict[str, Any] = {"brain_id": brain_id}

        if session_id is not None:
            sql += " AND session_id = $session_id"
            params["session_id"] = session_id
        if since is not None:
            sql += " AND created_at > $since"
            params["since"] = since

        sql += " ORDER BY created_at ASC LIMIT $limit"
        params["limit"] = safe_limit

        rows = await self._query(sql, **params)
        return [_row_to_action_event(r, brain_id) for r in rows]

    async def prune_action_events(self, older_than: datetime) -> int:
        """Delete action events older than older_than. Returns count deleted."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM action_log WHERE brain_id = $brain_id AND created_at < $older_than",
            brain_id=brain_id,
            older_than=older_than,
        )
        if not rows:
            return 0
        conn = self._ensure_conn()
        deleted = 0
        for r in rows:
            rid = str(r.get("id", ""))
            if rid:
                await conn.delete(rid)
                deleted += 1
        return deleted

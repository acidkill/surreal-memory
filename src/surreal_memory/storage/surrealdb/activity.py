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
        """Yield pair aggregates from bounded raw-event keyset pages.

        Aggregating in Python is deliberate: SurrealDB 3.2 applies LIMIT after
        GROUP BY, so a grouped page query retains every matching pair before
        returning the requested page. Raw rows are fetched in stable pair/id
        order and reduced incrementally, retaining only one event page and the
        current pair's counters.
        """
        if page_size < 1:
            raise ValueError("page_size must be positive")
        brain_id = self._get_brain_id()
        pair_cursor = after_pair
        event_cursor: tuple[str, str, str] | None = None
        current_pair: tuple[str, str] | None = None
        pair_count = 0
        pair_strength_total = 0.0
        event_page_size = min(int(page_size), 2000)
        while True:

            async def fetch_events(
                cursor_condition: str = "", cursor_params: dict[str, Any] | None = None
            ) -> list[dict[str, Any]]:
                conditions = [
                    "brain_id = $brain_id",
                    "created_at > $since",
                    "created_at <= $until",
                ]
                if cursor_condition:
                    conditions.append(cursor_condition)
                params: dict[str, Any] = {
                    "brain_id": brain_id,
                    "since": since,
                    "until": until,
                    "limit": event_page_size,
                }
                if cursor_params:
                    params.update(cursor_params)
                return await self._query(
                    "SELECT id, neuron_a, neuron_b, binding_strength "
                    "FROM co_activations WHERE "
                    + " AND ".join(conditions)
                    + " ORDER BY neuron_a, neuron_b, id LIMIT $limit",
                    **params,
                )

            event_rows: list[dict[str, Any]] = []
            if event_cursor is None and pair_cursor is None:
                event_rows = await fetch_events()
            else:
                after_a, after_b = event_cursor[:2] if event_cursor else pair_cursor or ("", "")
                if event_cursor is not None:
                    event_rows.extend(
                        await fetch_events(
                            "neuron_a = $after_a AND neuron_b = $after_b "
                            "AND id > type::record('co_activations', $after_id)",
                            {
                                "after_a": after_a,
                                "after_b": after_b,
                                "after_id": _to_surreal_id(event_cursor[2]),
                            },
                        )
                    )
                # SurrealDB 3.2's composite-index range scan can include its
                # lower-bound key when a residual created_at filter is present.
                # Keep explicit not-equal filters so the split keyset ranges
                # are strictly disjoint at their pair boundary.
                event_rows.extend(
                    await fetch_events(
                        "neuron_a = $after_a AND neuron_b > $after_b AND neuron_b != $after_b",
                        {"after_a": after_a, "after_b": after_b},
                    )
                )
                event_rows.extend(
                    await fetch_events(
                        "neuron_a > $after_a AND neuron_a != $after_a",
                        {"after_a": after_a},
                    )
                )
                event_rows.sort(
                    key=lambda row: (
                        str(row.get("neuron_a", "")),
                        str(row.get("neuron_b", "")),
                        str(row.get("id", "")).split(":", 1)[-1],
                    )
                )
                event_rows = event_rows[:event_page_size]

            rows = event_rows
            if not rows:
                if current_pair is not None and pair_count >= min_count:
                    yield (
                        current_pair[0],
                        current_pair[1],
                        pair_count,
                        pair_strength_total / pair_count,
                    )
                return
            for row in rows:
                neuron_a = str(row.get("neuron_a", ""))
                neuron_b = str(row.get("neuron_b", ""))
                raw_id = str(row.get("id", ""))
                event_id = raw_id.split(":", 1)[-1]
                pair = (neuron_a, neuron_b)
                event_key = (neuron_a, neuron_b, event_id)
                if not neuron_a or not neuron_b or not event_id:
                    raise RuntimeError("co-activation event is missing its stable key")
                if event_cursor is not None and event_key <= event_cursor:
                    raise RuntimeError("co-activation event page did not advance")
                if current_pair is not None and pair != current_pair:
                    if pair <= current_pair:
                        raise RuntimeError("co-activation pair order regressed")
                    if pair_count >= min_count:
                        yield (
                            current_pair[0],
                            current_pair[1],
                            pair_count,
                            pair_strength_total / pair_count,
                        )
                    pair_count = 0
                    pair_strength_total = 0.0
                current_pair = pair
                pair_count += 1
                pair_strength_total += float(row.get("binding_strength", 0.0))
                event_cursor = event_key
            if len(rows) < event_page_size:
                if current_pair is not None and pair_count >= min_count:
                    yield (
                        current_pair[0],
                        current_pair[1],
                        pair_count,
                        pair_strength_total / pair_count,
                    )
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

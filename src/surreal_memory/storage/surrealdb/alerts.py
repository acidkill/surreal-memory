"""SurrealDB alerts storage mixin.

Single-alert lookups bind the *sanitised* id and rebuild the record id inside
SurrealQL with ``type::record('alerts', $sid)``. Comparing ``id`` against a
plain ``"alerts:<sid>"`` string is not a slower spelling of the same thing —
``id`` holds a record id, so the predicate is unconditionally false and the row
can never match. That silently disabled ``mark_alerts_seen``,
``mark_alert_acknowledged`` and ``get_alert``. ``resolve_alerts_by_type`` was
unaffected because it reuses the ``id`` value its own SELECT returned. Same
lesson, same shape as the ``typed_memory`` / ``fiber`` lookups in this package.

The writes then reuse that same ``id`` object rather than rebuilding
``f"alerts:{sid}"``. Rebuilding it looks equivalent and is not: an all-digit
sid (``uuid4().hex[:16]`` is all digits about once in 1150 — character 12 is
always the version nibble ``4``, so only fifteen of the sixteen are random)
round-trips
through the SDK as a *numeric* record id, while ``record_alert`` stored a
*string* one — so the SELECT would find the row and the merge would write to a
different, empty record, and the call would report success having changed
nothing. Reading the id back from the query is the only spelling that cannot
drift from the row that was actually matched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from surreal_memory.core.alert import Alert, AlertStatus, AlertType
from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

_DEDUP_COOLDOWN = timedelta(hours=6)

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


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


def _row_to_alert(row: dict[str, Any]) -> Alert:
    """Convert a SurrealDB record to an Alert dataclass."""
    metadata = dict(row.get("metadata") or {})
    alert_id = _record_id_part(str(row.get("id", "")))

    return Alert(
        id=alert_id,
        brain_id=str(row["brain_id"]),
        alert_type=AlertType(str(row["alert_type"])),
        severity=str(row.get("severity", "low")),
        message=str(row.get("message", "")),
        recommended_action=str(row.get("recommended_action", "")),
        status=AlertStatus(str(row["status"])),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        seen_at=_parse_datetime(row.get("seen_at")),
        acknowledged_at=_parse_datetime(row.get("acknowledged_at")),
        resolved_at=_parse_datetime(row.get("resolved_at")),
        metadata=metadata,
    )


class SurrealDBAlertsMixin:
    """Mixin providing alert CRUD operations for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def record_alert(self, alert: Alert) -> str:
        """Insert a new alert, respecting the 6-hour dedup cooldown.

        Returns the alert ID if inserted, empty string if suppressed.
        """
        brain_id = self._get_brain_id()
        cutoff = utcnow() - _DEDUP_COOLDOWN

        existing = await self._query(
            "SELECT count() AS cnt FROM alerts"
            " WHERE brain_id = $brain_id AND alert_type = $alert_type"
            " AND created_at > $cutoff AND status IN ['active', 'seen']"
            " GROUP ALL",
            brain_id=brain_id,
            alert_type=alert.alert_type.value,
            cutoff=cutoff,
        )
        if existing and int(existing[0].get("cnt", 0)) > 0:
            return ""

        conn = self._ensure_conn()
        sid = _to_surreal_id(alert.id)

        record_data: dict[str, Any] = {
            "id": sid,
            "brain_id": brain_id,
            "alert_type": alert.alert_type.value,
            "severity": alert.severity,
            "message": alert.message,
            "recommended_action": alert.recommended_action,
            "status": alert.status.value,
            "created_at": alert.created_at,
            "seen_at": alert.seen_at,
            "acknowledged_at": alert.acknowledged_at,
            "resolved_at": alert.resolved_at,
            "metadata": dict(alert.metadata),
        }

        try:
            await conn.insert("alerts", record_data)
        except Exception:
            try:
                # Not conn.delete(f"alerts:{sid}"): an all-digit sid goes back
                # through the SDK as a numeric record id, so that call deletes a
                # different (absent) record, raises nothing, and leaves the
                # clashing row in place — the retry below then fails the same
                # way. Rebuilding the id in SurrealQL keeps it a string, which
                # is what the insert above stored.
                await self._query("DELETE type::record('alerts', $sid)", sid=sid)
            except Exception:
                logger.debug("alert insert retry: delete of the clashing row failed")
            await conn.insert("alerts", record_data)

        return alert.id

    async def get_active_alerts(self, limit: int = 50) -> list[Alert]:
        """Get active/seen/acknowledged alerts, sorted by severity then recency."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 200)

        rows = await self._query(
            "SELECT * FROM alerts"
            " WHERE brain_id = $brain_id"
            " AND status IN ['active', 'seen', 'acknowledged']"
            " ORDER BY created_at DESC"
            " LIMIT $limit",
            brain_id=brain_id,
            limit=safe_limit,
        )

        alerts = [_row_to_alert(r) for r in rows]
        alerts.sort(
            key=lambda a: (
                _SEVERITY_RANK.get(a.severity, 99),
                -(a.created_at.timestamp() if a.created_at else 0),
            )
        )
        return alerts

    async def count_pending_alerts(self) -> int:
        """Count active + seen alerts (not acknowledged or resolved)."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT count() AS cnt FROM alerts"
            " WHERE brain_id = $brain_id AND status IN ['active', 'seen']"
            " GROUP ALL",
            brain_id=brain_id,
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def mark_alerts_seen(self, alert_ids: list[str]) -> int:
        """Mark active alerts as seen. Returns count of updated rows."""
        if not alert_ids:
            return 0

        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        now = utcnow()
        updated = 0

        for aid in alert_ids:
            existing = await self._query(
                "SELECT id FROM alerts"
                " WHERE brain_id = $brain_id AND id = type::record('alerts', $sid)"
                " AND status = 'active' LIMIT 1",
                brain_id=brain_id,
                sid=_to_surreal_id(aid),
            )
            if not existing:
                continue
            await conn.merge(
                existing[0]["id"],
                {"status": AlertStatus.SEEN.value, "seen_at": now},
            )
            updated += 1
        return updated

    async def mark_alert_acknowledged(self, alert_id: str) -> bool:
        """Mark a single alert as acknowledged. Returns True if updated."""
        brain_id = self._get_brain_id()

        existing = await self._query(
            "SELECT id FROM alerts"
            " WHERE brain_id = $brain_id AND id = type::record('alerts', $sid)"
            " AND status IN ['active', 'seen'] LIMIT 1",
            brain_id=brain_id,
            sid=_to_surreal_id(alert_id),
        )
        if not existing:
            return False

        conn = self._ensure_conn()
        await conn.merge(
            existing[0]["id"],
            {"status": AlertStatus.ACKNOWLEDGED.value, "acknowledged_at": utcnow()},
        )
        return True

    async def resolve_alerts_by_type(self, alert_types: list[str]) -> int:
        """Resolve all active/seen alerts of given types. Returns count."""
        if not alert_types:
            return 0

        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM alerts"
            " WHERE brain_id = $brain_id"
            " AND alert_type IN $types"
            " AND status IN ['active', 'seen']",
            brain_id=brain_id,
            types=list(alert_types),
        )
        if not rows:
            return 0

        conn = self._ensure_conn()
        now = utcnow()
        updated = 0

        for r in rows:
            rid = str(r.get("id"))
            if not rid:
                continue
            await conn.merge(
                rid,
                {"status": AlertStatus.RESOLVED.value, "resolved_at": now},
            )
            updated += 1
        return updated

    async def get_alert(self, alert_id: str) -> Alert | None:
        """Get a single alert by ID."""
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(alert_id)

        rows = await self._query(
            "SELECT * FROM alerts WHERE brain_id = $brain_id"
            " AND id = type::record('alerts', $sid) LIMIT 1",
            brain_id=brain_id,
            sid=sid,
        )
        if not rows:
            return None
        return _row_to_alert(rows[0])

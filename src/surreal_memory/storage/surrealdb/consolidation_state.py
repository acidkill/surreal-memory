"""Durable checkpoint and per-brain lease primitives for consolidation.

The state is deliberately stored in SurrealDB so CLI, MCP, and scheduled
consolidation workers coordinate across hosts and process restarts.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from hashlib import sha256
from typing import Any

from surreal_memory.utils.timeutils import utcnow


class SurrealDBConsolidationStateMixin:
    """Storage mixin for consolidation checkpoints and distributed leases."""

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _consolidation_record_id(brain_id: str) -> str:
        return sha256(brain_id.encode("utf-8")).hexdigest()

    async def get_consolidation_progress(
        self, brain_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the current durable consolidation run for this brain."""
        resolved_brain_id = brain_id or self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM consolidation_progress WHERE brain_id = $brain_id LIMIT 1",
            brain_id=resolved_brain_id,
        )
        return dict(rows[0]) if rows else None

    async def create_consolidation_progress(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Create a new run record; record-ID uniqueness arbitrates duplicate starts."""
        brain_id = str(state["brain_id"])
        record_id = self._consolidation_record_id(brain_id)
        rows = await self._query(
            "CREATE type::record('consolidation_progress', $record_id) CONTENT $state RETURN AFTER",
            record_id=record_id,
            state=dict(state),
        )
        return dict(rows[0]) if rows else dict(state)

    async def start_consolidation_progress(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Create or replace the completed run record for a new brain-scoped run.

        Callers must hold the brain lease and must have verified that any
        existing record is complete before invoking this method.
        """
        brain_id = str(state["brain_id"])
        record_id = self._consolidation_record_id(brain_id)
        rows = await self._query(
            "UPSERT type::record('consolidation_progress', $record_id) CONTENT $state RETURN AFTER",
            record_id=record_id,
            state=dict(state),
        )
        return dict(rows[0]) if rows else dict(state)

    async def claim_consolidation_progress(
        self, brain_id: str, owner_token: str
    ) -> dict[str, Any] | None:
        """Fence a resumed run with the token of its current lease holder."""
        record_id = self._consolidation_record_id(brain_id)
        rows = await self._query(
            "UPDATE type::record('consolidation_progress', $record_id) "
            "SET owner_token = $owner_token, updated_at = $updated_at "
            "WHERE brain_id = $brain_id AND status IN ['queued', 'running', 'paused', 'failed'] "
            "RETURN AFTER",
            record_id=record_id,
            owner_token=owner_token,
            updated_at=utcnow(),
            brain_id=brain_id,
        )
        return dict(rows[0]) if rows else None

    async def save_consolidation_progress(
        self, brain_id: str, owner_token: str, state: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Save a checkpoint only while this worker still owns the run record."""
        record_id = self._consolidation_record_id(brain_id)
        data = dict(state)
        data.pop("id", None)
        data["brain_id"] = brain_id
        data["owner_token"] = owner_token
        data["updated_at"] = utcnow()
        rows = await self._query(
            "UPDATE type::record('consolidation_progress', $record_id) "
            "CONTENT $state WHERE brain_id = $brain_id AND owner_token = $owner_token "
            "RETURN AFTER",
            record_id=record_id,
            state=data,
            brain_id=brain_id,
            owner_token=owner_token,
        )
        return dict(rows[0]) if rows else None

    async def acquire_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int = 120
    ) -> bool:
        """Acquire a lease or atomically replace it after expiry."""
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 30 and 3600")
        record_id = self._consolidation_record_id(brain_id)
        now = utcnow()
        lease_data = {
            "brain_id": brain_id,
            "owner_token": owner_token,
            "acquired_at": now,
            "expires_at": now + timedelta(seconds=lease_seconds),
        }
        try:
            rows = await self._query(
                "CREATE type::record('consolidation_lease', $record_id) "
                "CONTENT $lease RETURN AFTER",
                record_id=record_id,
                lease=lease_data,
            )
            return bool(rows)
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "alreadyexists" not in message:
                raise

        rows = await self._query(
            "UPDATE type::record('consolidation_lease', $record_id) CONTENT $lease "
            "WHERE brain_id = $brain_id AND expires_at <= $now RETURN AFTER",
            record_id=record_id,
            lease=lease_data,
            brain_id=brain_id,
            now=now,
        )
        return bool(rows)

    async def renew_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int = 120
    ) -> bool:
        """Extend a lease only if it has not already been taken over."""
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 30 and 3600")
        record_id = self._consolidation_record_id(brain_id)
        now = utcnow()
        rows = await self._query(
            "UPDATE type::record('consolidation_lease', $record_id) "
            "SET expires_at = $expires_at WHERE brain_id = $brain_id "
            "AND owner_token = $owner_token AND expires_at > $now RETURN AFTER",
            record_id=record_id,
            expires_at=now + timedelta(seconds=lease_seconds),
            brain_id=brain_id,
            owner_token=owner_token,
            now=now,
        )
        return bool(rows)

    async def release_consolidation_lease(self, brain_id: str, owner_token: str) -> bool:
        """Release only the lease held by this owner."""
        record_id = self._consolidation_record_id(brain_id)
        rows = await self._query(
            "DELETE type::record('consolidation_lease', $record_id) "
            "WHERE brain_id = $brain_id AND owner_token = $owner_token RETURN BEFORE",
            record_id=record_id,
            brain_id=brain_id,
            owner_token=owner_token,
        )
        return bool(rows)

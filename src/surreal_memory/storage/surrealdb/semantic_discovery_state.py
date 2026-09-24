"""Immutable, bounded snapshots for resumable semantic discovery stages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

_MAX_SNAPSHOT_BYTES = 1_000_000


class SemanticDiscoveryStateError(RuntimeError):
    """A durable discovery snapshot is malformed or unavailable."""


class SemanticDiscoveryStateConflictError(SemanticDiscoveryStateError):
    """An immutable state revision was retried with different data."""


class SurrealDBSemanticDiscoveryStateMixin:
    """Storage API for immutable snapshots referenced by lease-fenced checkpoints."""

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _record_id(state_id: str, revision: int) -> str:
        return sha256(f"{state_id}\0{revision}".encode()).hexdigest()

    @staticmethod
    def _manifest_references(
        value: Any, *, state_id: str, revision: int, source_token: str
    ) -> bool:
        if isinstance(value, Mapping):
            if (
                value.get("state_id") == state_id
                and value.get("state_revision") == revision
                and value.get("source_token") == source_token
            ):
                return True
            return any(
                SurrealDBSemanticDiscoveryStateMixin._manifest_references(
                    child,
                    state_id=state_id,
                    revision=revision,
                    source_token=source_token,
                )
                for child in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                SurrealDBSemanticDiscoveryStateMixin._manifest_references(
                    child,
                    state_id=state_id,
                    revision=revision,
                    source_token=source_token,
                )
                for child in value
            )
        return False

    @staticmethod
    def _contains_vector_field(value: Any) -> bool:
        forbidden = {"embedding", "embedding_vec", "embedding_vector", "vector", "vectors"}
        if isinstance(value, Mapping):
            return any(
                str(key).lower() in forbidden
                or SurrealDBSemanticDiscoveryStateMixin._contains_vector_field(child)
                for key, child in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(
                SurrealDBSemanticDiscoveryStateMixin._contains_vector_field(child)
                for child in value
            )
        return False

    async def save_semantic_discovery_state(
        self, state_id: str, revision: int, payload: Mapping[str, Any]
    ) -> None:
        """Create an immutable revision; an exact retry is idempotent.

        Use a random state_id unique to each run/owner epoch. A stale writer may
        leave an orphan snapshot, but it cannot overwrite a revision or publish
        its checkpoint pointer; publishing remains the lease-fenced progress
        operation. Snapshots are limited to 1 MB and may not contain vectors.
        """
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a non-empty string")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("revision must be a positive integer")
        data = dict(payload)
        brain_id = self._get_brain_id()
        if data.get("brain_id") != brain_id:
            raise ValueError("snapshot brain_id must match the current brain")
        for name in ("run_id", "owner_token", "source_token"):
            if not isinstance(data.get(name), str) or not data[name]:
                raise ValueError(f"snapshot {name} must be a non-empty string")
        try:
            serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot payload must be JSON serializable") from exc
        if len(serialized.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
            raise ValueError(f"snapshot payload exceeds {_MAX_SNAPSHOT_BYTES} bytes")
        if self._contains_vector_field(data):
            raise ValueError("snapshot payload must not contain embedding vectors")
        # Compare and persist the same JSON-normalized value on exact retries.
        data = json.loads(serialized)

        record_id = self._record_id(state_id, revision)
        record = {
            "state_id": state_id,
            "revision": revision,
            "brain_id": brain_id,
            "run_id": data["run_id"],
            "owner_token": data["owner_token"],
            "source_token": data["source_token"],
            "payload": data,
        }
        try:
            rows = await self._query(
                "CREATE type::record('semantic_discovery_state', $record_id) "
                "CONTENT $record RETURN AFTER",
                record_id=record_id,
                record=record,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "alreadyexists" not in message:
                raise
            rows = await self._query(
                "SELECT * FROM semantic_discovery_state WHERE id = "
                "type::record('semantic_discovery_state', $record_id) LIMIT 1",
                record_id=record_id,
            )
            if not rows:
                raise SemanticDiscoveryStateError(
                    "immutable state revision exists but could not be read"
                ) from exc
            existing = rows[0]
            if (
                existing.get("state_id") != state_id
                or existing.get("revision") != revision
                or existing.get("brain_id") != brain_id
                or existing.get("run_id") != data["run_id"]
                or existing.get("owner_token") != data["owner_token"]
                or existing.get("source_token") != data["source_token"]
                or existing.get("payload") != data
            ):
                raise SemanticDiscoveryStateConflictError(
                    "immutable state revision was retried with conflicting content"
                ) from exc
            return
        if not rows:
            raise SemanticDiscoveryStateError("SurrealDB did not confirm the state snapshot write")

    async def load_semantic_discovery_state(
        self, state_id: str, revision: int
    ) -> Mapping[str, Any] | None:
        """Load only the exact snapshot referenced by the active run manifest."""
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a non-empty string")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("revision must be a positive integer")
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM semantic_discovery_state WITH INDEX idx_sds_state_revision "
            "WHERE brain_id = $brain_id AND state_id = $state_id "
            "AND revision = $revision LIMIT 1",
            brain_id=brain_id,
            state_id=state_id,
            revision=revision,
        )
        if not rows:
            return None
        row = rows[0]
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            return None
        if row.get("brain_id") != brain_id or row.get("state_id") != state_id:
            return None
        if row.get("revision") != revision or payload.get("brain_id") != brain_id:
            return None

        progress_rows = await self._query(
            "SELECT * FROM consolidation_progress WHERE brain_id = $brain_id LIMIT 1",
            brain_id=brain_id,
        )
        lease_rows = await self._query(
            "SELECT * FROM consolidation_lease WHERE brain_id = $brain_id "
            "AND expires_at > time::now() LIMIT 1",
            brain_id=brain_id,
        )
        if not progress_rows or not lease_rows:
            return None
        progress = progress_rows[0]
        lease = lease_rows[0]
        if (
            progress.get("owner_token") != lease.get("owner_token")
            or progress.get("run_id") != row.get("run_id")
            or payload.get("run_id") != row.get("run_id")
        ):
            return None
        if not self._manifest_references(
            progress.get("strategy_states"),
            state_id=state_id,
            revision=revision,
            source_token=str(row.get("source_token", "")),
        ):
            return None
        if payload.get("source_token") != row.get("source_token"):
            return None
        return dict(payload)

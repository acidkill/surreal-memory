"""Immutable, bounded snapshots for resumable semantic discovery stages."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Collection, Mapping, Sequence
from datetime import timedelta
from hashlib import sha256
from typing import Any, cast

from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

_MAX_SNAPSHOT_BYTES = 1_000_000
_PRUNE_SCAN_ROWS = 64
_PRUNE_SCAN_PAGES = 4
_PRUNE_DELETE_ROWS = 8
_PRUNE_SECONDS = 5.0


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
        value: Any,
        *,
        state_id: str,
        revision: int,
        source_token: str,
        _depth: int = 0,
    ) -> bool:
        if _depth > 64:
            return False
        if isinstance(value, str):
            candidate = value.lstrip()
            # A pending apply manifest can contain thousands of synapses; the
            # snapshot payload limit does not bound its serialized envelope.
            if not candidate.startswith(("{", "[")):
                return False
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                return False
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, RecursionError, ValueError):
                return False
            return SurrealDBSemanticDiscoveryStateMixin._manifest_references(
                decoded,
                state_id=state_id,
                revision=revision,
                source_token=source_token,
                _depth=_depth + 1,
            )
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
                    _depth=_depth + 1,
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
                    _depth=_depth + 1,
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

    async def _load_semantic_discovery_state_page(
        self,
        state_id: str,
        revision: int,
        *,
        parent_ref: Mapping[str, Any],
        ancestor_refs: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any] | None:
        """Load a bounded classifier page through a verified proof-index path."""
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a non-empty string")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(parent_ref, Mapping) or not isinstance(ancestor_refs, Sequence):
            return None
        if isinstance(ancestor_refs, (str, bytes)) or len(ancestor_refs) > 63:
            return None
        parent_state_id = parent_ref.get("state_id")
        parent_revision = parent_ref.get("revision")
        if (
            not isinstance(parent_state_id, str)
            or not parent_state_id
            or isinstance(parent_revision, bool)
            or not isinstance(parent_revision, int)
            or parent_revision < 1
        ):
            return None

        # The root loader enforces the live run, manifest pointer, and lease.
        root = await self.load_semantic_discovery_state(parent_state_id, parent_revision)
        if (
            root is None
            or root.get("kind") != "semantic_source_validation"
            or type(root.get("version")) is not int
            or root.get("version") != 1
            or root.get("source_token") != parent_ref.get("source_token")
        ):
            return None
        brain_id = self._get_brain_id()
        if brain_id != root.get("brain_id"):
            return None

        def matching_ref(parent: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
            refs = parent.get("classifier_pages")
            if not isinstance(refs, list):
                return False
            for ref in refs:
                if (
                    isinstance(ref, Mapping)
                    and ref.get("state_id") == expected.get("state_id")
                    and type(ref.get("revision")) is int
                    and ref.get("revision") == expected.get("revision")
                    and type(ref.get("state_revision")) is int
                    and ref.get("state_revision") == expected.get("state_revision")
                    and ref.get("source_token") == root.get("source_token")
                    and isinstance(ref.get("sha256"), str)
                    and ("sha256" not in expected or ref.get("sha256") == expected.get("sha256"))
                ):
                    return True
            return False

        async def load_referenced(
            expected: Mapping[str, Any], *, kind: str | tuple[str, ...]
        ) -> Mapping[str, Any] | None:
            allowed_kinds = (kind,) if isinstance(kind, str) else kind
            ref_state_id = expected.get("state_id")
            ref_revision = expected.get("revision")
            if (
                not isinstance(ref_state_id, str)
                or not ref_state_id
                or type(ref_revision) is not int
                or ref_revision < 1
                or type(expected.get("state_revision")) is not int
                or expected.get("state_revision") != ref_revision
                or expected.get("source_token") != root.get("source_token")
                or not isinstance(expected.get("sha256"), str)
            ):
                return None
            rows = await self._query(
                "SELECT * FROM semantic_discovery_state WITH INDEX idx_sds_state_revision "
                "WHERE brain_id = $brain_id AND state_id = $state_id "
                "AND revision = $revision LIMIT 1",
                brain_id=brain_id,
                state_id=ref_state_id,
                revision=ref_revision,
            )
            if not rows:
                return None
            row = rows[0]
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                return None
            if (
                row.get("brain_id") != brain_id
                or row.get("state_id") != ref_state_id
                or type(row.get("revision")) is not int
                or row.get("revision") != ref_revision
                or row.get("run_id") != root.get("run_id")
                or row.get("owner_token") != root.get("owner_token")
                or row.get("source_token") != root.get("source_token")
                or payload.get("brain_id") != row.get("brain_id")
                or payload.get("run_id") != row.get("run_id")
                or payload.get("owner_token") != row.get("owner_token")
                or payload.get("source_token") != row.get("source_token")
                or payload.get("reference_time") != root.get("reference_time")
                or payload.get("kind") not in allowed_kinds
                or type(payload.get("version")) is not int
                or payload.get("version") != 1
                or self._contains_vector_field(payload)
            ):
                return None
            try:
                serialized = json.dumps(
                    dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                encoded = serialized.encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError):
                return None
            if len(encoded) > _MAX_SNAPSHOT_BYTES or sha256(encoded).hexdigest() != expected.get(
                "sha256"
            ):
                return None
            payload_kind = payload.get("kind")
            if payload_kind == "semantic_source_validation_index":
                if not isinstance(payload.get("classifier_pages"), list):
                    return None
            elif payload_kind == "semantic_source_validation_ids":
                if not isinstance(payload.get("records"), list) or any(
                    not isinstance(record, list)
                    or len(record) != 2
                    or not all(isinstance(part, str) and part for part in record)
                    for record in payload.get("records", [])
                ):
                    return None
            return cast("Mapping[str, Any]", json.loads(serialized))

        current: Mapping[str, Any] = root
        for ancestor_ref in ancestor_refs:
            if not isinstance(ancestor_ref, Mapping) or not matching_ref(current, ancestor_ref):
                return None
            current_page = await load_referenced(
                ancestor_ref, kind="semantic_source_validation_index"
            )
            if current_page is None:
                return None
            current = current_page

        target_ref = {
            "state_id": state_id,
            "revision": revision,
            "state_revision": revision,
            "source_token": root.get("source_token"),
        }
        if not matching_ref(current, target_ref):
            return None
        final_ref = next(
            (
                ref
                for ref in current["classifier_pages"]
                if isinstance(ref, Mapping)
                and ref.get("state_id") == state_id
                and type(ref.get("revision")) is int
                and ref.get("revision") == revision
                and type(ref.get("state_revision")) is int
                and ref.get("state_revision") == revision
                and ref.get("source_token") == root.get("source_token")
            ),
            None,
        )
        if final_ref is None:
            return None
        return await load_referenced(
            final_ref,
            kind=("semantic_source_validation_ids", "semantic_source_validation_index"),
        )

    async def prune_semantic_discovery_state(
        self, *, older_than: timedelta, keep_run_ids: Collection[str], owner_token: str
    ) -> dict[str, Any]:
        """Make bounded, durable progress through discovery snapshots for this brain."""
        if not isinstance(older_than, timedelta) or older_than <= timedelta(0):
            raise ValueError("older_than must be a positive timedelta")
        if isinstance(keep_run_ids, str):
            raise TypeError("keep_run_ids must be a collection of run ids, not a string")
        if not owner_token:
            raise ValueError("owner_token must be non-empty")
        brain_id = self._get_brain_id()
        record_id = sha256(brain_id.encode()).hexdigest()
        keep = sorted({str(run_id) for run_id in keep_run_ids})
        cutoff = utcnow() - older_than
        deadline = time.monotonic() + _PRUNE_SECONDS
        deleted = 0
        more = True
        for _ in range(_PRUNE_SCAN_PAGES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            async with asyncio.timeout(remaining):
                saved = await self._query(
                    "SELECT cursor FROM type::record('semantic_discovery_prune_cursor', $record_id)",
                    record_id=record_id,
                )
                cursor = saved[0].get("cursor") if saved else None
                if cursor is None:
                    source = "semantic_discovery_state"
                else:
                    part = _record_id_part(str(cursor))
                    if not part or _to_surreal_id(part) != part:
                        raise SemanticDiscoveryStateError("invalid discovery prune cursor")
                    # On 3.2 a literal RecordIdScan materializes the tail before
                    # LIMIT. A runtime range passes LIMIT to the storage iterator.
                    source = "type::record('semantic_discovery_state', $cursor>..)"
                page = await self._query(
                    "SELECT id, (brain_id = $brain_id AND created_at < $cutoff "
                    "AND run_id NOT IN $keep) AS eligible FROM "
                    f"{source} LIMIT $limit",
                    brain_id=brain_id,
                    cutoff=cutoff,
                    keep=keep,
                    cursor=_record_id_part(str(cursor)) if cursor else None,
                    limit=_PRUNE_SCAN_ROWS,
                )
                candidates: list[Any] = []
                scanned = 0
                for row in page:
                    scanned += 1
                    if row.get("eligible"):
                        candidates.append(row["id"])
                        if len(candidates) >= _PRUNE_DELETE_ROWS - deleted:
                            break
                more = scanned < len(page) or len(page) == _PRUNE_SCAN_ROWS
                next_cursor = str(page[scanned - 1]["id"]) if scanned and more else None
                result = await self._query(
                    "RETURN { "
                    "LET $lease = (UPDATE type::record('consolidation_lease', $record_id) "
                    "SET sweep_fenced_at = time::now() WHERE brain_id = $brain_id "
                    "AND owner_token = $owner_token AND expires_at > time::now() RETURN AFTER); "
                    "IF array::len($lease) != 1 { THROW 'discovery prune lease lost'; }; "
                    "LET $protected = (SELECT VALUE run_id FROM consolidation_progress "
                    "WHERE brain_id = $brain_id AND status IN ['queued', 'running', 'paused', 'failed']); "
                    "LET $removed = (DELETE $candidates WHERE brain_id = $brain_id "
                    "AND created_at < $cutoff AND run_id NOT IN $keep "
                    "AND run_id NOT IN $protected RETURN VALUE $before.id); "
                    "UPSERT type::record('semantic_discovery_prune_cursor', $record_id) "
                    "CONTENT {brain_id: $brain_id, cursor: $next_cursor ?? NONE} RETURN NONE; "
                    "RETURN [{deleted: array::len($removed)}]; };",
                    record_id=record_id,
                    brain_id=brain_id,
                    owner_token=owner_token,
                    candidates=candidates,
                    cutoff=cutoff,
                    keep=keep,
                    next_cursor=next_cursor,
                )
                count = result[0].get("deleted") if len(result) == 1 else None
                if type(count) is not int or not 0 <= count <= len(candidates):
                    raise SemanticDiscoveryStateError(
                        "invalid discovery prune transaction response"
                    )
                deleted += count
            if not more or deleted >= _PRUNE_DELETE_ROWS:
                break
        return {"deleted": deleted, "more": more}

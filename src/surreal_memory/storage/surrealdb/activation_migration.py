"""Guarded, explicit activation changes for the default brain."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage


class ActivationMigrationError(RuntimeError):
    """Migration could not prove the requested state transition."""


def _is_config_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class SurrealDBActivationMigrationMixin:
    """Use existing consolidation ownership to serialize a full-config CAS.

    The brief coordination lease is the only write besides activation_strategy.
    No checkpoint, timestamp, metadata, or other brain setting is overwritten.
    """

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _run_query(self, sql: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    if TYPE_CHECKING:

        async def acquire_consolidation_lease(
            self, brain_id: str, owner_token: str, *, lease_seconds: int = 120
        ) -> bool:
            raise NotImplementedError

        async def release_consolidation_lease(self, brain_id: str, owner_token: str) -> bool:
            raise NotImplementedError

    async def get_activation_snapshot(self, *, brain_name: str = "default") -> dict[str, Any]:
        """Read raw persisted config without deserialization losing unknown keys."""
        if brain_name != "default" or self._get_brain_id() != "default":
            raise ActivationMigrationError("Activation migration is restricted to default")
        rows = await self._query(
            "SELECT *, crypto::sha256(type::string(config)) AS config_sha256 "
            "FROM brain WHERE name = $name",
            name=brain_name,
        )
        if len(rows) != 1:
            raise ActivationMigrationError("Expected exactly one default brain")
        row = rows[0]
        rid = row["id"]
        if not hasattr(rid, "id") or getattr(rid, "table_name", None) != "brain":
            raise ActivationMigrationError("Expected an exact brain RecordID")
        if row.get("name") != brain_name or not isinstance(row.get("config"), dict):
            raise ActivationMigrationError("Invalid default brain configuration")
        if not _is_config_sha256(row.get("config_sha256")):
            raise ActivationMigrationError("Invalid server config SHA-256 proof")
        return {
            "brain_id": str(rid.id),
            "name": brain_name,
            "config": deepcopy(row["config"]),
            "config_sha256": row["config_sha256"],
        }

    async def set_activation_strategy(
        self,
        *,
        brain_id: str,
        expected_config: dict[str, Any],
        expected_config_sha256: str,
        from_strategy: str,
        to_strategy: str,
    ) -> dict[str, Any]:
        """Change only activation, confirm readback, and reconcile a lost reply.

        Reversing the command uses the same expected-full-config and lease guards.
        A failed write is never retried; ambiguous outcomes require reconciliation.
        """
        if not _is_config_sha256(expected_config_sha256):
            raise ActivationMigrationError("Expected config SHA-256 proof is invalid")
        if {from_strategy, to_strategy} != {"classic", "hybrid"}:
            raise ActivationMigrationError("Only classic <-> hybrid transitions are supported")
        if expected_config.get("activation_strategy") != from_strategy:
            raise ActivationMigrationError("Expected config does not match --from")
        before = await self.get_activation_snapshot()
        if (
            before["brain_id"] != brain_id
            or before["config"] != expected_config
            or before["config_sha256"] != expected_config_sha256
        ):
            raise ActivationMigrationError("Brain identity or full configuration changed")
        scope = self._get_brain_id()
        token = "activation-" + uuid4().hex
        expected_after = deepcopy(expected_config)
        expected_after["activation_strategy"] = to_strategy
        reconciled = False
        acquired = False
        acquisition_returned = False
        try:
            acquired = await self.acquire_consolidation_lease(scope, token)
            acquisition_returned = True
            if not acquired:
                raise ActivationMigrationError("An active consolidation lease blocks migration")
            failure: Exception | None = None
            try:
                await self._run_query(
                    "UPDATE type::record('brain', $brain_id) "
                    "SET config.activation_strategy = $to_strategy "
                    "WHERE name = $name "
                    "AND crypto::sha256(type::string(config)) = $expected_config_sha256 "
                    "AND array::len((SELECT * FROM type::record('consolidation_lease', $lease_id) "
                    "WHERE brain_id = $scope AND owner_token = $migration_owner "
                    "AND expires_at > time::now())) = 1 RETURN AFTER",
                    {
                        "brain_id": brain_id,
                        "name": "default",
                        "expected_config_sha256": expected_config_sha256,
                        "to_strategy": to_strategy,
                        "lease_id": sha256(scope.encode("utf-8")).hexdigest(),
                        "scope": scope,
                        "migration_owner": token,
                    },
                )
            except Exception as exc:
                failure = exc
            try:
                after = await self.get_activation_snapshot()
            except Exception as exc:
                raise ActivationMigrationError(
                    "Migration outcome unknown: readback failed"
                ) from exc
            if after["brain_id"] != brain_id or after["config"] != expected_after:
                raise ActivationMigrationError(
                    "Activation CAS or full-config readback failed"
                ) from failure
            reconciled = failure is not None
            return {
                "brain_id": brain_id,
                "brain": "default",
                "from": from_strategy,
                "to": to_strategy,
                "confirmed": True,
                "reconciled": reconciled,
            }
        finally:
            # Also release after a lost acquisition reply; ownership makes this safe.
            # A failed acquisition returning False belongs to another owner.
            if acquired or not acquisition_returned:
                released = await self.release_consolidation_lease(scope, token)
                if acquired and not released:
                    raise ActivationMigrationError("Migration lease ownership was lost")


@asynccontextmanager
async def activation_migration_storage() -> AsyncIterator[SurrealDBStorage]:
    """Connect to an existing database without schema or brain auto-migrations."""
    from surreal_memory.storage.surrealdb.connection import (
        MIN_SERVER_VERSION,
        parse_server_version,
    )
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage()
    try:
        await storage._reconnect()
        version = parse_server_version(str(await storage._ensure_conn().version()))
        if version is None or version < MIN_SERVER_VERSION:
            raise ActivationMigrationError("Cannot confirm a supported SurrealDB server version")
        storage.set_brain("default")
        yield storage
    finally:
        await storage.close()

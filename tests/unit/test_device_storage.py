"""Tests for SQLiteDevicesMixin CRUD operations."""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest_asyncio

if TYPE_CHECKING:
    import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.sync_records import DeviceRecord
from surreal_memory.storage.memory_store import InMemoryStorage

# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage_with_brain(tmp_path: pathlib.Path) -> InMemoryStorage:
    """InMemoryStorage with one initialized brain, ready for device tests."""
    storage = InMemoryStorage()

    brain = Brain.create(name="device-test", config=BrainConfig())
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    yield storage

    await storage.close()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRegisterDevice:
    """Test register_device creates and returns a DeviceRecord."""

    async def test_register_device(self, storage_with_brain: InMemoryStorage) -> None:
        """Register a device — verify all returned fields."""
        record = await storage_with_brain.register_device(
            device_id="dev-001", device_name="my-laptop"
        )

        assert isinstance(record, DeviceRecord)
        assert record.device_id == "dev-001"
        assert record.device_name == "my-laptop"
        assert record.last_sync_at is None
        assert record.last_sync_sequence == 0
        # registered_at is populated
        assert record.registered_at is not None

    async def test_register_device_upsert(self, storage_with_brain: InMemoryStorage) -> None:
        """Registering the same device_id twice updates device_name."""
        await storage_with_brain.register_device("dev-001", "old-name")
        await storage_with_brain.register_device("dev-001", "new-name")

        fetched = await storage_with_brain.get_device("dev-001")
        assert fetched is not None
        assert fetched.device_name == "new-name"

    async def test_register_device_without_name(self, storage_with_brain: InMemoryStorage) -> None:
        """register_device with no name uses empty string."""
        record = await storage_with_brain.register_device("dev-no-name")
        assert record.device_name == ""

    async def test_register_device_stores_brain_id(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """Registered DeviceRecord carries the current brain_id."""
        record = await storage_with_brain.register_device("dev-002", "desktop")
        expected_brain_id = storage_with_brain._get_brain_id()
        assert record.brain_id == expected_brain_id


class TestGetDevice:
    """Test get_device retrieves or returns None."""

    async def test_get_device_not_found(self, storage_with_brain: InMemoryStorage) -> None:
        """get_device returns None when device_id is not registered."""
        result = await storage_with_brain.get_device("nonexistent-dev")
        assert result is None

    async def test_get_device_returns_record(self, storage_with_brain: InMemoryStorage) -> None:
        """get_device returns the correct DeviceRecord after registration."""
        await storage_with_brain.register_device("dev-abc", "work-machine")

        record = await storage_with_brain.get_device("dev-abc")
        assert record is not None
        assert record.device_id == "dev-abc"
        assert record.device_name == "work-machine"

    async def test_get_device_returns_device_record_type(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """get_device returns a DeviceRecord instance."""
        await storage_with_brain.register_device("dev-typed", "typed-machine")
        record = await storage_with_brain.get_device("dev-typed")
        assert isinstance(record, DeviceRecord)


class TestListDevices:
    """Test list_devices returns all devices sorted by registered_at."""

    async def test_list_devices_empty(self, storage_with_brain: InMemoryStorage) -> None:
        """list_devices returns empty list when no devices registered."""
        devices = await storage_with_brain.list_devices()
        assert devices == []

    async def test_list_devices_two_devices(self, storage_with_brain: InMemoryStorage) -> None:
        """register 2 devices, list returns 2 sorted by registered_at ASC."""
        await storage_with_brain.register_device("dev-first", "machine-a")
        await storage_with_brain.register_device("dev-second", "machine-b")

        devices = await storage_with_brain.list_devices()
        assert len(devices) == 2
        # Both device IDs present
        ids = {d.device_id for d in devices}
        assert "dev-first" in ids
        assert "dev-second" in ids

    async def test_list_devices_sorted_by_registered_at(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """Devices come back in ascending registered_at order."""
        await storage_with_brain.register_device("dev-a", "alpha")
        await storage_with_brain.register_device("dev-b", "beta")
        await storage_with_brain.register_device("dev-c", "gamma")

        devices = await storage_with_brain.list_devices()
        assert len(devices) == 3
        for i in range(len(devices) - 1):
            assert devices[i].registered_at <= devices[i + 1].registered_at

    async def test_list_devices_returns_device_record_instances(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """All items returned by list_devices are DeviceRecord instances."""
        await storage_with_brain.register_device("dev-x", "x")
        devices = await storage_with_brain.list_devices()
        assert all(isinstance(d, DeviceRecord) for d in devices)


class TestUpdateDeviceSync:
    """Test update_device_sync updates last_sync_at and last_sync_sequence."""

    async def test_update_device_sync(self, storage_with_brain: InMemoryStorage) -> None:
        """After update_device_sync, fetched record reflects new values."""
        await storage_with_brain.register_device("dev-sync", "sync-machine")

        await storage_with_brain.update_device_sync("dev-sync", last_sync_sequence=42)

        record = await storage_with_brain.get_device("dev-sync")
        assert record is not None
        assert record.last_sync_sequence == 42
        assert record.last_sync_at is not None

    async def test_update_device_sync_increases_sequence(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """Updating sync twice carries the latest sequence."""
        await storage_with_brain.register_device("dev-seq", "seq-machine")
        await storage_with_brain.update_device_sync("dev-seq", 10)
        await storage_with_brain.update_device_sync("dev-seq", 99)

        record = await storage_with_brain.get_device("dev-seq")
        assert record is not None
        assert record.last_sync_sequence == 99

    async def test_update_device_sync_sets_last_sync_at(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """update_device_sync sets last_sync_at to a recent timestamp."""
        from surreal_memory.utils.timeutils import utcnow

        await storage_with_brain.register_device("dev-ts", "ts-machine")
        before = utcnow()
        await storage_with_brain.update_device_sync("dev-ts", 1)
        after = utcnow()

        record = await storage_with_brain.get_device("dev-ts")
        assert record is not None
        assert record.last_sync_at is not None
        assert before <= record.last_sync_at <= after


class TestRemoveDevice:
    """Test remove_device deletes a registered device."""

    async def test_remove_device(self, storage_with_brain: InMemoryStorage) -> None:
        """remove_device returns True and the device is no longer found."""
        await storage_with_brain.register_device("dev-remove", "to-remove")

        removed = await storage_with_brain.remove_device("dev-remove")
        assert removed is True

        fetched = await storage_with_brain.get_device("dev-remove")
        assert fetched is None

    async def test_remove_device_not_found_returns_false(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """remove_device returns False if device_id does not exist."""
        result = await storage_with_brain.remove_device("ghost-device")
        assert result is False

    async def test_remove_device_does_not_affect_others(
        self, storage_with_brain: InMemoryStorage
    ) -> None:
        """Removing one device does not remove others."""
        await storage_with_brain.register_device("dev-keep", "keeper")
        await storage_with_brain.register_device("dev-gone", "gonner")

        await storage_with_brain.remove_device("dev-gone")

        assert await storage_with_brain.get_device("dev-keep") is not None
        assert await storage_with_brain.get_device("dev-gone") is None


class TestBrainIsolation:
    """Devices registered in brain A must not be visible from brain B."""

    async def test_brain_isolation(self, tmp_path: pathlib.Path) -> None:
        """Devices in brain A are invisible when brain B is the active context."""
        storage = InMemoryStorage()

        brain_a = Brain.create(name="brain-a", config=BrainConfig())
        brain_b = Brain.create(name="brain-b", config=BrainConfig())
        await storage.save_brain(brain_a)
        await storage.save_brain(brain_b)

        # Register a device under brain A
        storage.set_brain(brain_a.id)
        await storage.register_device("dev-in-a", "a-machine")

        # Switch to brain B — should have no devices
        storage.set_brain(brain_b.id)
        devices_b = await storage.list_devices()
        assert devices_b == []

        # Brain B get_device returns None for brain A's device
        assert await storage.get_device("dev-in-a") is None

        # Switch back to brain A — device still there
        storage.set_brain(brain_a.id)
        devices_a = await storage.list_devices()
        assert len(devices_a) == 1
        assert devices_a[0].device_id == "dev-in-a"

        await storage.close()


class TestSurrealDBDeviceRecords:
    """The SurrealDB backend must speak the same device shape as every other one.

    It used to return the local-identity ``DeviceInfo``, which has no
    ``last_sync_at`` and no ``last_sync_sequence``. Every reader — the dashboard
    sync card, the MCP sync status, the hub routes, the sync engine — reads
    those two fields inside a ``try``, so the missing attributes surfaced as
    "Devices (0)" rather than as an error.
    """

    @staticmethod
    def _store() -> Any:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("brain-1")
        return storage

    async def test_list_devices_returns_device_records_with_sync_fields(self) -> None:
        storage = self._store()
        storage._query = AsyncMock(
            return_value=[
                {
                    "device_id": "dev-001",
                    "brain_id": "brain-1",
                    "device_name": "my-laptop",
                    "registered_at": "2026-08-01T10:00:00Z",
                    "last_sync_at": "2026-08-11T09:30:00Z",
                    "last_sync_sequence": 42,
                }
            ]
        )

        devices = await storage.list_devices()

        assert len(devices) == 1
        device = devices[0]
        assert isinstance(device, DeviceRecord)
        assert device.device_id == "dev-001"
        assert device.brain_id == "brain-1"
        assert device.device_name == "my-laptop"
        assert device.last_sync_sequence == 42
        assert device.last_sync_at is not None
        # The five fields every consumer renders must survive the mapping.
        assert device.registered_at.isoformat()

    async def test_never_synced_device_reads_as_zero_not_as_missing(self) -> None:
        storage = self._store()
        storage._query = AsyncMock(return_value=[{"device_id": "dev-002", "device_name": "fresh"}])

        device = (await storage.list_devices())[0]

        assert device.last_sync_at is None
        assert device.last_sync_sequence == 0
        assert device.brain_id == "brain-1"

    async def test_register_device_returns_a_record_the_hub_route_can_render(self) -> None:
        storage = self._store()
        conn = AsyncMock()
        storage._conn = conn

        record = await storage.register_device("dev-003", "workstation")

        assert isinstance(record, DeviceRecord)
        # hub.py renders exactly these; a DeviceInfo made the route 500.
        assert record.device_id == "dev-003"
        assert record.device_name == "workstation"
        assert record.last_sync_sequence == 0
        assert record.registered_at.isoformat()

    async def test_a_failed_lookup_is_logged_not_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = self._store()
        conn = AsyncMock()
        conn.select.side_effect = RuntimeError("connection reset")
        storage._conn = conn

        with caplog.at_level(logging.WARNING):
            result = await storage.get_device("dev-004")

        assert result is None
        assert any("Device lookup failed" in r.message for r in caplog.records)

    async def test_a_failed_watermark_write_is_logged_not_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = self._store()
        conn = AsyncMock()
        conn.merge.side_effect = RuntimeError("connection reset")
        storage._conn = conn

        with caplog.at_level(logging.WARNING):
            await storage.update_device_sync("dev-005", 7)

        assert any("sync watermark" in r.message for r in caplog.records)

    async def test_a_malformed_watermark_degrades_to_zero_instead_of_raising(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # get_device is read on the sync hot path with no guard of its own, so
        # raising over a bookkeeping value would block the sync itself.
        storage = self._store()
        storage._query = AsyncMock(
            return_value=[{"device_id": "dev-006", "last_sync_sequence": "not-a-number"}]
        )

        with caplog.at_level(logging.WARNING):
            device = (await storage.list_devices())[0]

        assert device.last_sync_sequence == 0
        assert any("non-numeric last_sync_sequence" in r.message for r in caplog.records)


class TestSurrealDBRecordIdBinding:
    """Regression guard for a live bug found while testing U3.

    A brain id is a UUID (e.g. ``00313cb4-61ca-...``) and is deliberately NOT
    folded through ``_to_surreal_id`` (``_safe_brain_id`` keeps its dashes —
    brain ids may also contain '.'). Inlining it as an f-string resource, e.g.
    ``conn.select(f"device:{brain_id}_{did}")``, hits a real SurrealQL parser
    trap: a record-id part starting with a digit is parsed as a *number*, and
    the parser hard-fails at the first non-digit character. Confirmed live
    against the real SurrealDB container — every read/update/delete on the
    device registry raised ``ValidationError: Parse error ... unexpected
    character after number token``, and the same pattern in ``save_brain``'s
    merge/update fallback was equally broken.

    The fix binds the id via ``RecordID(table, id)`` instead, which the SDK
    sends as a query *variable* (CBOR-encoded), never through the SurrealQL
    text parser. These tests assert the TYPE passed to ``conn.select`` /
    ``merge`` / ``delete`` is a ``RecordID``, not a string — a mock accepts
    either silently, so only an explicit type assertion catches a regression
    back to raw f-string interpolation.
    """

    @staticmethod
    def _store() -> Any:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        # A brain id shaped like the real live 'default' brain's UUID — the
        # exact shape that reproduced the parse error (digits then a hex
        # letter inside the first UUID segment).
        storage.set_brain("00313cb4-61ca-4e69-9784-e51431e99ad7")
        return storage

    async def test_get_device_binds_a_record_id_not_a_raw_string(self) -> None:
        from surrealdb import RecordID

        storage = self._store()
        conn = AsyncMock()
        conn.select.return_value = []
        storage._conn = conn

        await storage.get_device("dev-001")

        (resource,), _ = conn.select.call_args
        assert isinstance(resource, RecordID), (
            f"get_device must bind a RecordID, not {type(resource).__name__}"
        )
        assert resource.table_name == "device"

    async def test_update_device_sync_binds_a_record_id_not_a_raw_string(self) -> None:
        from surrealdb import RecordID

        storage = self._store()
        conn = AsyncMock()
        storage._conn = conn

        await storage.update_device_sync("dev-001", 7)

        (resource, _data), _ = conn.merge.call_args
        assert isinstance(resource, RecordID), (
            f"update_device_sync must bind a RecordID, not {type(resource).__name__}"
        )
        assert resource.table_name == "device"

    async def test_remove_device_binds_a_record_id_not_a_raw_string(self) -> None:
        from surrealdb import RecordID

        storage = self._store()
        conn = AsyncMock()
        storage._conn = conn

        await storage.remove_device("dev-001")

        (resource,), _ = conn.delete.call_args
        assert isinstance(resource, RecordID), (
            f"remove_device must bind a RecordID, not {type(resource).__name__}"
        )
        assert resource.table_name == "device"

    async def test_register_device_merge_fallback_binds_a_record_id(self) -> None:
        from surrealdb import RecordID

        storage = self._store()
        conn = AsyncMock()
        conn.insert.side_effect = RuntimeError("already exists")
        storage._conn = conn

        await storage.register_device("dev-001", "laptop")

        (resource, _data), _ = conn.merge.call_args
        assert isinstance(resource, RecordID), (
            f"register_device's merge fallback must bind a RecordID, not {type(resource).__name__}"
        )
        assert resource.table_name == "device"

    async def test_save_brain_merge_fallback_binds_a_record_id(self) -> None:
        import dataclasses

        from surrealdb import RecordID

        from surreal_memory.core.brain import Brain, BrainConfig

        storage = self._store()
        conn = AsyncMock()
        conn.insert.side_effect = RuntimeError("already exists")
        storage._conn = conn

        brain = dataclasses.replace(
            Brain.create(name="default", config=BrainConfig()),
            id="00313cb4-61ca-4e69-9784-e51431e99ad7",
        )

        await storage.save_brain(brain)

        (resource, _data), _ = conn.merge.call_args
        assert isinstance(resource, RecordID), (
            f"save_brain's merge fallback must bind a RecordID, not {type(resource).__name__}"
        )
        assert resource.table_name == "brain"
        assert resource.id == brain.id

    async def test_save_brain_select_and_update_fallback_binds_a_record_id(self) -> None:
        # Deepest fallback: both insert() AND merge() fail, so save_brain falls
        # through to a raw SELECT + per-field UPDATE. This is the OTHER call
        # site that carried the original bug — database-reviewer flagged that
        # the merge-fallback test above (conn.merge auto-succeeding on a bare
        # AsyncMock) never actually exercises this branch.
        import dataclasses

        from surrealdb import RecordID

        from surreal_memory.core.brain import Brain, BrainConfig

        storage = self._store()
        conn = AsyncMock()
        conn.insert.side_effect = RuntimeError("already exists")
        conn.merge.side_effect = RuntimeError("merge also unavailable")
        # _query() calls conn.query(sql, params); shape a response so the
        # SELECT branch sees one row and proceeds to the UPDATE loop.
        conn.query.return_value = [{"id": "00313cb4-61ca-4e69-9784-e51431e99ad7"}]
        storage._conn = conn

        brain = dataclasses.replace(
            Brain.create(name="default", config=BrainConfig()),
            id="00313cb4-61ca-4e69-9784-e51431e99ad7",
        )

        await storage.save_brain(brain)

        select_call, *update_calls = conn.query.call_args_list
        select_sql, select_params = select_call.args
        assert "SELECT" in select_sql
        assert isinstance(select_params["id"], RecordID)
        assert select_params["id"].table_name == "brain"
        assert select_params["id"].id == brain.id

        assert update_calls, "the per-field UPDATE loop must have run"
        for call in update_calls:
            update_sql, update_params = call.args
            assert update_sql.startswith("UPDATE $rid SET ")
            assert isinstance(update_params["rid"], RecordID), (
                "the UPDATE fallback must bind a RecordID via $rid, not a raw "
                f"f-string — got {type(update_params['rid']).__name__}"
            )
            assert update_params["rid"].table_name == "brain"
            assert update_params["rid"].id == brain.id

"""Live-DB regression: the SurrealDB change readers must return what sync consumes.

``SyncEngine`` reads ``change.id`` and calls ``change.changed_at.isoformat()`` on
whatever ``get_unsynced_changes`` and ``get_changes_since`` return, and the
in-memory backend returns ``ChangeEntry``. The SurrealDB backend returned
``SyncChange`` instead — no ``id``, and ``changed_at`` a ``str`` — so both sync
paths raised ``AttributeError`` on the only production backend. Skipped unless
SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.sync_records import ChangeEntry
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name="sync-change-entry-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="sync entry content"))
    yield store
    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await store.close()
    except Exception:
        pass


class TestChangeReadersReturnChangeEntry:
    async def test_get_changes_since_returns_change_entry_instances(self, storage) -> None:  # type: ignore[no-untyped-def]
        """The live twin of the in-memory test that already pins this contract."""
        changes = await storage.get_changes_since(0)

        assert changes, "expected at least one change row for the neuron just written"
        assert all(isinstance(c, ChangeEntry) for c in changes), (
            f"got {[type(c).__name__ for c in changes]}"
        )

    async def test_get_unsynced_changes_gives_sync_what_it_reads(self, storage) -> None:  # type: ignore[no-untyped-def]
        """SyncEngine reads `.id` and calls `.changed_at.isoformat()` — both must work."""
        changes = await storage.get_unsynced_changes()

        assert changes, "expected at least one unsynced change row"
        change = changes[0]
        assert isinstance(change.id, int), (
            f"sync reads change.id as the sequence; got {change.id!r}"
        )
        assert isinstance(change.changed_at, datetime), (
            f"sync calls change.changed_at.isoformat(); got {type(change.changed_at).__name__}"
        )
        # The two attribute accesses SyncEngine actually performs, run for real.
        assert change.changed_at.isoformat()
        assert change.brain_id == storage.current_brain_id
        assert change.synced is False

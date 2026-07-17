"""Live-DB regression for the SurrealDB RecordID-vs-string id comparison fix (UB1).

get_source / update_source / delete_source compared `id = $string`, which never
matches a `RecordID`-typed id on SurrealDB 3.2.0, so they always returned
None / False. Fixed by comparing against `type::record("source", $sid)`.
Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.source import Source, SourceStatus
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
    brain = Brain.create(name="ub1-recordid-fix-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await store.close()
    except Exception:
        pass


class TestSourceRecordIdFix:
    async def test_get_source_finds_row_and_trust_round_trips(self, storage) -> None:  # type: ignore[no-untyped-def]
        src = Source.create(brain_id=storage._get_brain_id(), name="doc.pdf", trust=0.8)
        await storage.add_source(src)
        got = await storage.get_source(src.id)
        assert got is not None  # UB1: previously always None (id=$string never matched RecordID)
        assert got.trust == 0.8
        assert got.name == "doc.pdf"

    async def test_update_source_matches(self, storage) -> None:  # type: ignore[no-untyped-def]
        src = Source.create(brain_id=storage._get_brain_id(), name="doc2.pdf")
        await storage.add_source(src)
        updated = await storage.update_source(src.id, status="superseded")
        assert updated is True  # UB1: previously False
        got = await storage.get_source(src.id)
        assert got is not None
        assert got.status == SourceStatus.SUPERSEDED

    async def test_delete_source_matches(self, storage) -> None:  # type: ignore[no-untyped-def]
        src = Source.create(brain_id=storage._get_brain_id(), name="doc3.pdf")
        await storage.add_source(src)
        deleted = await storage.delete_source(src.id)
        assert deleted is True  # UB1: previously False
        assert await storage.get_source(src.id) is None

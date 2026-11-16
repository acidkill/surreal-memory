"""Live regressions for typed entity-ref record IDs in SurrealDB."""

from __future__ import annotations

import ipaddress
import os
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_entity_ref_record_ids_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        return hostname == "localhost" or bool(
            hostname and ipaddress.ip_address(hostname).is_loopback
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires an explicit loopback URL for disposable SurrealDB 3.2.x",
    ),
]


@pytest_asyncio.fixture
async def store():
    database = "it_" + uuid.uuid4().hex[:12]
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database=database,
    )
    try:
        await storage.initialize()
        brain = Brain.create(name="entity-ref-record-ids-it")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        yield storage
    finally:
        try:
            await SurrealDBStorage._query_response(storage, f"REMOVE DATABASE `{database}`")
        finally:
            await storage.close()


async def _create_ref(
    store: SurrealDBStorage,
    *,
    record_id: str,
    brain_id: str,
    entity_text: str,
    fiber_id: str,
    created_at: datetime,
    promoted: bool = False,
) -> None:
    await store._query(
        "CREATE type::record('entity_refs', $rid)"
        " SET brain_id = $bid, entity_text = $entity, fiber_id = $fiber,"
        " created_at = $created, promoted = $promoted",
        rid=record_id,
        bid=brain_id,
        entity=entity_text,
        fiber=fiber_id,
        created=created_at,
        promoted=promoted,
    )


async def _fiber_ids(store: SurrealDBStorage, brain_id: str, entity_text: str) -> set[str]:
    rows = await store._query(
        "SELECT fiber_id FROM entity_refs WHERE brain_id = $bid AND entity_text = $entity",
        bid=brain_id,
        entity=entity_text,
    )
    return {str(row["fiber_id"]) for row in rows}


@pytest.mark.asyncio
async def test_prune_old_entity_refs_uses_typed_ids_and_keeps_scope_and_filters(
    store: SurrealDBStorage,
) -> None:
    active_brain_id = store._get_brain_id()
    foreign_brain = Brain.create(name="entity-ref-record-ids-foreign")
    await store.save_brain(foreign_brain)

    old = datetime.now(UTC) - timedelta(days=30)
    fresh = datetime.now(UTC)
    entity = "Łódź-retention"
    for record_id, fiber_id in [
        ("ascii-old", "fiber-ascii-old"),
        ("123456", "fiber-numeric-old"),
        ("default_łódź_old_123", "fiber-unicode-old"),
    ]:
        await _create_ref(
            store,
            record_id=record_id,
            brain_id=active_brain_id,
            entity_text=entity,
            fiber_id=fiber_id,
            created_at=old,
        )
    await _create_ref(
        store,
        record_id="promoted-old",
        brain_id=active_brain_id,
        entity_text=entity,
        fiber_id="fiber-promoted-old",
        created_at=old,
        promoted=True,
    )
    await _create_ref(
        store,
        record_id="fresh-oldname",
        brain_id=active_brain_id,
        entity_text=entity,
        fiber_id="fiber-fresh",
        created_at=fresh,
    )
    await _create_ref(
        store,
        record_id="foreign-old",
        brain_id=foreign_brain.id,
        entity_text=entity,
        fiber_id="fiber-foreign-old",
        created_at=old,
    )

    assert await store.prune_old_entity_refs(max_age_days=1) == 3
    assert await store.prune_old_entity_refs(max_age_days=1) == 0
    assert await _fiber_ids(store, active_brain_id, entity) == {
        "fiber-promoted-old",
        "fiber-fresh",
    }
    assert await _fiber_ids(store, foreign_brain.id, entity) == {"fiber-foreign-old"}


@pytest.mark.asyncio
async def test_mark_entity_refs_promoted_uses_typed_ids_and_keeps_scope(
    store: SurrealDBStorage,
) -> None:
    active_brain_id = store._get_brain_id()
    foreign_brain = Brain.create(name="entity-ref-promote-foreign")
    await store.save_brain(foreign_brain)
    entity = "Żółć promotion"
    now = datetime.now(UTC)

    for record_id, fiber_id in [
        ("ascii", "fiber-ascii"),
        ("123456", "fiber-numeric"),
        ("Łódź", "fiber-unicode"),
    ]:
        await _create_ref(
            store,
            record_id=record_id,
            brain_id=active_brain_id,
            entity_text=entity,
            fiber_id=fiber_id,
            created_at=now,
        )
    await _create_ref(
        store,
        record_id="already-promoted",
        brain_id=active_brain_id,
        entity_text=entity,
        fiber_id="fiber-already-promoted",
        created_at=now,
        promoted=True,
    )
    await _create_ref(
        store,
        record_id="foreign-unpromoted",
        brain_id=foreign_brain.id,
        entity_text=entity,
        fiber_id="fiber-foreign",
        created_at=now,
    )

    assert await store.mark_entity_refs_promoted(entity) == 3
    assert await store.mark_entity_refs_promoted(entity) == 0
    active_rows = await store._query(
        "SELECT fiber_id, promoted FROM entity_refs"
        " WHERE brain_id = $bid AND entity_text = $entity",
        bid=active_brain_id,
        entity=entity,
    )
    assert {str(row["fiber_id"]): bool(row["promoted"]) for row in active_rows} == {
        "fiber-ascii": True,
        "fiber-numeric": True,
        "fiber-unicode": True,
        "fiber-already-promoted": True,
    }
    foreign_rows = await store._query(
        "SELECT fiber_id, promoted FROM entity_refs"
        " WHERE brain_id = $bid AND entity_text = $entity",
        bid=foreign_brain.id,
        entity=entity,
    )
    assert [(str(row["fiber_id"]), bool(row["promoted"])) for row in foreign_rows] == [
        ("fiber-foreign", False)
    ]

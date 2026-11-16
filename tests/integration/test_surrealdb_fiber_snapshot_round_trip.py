"""Live-DB pinning tests: a fiber must survive export_brain -> import_brain intact.

The SurrealDB snapshot used to carry seven fiber fields — id, neuron_ids, synapse_ids,
anchor_neuron_id, pathway, conductivity, salience — and ``import_brain`` rebuilt the
``Fiber`` from exactly those seven. Everything else was reset to its dataclass default
on the way through, silently: measured on the live brain, one round trip dropped 788
summaries, 2174 auto_tags, 1777 agent_tags, 2010 time_starts, 1910 frequencies and
2299 metadata dicts. Nothing errored; the fibers simply came out blank.

Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]

# Naive UTC: `_parse_datetime` strips tzinfo on the way out of the store
# ("naive for consistency across codebase"), so a tz-aware seed could never
# compare equal and the test would be pinning the convention, not the round trip.
_T0 = datetime(2026, 3, 4, 5, 6, 7)
_VECTOR = [0.6, 0.8, 0.0, 0.0]


def _new_store(database: str) -> SurrealDBStorage:
    return SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database=database,
        embedding_dim=4,
    )


@pytest_asyncio.fixture
async def store():
    database = "it_" + uuid.uuid4().hex[:12]
    storage = _new_store(database)
    try:
        await storage.initialize()
        brain = Brain.create(name="fiber-snapshot-src")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        yield storage
    finally:
        try:
            await SurrealDBStorage._query_response(storage, f"REMOVE DATABASE `{database}`")
        finally:
            await storage.close()


@pytest_asyncio.fixture
async def target():
    """The import side lives in its own DATABASE.

    ``_to_surreal_id`` folds a fiber id to ``fiber:<id>`` with no brain component, so
    importing a snapshot into a second brain of the same database collides with the
    rows it was exported from and ``import_brain`` skips them — ``store.py`` says so in
    its own comment. A transplant between databases is both the realistic shape and the
    only one that exercises the writes.
    """
    database = "it_" + uuid.uuid4().hex[:12]
    storage = _new_store(database)
    try:
        await storage.initialize()
        yield storage
    finally:
        try:
            await SurrealDBStorage._query_response(storage, f"REMOVE DATABASE `{database}`")
        finally:
            await storage.close()


async def _seed_rich_fiber(store: SurrealDBStorage) -> Fiber:
    """A fiber with every persisted field set to something distinguishable from its default."""
    anchor = Neuron.create(type=NeuronType.CONCEPT, content="snapshot anchor")
    await store.add_neuron(anchor)
    fiber = Fiber(
        id=str(uuid.uuid4()),
        neuron_ids={anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=anchor.id,
        pathway=[anchor.id],
        conductivity=0.75,
        salience=0.42,
        coherence=0.31,
        frequency=7,
        summary="a summary that must survive the trip",
        essence="an essence that must survive too",
        auto_tags={"kb", "auto-one"},
        agent_tags={"agent-one"},
        metadata={"source": "pinning-test", "n": 3},
        compression_tier=2,
        pinned=True,
        last_conducted=_T0,
        time_start=_T0 - timedelta(days=2),
        time_end=_T0 + timedelta(days=1),
        created_at=_T0 - timedelta(days=10),
    )
    await store.add_fiber(fiber)
    return fiber


async def test_every_persisted_fiber_field_survives_export_then_import(
    store: SurrealDBStorage, target: SurrealDBStorage
) -> None:
    original = await _seed_rich_fiber(store)
    snapshot = await store.export_brain(store._get_brain_id())

    await target.import_brain(snapshot, target_brain_id="imported-" + uuid.uuid4().hex[:8])

    loaded = await target.get_fiber(original.id)
    assert loaded is not None

    # Compare read-to-read: the source store's own view of the fiber is the honest
    # reference for "the snapshot preserved it", since a comparison against the
    # in-memory object would also be asserting the store's storage conventions.
    source_view = await store.get_fiber(original.id)
    assert source_view is not None
    for field in (
        "summary",
        "essence",
        "auto_tags",
        "agent_tags",
        "metadata",
        "frequency",
        "coherence",
        "conductivity",
        "salience",
        "compression_tier",
        "pinned",
        "time_start",
        "time_end",
        "last_conducted",
        "created_at",
        "anchor_neuron_id",
        "neuron_ids",
        "pathway",
    ):
        assert getattr(loaded, field) == getattr(source_view, field), field

    # ...and the values are the ones seeded, not defaults that happen to agree.
    assert loaded.summary == original.summary
    assert loaded.auto_tags == original.auto_tags
    assert loaded.agent_tags == original.agent_tags
    assert loaded.metadata == original.metadata
    assert loaded.frequency == original.frequency
    assert loaded.pinned is True
    assert loaded.compression_tier == 2


async def test_storage_only_fiber_vector_survives_export_then_import(
    store: SurrealDBStorage, target: SurrealDBStorage
) -> None:
    original = await _seed_rich_fiber(store)
    await store.update_fiber_embeddings([(original.id, _VECTOR)])

    snapshot = await store.export_brain(store._get_brain_id())
    (fiber_record,) = [f for f in snapshot.fibers if f["id"] == original.id]
    assert fiber_record["fiber_vec"] == _VECTOR

    await target.import_brain(snapshot, target_brain_id="vector-restore-" + uuid.uuid4().hex[:8])
    matches = await target.find_fibers_by_embedding(_VECTOR, limit=1)
    assert len(matches) == 1
    assert matches[0][0].id == original.id
    assert matches[0][1] == pytest.approx(1.0)


async def test_snapshot_carries_the_fields_rather_than_the_reader_guessing(
    store: SurrealDBStorage,
) -> None:
    """The snapshot dict itself must hold them — an import cannot restore what was never written."""
    original = await _seed_rich_fiber(store)
    snapshot = await store.export_brain(store._get_brain_id())

    (fd,) = [f for f in snapshot.fibers if f["id"] == original.id]
    for key in (
        "summary",
        "essence",
        "auto_tags",
        "agent_tags",
        "metadata",
        "fiber_vec",
        "frequency",
        "coherence",
        "compression_tier",
        "pinned",
        "time_start",
        "time_end",
        "last_conducted",
        "created_at",
    ):
        assert key in fd, f"snapshot drops fiber field {key!r}"
    assert fd["summary"] == original.summary
    assert set(fd["auto_tags"]) == original.auto_tags


async def test_a_pre_round_trip_snapshot_still_imports(
    store: SurrealDBStorage, target: SurrealDBStorage
) -> None:
    """Backward compatibility: a seven-field snapshot must import, not raise."""
    original = await _seed_rich_fiber(store)
    snapshot = await store.export_brain(store._get_brain_id())
    legacy_keys = {
        "id",
        "neuron_ids",
        "synapse_ids",
        "anchor_neuron_id",
        "pathway",
        "conductivity",
        "salience",
    }
    legacy = dataclasses.replace(
        snapshot,
        fibers=[{k: v for k, v in f.items() if k in legacy_keys} for f in snapshot.fibers],
    )

    await target.import_brain(legacy, target_brain_id="legacy-" + uuid.uuid4().hex[:8])

    loaded = await target.get_fiber(original.id)
    assert loaded is not None
    assert loaded.salience == pytest.approx(original.salience)
    assert loaded.summary is None  # the field simply was not in that snapshot


async def test_fixture_failure_removes_only_its_disposable_database(target):
    """A failed fixture body leaves the independently scoped control database intact."""
    fixture = store.__wrapped__()
    temporary = await anext(fixture)
    temporary_database = temporary._database
    control_database = target._database
    assert temporary_database != control_database
    before = await SurrealDBStorage._query_response(target, "INFO FOR NS")
    assert temporary_database in before["databases"]
    assert control_database in before["databases"]
    with pytest.raises(RuntimeError, match="synthetic body"):
        await fixture.athrow(RuntimeError("synthetic body"))
    after = await SurrealDBStorage._query_response(target, "INFO FOR NS")
    assert temporary_database not in after["databases"]
    assert control_database in after["databases"]

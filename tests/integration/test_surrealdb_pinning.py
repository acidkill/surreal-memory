"""Live integration tests for SurrealDB pinning, training files and graph density.

These run against a REAL SurrealDB >= 3.2.0 (skipped unless SURREALDB_URL is set).

Every operation exercised here used to exist on ``SQLiteStorage`` alone, guarded
at each call site by ``hasattr``, so on SurrealDB decay and prune deleted pinned
memories, ``smem_pin`` refused every action, ``smem train`` re-encoded its whole
corpus each run, and ``activation_strategy="auto"`` never left classic BFS. The
unit suite covers the contract on SQLite and the in-memory backend; only a live
server can check the SurrealQL itself, which is what this module is for. Two
traps in particular need a real database to catch:

* record-id set membership — ``id IN ['fiber:x']`` compares a ``RecordID``
  against strings and silently matches nothing, so pinning must go through an
  interpolated ``FROM`` record list;
* ``fiber``'s record id carries the underscore form of the uuid while
  ``typed_memory.fiber_id`` keeps the dash form, so a naive join for
  type/priority silently returns nothing.

Run (real-db-test-runner drives this):
    SURREALDB_URL=http://localhost:<port> SURREALDB_USER=root SURREALDB_PASS=... \
        uv run --extra dev --extra server --extra surrealdb \
        pytest tests/integration/test_surrealdb_pinning.py -m integration -q
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
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


@pytest_asyncio.fixture
async def store():
    """A fresh, initialised store scoped to its own brain."""
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="pin-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _fiber_with_neuron(store: SurrealDBStorage, content: str, **kwargs) -> Fiber:
    neuron = Neuron.create(type=NeuronType.ENTITY, content=content)
    await store.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        **kwargs,
    )
    await store.add_fiber(fiber)
    return fiber


class TestPinning:
    async def test_pin_and_unpin_round_trip(self, store: SurrealDBStorage) -> None:
        fiber = await _fiber_with_neuron(store, "kb fact")
        assert (await store.get_fiber(fiber.id)).pinned is False

        assert await store.pin_fibers([fiber.id], pinned=True) == 1
        assert (await store.get_fiber(fiber.id)).pinned is True

        assert await store.pin_fibers([fiber.id], pinned=False) == 1
        assert (await store.get_fiber(fiber.id)).pinned is False

    async def test_pin_is_scoped_to_the_current_brain(self, store: SurrealDBStorage) -> None:
        """A fiber id from another brain must not be pinnable."""
        fiber = await _fiber_with_neuron(store, "brain one")

        other = Brain.create(name="pin-it-other")
        await store.save_brain(other)
        store.set_brain(other.id)

        assert await store.pin_fibers([fiber.id], pinned=True) == 0
        assert await store.get_pinned_neuron_ids() == set()

    async def test_pin_unknown_id_creates_nothing(self, store: SurrealDBStorage) -> None:
        """UPDATE (not UPSERT) must not conjure a fiber row."""
        assert await store.pin_fibers([str(uuid.uuid4())], pinned=True) == 0
        assert await store.get_fibers(limit=100) == []

    async def test_get_pinned_neuron_ids(self, store: SurrealDBStorage) -> None:
        pinned = await _fiber_with_neuron(store, "pinned")
        loose = await _fiber_with_neuron(store, "not pinned")
        await store.pin_fibers([pinned.id], pinned=True)

        ids = await store.get_pinned_neuron_ids()
        assert ids == set(pinned.neuron_ids)
        assert not (ids & set(loose.neuron_ids))

    async def test_pin_many_fibers_at_once(self, store: SurrealDBStorage) -> None:
        """The FROM record-list form has to handle a batch, not just one id."""
        fibers = [await _fiber_with_neuron(store, f"fact {i}") for i in range(5)]

        assert await store.pin_fibers([f.id for f in fibers], pinned=True) == 5
        expected = {nid for f in fibers for nid in f.neuron_ids}
        assert await store.get_pinned_neuron_ids() == expected


class TestListPinnedFibers:
    async def test_lists_only_pinned(self, store: SurrealDBStorage) -> None:
        pinned = await _fiber_with_neuron(store, "kb", summary="a pinned summary")
        await _fiber_with_neuron(store, "ordinary", summary="not pinned")
        await store.pin_fibers([pinned.id], pinned=True)

        listed = await store.list_pinned_fibers()
        assert len(listed) == 1
        assert listed[0]["summary"] == "a pinned summary"
        assert listed[0]["created_at"]

    async def test_type_and_priority_come_from_typed_memory(self, store: SurrealDBStorage) -> None:
        """Regression: fiber ids carry the underscore uuid, typed_memory.fiber_id
        the dash form, so joining on that field returns nothing at all.
        """
        fiber = await _fiber_with_neuron(store, "decision", summary="chose Postgres")
        await store.add_typed_memory(
            TypedMemory.create(
                fiber_id=fiber.id,
                memory_type=MemoryType.DECISION,
                priority=Priority.HIGH,
                tags={"db", "architecture"},
            )
        )
        await store.pin_fibers([fiber.id], pinned=True)

        entry = (await store.list_pinned_fibers())[0]
        assert entry["type"] == MemoryType.DECISION.value
        assert entry["priority"] == int(Priority.HIGH)
        assert sorted(entry["tags"]) == ["architecture", "db"]

    async def test_defaults_without_a_typed_memory(self, store: SurrealDBStorage) -> None:
        fiber = await _fiber_with_neuron(store, "bare", summary="no typed memory")
        await store.pin_fibers([fiber.id], pinned=True)

        entry = (await store.list_pinned_fibers())[0]
        assert entry["type"] == "unknown"
        assert entry["priority"] == 5

    async def test_empty_brain(self, store: SurrealDBStorage) -> None:
        assert await store.list_pinned_fibers() == []


class TestGraphDensity:
    async def test_empty_brain_is_zero(self, store: SurrealDBStorage) -> None:
        assert await store.get_graph_density() == 0.0

    async def test_counts_synapses_per_neuron(self, store: SurrealDBStorage) -> None:
        neurons = []
        for i in range(4):
            n = Neuron.create(type=NeuronType.CONCEPT, content=f"n{i}")
            await store.add_neuron(n)
            neurons.append(n)

        for src, dst in ((0, 1), (1, 2)):
            await store.add_synapse(
                Synapse.create(
                    source_id=neurons[src].id,
                    target_id=neurons[dst].id,
                    type=SynapseType.RELATED_TO,
                )
            )

        assert await store.get_graph_density() == pytest.approx(2 / 4)


class TestTrainingFiles:
    async def test_upsert_and_lookup(self, store: SurrealDBStorage) -> None:
        record_id = await store.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=10,
            chunks_completed=10,
            status="completed",
            domain_tag="react",
        )
        assert record_id

        record = await store.get_training_file_by_hash("abc123")
        assert record is not None
        assert record["status"] == "completed"
        assert record["file_path"] == "/docs/test.md"
        assert record["id"] == record_id

        assert await store.get_training_file_by_hash("nope") is None

    async def test_upsert_updates_in_place(self, store: SurrealDBStorage) -> None:
        first = await store.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=5,
            chunks_completed=3,
            status="in_progress",
        )
        second = await store.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=5,
            chunks_completed=5,
            status="completed",
        )
        assert first == second

        record = await store.get_training_file_by_hash("abc123")
        assert record["chunks_completed"] == 5
        assert record["status"] == "completed"
        assert record["trained_at"]
        assert await store.get_training_stats() == {
            "total_files": 1,
            "completed": 1,
            "in_progress": 0,
            "failed": 0,
            "total_chunks": 5,
        }

    async def test_update_progress_supports_resume(self, store: SurrealDBStorage) -> None:
        record_id = await store.upsert_training_file(
            file_hash="resume-me",
            file_path="/docs/big.md",
            file_size=4096,
            chunks_total=10,
            status="in_progress",
        )

        await store.update_training_file_progress(record_id, chunks_completed=4)
        record = await store.get_training_file_by_hash("resume-me")
        assert record["chunks_completed"] == 4
        assert record["status"] == "in_progress"
        assert not record["trained_at"]

        await store.update_training_file_progress(
            record_id, chunks_completed=10, status="completed"
        )
        record = await store.get_training_file_by_hash("resume-me")
        assert record["chunks_completed"] == 10
        assert record["status"] == "completed"
        assert record["trained_at"]

    async def test_stats_across_statuses(self, store: SurrealDBStorage) -> None:
        await store.upsert_training_file(
            file_hash="h1",
            file_path="a.md",
            file_size=100,
            chunks_total=5,
            chunks_completed=5,
            status="completed",
        )
        await store.upsert_training_file(
            file_hash="h2",
            file_path="b.md",
            file_size=200,
            chunks_total=3,
            chunks_completed=1,
            status="in_progress",
        )
        await store.upsert_training_file(
            file_hash="h3",
            file_path="c.md",
            file_size=300,
            chunks_total=4,
            chunks_completed=0,
            status="failed",
        )

        assert await store.get_training_stats() == {
            "total_files": 3,
            "completed": 1,
            "in_progress": 1,
            "failed": 1,
            "total_chunks": 6,
        }

    async def test_stats_empty_brain(self, store: SurrealDBStorage) -> None:
        assert await store.get_training_stats() == {
            "total_files": 0,
            "completed": 0,
            "in_progress": 0,
            "failed": 0,
            "total_chunks": 0,
        }

    async def test_records_are_scoped_to_the_brain(self, store: SurrealDBStorage) -> None:
        await store.upsert_training_file(
            file_hash="shared-hash", file_path="a.md", file_size=1, status="completed"
        )

        other = Brain.create(name="pin-it-other")
        await store.save_brain(other)
        store.set_brain(other.id)

        assert await store.get_training_file_by_hash("shared-hash") is None
        assert (await store.get_training_stats())["total_files"] == 0

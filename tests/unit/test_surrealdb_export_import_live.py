"""Live-DB regression: SurrealDB snapshots must carry typed memories and projects.

Typed memories and projects live in their own SurrealDB tables rather than as
neurons, and export_brain used to skip both. Every caller of the snapshot —
`smem export`/`smem import`, `smem brain export/import`, and multi-device sync —
therefore moved the graph while silently dropping memory type, priority, tags,
trust score, expiry, tier and project scoping. Skipped unless SURREALDB_URL
points at a running SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.project import Project
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)

_SOURCE_BRAIN = "snapshot-roundtrip-live"
_TARGET_BRAIN = "snapshot-roundtrip-live-target"


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name=_SOURCE_BRAIN, brain_id=_SOURCE_BRAIN)
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    for leftover in (_TARGET_BRAIN, brain.id):
        try:
            await cleanup_live_brains(store, own_brain_id=leftover)
        except Exception:
            pass
    try:
        await store.close()
    except Exception:
        pass


class TestSnapshotRoundTripLive:
    async def test_typed_memories_and_projects_survive_export_import(self, storage) -> None:  # type: ignore[no-untyped-def]
        project = Project.create(name="live-project", description="snapshot scope")
        await storage.add_project(project)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="live snapshot content")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="live snapshot fiber",
        )
        await storage.add_fiber(fiber)
        await storage.add_typed_memory(
            TypedMemory.create(
                fiber_id=fiber.id,
                memory_type=MemoryType.DECISION,
                priority=Priority.HIGH,
                project_id=project.id,
                tags={"alpha"},
                trust_score=0.9,
                tier="hot",
            )
        )

        snapshot = await storage.export_brain(_SOURCE_BRAIN)

        assert len(snapshot.metadata.get("typed_memories", [])) == 1
        assert len(snapshot.metadata.get("projects", [])) == 1

        await storage.import_brain(snapshot, _TARGET_BRAIN)
        storage.set_brain(_TARGET_BRAIN)

        restored = await storage.get_typed_memory(fiber.id)
        assert restored is not None
        assert restored.memory_type == MemoryType.DECISION
        assert restored.priority == Priority.HIGH
        assert restored.tags == frozenset({"alpha"})
        assert restored.trust_score == 0.9
        assert restored.tier == "hot"
        assert restored.project_id == project.id

        restored_projects = await storage.list_projects()
        assert [p.name for p in restored_projects] == ["live-project"]

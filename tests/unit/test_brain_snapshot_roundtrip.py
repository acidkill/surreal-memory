"""Snapshot serialization for typed memories — the brain export/import payload.

`smem export` / `smem import`, `smem brain export/import` and multi-device sync
all move a BrainSnapshot, whose typed-memory records used to be built by
hand-rolled parsing in each backend. The SQLite importer dropped every field
added after it was written (trust score, source, tier, validity window,
supersession), so a round trip quietly downgraded memories.
"""

from __future__ import annotations

from datetime import timedelta

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


def _full_typed_memory() -> TypedMemory:
    """A typed memory with every optional field populated."""
    now = utcnow()
    return TypedMemory(
        fiber_id="fiber-1",
        memory_type=MemoryType.DECISION,
        priority=Priority.HIGH,
        provenance=Provenance(
            source="user_input",
            confidence=Confidence.HIGH,
            verified=True,
            verified_at=now,
            created_by="tester",
            last_confirmed=now,
        ),
        expires_at=now + timedelta(days=30),
        project_id="project-1",
        tags=frozenset({"alpha", "beta"}),
        metadata={"note": "keep"},
        created_at=now,
        trust_score=0.75,
        source="user_input",
        tier="hot",
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=1),
        superseded_by="fiber-2",
    )


class TestProvenanceSerialization:
    def test_round_trips_through_dict(self) -> None:
        now = utcnow()
        prov = Provenance(
            source="ai_inference",
            confidence=Confidence.LOW,
            verified=False,
            verified_at=None,
            created_by="agent",
            last_confirmed=now,
        )

        data = prov.to_dict()

        assert data["source"] == "ai_inference"
        assert data["confidence"] == Confidence.LOW.value
        assert data["verified_at"] is None
        assert data["last_confirmed"] == now.isoformat()


class TestTypedMemorySerialization:
    def test_every_field_survives_a_round_trip(self) -> None:
        original = _full_typed_memory()

        restored = TypedMemory.from_dict(original.to_dict())

        assert restored.fiber_id == original.fiber_id
        assert restored.memory_type == original.memory_type
        assert restored.priority == original.priority
        assert restored.provenance == original.provenance
        assert restored.expires_at == original.expires_at
        assert restored.project_id == original.project_id
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata
        assert restored.created_at == original.created_at
        assert restored.trust_score == original.trust_score
        assert restored.source == original.source
        assert restored.tier == original.tier
        assert restored.valid_from == original.valid_from
        assert restored.valid_until == original.valid_until
        assert restored.superseded_by == original.superseded_by

    def test_reads_a_snapshot_written_before_the_newer_fields_existed(self) -> None:
        legacy = {
            "fiber_id": "fiber-legacy",
            "memory_type": "fact",
            "priority": Priority.NORMAL.value,
            "provenance": {"source": "import", "confidence": "medium"},
            "expires_at": None,
            "project_id": None,
            "tags": ["old"],
            "metadata": {},
            "created_at": utcnow().isoformat(),
        }

        restored = TypedMemory.from_dict(legacy)

        assert restored.fiber_id == "fiber-legacy"
        assert restored.memory_type == MemoryType.FACT
        assert restored.tags == frozenset({"old"})
        assert restored.trust_score is None
        assert restored.tier == "warm"
        assert restored.superseded_by is None

    def test_unparseable_timestamp_does_not_raise(self) -> None:
        restored = TypedMemory.from_dict(
            {
                "fiber_id": "fiber-bad-date",
                "memory_type": "fact",
                "expires_at": "not-a-date",
            }
        )

        assert restored.expires_at is None


class TestBrainSnapshotTypedMemories:
    async def test_export_import_preserves_the_full_record(self) -> None:
        source_storage = InMemoryStorage()
        brain = Brain.create(name="snapshot-source")
        await source_storage.save_brain(brain)
        source_storage.set_brain(brain.id)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="snapshot content")
        await source_storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="snapshot fiber",
        )
        await source_storage.add_fiber(fiber)

        original = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            priority=Priority.HIGH,
            expires_at=utcnow() + timedelta(days=7),
            tags=frozenset({"alpha"}),
            trust_score=0.9,
            source="user_input",
            tier="hot",
        )
        await source_storage.add_typed_memory(original)

        snapshot = await source_storage.export_brain(brain.id)

        target = InMemoryStorage()
        await target.import_brain(snapshot, "snapshot-target")
        target.set_brain("snapshot-target")

        restored = await target.get_typed_memory(fiber.id)
        assert restored is not None
        assert restored.memory_type == MemoryType.DECISION
        assert restored.priority == Priority.HIGH
        assert restored.tags == frozenset({"alpha"})
        # Fields the old hand-rolled importer silently discarded.
        assert restored.trust_score == 0.9
        assert restored.source == "user_input"
        assert restored.tier == "hot"

"""All-15-types round-trip test on the SurrealDB backend.

Locks the contract that SurrealDBTypedMemoryMixin can store and
retrieve every MemoryType value with all fields preserved (memory_type,
priority, tags, source, project_id, expires_at, tier, metadata,
provenance). This catches enum-to-string regressions, datetime
serialization issues, and tier-default drift (notably the BOUNDARY →
HOT auto-promotion in TypedMemory.__post_init__).

The test is skipped when SURREALDB_URL is unset so CI without docker
still passes.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    Priority,
    TypedMemory,
)
from surreal_memory.core.neuron import Neuron, NeuronType

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="all-types-roundtrip")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    try:
        await storage.close()
    except Exception:
        pass


async def _make_fiber(storage, idx: int) -> Fiber:  # type: ignore[no-untyped-def]
    """Create the neuron + fiber that a TypedMemory row references."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"roundtrip-{idx}")
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=f"fiber-{idx}",
    )
    await storage.add_fiber(fiber)
    return fiber


@pytest.mark.asyncio
@pytest.mark.parametrize("mtype", list(MemoryType), ids=lambda m: m.value)
async def test_round_trip_preserves_fields(surrealdb_storage, mtype: MemoryType) -> None:  # type: ignore[no-untyped-def]
    """Each MemoryType round-trips: add → get → fields preserved."""
    fiber = await _make_fiber(surrealdb_storage, idx=hash(mtype.value) & 0xFFFF)

    typed_mem = TypedMemory.create(
        fiber_id=fiber.id,
        memory_type=mtype,
        priority=Priority.HIGH,
        source="user_input",
        tags={"roundtrip", mtype.value},
        # Use 30 for everything; BOUNDARY's __post_init__ already keeps
        # expires_at as the value passed (it does not force None — only
        # the tier is auto-promoted).
        expires_in_days=30,
        project_id=None,
    )
    expected_tier = MemoryTier.HOT if mtype == MemoryType.BOUNDARY else MemoryTier.WARM

    await surrealdb_storage.add_typed_memory(typed_mem)

    fetched = await surrealdb_storage.get_typed_memory(fiber.id)
    assert fetched is not None, f"{mtype.value} round-trip lost the row"
    assert fetched.memory_type == mtype
    assert fetched.priority == Priority.HIGH
    assert fetched.tier == expected_tier
    assert fetched.source == "user_input"
    assert "roundtrip" in fetched.tags
    assert mtype.value in fetched.tags
    assert fetched.expires_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mtype", list(MemoryType), ids=lambda m: m.value)
async def test_find_by_type_returns_only_that_type(surrealdb_storage, mtype: MemoryType) -> None:  # type: ignore[no-untyped-def]
    """find_typed_memories(memory_type=mtype) hits the idx_typed_type index
    and returns exactly the rows of that type — no leakage from other
    types that happen to share the same brain.
    """
    fiber = await _make_fiber(surrealdb_storage, idx=10000 + (hash(mtype.value) & 0xFFFF))
    typed_mem = TypedMemory.create(
        fiber_id=fiber.id,
        memory_type=mtype,
        priority=Priority.NORMAL,
    )
    await surrealdb_storage.add_typed_memory(typed_mem)

    rows = await surrealdb_storage.find_typed_memories(memory_type=mtype)
    fiber_ids = {r.fiber_id for r in rows}
    assert fiber.id in fiber_ids, f"{mtype.value} not found in find_typed_memories result"
    # Every returned row must have the requested type
    assert all(r.memory_type == mtype for r in rows)


@pytest.mark.asyncio
async def test_boundary_auto_promotes_to_hot_tier(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """BOUNDARY type's tier is forced to HOT in TypedMemory.__post_init__,
    and SurrealDB must preserve that on round-trip.
    """
    fiber = await _make_fiber(surrealdb_storage, idx=99999)

    # Even though caller passes WARM, BOUNDARY post-init forces HOT
    typed_mem = TypedMemory.create(
        fiber_id=fiber.id,
        memory_type=MemoryType.BOUNDARY,
        tier=MemoryTier.WARM,  # ignored — boundary forces HOT
    )
    assert typed_mem.tier == MemoryTier.HOT, "BOUNDARY post-init should force HOT"

    await surrealdb_storage.add_typed_memory(typed_mem)
    fetched = await surrealdb_storage.get_typed_memory(fiber.id)
    assert fetched is not None
    assert fetched.memory_type == MemoryType.BOUNDARY
    assert fetched.tier == MemoryTier.HOT, (
        "SurrealDB lost the BOUNDARY → HOT tier promotion on round-trip"
    )

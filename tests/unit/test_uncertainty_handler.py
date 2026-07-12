"""U5: smem_uncertainty tool — overview + per-signal actions."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.mcp.conflict_handler import ConflictHandler
from surreal_memory.mcp.uncertainty_handler import UncertaintyHandler
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


class _FakeServer(UncertaintyHandler, ConflictHandler):
    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


async def _add_tm(storage: InMemoryStorage, label: str, **fields: object) -> str:
    """Create a neuron + fiber + typed memory; return the (real) fiber id."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"c-{label}")
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id, summary=label
    )
    await storage.add_fiber(fiber)
    tm = TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT)
    if fields:
        tm = dataclasses.replace(tm, **fields)
    await storage.add_typed_memory(tm)
    return fiber.id


@pytest.fixture
async def server() -> _FakeServer:
    storage = InMemoryStorage()
    brain = Brain.create(name="uncertainty")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    now = utcnow()
    await _add_tm(storage, "low", trust_score=0.3)
    await _add_tm(storage, "sup", superseded_by="f-new", valid_until=now)
    await _add_tm(storage, "exp", expires_at=now + timedelta(days=3))
    await _add_tm(storage, "ok", trust_score=0.9)

    # A CONTRADICTS conflict (two neurons + an unresolved synapse).
    n_old = Neuron.create(type=NeuronType.CONCEPT, content="old", metadata={"_disputed": True})
    n_new = Neuron.create(type=NeuronType.CONCEPT, content="new")
    await storage.add_neuron(n_old)
    await storage.add_neuron(n_new)
    await storage.add_synapse(
        Synapse.create(
            source_id=n_new.id, target_id=n_old.id, type=SynapseType.CONTRADICTS, weight=0.8
        )
    )
    await storage.add_fiber(
        Fiber.create(
            neuron_ids={n_old.id}, synapse_ids=set(), anchor_neuron_id=n_old.id, summary="f-old"
        )
    )
    return _FakeServer(storage)


class TestUncertaintyTool:
    async def test_overview_aggregates_signals(self, server: _FakeServer) -> None:
        res = await server._uncertainty({"action": "overview"})
        assert res["level"] == "high"  # contradictions + low_evidence present
        c = res["counts"]
        assert c["contradictions"] == 1
        assert c["low_evidence"] == 1
        assert c["superseded"] == 1
        assert c["expiring"] == 1
        assert c["drift_clusters"] == 0  # InMemory has no drift backend
        assert res["contradiction_rate"] > 0.0
        assert res["total_memories"] >= 4

    async def test_overview_is_default_action(self, server: _FakeServer) -> None:
        assert (await server._uncertainty({}))["level"] == "high"

    async def test_low_evidence_action(self, server: _FakeServer) -> None:
        res = await server._uncertainty({"action": "low_evidence"})
        assert res["count"] == 1
        assert res["low_evidence"][0]["trust_score"] == 0.3

    async def test_expiring_action(self, server: _FakeServer) -> None:
        res = await server._uncertainty({"action": "expiring", "within_days": 14})
        assert res["count"] == 1
        assert res["expiring"][0]["memory_type"] == "fact"

    async def test_drift_action_empty_without_backend(self, server: _FakeServer) -> None:
        res = await server._uncertainty({"action": "drift"})
        assert res["drift_clusters"] == []
        assert res["count"] == 0

    async def test_contradictions_delegates_to_conflicts(self, server: _FakeServer) -> None:
        res = await server._uncertainty({"action": "contradictions"})
        # ConflictHandler._conflicts_list shape.
        assert "conflicts" in res
        assert res["count"] == 1

    async def test_unknown_action_errors(self, server: _FakeServer) -> None:
        assert "error" in await server._uncertainty({"action": "bogus"})

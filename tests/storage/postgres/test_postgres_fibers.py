"""CRUD tests for PostgreSQL fibers."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_add_and_get_fiber(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Add neurons, synapses, then fiber, and retrieve."""
    deps = __import__("neural_memory.core.fiber", fromlist=["Fiber"]).Fiber
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    fiber = deps.create(
        neuron_ids={n.id for n in sample_neurons[:3]},
        synapse_ids={s.id for s in sample_synapses[:2]},
        anchor_neuron_id=sample_neurons[0].id,
        summary="Test fiber",
    )
    await storage.add_fiber(fiber)

    got = await storage.get_fiber(fiber.id)
    assert got is not None
    assert got.id == fiber.id
    assert got.summary == "Test fiber"


@pytest.mark.asyncio
async def test_get_fiber_not_found(storage: Any) -> None:
    """Get non-existent fiber returns None."""
    got = await storage.get_fiber("nonexistent-fiber")
    assert got is None


@pytest.mark.asyncio
async def test_get_fibers_ordered(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Get fibers with ordering."""
    deps = __import__("neural_memory.core.fiber", fromlist=["Fiber"]).Fiber
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    fiber = deps.create(
        neuron_ids={n.id for n in sample_neurons[:2]},
        synapse_ids={s.id for s in sample_synapses[:1]},
        anchor_neuron_id=sample_neurons[0].id,
    )
    await storage.add_fiber(fiber)

    fibers = await storage.get_fibers(limit=10, order_by="created_at", descending=True)
    assert len(fibers) >= 1


@pytest.mark.asyncio
async def test_find_fibers_by_neuron(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Find fibers containing a neuron."""
    deps = __import__("neural_memory.core.fiber", fromlist=["Fiber"]).Fiber
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    fiber = deps.create(
        neuron_ids={n.id for n in sample_neurons[:2]},
        synapse_ids={s.id for s in sample_synapses[:1]},
        anchor_neuron_id=sample_neurons[0].id,
    )
    await storage.add_fiber(fiber)

    found = await storage.find_fibers(contains_neuron=sample_neurons[0].id, limit=10)
    assert len(found) >= 1


@pytest.mark.asyncio
async def test_pin_fibers(
    storage: Any, sample_neurons: list[Any], sample_synapses: list[Any]
) -> None:
    """Pin and unpin fibers."""
    deps = __import__("neural_memory.core.fiber", fromlist=["Fiber"]).Fiber
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)

    fiber = deps.create(
        neuron_ids={n.id for n in sample_neurons[:2]},
        synapse_ids={s.id for s in sample_synapses[:1]},
        anchor_neuron_id=sample_neurons[0].id,
        summary="Pinnable fiber",
    )
    await storage.add_fiber(fiber)

    # Pin
    count = await storage.pin_fibers([fiber.id], pinned=True)
    assert count == 1

    # Verify pinned
    pinned_ids = await storage.get_pinned_neuron_ids()
    assert sample_neurons[0].id in pinned_ids

    pinned_list = await storage.list_pinned_fibers()
    assert len(pinned_list) >= 1
    assert pinned_list[0]["fiber_id"] == fiber.id

    # Unpin
    count = await storage.pin_fibers([fiber.id], pinned=False)
    assert count == 1
    pinned_ids = await storage.get_pinned_neuron_ids()
    assert sample_neurons[0].id not in pinned_ids


@pytest.mark.asyncio
async def test_pin_fibers_empty_list(storage: Any) -> None:
    """Pin with empty list returns 0."""
    count = await storage.pin_fibers([], pinned=True)
    assert count == 0

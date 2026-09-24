from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


@pytest.mark.asyncio
async def test_pair_membership_is_undirected_brain_scoped_and_indexed() -> None:
    storage = SurrealDBStorage(url="http://localhost:8001")
    storage._current_brain_id = "brain-a"
    storage._query = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [],
            [{"id": "synapse:edge-1"}],
            [],
            [{"id": "synapse:edge-2"}],
            [],
            [],
        ]
    )

    existing = await storage.find_existing_synapse_pairs(
        [
            ("neuron-a", "neuron-b"),
            ("neuron-a", "neuron-b"),
            ("neuron-b", "neuron-a"),
            ("missing", "pair"),
        ]
    )

    assert existing == {("neuron-a", "neuron-b"), ("neuron-b", "neuron-a")}
    assert storage._query.await_count == 6
    calls = storage._query.await_args_list
    assert calls[0].kwargs["brain_id"] == "brain-a"
    assert "WITH INDEX idx_synapse_pair_in_out" in calls[0].args[0]
    assert "WITH INDEX idx_synapse_pair_out_in" in calls[1].args[0]
    assert "type =" not in calls[0].args[0]
    assert all("LIMIT 1" in call.args[0] for call in calls)
    assert calls[0].kwargs["left_id"] == "neuron_a"
    assert calls[0].kwargs["right_id"] == "neuron_b"


@pytest.mark.asyncio
async def test_pair_membership_bounds_empty_and_large_inputs() -> None:
    storage = SurrealDBStorage(url="http://localhost:8001")
    storage._current_brain_id = "brain-a"
    storage._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

    assert await storage.find_existing_synapse_pairs([]) == set()
    assert storage._query.await_count == 0

    pairs = [(f"source-{index}", f"target-{index}") for index in range(130)]
    assert await storage.find_existing_synapse_pairs(pairs) == set()
    assert storage._query.await_count == 260

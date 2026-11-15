"""Full retrieval must perform no storage or session writes during evaluation."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import patch

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.engine.retrieval_eval import ReadOnlyStorage
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest.mark.parametrize("strategy", ["classic", "hybrid"])
@pytest.mark.parametrize("threshold", [0.0, 1.0])
async def test_full_pipeline_evaluation_denies_all_writes(strategy: str, threshold: float) -> None:
    storage = InMemoryStorage()
    brain = Brain.create(name="synthetic-evaluation")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    for index in range(3):
        neuron = Neuron.create(
            type=NeuronType.CONCEPT, content=f"graph activation synthetic evidence record {index}"
        )
        await storage.add_neuron(neuron)
        await storage.add_fiber(
            Fiber.create(
                neuron_ids={neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=neuron.id,
                summary=f"graph activation evidence {index}",
                fiber_id=f"eval-{index}",
            )
        )
    before = asdict(await storage.export_brain(brain.id))
    before.pop("exported_at")
    boundary = ReadOnlyStorage(storage)
    config = BrainConfig(
        activation_strategy=strategy,
        sufficiency_threshold=threshold,
        adaptive_depth_enabled=True,
        reconsolidation_enabled=True,
    )
    pipeline = ReflexPipeline(boundary, config, read_only=True)  # type: ignore[arg-type]
    with patch("surreal_memory.engine.session_state.SessionManager.get_instance") as sessions:
        result = await pipeline.query("graph activation", session_id="must-not-mutate")
    assert boundary.write_attempts == [], (boundary.write_attempts, boundary.read_counts)
    assert result.neurons_activated > 0, (boundary.read_counts, result.metadata)
    assert boundary.read_counts["find_fibers_batch"] > 0
    assert boundary.write_attempts == []
    assert pipeline._write_queue.pending_count == 0
    sessions.assert_not_called()
    after = asdict(await storage.export_brain(brain.id))
    after.pop("exported_at")
    assert after == before


async def test_boundary_denies_unknown_and_raw_transport_and_copies_results() -> None:
    storage = InMemoryStorage()
    storage.set_brain("test")
    neuron = Neuron.create(type=NeuronType.CONCEPT, content="synthetic")
    await storage.add_neuron(neuron)
    boundary = ReadOnlyStorage(storage)
    for method in ("add_neuron", "_query", "_ensure_conn", "save_brain"):
        with pytest.raises(RuntimeError, match="denied"):
            getattr(boundary, method)
    assert len(boundary.write_attempts) == 4
    assert await boundary.get_neuron(neuron.id) is not neuron

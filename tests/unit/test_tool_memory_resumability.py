"""Crash-replay coverage for tool-event graph processing."""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.tool_memory import process_events
from surreal_memory.unified_config import ToolMemoryConfig


class _ReplayStore:
    """In-memory store that fails once after graph writes, before event markers."""

    current_brain_id = "test-brain"

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = [
            {
                "id": "event-a",
                "tool_name": "Read",
                "server_name": "fs",
                "session_id": "s1",
                "task_context": "inspect files",
                "success": True,
                "created_at": "2026-09-23T10:00:00",
            },
            {
                "id": "event-b",
                "tool_name": "Grep",
                "server_name": "fs",
                "session_id": "s1",
                "task_context": "inspect files",
                "success": True,
                "created_at": "2026-09-23T10:00:10",
            },
        ]
        self.processed: set[str] = set()
        self.neurons: dict[str, Neuron] = {}
        self.synapses: list[Synapse] = []
        self.mark_calls = 0
        self.update_calls = 0

    def set_brain(self, brain_id: str) -> None:
        self.current_brain_id = brain_id

    async def get_unprocessed_events(self, brain_id: str, limit: int) -> list[dict[str, Any]]:
        del brain_id
        return [event for event in self.events if event["id"] not in self.processed][:limit]

    async def mark_events_processed(self, brain_id: str, event_ids: list[Any]) -> None:
        del brain_id
        self.mark_calls += 1
        if self.mark_calls == 1:
            raise RuntimeError("simulated interruption before event marker commit")
        self.processed.update(str(event_id) for event_id in event_ids)

    async def find_neurons(self, *, content_exact: str, limit: int = 1) -> list[Neuron]:
        neuron = self.neurons.get(content_exact)
        return [neuron][:limit] if neuron is not None else []

    async def add_neuron(self, neuron: Neuron) -> None:
        self.neurons[neuron.content] = neuron

    async def get_synapses(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
    ) -> list[Synapse]:
        return [
            synapse
            for synapse in self.synapses
            if (source_id is None or synapse.source_id == source_id)
            and (target_id is None or synapse.target_id == target_id)
            and (type is None or synapse.type == type)
        ]

    async def add_synapse(self, synapse: Synapse) -> None:
        self.synapses.append(synapse)

    async def update_synapse(self, synapse: Synapse) -> None:
        self.update_calls += 1
        for index, current in enumerate(self.synapses):
            if current.id == synapse.id:
                self.synapses[index] = synapse
                return
        raise AssertionError("cannot update a missing test synapse")


@pytest.mark.asyncio
async def test_replay_after_graph_writes_does_not_duplicate_or_reinforce() -> None:
    store = _ReplayStore()
    config = ToolMemoryConfig(enabled=True, min_frequency=1, cooccurrence_window_s=60)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        await process_events(store, "test-brain", config)  # type: ignore[arg-type]

    first_weights = sorted((synapse.type.value, synapse.weight) for synapse in store.synapses)
    assert len(store.neurons) == 3
    assert len(store.synapses) == 3
    assert store.processed == set()

    replay = await process_events(store, "test-brain", config)  # type: ignore[arg-type]

    assert replay.events_processed == 2
    assert replay.last_event_id == "event-b"
    assert replay.synapses_created == 0
    assert replay.synapses_reinforced == 0
    assert len(store.neurons) == 3
    assert len(store.synapses) == 3
    assert (
        sorted((synapse.type.value, synapse.weight) for synapse in store.synapses) == first_weights
    )
    assert store.update_calls == 0
    assert store.processed == {"event-a", "event-b"}


@pytest.mark.asyncio
async def test_strategy_checkpoints_each_marked_batch_and_drains_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from types import SimpleNamespace

    from surreal_memory.engine.consolidation import (
        ConsolidationEngine,
        ConsolidationReport,
        ConsolidationStrategy,
    )
    from surreal_memory.unified_config import UnifiedConfig

    store = _ReplayStore()
    store.mark_calls = 1  # Disable the one-shot interruption for this strategy-level test.
    tool_config = ToolMemoryConfig(
        enabled=True,
        min_frequency=1,
        process_batch_size=1,
        cooccurrence_window_s=60,
    )
    monkeypatch.setattr(
        UnifiedConfig,
        "load",
        classmethod(lambda cls: SimpleNamespace(data_dir=tmp_path, tool_memory=tool_config)),
    )

    class _Progress:
        checkpoints: list[tuple[str | None, set[str]]] = []

        def strategy_state(self, strategy: str) -> dict[str, Any]:
            del strategy
            return {}

        async def checkpoint(
            self,
            strategy: str,
            phase: str,
            *,
            cursor: str | None,
            counters: dict[str, int | float],
            pending: list[str] | None = None,
        ) -> None:
            del pending
            assert strategy == ConsolidationStrategy.PROCESS_TOOL_EVENTS.value
            assert phase == "batch_committed"
            assert counters["events_processed"] == len(self.checkpoints) + 1
            self.checkpoints.append((cursor, set(store.processed)))

    engine = ConsolidationEngine(store)  # type: ignore[arg-type]
    progress = _Progress()
    engine._progress_session = progress  # type: ignore[assignment]
    engine._active_strategy = ConsolidationStrategy.PROCESS_TOOL_EVENTS

    await engine._process_tool_events(ConsolidationReport(), dry_run=False)

    assert progress.checkpoints == [
        ("event-a", {"event-a"}),
        ("event-b", {"event-a", "event-b"}),
    ]

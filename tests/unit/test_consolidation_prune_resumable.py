from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)

REFERENCE_TIME = datetime(2026, 9, 20, 12, 30)


def _weak_synapse() -> Synapse:
    return Synapse(
        id="edge-1",
        source_id="source-1",
        target_id="target-1",
        type=SynapseType.RELATED_TO,
        weight=0.001,
        created_at=REFERENCE_TIME - timedelta(days=60),
    )


class _Progress:
    def __init__(self, *, pause_before_delete: bool = False) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"prune": {}}}
        self.pause_before_delete = pause_before_delete
        self.writes: list[dict[str, Any]] = []

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return self.state["strategy_states"].setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        current = self.strategy_state(strategy)
        current.update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
            updated_at=REFERENCE_TIME,
        )
        self.writes.append(dict(current))
        if self.pause_before_delete and phase == "synapse_pending":
            self.pause_before_delete = False
            raise ConsolidationPausedError("simulated budget exhaustion")


class _PruneStorage:
    current_brain_id = "default"

    def __init__(self) -> None:
        self.synapses = {"edge-1": _weak_synapse()}
        self.removed_fiber_refs: list[set[str]] = []
        self.prune_page_calls = 0

    async def get_pinned_neuron_ids(self) -> set[str]:
        return set()

    async def get_synapse_prune_page(
        self,
        cursor_created_at: datetime | None,
        cursor_id: str | None,
        *,
        limit: int,
    ) -> list[Synapse]:
        self.prune_page_calls += 1
        rows = sorted(self.synapses.values(), key=lambda item: (item.created_at, item.id))
        if cursor_id is not None and cursor_created_at is not None:
            rows = [
                row for row in rows if (row.created_at, row.id) > (cursor_created_at, cursor_id)
            ]
        return rows[:limit]

    async def get_synapses_by_ids(self, synapse_ids: list[str] | set[str]) -> list[Synapse]:
        return [self.synapses[sid] for sid in synapse_ids if sid in self.synapses]

    async def get_synapses_for_sources(self, source_ids: list[str] | set[str]) -> list[Synapse]:
        sources = set(source_ids)
        return [row for row in self.synapses.values() if row.source_id in sources]

    async def get_synapse_target_counts_for_sources(
        self, source_ids: list[str] | set[str]
    ) -> dict[str, int]:
        targets: dict[str, set[str]] = {}
        for synapse in self.synapses.values():
            if synapse.source_id in source_ids:
                targets.setdefault(synapse.source_id, set()).add(synapse.target_id)
        return {source_id: len(target_ids) for source_id, target_ids in targets.items()}

    async def get_fiber_neuron_ids_for(
        self, neuron_ids: list[str] | set[str], *, min_salience: float | None = None
    ) -> set[str]:
        return set()

    async def delete_synapses_batch(self, synapse_ids: set[str] | list[str]) -> int:
        deleted = 0
        for synapse_id in synapse_ids:
            deleted += int(self.synapses.pop(synapse_id, None) is not None)
        return deleted

    async def remove_synapse_refs_from_fibers(self, synapse_ids: set[str]) -> int:
        self.removed_fiber_refs.append(set(synapse_ids))
        return 0

    async def find_neurons_after_id(
        self,
        cursor_id: str | None,
        *,
        limit: int,
        created_before: datetime,
        ephemeral: bool | None,
        include_embedding: bool,
    ) -> list[Any]:
        return []

    async def find_neurons_by_ids(
        self, neuron_ids: list[str], *, include_embedding: bool = False
    ) -> list[Any]:
        return []

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, Any]:
        return {}

    async def get_connected_neuron_ids_for(self, neuron_ids: list[str] | set[str]) -> set[str]:
        return set()

    async def delete_neurons_batch(self, neuron_ids: list[str]) -> int:
        return 0


def _engine(storage: _PruneStorage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(
        storage,
        ConsolidationConfig(prune_isolated_neurons=False),
    )
    engine._active_strategy = ConsolidationStrategy.PRUNE
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest.mark.parametrize(
    ("has_second_target", "weak_edge_survives"), [(False, True), (True, False)]
)
@pytest.mark.asyncio
async def test_bridge_protection_uses_distinct_target_count(
    has_second_target: bool, weak_edge_survives: bool
) -> None:
    storage = _PruneStorage()
    weak = replace(_weak_synapse(), weight=0.03)
    storage.synapses = {weak.id: weak}
    if has_second_target:
        strong = replace(_weak_synapse(), id="edge-2", target_id="target-2", weight=0.5)
        storage.synapses[strong.id] = strong

    await _engine(storage, _Progress())._prune(ConsolidationReport(), REFERENCE_TIME, dry_run=False)

    assert (weak.id in storage.synapses) is weak_edge_survives


@pytest.mark.asyncio
async def test_prune_replays_pending_synapse_delete_and_fiber_cleanup() -> None:
    storage = _PruneStorage()
    after_reference = replace(
        _weak_synapse(),
        id="edge-after-reference",
        created_at=REFERENCE_TIME + timedelta(seconds=1),
    )
    storage.synapses[after_reference.id] = after_reference
    progress = _Progress(pause_before_delete=True)
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated budget"):
        await engine._prune(ConsolidationReport(), REFERENCE_TIME, dry_run=False)

    pending = progress.strategy_state("prune")
    assert pending["phase"] == "synapse_pending"
    assert pending["pending"] == ["edge-1"]
    assert "edge-1" in storage.synapses

    report = ConsolidationReport()
    await engine._prune(report, REFERENCE_TIME, dry_run=False)

    resumed = progress.strategy_state("prune")
    assert resumed["phase"] == "synapse_scan"
    saved_cursor = json.loads(resumed["cursor"])
    assert saved_cursor == {
        "version": 1,
        "created_at": after_reference.created_at.isoformat(),
        "id": "edge-after-reference",
    }
    assert resumed["pending"] == []
    assert storage.synapses == {after_reference.id: after_reference}
    assert storage.removed_fiber_refs == [{"edge-1"}]
    assert report.synapses_pruned == 1
    assert storage.prune_page_calls == 1


@pytest.mark.asyncio
async def test_prune_rejects_unknown_synapse_cursor_format_without_advancing() -> None:
    storage = _PruneStorage()
    progress = _Progress()
    state = progress.strategy_state("prune")
    state.update(phase="synapse_scan", cursor="edge-legacy", pending=[])

    with pytest.raises(ConsolidationProgressError, match="cursor is incompatible"):
        await _engine(storage, progress)._prune(
            ConsolidationReport(), REFERENCE_TIME, dry_run=False
        )

    assert state["cursor"] == "edge-legacy"
    assert set(storage.synapses) == {"edge-1"}


@pytest.mark.asyncio
async def test_prune_revalidates_pending_candidate_before_delete() -> None:
    storage = _PruneStorage()
    progress = _Progress(pause_before_delete=True)
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError):
        await engine._prune(ConsolidationReport(), REFERENCE_TIME, dry_run=False)

    storage.synapses["edge-1"] = replace(storage.synapses["edge-1"], weight=0.8)
    report = ConsolidationReport()
    await engine._prune(report, REFERENCE_TIME, dry_run=False)

    assert "edge-1" in storage.synapses
    assert storage.removed_fiber_refs == []
    assert progress.strategy_state("prune")["pending"] == []
    assert report.synapses_pruned == 0


@pytest.mark.asyncio
async def test_prune_dry_run_never_mutates_checkpoint_or_graph() -> None:
    storage = _PruneStorage()
    progress = _Progress(pause_before_delete=True)
    engine = _engine(storage, progress)

    report = ConsolidationReport()
    await engine._prune(report, REFERENCE_TIME, dry_run=True)

    assert report.synapses_pruned == 1
    assert "edge-1" in storage.synapses
    assert storage.removed_fiber_refs == []
    assert progress.writes == []

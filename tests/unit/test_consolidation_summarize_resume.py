from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.fiber import Fiber
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


class _Progress:
    def __init__(self, pause_phase: str | None = None) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"summarize": {}}}
        self.pause_phase = pause_phase
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
            updated_at=datetime.now(UTC),
        )
        self.writes.append(dict(current))
        if phase == self.pause_phase:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated interruption")


class _Storage:
    def __init__(self, fibers: list[Fiber], *, crash_after: str | None = None) -> None:
        self.fibers = {fiber.id: fiber for fiber in fibers}
        self.neurons = {
            fiber.anchor_neuron_id: SimpleNamespace(
                id=fiber.anchor_neuron_id,
                content=f"content for {fiber.anchor_neuron_id}",
            )
            for fiber in fibers
        }
        self.synapses: dict[str, Any] = {}
        self.added_neurons: list[str] = []
        self.added_synapses: list[str] = []
        self.added_fibers: list[str] = []
        self.crash_after = crash_after

    async def get_fibers(self, *, limit: int) -> list[Fiber]:
        return list(self.fibers.values())[:limit]

    async def get_neuron(self, neuron_id: str) -> Any:
        return self.neurons.get(neuron_id)

    async def add_neuron(self, neuron: Any) -> str:
        self.added_neurons.append(neuron.id)
        self.neurons[neuron.id] = neuron
        if self.crash_after == "neuron":
            self.crash_after = None
            raise RuntimeError("simulated crash after neuron write")
        return neuron.id

    async def get_synapse(self, synapse_id: str) -> Any:
        return self.synapses.get(synapse_id)

    async def add_synapse(self, synapse: Any) -> str:
        self.added_synapses.append(synapse.id)
        self.synapses[synapse.id] = synapse
        if self.crash_after == "synapse":
            self.crash_after = None
            raise RuntimeError("simulated crash after synapse write")
        return synapse.id

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        return self.fibers.get(fiber_id)

    async def add_fiber(self, fiber: Fiber) -> str:
        self.added_fibers.append(fiber.id)
        self.fibers[fiber.id] = fiber
        if self.crash_after == "fiber":
            self.crash_after = None
            raise RuntimeError("simulated crash after summary fiber write")
        return fiber.id


def _source_fibers() -> list[Fiber]:
    return [
        Fiber.create(
            neuron_ids={"anchor-a"},
            synapse_ids=set(),
            anchor_neuron_id="anchor-a",
            summary="first source summary",
            tags={"shared", "alpha"},
            fiber_id="fiber-a",
        ),
        Fiber.create(
            neuron_ids={"anchor-b"},
            synapse_ids=set(),
            anchor_neuron_id="anchor-b",
            summary="second source summary",
            tags={"shared", "beta"},
            fiber_id="fiber-b",
        ),
    ]


def _engine(
    storage: _Storage,
    progress: _Progress | None = None,
    *,
    config: ConsolidationConfig | None = None,
) -> ConsolidationEngine:
    engine = ConsolidationEngine(
        storage,
        config
        or ConsolidationConfig(
            summarize_min_cluster_size=2,
            summarize_tag_overlap_threshold=0.2,
        ),
    )
    engine._active_strategy = ConsolidationStrategy.SUMMARIZE
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


def _pending(progress: _Progress) -> dict[str, Any]:
    serialized = progress.strategy_state("summarize")["pending"]
    assert len(serialized) == 2
    assert json.loads(serialized[0])["kind"] == "summary_pairs"
    return json.loads(serialized[1])


@pytest.mark.asyncio
async def test_pending_cluster_resumes_from_saved_output_and_ids() -> None:
    storage = _Storage(_source_fibers())
    progress = _Progress(pause_phase="summarize_pending")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await engine._summarize(ConsolidationReport(), dry_run=False)

    snapshot = _pending(progress)
    assert snapshot["kind"] == "summary_cluster"
    assert snapshot["version"] == 1
    assert snapshot["source_fiber_ids"] == ["fiber-a", "fiber-b"]
    assert (
        snapshot["concept_content"]
        == "[alpha, beta, shared] first source summary; second source summary"
    )
    assert storage.added_neurons == []
    assert storage.added_synapses == []
    assert storage.added_fibers == []

    report = ConsolidationReport()
    await _engine(storage, progress)._summarize(report, dry_run=False)

    assert storage.neurons[snapshot["concept_neuron_id"]].content == snapshot["concept_content"]
    assert len(storage.added_neurons) == 1
    assert storage.added_synapses == [edge["id"] for edge in snapshot["synapses"]]
    assert storage.added_fibers == [snapshot["summary_fiber_id"]]
    assert report.summaries_created == 1
    assert len(progress.strategy_state("summarize")["pending"]) == 1
    assert progress.strategy_state("summarize")["cursor"].endswith(snapshot["cluster_key"])


@pytest.mark.asyncio
async def test_crash_after_summary_fiber_write_replays_without_duplicate_effects() -> None:
    storage = _Storage(_source_fibers(), crash_after="fiber")
    progress = _Progress()
    engine = _engine(storage, progress)

    with pytest.raises(RuntimeError, match="crash after summary fiber"):
        await engine._summarize(ConsolidationReport(), dry_run=False)

    snapshot = _pending(progress)
    assert snapshot["summary_fiber_id"] in storage.fibers
    assert progress.strategy_state("summarize")["cursor"].endswith("|")

    report = ConsolidationReport()
    await _engine(storage, progress)._summarize(report, dry_run=False)

    assert storage.added_neurons == [snapshot["concept_neuron_id"]]
    assert storage.added_synapses == [edge["id"] for edge in snapshot["synapses"]]
    assert storage.added_fibers == [snapshot["summary_fiber_id"]]
    assert report.summaries_created == 1
    assert len(progress.strategy_state("summarize")["pending"]) == 1


@pytest.mark.parametrize("change", ["source_data", "parameter", "pending_version"])
@pytest.mark.asyncio
async def test_incompatible_snapshot_is_rejected_without_mutating_storage(change: str) -> None:
    storage = _Storage(_source_fibers())
    progress = _Progress(pause_phase="summarize_pending")
    engine = _engine(storage, progress)

    with pytest.raises(ConsolidationPausedError):
        await engine._summarize(ConsolidationReport(), dry_run=False)

    if change == "source_data":
        changed = Fiber.create(
            neuron_ids={"anchor-a"},
            synapse_ids=set(),
            anchor_neuron_id="anchor-a",
            summary="changed source summary",
            tags={"shared", "alpha"},
            fiber_id="fiber-a",
        )
        storage.fibers[changed.id] = changed
    elif change == "pending_version":
        snapshot = _pending(progress)
        snapshot["version"] = 99
        progress.strategy_state("summarize")["pending"] = [
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ]
    else:
        engine = _engine(
            storage,
            progress,
            config=ConsolidationConfig(
                summarize_min_cluster_size=2,
                summarize_tag_overlap_threshold=0.4,
            ),
        )

    before_neurons = dict(storage.neurons)
    before_synapses = dict(storage.synapses)
    before_fibers = dict(storage.fibers)

    with pytest.raises(ConsolidationProgressError, match="summarize"):
        await engine._summarize(ConsolidationReport(), dry_run=False)

    assert storage.neurons == before_neurons
    assert storage.synapses == before_synapses
    assert storage.fibers == before_fibers


@pytest.mark.asyncio
async def test_dry_run_does_not_change_progress_or_graph_state() -> None:
    storage = _Storage(_source_fibers())
    progress = _Progress()
    engine = _engine(storage, progress)
    state_before = json.dumps(progress.state, sort_keys=True)
    writes_before = list(progress.writes)
    graph_before = (dict(storage.neurons), dict(storage.synapses), dict(storage.fibers))

    report = ConsolidationReport()
    await engine._summarize(report, dry_run=True)

    assert report.summaries_created == 1
    assert json.dumps(progress.state, sort_keys=True) == state_before
    assert progress.writes == writes_before
    assert (storage.neurons, storage.synapses, storage.fibers) == graph_before
    assert storage.added_neurons == []
    assert storage.added_synapses == []
    assert storage.added_fibers == []


@pytest.mark.asyncio
async def test_summarize_considers_fibers_after_the_former_top_1000_limit() -> None:
    fibers = [
        Fiber.create(
            neuron_ids={f"anchor-{i}"},
            synapse_ids=set(),
            anchor_neuron_id=f"anchor-{i}",
            summary=f"source {i}",
            tags={f"unique-{i}"},
            fiber_id=f"fiber-{i:04d}",
        )
        for i in range(1000)
    ]
    fibers.extend(
        Fiber.create(
            neuron_ids={f"anchor-{i}"},
            synapse_ids=set(),
            anchor_neuron_id=f"anchor-{i}",
            summary=f"source {i}",
            tags={"z-shared"},
            fiber_id=f"fiber-{i:04d}",
        )
        for i in (1000, 1001)
    )
    report = ConsolidationReport()

    await _engine(_Storage(fibers))._summarize(report, dry_run=True)

    assert report.summaries_created == 1
    assert "summarize_fibers_scanned" not in report.extra


@pytest.mark.asyncio
async def test_summarize_examines_pairs_after_the_former_50000_limit() -> None:
    # Ten groups contribute 49,500 pairs. The eleventh only existed after
    # the previous global cap and must also produce a cluster.
    fibers = [
        Fiber.create(
            neuron_ids={f"anchor-{group}-{member}"},
            synapse_ids=set(),
            anchor_neuron_id=f"anchor-{group}-{member}",
            summary=f"source {group}-{member}",
            tags={f"group-{group:02d}"},
            fiber_id=f"fiber-{group:02d}-{member:03d}",
        )
        for group in range(11)
        for member in range(100)
    ]
    report = ConsolidationReport()

    await _engine(_Storage(fibers))._summarize(report, dry_run=True)

    assert report.summaries_created == 11


@pytest.mark.asyncio
async def test_old_summarize_cursor_refuses_incompatible_resume() -> None:
    storage = _Storage(_source_fibers())
    progress = _Progress()
    progress.strategy_state("summarize").update(
        cursor="summarize-v1|old-fingerprint|",
        pending=[],
        counters={},
    )

    with pytest.raises(ConsolidationProgressError, match="summarize inputs or algorithm changed"):
        await _engine(storage, progress)._summarize(ConsolidationReport(), dry_run=False)

    assert storage.added_neurons == []
    assert storage.added_synapses == []
    assert storage.added_fibers == []


@pytest.mark.asyncio
async def test_pair_scan_resumes_from_committed_tag_and_union_state() -> None:
    class PauseAfterHundredTags(_Progress):
        async def checkpoint(
            self,
            strategy: str,
            phase: str,
            *,
            cursor: str | None = None,
            pending: list[str] | None = None,
            counters: dict[str, int | float] | None = None,
        ) -> None:
            await super().checkpoint(
                strategy, phase, cursor=cursor, pending=pending, counters=counters
            )
            if cursor is not None and cursor.endswith("pairs:100"):
                raise ConsolidationPausedError("after hundred tags")

    fibers = [
        Fiber.create(
            neuron_ids={f"anchor-{i}"},
            synapse_ids=set(),
            anchor_neuron_id=f"anchor-{i}",
            summary=f"source {i}",
            tags={"a-tag"} if i < 2 else {f"b-{i:03d}"} if i < 101 else {"z-tag"},
            fiber_id=f"fiber-{i:03d}",
        )
        for i in range(103)
    ]
    storage = _Storage(fibers)
    progress = PauseAfterHundredTags()

    with pytest.raises(ConsolidationPausedError, match="after hundred tags"):
        await _engine(storage, progress)._summarize(ConsolidationReport(), dry_run=False)

    state = progress.strategy_state("summarize")
    assert state["cursor"].endswith("pairs:100")
    snapshot = json.loads(state["pending"][0])
    assert snapshot["next_tag"] == 100
    assert snapshot["pairs_examined"] == 1
    assert snapshot["parent"]
    assert storage.added_neurons == []

    report = ConsolidationReport()
    await _engine(storage, progress)._summarize(report, dry_run=False)

    assert report.summaries_created == 2
    assert state["pending"] and len(state["pending"]) == 1
    assert progress.strategy_state("summarize")["cursor"].split("|", 2)[2] != "pairs:100"

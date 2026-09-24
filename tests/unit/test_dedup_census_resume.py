"""Durable keyset snapshot and interruption coverage for dedup census."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

_REFERENCE_TIME = datetime(2026, 9, 24, 12, 0)


def test_dedup_census_schema_is_additive_and_run_scoped() -> None:
    from surreal_memory.storage.surrealdb.schema import SOURCE_REVISION_DDL

    assert "DEFINE TABLE IF NOT EXISTS consolidation_dedup_census SCHEMALESS" in SOURCE_REVISION_DDL
    assert any(
        "idx_dedup_census_run_page" in statement
        and "run_id, brain_id, strategy, filter_fingerprint, page_index UNIQUE" in statement
        for statement in SOURCE_REVISION_DDL
    )


class _AlreadyExistsError(Exception):
    pass


class _Progress:
    reference_time = _REFERENCE_TIME
    brain_id = "dedup-census-brain"

    def __init__(self, *, fail_first_checkpoint: bool = False) -> None:
        self.fail_first_checkpoint = fail_first_checkpoint
        self.state: dict[str, Any] = {
            "run_id": "dedup-census-run",
            "strategy_states": {"dedup": {"phase": "starting"}},
        }

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
        if self.fail_first_checkpoint:
            self.fail_first_checkpoint = False
            raise RuntimeError("simulated checkpoint interruption")
        self.strategy_state(strategy).update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )


class _DurableStorage:
    current_brain_id = _Progress.brain_id

    def __init__(self, neurons: list[Neuron]) -> None:
        self.neurons = sorted(neurons, key=lambda neuron: neuron.id)
        self.source_cursors: list[str | None] = []
        self.staged: dict[str, dict[str, Any]] = {}
        self.synapses: list[Synapse] = []
        self.brain = Brain.create(name="dedup-census-test")

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def find_neurons_after_id(
        self,
        cursor_id: str | None,
        *,
        limit: int = 1000,
        created_before: datetime | None = None,
        ephemeral: bool | None = False,
        include_embedding: bool = False,
    ) -> list[Neuron]:
        self.source_cursors.append(cursor_id)
        rows = [
            neuron
            for neuron in self.neurons
            if (cursor_id is None or neuron.id > cursor_id)
            and (created_before is None or neuron.created_at <= created_before)
            and (ephemeral is None or neuron.ephemeral is ephemeral)
        ]
        return rows[:limit]

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        if sql.startswith("CREATE type::record('consolidation_dedup_census'"):
            stage_id = str(params["stage_id"])
            if stage_id in self.staged:
                raise _AlreadyExistsError("record already exists")
            row = dict(params["row"])
            row["id"] = f"consolidation_dedup_census:{stage_id}"
            self.staged[stage_id] = row
            return [row]
        if sql.startswith("SELECT * FROM type::record('consolidation_dedup_census'"):
            row = self.staged.get(str(params["stage_id"]))
            return [dict(row)] if row is not None else []
        if sql.startswith("SELECT * FROM consolidation_dedup_census"):
            filters = (
                "run_id",
                "brain_id",
                "strategy",
                "filter_fingerprint",
            )
            rows = [
                row
                for row in self.staged.values()
                if all(row.get(key) == params[key] for key in filters)
            ]
            if "complete = true" in sql:
                rows = [row for row in rows if row.get("complete") is True]
            if "page_index = $page_index" in sql:
                rows = [row for row in rows if row.get("page_index") == params["page_index"]]
            rows.sort(key=lambda row: int(row["page_index"]))
            return [dict(row) for row in rows[: int(params.get("limit", 1))]]
        raise AssertionError(f"unexpected query: {sql}")

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
        limit: int | None = None,
    ) -> list[Synapse]:
        rows = [
            edge
            for edge in self.synapses
            if (source_id is None or edge.source_id == source_id)
            and (target_id is None or edge.target_id == target_id)
            and (type is None or edge.type == type)
        ]
        return rows[:limit] if limit is not None else rows

    async def add_synapse(self, synapse: Synapse) -> str:
        if any(row.id == synapse.id for row in self.synapses):
            raise ValueError("duplicate key")
        self.synapses.append(synapse)
        return synapse.id

    async def get_brain(self, brain_id: str) -> Brain:
        return self.brain

    async def save_brain(self, brain: Brain) -> None:
        self.brain = brain


def _anchor(index: int, *, content_hash: int = 123) -> Neuron:
    return Neuron(
        id=f"neuron-{index:04d}",
        type=NeuronType.CONCEPT,
        content=f"memory-{index}",
        metadata={"is_anchor": True},
        content_hash=content_hash,
        created_at=_REFERENCE_TIME,
    )


def _engine(storage: _DurableStorage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, config=ConsolidationConfig(dedup_max_anchors=5))
    engine._progress_session = progress  # type: ignore[assignment]
    engine._active_strategy = ConsolidationStrategy.DEDUP
    return engine


@pytest.mark.asyncio
async def test_census_resume_uses_immutable_pages_and_excludes_late_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.utils import simhash

    storage = _DurableStorage([_anchor(index) for index in range(501)])
    progress = _Progress()
    engine = _engine(storage, progress)
    budget_checks = 0

    async def pause_after_first_page_checkpoint() -> None:
        nonlocal budget_checks
        budget_checks += 1
        if budget_checks == 2:
            raise ConsolidationPausedError("pause after committed census page")

    engine._check_progress_budget = pause_after_first_page_checkpoint  # type: ignore[method-assign]
    with pytest.raises(ConsolidationPausedError, match="committed census page"):
        await engine._dedup(ConsolidationReport(), dry_run=False)

    checkpoint = progress.strategy_state("dedup")
    assert checkpoint["phase"] == "dedup_census"
    assert len(storage.staged) == 1
    assert storage.source_cursors == [None]

    # Rows already frozen into page zero can change without changing the resumed
    # comparisons. New anchors created after the run's reference time are outside
    # this run's source boundary, even if their IDs follow the saved cursor.
    storage.neurons[:500] = [
        replace(neuron, metadata={"is_anchor": False}, content_hash=0)
        for neuron in storage.neurons[:500]
    ]
    storage.neurons.append(
        replace(_anchor(501), id="neuron-0501", created_at=_REFERENCE_TIME + timedelta(seconds=1))
    )
    storage.neurons.sort(key=lambda neuron: neuron.id)

    comparisons = 0

    def count_comparison(*_args: object, **_kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return False

    monkeypatch.setattr(simhash, "is_near_duplicate", count_comparison)
    resumed = _engine(storage, progress)
    resumed._check_progress_budget = _done  # type: ignore[method-assign]
    report = ConsolidationReport()
    await resumed._dedup(report, dry_run=False)

    assert report.extra["dedup_anchors_total"] == 501
    assert report.extra["dedup_anchors_scanned"] == 5
    assert report.duplicates_found == 0
    assert comparisons == 10  # each of the five frozen anchors compares each later one once
    assert storage.source_cursors == [None, "neuron-0499"]
    assert progress.strategy_state("dedup")["phase"] == "dedup_window_complete"
    assert len(storage.staged) == 3  # two immutable data pages plus one completion marker


@pytest.mark.asyncio
async def test_crash_after_page_write_replays_and_verifies_same_immutable_page() -> None:
    storage = _DurableStorage([_anchor(0), _anchor(1)])
    progress = _Progress(fail_first_checkpoint=True)
    engine = _engine(storage, progress)
    engine._check_progress_budget = _done  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        await engine._dedup(ConsolidationReport(), dry_run=False)
    assert len(storage.staged) == 1  # page 0 committed before progress did

    await _engine(storage, progress)._dedup(ConsolidationReport(), dry_run=False)
    assert storage.source_cursors[:2] == [None, None]
    assert len(storage.staged) == 2  # identical page 0 plus the completion marker
    assert progress.strategy_state("dedup")["phase"] == "dedup_window_complete"


@pytest.mark.asyncio
async def test_dry_run_with_progress_context_does_not_stage_or_checkpoint() -> None:
    storage = _DurableStorage([_anchor(0), _anchor(1)])
    progress = _Progress()
    engine = _engine(storage, progress)

    report = ConsolidationReport(dry_run=True)
    await engine._dedup(report, dry_run=True)

    assert report.duplicates_found == 1
    assert storage.staged == {}
    assert storage.synapses == []
    assert progress.strategy_state("dedup")["phase"] == "starting"


async def _done() -> None:
    return None

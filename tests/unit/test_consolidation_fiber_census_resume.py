from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)

REFERENCE_TIME = datetime(2026, 9, 24, 12, 0)


class _AlreadyExistsError(Exception):
    pass


class _SimulatedCrashError(RuntimeError):
    pass


class _Progress:
    def __init__(
        self,
        run_id: str = "run-census-test",
        *,
        fail_first_checkpoint: bool = False,
    ) -> None:
        self.reference_time = REFERENCE_TIME
        self.fail_first_checkpoint = fail_first_checkpoint
        self.state: dict[str, Any] = {
            "run_id": run_id,
            "strategy_states": {"merge": {}},
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
            raise _SimulatedCrashError("simulated interruption after stage write")
        value = self.strategy_state(strategy)
        value.update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )


class _PagedStorage:
    current_brain_id = "synthetic-census-brain"

    def __init__(self, fibers: list[Fiber]) -> None:
        self.fibers = sorted(fibers, key=lambda item: item.id)
        self.source_cursors: list[str | None] = []
        self.staged: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.stage_ids: dict[str, tuple[str, str, int]] = {}

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def get_fibers_after_id(
        self,
        cursor: str | None,
        *,
        limit: int,
        created_before: datetime | None = None,
    ) -> list[Fiber]:
        self.source_cursors.append(cursor)
        remaining = [
            fiber
            for fiber in self.fibers
            if (cursor is None or fiber.id > cursor)
            and (created_before is None or fiber.created_at <= created_before)
        ]
        return remaining[:limit]

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        if sql.startswith("CREATE type::record('consolidation_fiber_census'"):
            stage_id = str(params["stage_id"])
            row = dict(params["row"])
            key = (str(row["run_id"]), str(row["strategy"]), int(row["page_index"]))
            if stage_id in self.stage_ids:
                raise _AlreadyExistsError(f"record {stage_id} already exists")
            self.staged[key] = row
            self.stage_ids[stage_id] = key
            return [row]
        if sql.startswith("SELECT * FROM type::record('consolidation_fiber_census'"):
            key = self.stage_ids.get(str(params["stage_id"]))
            return [self.staged[key]] if key is not None else []
        if sql.startswith("SELECT * FROM consolidation_fiber_census"):
            run_id = str(params.get("run_id", ""))
            strategy = str(params["strategy"])
            after_page = int(params.get("after_page", -1))
            pages = [
                row
                for (row_run_id, row_strategy, page_index), row in self.staged.items()
                if row_run_id == run_id and row_strategy == strategy and page_index > after_page
            ]
            pages.sort(key=lambda row: int(row["page_index"]))
            return pages[: int(params.get("limit", 1))]
        raise AssertionError(f"unexpected staging query: {sql}")

    async def get_fibers(self, *, limit: int = 10000) -> list[Fiber]:
        return self.fibers[:limit]


def _fibers(count: int) -> list[Fiber]:
    return [
        Fiber(
            id=f"fiber-{index:04d}",
            neuron_ids={f"neuron-{index:04d}"},
            synapse_ids=set(),
            anchor_neuron_id=f"neuron-{index:04d}",
            created_at=REFERENCE_TIME,
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_fiber_census_budget_pause_resumes_after_staged_page() -> None:
    storage = _PagedStorage(_fibers(501))
    progress = _Progress()
    first = ConsolidationEngine(storage, ConsolidationConfig())
    first._progress_session = progress  # type: ignore[assignment]
    first._active_strategy = ConsolidationStrategy.MERGE
    pause_after_first_page = True

    async def pause_at_page_boundary() -> None:
        nonlocal pause_after_first_page
        if pause_after_first_page and storage.source_cursors:
            pause_after_first_page = False
            raise ConsolidationPausedError("simulated census budget pause")

    first._check_progress_budget = pause_at_page_boundary  # type: ignore[method-assign]
    with pytest.raises(ConsolidationPausedError, match="census budget pause"):
        await first._all_fibers_paged()

    storage.fibers.append(
        Fiber(
            id="fiber-0500a",
            neuron_ids={"neuron-late"},
            synapse_ids=set(),
            anchor_neuron_id="neuron-late",
            created_at=REFERENCE_TIME.replace(hour=13),
        )
    )
    storage.fibers.sort(key=lambda item: item.id)

    state = progress.strategy_state("merge")
    assert state["phase"] == "fiber_census"
    checkpoint = json.loads(state["cursor"])
    assert checkpoint["version"] == 1
    first_stage_page = storage.staged[("run-census-test", "merge", 0)]
    assert checkpoint["last_fiber_id"] == first_stage_page["last_fiber_id"]
    assert len(storage.staged) == 1

    resumed = ConsolidationEngine(storage, ConsolidationConfig())
    resumed._progress_session = progress  # type: ignore[assignment]
    resumed._active_strategy = ConsolidationStrategy.MERGE
    resumed._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]

    fibers = await resumed._all_fibers_paged()

    assert [fiber.id for fiber in fibers] == [fiber.id for fiber in _fibers(501)]
    assert storage.source_cursors[0] is None
    assert storage.source_cursors[1] == checkpoint["last_fiber_id"]
    assert None not in storage.source_cursors[1:]
    assert len(storage.staged) >= 2


@pytest.mark.asyncio
async def test_fiber_census_staging_is_isolated_between_run_ids() -> None:
    storage = _PagedStorage(_fibers(501))

    async def collect(run_id: str) -> list[Fiber]:
        engine = ConsolidationEngine(storage, ConsolidationConfig())
        engine._progress_session = _Progress(run_id)  # type: ignore[assignment]
        engine._active_strategy = ConsolidationStrategy.MERGE
        engine._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]
        return await engine._all_fibers_paged()

    first = await collect("run-one")
    source_cursor_count = len(storage.source_cursors)
    second = await collect("run-two")

    assert [fiber.id for fiber in first] == [fiber.id for fiber in second]
    assert len(storage.source_cursors) == source_cursor_count + 2
    assert storage.source_cursors[source_cursor_count] is None
    assert {key[0] for key in storage.staged} == {"run-one", "run-two"}


@pytest.mark.asyncio
async def test_fiber_census_streams_more_than_ten_thousand_fibers_in_bounded_pages() -> None:
    storage = _PagedStorage(_fibers(10_001))
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._progress_session = _Progress()  # type: ignore[assignment]
    engine._active_strategy = ConsolidationStrategy.MERGE
    engine._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]

    fiber_ids: list[str] = []
    largest_page = 0
    page_count = 0
    async for page in engine._iter_fiber_census_pages():
        page_count += 1
        largest_page = max(largest_page, len(page))
        fiber_ids.extend(fiber.id for fiber in page)

    assert len(fiber_ids) == 10_001
    assert fiber_ids == sorted(fiber_ids)
    assert largest_page == 500
    assert page_count == 21
    assert len(storage.staged) == page_count + 1  # one immutable completion marker


@pytest.mark.asyncio
async def test_fiber_census_rejects_checkpoint_from_another_run() -> None:
    storage = _PagedStorage(_fibers(1))
    progress = _Progress("current-run")
    progress.strategy_state("merge").update(
        phase="fiber_census",
        cursor=json.dumps(
            {
                "version": 1,
                "run_id": "different-run",
                "brain_id": storage.current_brain_id,
                "strategy": "merge",
                "filter_fingerprint": "irrelevant-because-run-mismatch",
                "last_fiber_id": None,
                "next_page_index": 0,
                "complete": False,
            }
        ),
    )
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._progress_session = progress  # type: ignore[assignment]
    engine._active_strategy = ConsolidationStrategy.MERGE

    with pytest.raises(ConsolidationProgressError, match="does not match this run"):
        await engine._all_fibers_paged()

    assert storage.source_cursors == []


@pytest.mark.asyncio
async def test_fiber_census_crash_after_stage_write_retries_same_immutable_page() -> None:
    storage = _PagedStorage(_fibers(501))
    progress = _Progress(fail_first_checkpoint=True)
    first = ConsolidationEngine(storage, ConsolidationConfig())
    first._progress_session = progress  # type: ignore[assignment]
    first._active_strategy = ConsolidationStrategy.MERGE
    first._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]

    with pytest.raises(_SimulatedCrashError, match="after stage write"):
        await first._all_fibers_paged()

    staged_before_retry = dict(storage.staged[("run-census-test", "merge", 0)])
    resumed_progress = _Progress("run-census-test")
    resumed = ConsolidationEngine(storage, ConsolidationConfig())
    resumed._progress_session = resumed_progress  # type: ignore[assignment]
    resumed._active_strategy = ConsolidationStrategy.MERGE
    resumed._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]

    fibers = await resumed._all_fibers_paged()

    assert [fiber.id for fiber in fibers] == [fiber.id for fiber in _fibers(501)]
    assert storage.staged[("run-census-test", "merge", 0)] == staged_before_retry
    assert storage.source_cursors[1] is None


@pytest.mark.asyncio
async def test_fiber_census_concurrent_identical_writers_are_idempotent() -> None:
    storage = _PagedStorage(_fibers(501))

    def make_engine() -> ConsolidationEngine:
        engine = ConsolidationEngine(storage, ConsolidationConfig())
        engine._progress_session = _Progress("shared-run")  # type: ignore[assignment]
        engine._active_strategy = ConsolidationStrategy.MERGE
        engine._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]
        return engine

    first, second = await asyncio.gather(
        make_engine()._all_fibers_paged(),
        make_engine()._all_fibers_paged(),
    )

    assert [fiber.id for fiber in first] == [fiber.id for fiber in second]
    assert len(storage.staged) == 3
    markers = [row for row in storage.staged.values() if row.get("complete") is True]
    assert len(markers) == 1
    assert markers[0]["fibers"] == []


@pytest.mark.asyncio
async def test_fiber_census_stale_writer_cannot_replace_page_after_crash() -> None:
    storage = _PagedStorage(_fibers(501))
    stale_progress = _Progress(fail_first_checkpoint=True)
    stale = ConsolidationEngine(storage, ConsolidationConfig())
    stale._progress_session = stale_progress  # type: ignore[assignment]
    stale._active_strategy = ConsolidationStrategy.MERGE
    stale._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]

    with pytest.raises(_SimulatedCrashError, match="after stage write"):
        await stale._all_fibers_paged()

    stored_page = storage.staged[("run-census-test", "merge", 0)]
    storage.fibers[0] = storage.fibers[0].with_summary("changed by current owner")
    current = ConsolidationEngine(storage, ConsolidationConfig())
    current._progress_session = _Progress()  # type: ignore[assignment]
    current._active_strategy = ConsolidationStrategy.MERGE
    current._check_progress_budget = _completed_budget_check  # type: ignore[method-assign]

    with pytest.raises(ConsolidationProgressError, match="refusing a stale-owner overwrite"):
        await current._all_fibers_paged()

    assert storage.staged[("run-census-test", "merge", 0)] == stored_page


async def _completed_budget_check() -> None:
    return None

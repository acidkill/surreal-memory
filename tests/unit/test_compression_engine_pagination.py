"""Bounded compression census and durable intent precedence."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.compression import (
    CompressionEngine,
    CompressionResult,
    CompressionTier,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


async def _store() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="compression-pagination-test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


def _fiber(number: int, *, pending: bool = False) -> Fiber:
    return Fiber(
        id=f"fiber-{number:05d}",
        neuron_ids={f"anchor-{number}"},
        synapse_ids=set(),
        anchor_neuron_id=f"anchor-{number}",
        pinned=True,
        compression_tier=4 if pending else 0,
        metadata={"_compression_pending": {"target_tier": 1}} if pending else {},
        created_at=utcnow() - timedelta(days=30),
    )


@pytest.mark.asyncio
async def test_run_scans_beyond_ten_thousand_in_bounded_pages() -> None:
    storage = await _store()
    for number in range(10001):
        await storage.add_fiber(_fiber(number))
    get_page = storage.get_fibers_after_id
    storage.get_fibers_after_id = AsyncMock(side_effect=get_page)  # type: ignore[method-assign]
    storage.get_fibers = AsyncMock(side_effect=AssertionError("legacy top-N read"))  # type: ignore[method-assign]

    report = await CompressionEngine(storage).run()

    assert report.fibers_skipped == 10001
    assert report.fibers_deferred == 0
    assert storage.get_fibers_after_id.await_count > 40  # type: ignore[attr-defined]
    assert all(
        call.kwargs["limit"] <= 250
        for call in storage.get_fibers_after_id.await_args_list  # type: ignore[attr-defined]
    )
    await storage.close()


@pytest.mark.asyncio
async def test_run_reports_unread_page_when_budget_elapsed() -> None:
    storage = await _store()
    await storage.add_fiber(_fiber(0))
    await storage.add_fiber(_fiber(1))

    report = await CompressionEngine(storage).run(time_budget_seconds=-1)

    assert report.fibers_deferred >= 2
    assert report.fibers_skipped == 0
    assert "deferred" in report.summary()
    await storage.close()


@pytest.mark.asyncio
async def test_run_resumes_pending_intent_before_pinned_and_tier_checks() -> None:
    storage = await _store()
    fiber = _fiber(0, pending=True)
    await storage.add_fiber(fiber)
    engine = CompressionEngine(storage)
    expected = CompressionResult(
        fiber_id=fiber.id,
        original_tier=0,
        new_tier=1,
        original_token_count=10,
        compressed_token_count=5,
        entities_preserved=0,
        backup_created=False,
    )
    engine.compress_fiber = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    report = await engine.run()

    engine.compress_fiber.assert_awaited_once()  # type: ignore[attr-defined]
    assert engine.compress_fiber.await_args.args[:2] == (fiber, CompressionTier.EXTRACTIVE)  # type: ignore[attr-defined]
    assert report.fibers_compressed == 1
    await storage.close()


@pytest.mark.asyncio
async def test_legacy_storage_fails_visibly_at_ten_thousand_ceiling() -> None:
    storage = MagicMock()
    storage.current_brain_id = "brain-id"
    storage.get_fibers = AsyncMock(return_value=[_fiber(i) for i in range(10000)])
    del storage.get_fibers_after_id

    with pytest.raises(RuntimeError, match="requires get_fibers_after_id"):
        await CompressionEngine(storage).run()

"""Bound on concurrent activation traversals (#181).

`activate_from_multiple` gathered one traversal per anchor set with no limit, so
simultaneous traversals — and pending asyncio tasks — grew directly with the
number of inputs. The reporter measured 512 anchor sets producing 512 concurrent
activations.

The test measures peak concurrency the same way rather than asserting on the
call shape: a semaphore that is created but never awaited would pass a
structural check and bound nothing.
"""

from __future__ import annotations

import asyncio

import pytest

from surreal_memory.core.brain import BrainConfig
from surreal_memory.engine.activation import SpreadingActivation
from surreal_memory.storage.memory_store import InMemoryStorage

ANCHOR_SETS = 128


class _ConcurrencyProbe:
    """Replaces a single activation with a coroutine that stays pending."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self._release = asyncio.Event()

    async def __call__(self, anchors, max_hops=None, anchor_activations=None):
        self.active += 1
        self.peak = max(self.peak, self.active)
        # Yield so every task that CAN start does start before any finishes —
        # without this the loop could serialise them and hide the defect.
        await asyncio.sleep(0)
        self.active -= 1
        return {}, []


@pytest.mark.asyncio
async def test_concurrent_traversals_are_bounded() -> None:
    engine = SpreadingActivation(InMemoryStorage(), BrainConfig())
    probe = _ConcurrencyProbe()
    engine.activate = probe  # type: ignore[method-assign]

    anchor_sets = [[f"n-{i}"] for i in range(ANCHOR_SETS)]
    await engine.activate_from_multiple(anchor_sets)

    assert probe.peak < ANCHOR_SETS, (
        f"all {ANCHOR_SETS} traversals ran at once — concurrency grows with the "
        "number of anchor sets, which is the defect"
    )


@pytest.mark.asyncio
async def test_every_anchor_set_is_still_processed() -> None:
    """A bound must throttle the work, not drop it."""
    engine = SpreadingActivation(InMemoryStorage(), BrainConfig())
    seen: list[str] = []

    async def _record(anchors, max_hops=None, anchor_activations=None):
        seen.extend(anchors)
        return {}, []

    engine.activate = _record  # type: ignore[method-assign]

    anchor_sets = [[f"n-{i}"] for i in range(ANCHOR_SETS)]
    await engine.activate_from_multiple(anchor_sets)

    assert len(seen) == ANCHOR_SETS

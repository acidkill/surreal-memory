"""Regression: SurrealDB import_brain must not silently drop colliding records.

Each neuron/synapse/fiber insert used to be wrapped in a bare `except: pass`,
so a duplicate record id — trivially reachable by importing a snapshot into a
newly-named brain while the original was still around, since `_to_surreal_id`
folds neuron id to `neuron:<id>` without a brain component — vanished with no
counter, no log line, and no signal to the caller.

The fix keeps the return type stable (interface `NeuralStorage.import_brain`
must stay `-> str`) and adds a `logger.warning` with the shortfall per loop
when at least one record failed to land.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import BrainSnapshot
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _snapshot_with(neurons: int, synapses: int, fibers: int) -> BrainSnapshot:
    """Minimal snapshot: unique ids so a bare add_* would happily accept them."""
    now = datetime(2026, 9, 3, tzinfo=UTC)
    return BrainSnapshot(
        brain_id="src-brain",
        brain_name="src",
        exported_at=now,
        version="1",
        neurons=[
            {
                "id": f"n{i}",
                "type": "concept",
                "content": f"neuron {i}",
                "metadata": {},
                "created_at": now.isoformat(),
            }
            for i in range(neurons)
        ],
        synapses=[
            {
                "id": f"s{i}",
                "source_id": "n0",
                "target_id": "n0",
                "type": "similar_to",
                "weight": 1.0,
                "direction": "forward",
                "metadata": {},
                "created_at": now.isoformat(),
            }
            for i in range(synapses)
        ],
        fibers=[
            {
                "id": f"f{i}",
                "neuron_ids": [],
                "synapse_ids": [],
                "anchor_neuron_id": "",
                "pathway": [],
                "conductivity": 1.0,
                "salience": 0.0,
            }
            for i in range(fibers)
        ],
        config={},
        metadata={},
    )


def _bare_storage() -> SurrealDBStorage:
    """Instance without connecting — __init__ needs env; tests only need methods."""
    store: SurrealDBStorage = object.__new__(SurrealDBStorage)
    store._current_brain_id = None  # type: ignore[attr-defined]
    return store


@pytest.mark.asyncio
async def test_import_brain_warns_when_neurons_are_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _bare_storage()
    store.set_brain = lambda bid: setattr(store, "_current_brain_id", bid)  # type: ignore[assignment,method-assign]
    store.save_brain = AsyncMock()  # type: ignore[method-assign]

    calls: dict[str, int] = {"neuron": 0, "synapse": 0, "fiber": 0}

    async def _add_neuron(_neuron: Any) -> None:
        calls["neuron"] += 1
        if calls["neuron"] == 2:
            raise RuntimeError("duplicate id n1")

    async def _add_synapse(_synapse: Any) -> None:
        calls["synapse"] += 1

    async def _add_fiber(_fiber: Any) -> None:
        calls["fiber"] += 1

    store.add_neuron = _add_neuron  # type: ignore[method-assign]
    store.add_synapse = _add_synapse  # type: ignore[method-assign]
    store.add_fiber = _add_fiber  # type: ignore[method-assign]

    snapshot = _snapshot_with(neurons=3, synapses=1, fibers=1)

    with caplog.at_level(logging.WARNING, logger="surreal_memory.storage.surrealdb.store"):
        bid = await store.import_brain(snapshot, "restore-test")

    # Interface stays str — no breaking change.
    assert bid == "restore-test"

    # Fix: the shortfall MUST surface as a warning that names both counts.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "neuron" in r.getMessage().lower() and "2" in r.getMessage() and "3" in r.getMessage()
        for r in warnings
    ), f"expected 'imported 2 of 3 neurons' warning; got: {[r.getMessage() for r in warnings]}"


@pytest.mark.asyncio
async def test_import_brain_silent_when_everything_lands(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _bare_storage()
    store.set_brain = lambda bid: setattr(store, "_current_brain_id", bid)  # type: ignore[assignment,method-assign]
    store.save_brain = AsyncMock()  # type: ignore[method-assign]
    store.add_neuron = AsyncMock()  # type: ignore[method-assign]
    store.add_synapse = AsyncMock()  # type: ignore[method-assign]
    store.add_fiber = AsyncMock()  # type: ignore[method-assign]

    snapshot = _snapshot_with(neurons=2, synapses=1, fibers=1)

    with caplog.at_level(logging.WARNING, logger="surreal_memory.storage.surrealdb.store"):
        await store.import_brain(snapshot, "clean-brain")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [], f"no drops → no warning; got: {[r.getMessage() for r in warnings]}"

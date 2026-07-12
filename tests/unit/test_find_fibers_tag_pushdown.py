"""Regression: find_fibers must apply the tag filter BEFORE the LIMIT.

Motivated by the LangChain chat-history adapter (U9), which reads a session's turns via
find_fibers(tags={lc-session:<id>}). If the backend truncates by LIMIT first and only
then filters by tag, a tagged subset can fall outside the fetch window on a large brain
and silently vanish. This pins the fix for the in-process backends (SQLite pushes an
EXISTS json_each predicate; InMemory filters before truncating). SurrealDB is covered by
the sibling live tests.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.sqlite_store import SQLiteStorage

_TAG = "lc-session:keep-me"


@pytest.fixture(params=["memory", "sqlite"])
async def storage(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    if request.param == "memory":
        s: object = InMemoryStorage()
        brain = Brain.create(name="tagpush")
        await s.save_brain(brain)  # type: ignore[attr-defined]
        s.set_brain(brain.id)  # type: ignore[attr-defined]
        yield s
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            sq = SQLiteStorage(Path(tmpdir) / "t.db")
            await sq.initialize()
            brain = Brain.create(name="tagpush")
            await sq.save_brain(brain)
            sq.set_brain(brain.id)
            yield sq
            await sq.close()


async def _add(storage: object, summary: str, salience: float, tag: str | None) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=summary)
    await storage.add_neuron(neuron)  # type: ignore[attr-defined]
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=summary,
        agent_tags={tag} if tag else set(),
    ).with_salience(salience)
    await storage.add_fiber(fiber)  # type: ignore[attr-defined]
    return fiber


async def test_tag_filter_applies_before_limit(storage: object) -> None:
    # High-salience decoys (no tag) would win a naive `ORDER BY salience DESC LIMIT k`
    # and starve the low-salience tagged fibers — which is exactly the bug.
    for i in range(3):
        await _add(storage, f"decoy {i}", salience=0.9, tag=None)
    kept = [
        await _add(storage, "keep 1", salience=0.1, tag=_TAG),
        await _add(storage, "keep 2", salience=0.1, tag=_TAG),
    ]

    found = await storage.find_fibers(tags={_TAG}, limit=2)  # type: ignore[attr-defined]
    ids = {f.id for f in found}
    assert ids == {k.id for k in kept}  # tagged rows survive the tight LIMIT


async def test_no_tag_still_returns_by_salience(storage: object) -> None:
    # Control: without a tag filter, the LIMIT is a plain top-k by salience.
    await _add(storage, "high", salience=0.9, tag=None)
    await _add(storage, "low", salience=0.1, tag=None)

    found = await storage.find_fibers(limit=1)  # type: ignore[attr-defined]
    assert len(found) == 1
    assert found[0].summary == "high"

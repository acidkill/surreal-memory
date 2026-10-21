"""Live SurrealDB regression: census rows are fingerprinted and compared as stored.

The Python SDK sends ``None`` as NONE, and SurrealDB keeps no object key whose value
is NONE, at any depth. A census row that is fingerprinted or compared with such keys
never matches the row read back. These tests cover two cases that a fixed list of
restored fields does not reach: fibers whose stored ``metadata`` holds a null, taken
through a real merge, and a dedup completion marker written a second time by a late
owner.

Runs only against the explicitly opted-in loopback SMEM_TEST_SURREALDB_URL; the fixture
creates a unique database and uses the disposable test credentials.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.dedup_census import DedupAnchorCensus
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

TEST_SURREALDB_URL = os.getenv("SMEM_TEST_SURREALDB_URL")
TEST_AUTH = ("root", "root")


def _is_loopback_test_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_test_url(TEST_SURREALDB_URL),
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL",
    ),
]


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SurrealDBStorage]:
    assert TEST_SURREALDB_URL is not None
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace="smem_ci",
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="census-stored-form-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def test_merge_merges_fibers_whose_stored_metadata_holds_null(
    store: SurrealDBStorage,
) -> None:
    neurons = [Neuron.create(NeuronType.CONCEPT, f"census stored shared {i}") for i in range(3)]
    for neuron in neurons:
        await store.add_neuron(neuron)
    for rank, salience in (("low", 0.2), ("high", 0.8)):
        # The same neurons twice, so the two fibers form one merge group.
        fiber = Fiber.create(
            neuron_ids={neuron.id for neuron in neurons},
            synapse_ids=set(),
            anchor_neuron_id=neurons[0].id,
            metadata={"rank": rank},
        )
        await store.add_fiber(replace(fiber, salience=salience))
    # A null written by other means (plain SurrealQL, an import) is stored and read back.
    await store._query(
        "UPDATE fiber SET metadata.note = NULL, metadata.nested = { gone: NULL, kept: 1 }, "
        "metadata.items = [{ gone: NULL, kept: 2 }, NULL]"
    )
    await store._query("UPDATE fiber SET metadata.shared = 'value' WHERE metadata.rank = 'low'")
    await store._query("UPDATE fiber SET metadata.shared = NULL WHERE metadata.rank = 'high'")
    fibers = await store.get_fibers_after_id(None, limit=10)
    assert len(fibers) == 2
    for fiber in fibers:
        assert fiber.metadata["note"] is None
        assert fiber.metadata["nested"] == {"gone": None, "kept": 1}
        assert fiber.metadata["items"] == [{"gone": None, "kept": 2}, None]
    assert {fiber.metadata["rank"]: fiber.metadata["shared"] for fiber in fibers} == {
        "low": "value",
        "high": None,
    }

    report = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.MERGE]
    )

    assert report.extra.get("failed_strategies") in (None, []), report.extra
    assert report.extra.get("consolidation_status") == "completed", report.extra
    assert report.fibers_merged == 2
    merged = await store.get_fibers_after_id(None, limit=10)
    assert len(merged) == 1
    assert merged[0].metadata["rank"] == "high"
    # The more salient source held a null for this key; a null never wins a collision.
    assert merged[0].metadata["shared"] == "value"
    assert "note" not in merged[0].metadata


class _MarkerNotVisible:
    """A late owner: its lookup of the completion marker returns nothing."""

    def __init__(self, storage: SurrealDBStorage) -> None:
        self._storage = storage
        self.hidden = 0

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        rows = await self._storage._query(sql, **params)
        if "complete = true" in sql:
            self.hidden += len(rows)
            return []
        return rows

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage, name)


async def test_dedup_census_accepts_its_completion_marker_written_again(
    store: SurrealDBStorage,
) -> None:
    for i in range(3):
        anchor = Neuron.create(NeuronType.CONCEPT, f"census stored anchor {i}")
        anchor.metadata["is_anchor"] = True
        await store.add_neuron(anchor)
    checkpoints: list[dict[str, Any]] = []

    async def checkpoint(phase: str, **state: Any) -> None:
        checkpoints.append({"phase": phase, **state})

    async def within_budget() -> None:
        return None

    def census(storage: Any, state: dict[str, Any]) -> DedupAnchorCensus:
        return DedupAnchorCensus(
            storage,
            run_id="run-census-stored-form",
            brain_id=str(store.current_brain_id),
            strategy=ConsolidationStrategy.DEDUP.value,
            reference_time=None,
            checkpoint_state=state,
            checkpoint=checkpoint,
            check_budget=within_budget,
        )

    assert await census(store, {}).build() == 3
    late_owner = _MarkerNotVisible(store)

    assert await census(late_owner, dict(checkpoints[-1])).build() == 3
    assert late_owner.hidden == 1

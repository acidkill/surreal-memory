"""Paging tests for ``get_synapses(..., offset=...)`` across backends.

The consolidation passes used to snapshot the whole synapse table in one
response. On a brain large enough that is how "[Errno 104] Connection reset by
peer" is earned. Paging is only a fix if consecutive pages neither overlap nor
skip, which requires a stable order -- so these tests assert the partition
property, not merely that an offset was accepted.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.memory_store import InMemoryStorage

try:  # pragma: no cover - mirrors tests/unit/test_surrealdb_store.py
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()

TOTAL = 10
PAGE = 3


def _make_neurons(count: int) -> list[Neuron]:
    return [Neuron.create(type=NeuronType.CONCEPT, content=f"node-{i}") for i in range(count)]


def _make_synapses(neurons: list[Neuron]) -> list[Synapse]:
    """A simple chain n0->n1->n2->... so every synapse is a distinct pair."""
    return [
        Synapse.create(
            source_id=neurons[i].id,
            target_id=neurons[i + 1].id,
            type=SynapseType.SIMILAR_TO,
            weight=0.5,
        )
        for i in range(len(neurons) - 1)
    ]


def _assert_pages_partition(pages: list[list[Synapse]], expected_total: int) -> None:
    """Every row appears exactly once across the pages."""
    seen = [s.id for page in pages for s in page]
    assert len(seen) == expected_total, f"expected {expected_total} rows, paged {len(seen)}"
    assert len(set(seen)) == expected_total, "pages overlap -- a row was returned twice"


async def _page_all(storage: object, page_size: int) -> list[list[Synapse]]:
    pages: list[list[Synapse]] = []
    offset = 0
    while True:
        page = await storage.get_synapses(limit=page_size, offset=offset)  # type: ignore[attr-defined]
        if not page:
            break
        pages.append(page)
        offset += page_size
        if len(page) < page_size:
            break
    return pages


class TestInMemoryPaging:
    @pytest.fixture
    def storage(self) -> InMemoryStorage:
        store = InMemoryStorage()
        brain = Brain.create(name="paging", config=BrainConfig(), owner_id="test")
        store._brains[brain.id] = brain
        store.set_brain(brain.id)
        return store

    async def test_pages_partition_the_result_set(self, storage: InMemoryStorage) -> None:
        neurons = _make_neurons(TOTAL)
        for neuron in neurons:
            await storage.add_neuron(neuron)
        synapses = _make_synapses(neurons)
        for synapse in synapses:
            await storage.add_synapse(synapse)

        pages = await _page_all(storage, PAGE)

        _assert_pages_partition(pages, len(synapses))

    async def test_offset_beyond_the_end_is_empty(self, storage: InMemoryStorage) -> None:
        neurons = _make_neurons(3)
        for neuron in neurons:
            await storage.add_neuron(neuron)
        for synapse in _make_synapses(neurons):
            await storage.add_synapse(synapse)

        assert await storage.get_synapses(limit=PAGE, offset=999) == []

    async def test_default_offset_is_unchanged_behaviour(self, storage: InMemoryStorage) -> None:
        neurons = _make_neurons(5)
        for neuron in neurons:
            await storage.add_neuron(neuron)
        for synapse in _make_synapses(neurons):
            await storage.add_synapse(synapse)

        assert await storage.get_synapses() == await storage.get_synapses(offset=0)


class TestSQLitePaging:
    @pytest_asyncio.fixture
    async def storage(self, tmp_path: object):
        store = InMemoryStorage()
        brain = Brain.create(name="paging", config=BrainConfig(), owner_id="test")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        return store

    async def test_pages_partition_the_result_set(self, storage: InMemoryStorage) -> None:
        neurons = _make_neurons(TOTAL)
        for neuron in neurons:
            await storage.add_neuron(neuron)
        synapses = _make_synapses(neurons)
        for synapse in synapses:
            await storage.add_synapse(synapse)

        pages = await _page_all(storage, PAGE)

        _assert_pages_partition(pages, len(synapses))

    async def test_offset_beyond_the_end_is_empty(self, storage: InMemoryStorage) -> None:
        neurons = _make_neurons(3)
        for neuron in neurons:
            await storage.add_neuron(neuron)
        for synapse in _make_synapses(neurons):
            await storage.add_synapse(synapse)

        assert await storage.get_synapses(limit=PAGE, offset=999) == []

    async def test_filters_still_apply_while_paging(self, storage: InMemoryStorage) -> None:
        neurons = _make_neurons(5)
        for neuron in neurons:
            await storage.add_neuron(neuron)
        for synapse in _make_synapses(neurons):
            await storage.add_synapse(synapse)

        page = await storage.get_synapses(type=SynapseType.ALIAS, limit=PAGE, offset=0)

        assert page == []


class TestSurrealDBQueryShape:
    """The SurrealDB backend builds SurrealQL by hand, so assert the text."""

    def _storage(self) -> object:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage(url="http://localhost:8001")
        storage._get_brain_id = lambda: "brain-1"  # type: ignore[method-assign]
        storage._query = AsyncMock(return_value=[])  # type: ignore[method-assign]
        return storage

    async def test_paged_read_orders_and_starts(self) -> None:
        storage = self._storage()

        await storage.get_synapses(limit=5000, offset=5000)  # type: ignore[attr-defined]

        query = storage._query.await_args.args[0]
        assert "ORDER BY id" in query
        assert "LIMIT 5000" in query
        assert "START 5000" in query
        # START must follow LIMIT -- SurrealQL rejects the other order.
        assert query.index("LIMIT") < query.index("START")

    async def test_first_page_has_no_start_clause(self) -> None:
        storage = self._storage()

        await storage.get_synapses(limit=5000)  # type: ignore[attr-defined]

        query = storage._query.await_args.args[0]
        assert "ORDER BY id" in query
        assert "START" not in query

    async def test_unbounded_read_is_left_alone(self) -> None:
        """No limit means no window, so ordering it would only cost time."""
        storage = self._storage()

        await storage.get_synapses()  # type: ignore[attr-defined]

        query = storage._query.await_args.args[0]
        assert "ORDER BY" not in query
        assert "LIMIT" not in query

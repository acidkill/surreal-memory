"""Pytest configuration and fixtures."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Generator
from datetime import datetime

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.memory_store import InMemoryStorage

# Import the real surrealdb SDK early (when installed) so it is present in
# sys.modules before any unit test module runs its
# `if "surrealdb" not in sys.modules: sys.modules["surrealdb"] = MagicMock()`
# stub. Otherwise that MagicMock leaks into the live SurrealDB tests in the same
# serial session, breaking `from surrealdb import AsyncSurreal` and awaits on the
# fake connection. No-op when surrealdb isn't installed (the stubs then apply).
with contextlib.suppress(ImportError):
    import surrealdb  # noqa: F401


@pytest.fixture
def brain_config() -> BrainConfig:
    """Create a test brain configuration."""
    return BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.2,
        max_spread_hops=4,
        max_context_tokens=500,
    )


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    """Create a test brain."""
    return Brain.create(
        name="test_brain",
        config=brain_config,
        owner_id="test_user",
    )


@pytest_asyncio.fixture
async def storage(brain: Brain) -> AsyncGenerator[InMemoryStorage, None]:
    """Create an in-memory storage instance with brain context."""
    store = InMemoryStorage()
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store


@pytest.fixture
def sample_neurons() -> list[Neuron]:
    """Create sample neurons for testing."""
    return [
        Neuron.create(
            type=NeuronType.TIME,
            content="3pm",
            metadata={"hour": 15},
            neuron_id="time-1",
        ),
        Neuron.create(
            type=NeuronType.SPATIAL,
            content="coffee shop",
            metadata={},
            neuron_id="spatial-1",
        ),
        Neuron.create(
            type=NeuronType.ENTITY,
            content="Alice",
            metadata={"entity_type": "person"},
            neuron_id="entity-1",
        ),
        Neuron.create(
            type=NeuronType.ACTION,
            content="discussed",
            metadata={},
            neuron_id="action-1",
        ),
        Neuron.create(
            type=NeuronType.CONCEPT,
            content="API design",
            metadata={},
            neuron_id="concept-1",
        ),
    ]


@pytest.fixture
def sample_synapses(sample_neurons: list[Neuron]) -> list[Synapse]:
    """Create sample synapses connecting the sample neurons."""
    # Get neuron IDs
    time_n = sample_neurons[0]
    spatial_n = sample_neurons[1]
    entity_n = sample_neurons[2]
    action_n = sample_neurons[3]
    concept_n = sample_neurons[4]

    return [
        Synapse.create(
            source_id=action_n.id,
            target_id=time_n.id,
            type=SynapseType.HAPPENED_AT,
            weight=0.9,
            synapse_id="syn-1",
        ),
        Synapse.create(
            source_id=action_n.id,
            target_id=spatial_n.id,
            type=SynapseType.AT_LOCATION,
            weight=0.8,
            synapse_id="syn-2",
        ),
        Synapse.create(
            source_id=action_n.id,
            target_id=entity_n.id,
            type=SynapseType.INVOLVES,
            weight=0.9,
            synapse_id="syn-3",
        ),
        Synapse.create(
            source_id=action_n.id,
            target_id=concept_n.id,
            type=SynapseType.RELATED_TO,
            weight=0.7,
            synapse_id="syn-4",
        ),
        Synapse.create(
            source_id=entity_n.id,
            target_id=concept_n.id,
            type=SynapseType.RELATED_TO,
            weight=0.6,
            synapse_id="syn-5",
        ),
    ]


@pytest_asyncio.fixture
async def populated_storage(
    storage: InMemoryStorage,
    sample_neurons: list[Neuron],
    sample_synapses: list[Synapse],
) -> InMemoryStorage:
    """Create storage populated with sample data."""
    for neuron in sample_neurons:
        await storage.add_neuron(neuron)

    for synapse in sample_synapses:
        await storage.add_synapse(synapse)

    # Create a fiber
    fiber = Fiber.create(
        neuron_ids={n.id for n in sample_neurons},
        synapse_ids={s.id for s in sample_synapses},
        anchor_neuron_id=sample_neurons[3].id,  # action neuron
        time_start=datetime(2024, 1, 1, 15, 0),
        time_end=datetime(2024, 1, 1, 16, 0),
        fiber_id="fiber-1",
    )
    await storage.add_fiber(fiber)

    return storage


@pytest.fixture
def reference_time() -> datetime:
    """Standard reference time for tests."""
    return datetime(2024, 2, 4, 14, 30, 0)


@pytest.fixture(autouse=True)
def _close_leaked_aiosqlite_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Stop aiosqlite worker threads a test leaves behind.

    aiosqlite runs every connection on a dedicated NON-daemon worker thread
    that blocks on its queue until ``close()`` delivers the stop sentinel. A
    fixture that initialises ``SQLiteStorage`` without closing it therefore
    leaks a live thread, and any such thread still referenced at interpreter
    exit is joined in ``threading._shutdown`` BEFORE the final garbage
    collection — so a fully green run hangs forever after the summary (and the
    stray ``RuntimeError: Event loop is closed`` warnings in the suite are the
    same leak delivering late results onto closed per-test loops).

    Track connections opened during each test and stop the stragglers at
    teardown. ``Connection.stop()`` enqueues the sentinel without needing an
    event loop, so this works from a synchronous fixture.
    """
    import aiosqlite

    live: list[aiosqlite.Connection] = []
    orig_init = aiosqlite.Connection.__init__

    def tracking_init(self: aiosqlite.Connection, *args: object, **kwargs: object) -> None:
        orig_init(self, *args, **kwargs)
        live.append(self)

    monkeypatch.setattr(aiosqlite.Connection, "__init__", tracking_init)
    yield
    for conn in live:
        thread = getattr(conn, "_thread", None)
        if thread is None or not thread.is_alive():
            continue
        with contextlib.suppress(Exception):
            conn.stop()
        thread.join(timeout=2.0)

"""Shared fixtures for FalkorDB storage tests.

All tests in this directory require a running FalkorDB instance.
Set FALKORDB_TEST_HOST / FALKORDB_TEST_PORT env vars to override defaults.
Tests are auto-skipped when FalkorDB is not reachable or not installed.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
import pytest_asyncio

FALKORDB_HOST = os.environ.get("FALKORDB_TEST_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_TEST_PORT", "6379"))


def falkordb_available(host: str = FALKORDB_HOST, port: int = FALKORDB_PORT) -> bool:
    """Check if FalkorDB package + server are both available."""
    try:
        from falkordb import FalkorDB as FalkorDBClient

        db = FalkorDBClient(host=host, port=port)
        g = db.select_graph("__ping__")
        g.query("RETURN 1")
        g.delete()
        return True
    except Exception:
        return False


_FALKORDB_AVAILABLE = falkordb_available()
_SKIP_REASON = f"FalkorDB not available at {FALKORDB_HOST}:{FALKORDB_PORT}"


@pytest.fixture(autouse=True)
def _require_falkordb() -> None:
    """Skip all tests when FalkorDB is not available.

    Using autouse fixture instead of pytestmark ensures the skip
    happens before async fixtures are set up.
    """
    if not _FALKORDB_AVAILABLE:
        pytest.skip(_SKIP_REASON)


def _import_deps() -> dict[str, Any]:
    """Lazy-import FalkorDB storage deps. Only called when tests actually run."""
    from neural_memory.core.brain import Brain
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.core.synapse import Synapse, SynapseType
    from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage

    return {
        "Brain": Brain,
        "Neuron": Neuron,
        "NeuronType": NeuronType,
        "Synapse": Synapse,
        "SynapseType": SynapseType,
        "FalkorDBStorage": FalkorDBStorage,
    }


@pytest.fixture
def brain_id() -> str:
    """Unique brain ID per test to avoid collisions."""
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest_asyncio.fixture
async def storage(brain_id: str) -> Any:
    """FalkorDB storage with a clean test brain."""
    deps = _import_deps()
    store = deps["FalkorDBStorage"](host=FALKORDB_HOST, port=FALKORDB_PORT)
    await store.initialize()

    brain = deps["Brain"].create(name=brain_id, brain_id=brain_id)
    await store.save_brain(brain)
    await store.set_brain_with_indexes(brain_id)

    yield store

    # Cleanup: clear test brain graph
    try:
        await store.clear(brain_id)
    except Exception:
        pass
    await store.close()


@pytest.fixture
def make_neuron() -> Any:
    """Factory for creating test neurons."""
    deps = _import_deps()
    neuron_cls = deps["Neuron"]
    neuron_type_cls = deps["NeuronType"]

    def _make(
        type: Any = None,
        content: str = "test neuron",
        metadata: dict | None = None,
    ) -> Any:
        return neuron_cls.create(
            type=type or neuron_type_cls.CONCEPT,
            content=content,
            metadata=metadata or {},
        )

    return _make


@pytest.fixture
def sample_neurons() -> list[Any]:
    """Five neurons of different types."""
    deps = _import_deps()
    neuron_cls = deps["Neuron"]
    nt = deps["NeuronType"]
    return [
        neuron_cls.create(type=nt.TIME, content="Monday morning"),
        neuron_cls.create(type=nt.SPATIAL, content="Office building"),
        neuron_cls.create(type=nt.ENTITY, content="Python language"),
        neuron_cls.create(type=nt.ACTION, content="Write code"),
        neuron_cls.create(type=nt.CONCEPT, content="Neural memory"),
    ]


@pytest.fixture
def sample_synapses(sample_neurons: list[Any]) -> list[Any]:
    """Synapses connecting sample neurons in a chain."""
    deps = _import_deps()
    synapse_cls = deps["Synapse"]
    st = deps["SynapseType"]
    n = sample_neurons
    return [
        synapse_cls.create(n[0].id, n[1].id, st.HAPPENED_AT, weight=0.8),
        synapse_cls.create(n[1].id, n[2].id, st.AT_LOCATION, weight=0.6),
        synapse_cls.create(n[2].id, n[3].id, st.CAUSED_BY, weight=0.7),
        synapse_cls.create(n[3].id, n[4].id, st.RELATES_TO, weight=0.9),
    ]

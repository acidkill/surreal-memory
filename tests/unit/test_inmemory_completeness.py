"""InMemoryStorage has to implement the whole NeuralStorage interface.

It is the test double the suite runs on, so a method it leaves as an inherited
stub is a feature that silently loses its coverage: `hasattr` says yes, the
call raises. Sixty-four methods sat in that state — sources, alerts, cognitive
state, knowledge gaps, devices, change log, Merkle sync, depth priors,
compression backups, neuron snapshots, lifecycle flags, hot index — tolerable
only while SQLite was around to back the tests that needed them.
"""

from __future__ import annotations

import inspect

from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage


def _interface_methods() -> set[str]:
    return {
        name
        for name, member in inspect.getmembers(NeuralStorage, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def _inherited_stubs(cls: type) -> set[str]:
    """Interface methods `cls` never overrode, still raising NotImplementedError."""
    stubs: set[str] = set()
    for name in _interface_methods():
        impl = getattr(cls, name, None)
        if impl is None:
            stubs.add(name)
            continue
        if not impl.__qualname__.startswith("NeuralStorage"):
            continue
        try:
            if "NotImplementedError" in inspect.getsource(impl):
                stubs.add(name)
        except OSError:  # pragma: no cover - source always available in-tree
            continue
    return stubs


def test_the_interface_is_non_trivial() -> None:
    # Guards the introspection itself: an empty set here would make the
    # assertion below pass without having checked anything.
    assert len(_interface_methods()) > 100


def test_in_memory_storage_implements_every_method() -> None:
    # These optimized SurrealDB operations and process-durable checkpoints are
    # intentionally optional; an in-process adapter must not claim durability.
    # Semantic source fences depend on a retained database changefeed, while
    # staged semantic snapshots must survive process restarts.
    optional_capabilities = {
        "assert_semantic_source_unchanged",
        "capture_semantic_source_token",
        "load_semantic_discovery_state",
        "save_semantic_discovery_state",
        "acquire_consolidation_lease",
        "claim_consolidation_progress",
        "create_consolidation_progress",
        "find_neurons_after_id",
        "get_connected_neuron_ids_for",
        "get_consolidation_progress",
        "get_fiber_neuron_ids_for",
        "get_synapses_after_id",
        "get_synapses_by_ids",
        "get_synapses_for_sources",
        "get_synapse_prune_page",
        "get_synapse_target_counts_for_sources",
        "release_consolidation_lease",
        "remove_synapse_refs_from_fibers",
        "renew_consolidation_lease",
        "save_consolidation_progress",
        "start_consolidation_progress",
    }
    missing = _inherited_stubs(InMemoryStorage)

    assert missing == optional_capabilities, (
        "only the explicitly optional SurrealDB capabilities may remain stubs; "
        f"missing={sorted(missing)}, expected={sorted(optional_capabilities)}"
    )

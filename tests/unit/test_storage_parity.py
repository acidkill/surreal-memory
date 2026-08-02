"""Every storage backend must implement the whole NeuralStorage interface.

Six methods — pin/unpin, pinned-neuron lookup, graph density and the
document-training file tracking — used to live on the old SQLite backend
alone, and were never declared on ``NeuralStorage``. Every caller reached for
them through ``hasattr`` and silently did nothing when they were missing, so
on SurrealDB (the production backend since 2.0.0) decay and prune deleted
pinned memories, ``smem_pin`` refused every action, ``smem train`` re-encoded
its whole corpus on each run, and ``activation_strategy="auto"`` never left
classic BFS. Nothing failed, because nothing checked.

These tests are the check. Adding a capability to one backend and forgetting the
others now fails here instead of going dark in production. SQLite is gone as of
3.0.0; the same protection now compares the two remaining backends directly.
"""

from __future__ import annotations

import inspect

from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.shared_store import SharedStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

# The capabilities this module exists to protect. Each must resolve to a real
# implementation — not the interface's inert fallback — on every backend that
# actually stores memories.
_CONTRACT = (
    "pin_fibers",
    "get_pinned_neuron_ids",
    "list_pinned_fibers",
    "get_graph_density",
    "upsert_training_file",
    "get_training_file_by_hash",
    "update_training_file_progress",
    "get_training_stats",
)

# SharedStorage is a thin HTTP client against a remote server, with no local
# tables to keep any of this in; it inherits the interface's no-op fallbacks.
_PERSISTENT_BACKENDS = (SurrealDBStorage, InMemoryStorage)
_ALL_BACKENDS = (*_PERSISTENT_BACKENDS, SharedStorage)


def _public_methods(cls: type) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def test_contract_is_declared_on_the_interface() -> None:
    """The whole point: these are interface methods, not one-backend trivia."""
    missing = set(_CONTRACT) - _public_methods(NeuralStorage)
    assert not missing, f"not declared on NeuralStorage: {sorted(missing)}"


def test_persistent_backends_override_every_contract_method() -> None:
    """A backend inheriting the base no-op silently loses the feature."""
    for backend in _PERSISTENT_BACKENDS:
        inherited = [
            name for name in _CONTRACT if getattr(backend, name) is getattr(NeuralStorage, name)
        ]
        assert not inherited, (
            f"{backend.__name__} inherits NeuralStorage's inert fallback for "
            f"{sorted(inherited)} — the feature is dead on that backend"
        )


def test_every_backend_implements_the_full_interface() -> None:
    """No backend may be missing an interface method outright."""
    declared = _public_methods(NeuralStorage)
    for backend in _ALL_BACKENDS:
        missing = declared - _public_methods(backend)
        assert not missing, f"{backend.__name__} is missing: {sorted(missing)}"


# Methods reachable on exactly one of the two persistent backends, undeclared
# on NeuralStorage. Each either has a caller that guards with try/except
# (degrades quietly — a real but non-crashing gap, tracked separately) or is a
# backend-specific internal/optimization with no cross-backend caller at all.
# This list is a ratchet: it may shrink, never grow.
_KNOWN_ASYMMETRIC_ONLY = frozenset(
    {
        # InMemoryStorage-only: a thin convenience wrapper around the batched
        # get_depth_priors_batch, which SurrealDB implements instead.
        "get_depth_priors",
        # SurrealDB-only: connection lifecycle / batch-optimization / stats
        # internals with no direct InMemoryStorage equivalent needed.
        "cap_tool_events",
        "count_activated_neuron_states",
        "delete_neurons_batch",
        "delete_synapses_batch",
        "find_neurons_by_embedding",
        "find_neurons_by_ids",
        "get_connected_neuron_ids",
        "get_edges_for_neurons",
        "get_synapse_degrees",
        "get_tool_stats",
        "get_tool_stats_by_period",
        "initialize",
        "list_brain_names",
        "prune_old_events",
        "update_neuron_embeddings",
    }
)


def test_no_new_undeclared_asymmetric_capability() -> None:
    """Nothing NEW may become reachable on exactly one persistent backend.

    A method that exists on SurrealDB or InMemoryStorage but not both, and
    isn't declared on NeuralStorage, is precisely the shape of the original
    bug: callers can only reach it via ``hasattr`` or a swallowed exception on
    whichever backend lacks it. Anything genuinely backend-specific belongs
    behind a leading underscore.
    """
    declared = _public_methods(NeuralStorage)
    surreal = _public_methods(SurrealDBStorage) - declared
    memory = _public_methods(InMemoryStorage) - declared

    asymmetric = surreal.symmetric_difference(memory)
    unexpected = asymmetric - _KNOWN_ASYMMETRIC_ONLY
    assert not unexpected, (
        "these exist on exactly one persistent backend and aren't declared on "
        f"NeuralStorage: {sorted(unexpected)}"
    )


def test_known_asymmetric_list_has_no_stale_entries() -> None:
    """Keep the ratchet honest — a method both backends now share must leave the list."""
    declared = _public_methods(NeuralStorage)
    surreal = _public_methods(SurrealDBStorage) - declared
    memory = _public_methods(InMemoryStorage) - declared
    asymmetric = surreal.symmetric_difference(memory)

    stale = _KNOWN_ASYMMETRIC_ONLY - asymmetric
    assert not stale, (
        "these are listed as single-backend-only but no longer are — drop "
        f"them from _KNOWN_ASYMMETRIC_ONLY: {sorted(stale)}"
    )

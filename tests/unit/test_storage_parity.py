"""Every storage backend must implement the whole NeuralStorage interface.

Six methods — pin/unpin, pinned-neuron lookup, graph density and the
document-training file tracking — used to live on ``SQLiteStorage`` alone, and
were never declared on ``NeuralStorage``. Every caller reached for them through
``hasattr`` and silently did nothing when they were missing, so on SurrealDB
(the production backend since 2.0.0) decay and prune deleted pinned memories,
``smem_pin`` refused every action, ``smem train`` re-encoded its whole corpus on
each run, and ``activation_strategy="auto"`` never left classic BFS. Nothing
failed, because nothing checked.

These tests are the check. Adding a capability to one backend and forgetting the
others now fails here instead of going dark in production.
"""

from __future__ import annotations

import inspect

from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.shared_store import SharedStorage
from surreal_memory.storage.sqlite_store import SQLiteStorage
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
_PERSISTENT_BACKENDS = (SQLiteStorage, SurrealDBStorage, InMemoryStorage)
_ALL_BACKENDS = (*_PERSISTENT_BACKENDS, SharedStorage)


def _public_methods(cls: type) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def test_contract_is_declared_on_the_interface() -> None:
    """The whole point: these are interface methods, not SQLite trivia."""
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


# Methods still reachable on SQLite alone. Each has live callers that do NOT
# guard with hasattr, so unlike the pinning/training gap this module was written
# for, they raise AttributeError on SurrealDB rather than degrading quietly —
# a different failure mode needing its own investigation, tracked separately.
# This list is a ratchet: it may shrink, never grow.
_KNOWN_SQLITE_ONLY = frozenset(
    {
        # Drift clustering
        "get_drift_clusters",
        "save_drift_cluster",
        "resolve_drift_cluster",
        # Session summaries
        "get_recent_session_summaries",
        "save_session_summary",
        # Sync cursors
        "get_sync_state",
        "save_sync_state",
        # Tag co-occurrence
        "get_tag_cooccurrence",
        "get_tag_fiber_counts",
        "record_tag_cooccurrence",
        # SurrealDB implements only the batched get_depth_priors_batch.
        "get_depth_priors",
    }
)


def test_sqlite_exposes_no_new_undeclared_capability() -> None:
    """Nothing NEW may become reachable on SQLite alone.

    A method on SQLiteStorage that the interface does not declare is precisely
    the shape of the original bug: callers can only reach it via ``hasattr`` or
    a swallowed exception, and it dies outright with the SQLite backend in
    3.0.0. Anything genuinely SQLite-specific belongs behind a leading
    underscore.
    """
    undeclared = _public_methods(SQLiteStorage) - _public_methods(NeuralStorage)
    surreal = _public_methods(SurrealDBStorage)

    orphaned = {name for name in undeclared if name not in surreal}
    assert not (orphaned - _KNOWN_SQLITE_ONLY), (
        "these exist on SQLite but on neither the NeuralStorage interface nor "
        "SurrealDB, so they go dark when SQLite is removed in 3.0.0: "
        f"{sorted(orphaned - _KNOWN_SQLITE_ONLY)}"
    )


def test_known_sqlite_only_list_has_no_stale_entries() -> None:
    """Keep the ratchet honest — an implemented method must leave the list."""
    undeclared = _public_methods(SQLiteStorage) - _public_methods(NeuralStorage)
    surreal = _public_methods(SurrealDBStorage)
    orphaned = {name for name in undeclared if name not in surreal}

    assert not (_KNOWN_SQLITE_ONLY - orphaned), (
        "these are listed as SQLite-only but no longer are — drop them from "
        f"_KNOWN_SQLITE_ONLY: {sorted(_KNOWN_SQLITE_ONLY - orphaned)}"
    )

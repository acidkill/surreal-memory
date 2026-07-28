"""Deterministic synapse ids for edges whose identity is the *pair*, not the row.

Some synapse types state a fact about a pair of neurons: "these two are
similar", "this anchor aliases that one". A second row over the same pair
carries no information — it is a duplicate by construction. But
``Synapse.create()`` mints a fresh UUID on every call, so the only uniqueness
either backend enforces (the synapse primary key) can never collide, and every
re-run of a pass that creates such edges is free to re-insert its whole set.

``engine/dedup/alias_edges.py`` already solved this for ALIAS edges by deriving
the id from the pair. This module generalises that derivation to any
``SynapseType``, so a pass can make idempotency **structural** instead of
depending on a full-table read of what already exists.

Two properties matter:

* For a type in :data:`~surreal_memory.core.synapse.BIDIRECTIONAL_TYPES`, the
  edge means the same thing in either direction, so the endpoints are sorted
  before hashing: ``(A, B)`` and ``(B, A)`` produce one id, not two.
* For every other type direction is meaningful, so the endpoints are hashed in
  the order given.

The digest layout deliberately matches
:func:`~surreal_memory.engine.dedup.alias_edges.alias_edge_id`, which means
``deterministic_edge_id(SynapseType.ALIAS, a, b) == alias_edge_id(a, b)``.
ALIAS is not a bidirectional type, so no sorting intervenes. That equivalence
is pinned by a test: ``alias_edge_id`` must stay byte-for-byte unchanged
because live rows already carry its digest, and this module must never drift
away from it.
"""

from __future__ import annotations

import hashlib

from surreal_memory.core.synapse import BIDIRECTIONAL_TYPES, SynapseType

__all__ = ["deterministic_edge_id"]

# Length of the hex digest slice kept in the id. Matches ``alias_edge_id`` so
# the two derivations stay interchangeable, and matches the 32-hex shape of the
# UUID-derived ids the rest of the synapse table uses.
_DIGEST_CHARS = 32


def deterministic_edge_id(
    type: SynapseType,
    source_id: str,
    target_id: str,
) -> str:
    """Return the stable synapse id for a ``source -> target`` edge of ``type``.

    Neuron ids are UUIDs, so the digest is unique across brains without mixing
    ``brain_id`` in — which keeps the id stable if a neuron is ever
    transplanted.

    Args:
        type: The synapse type. Types in ``BIDIRECTIONAL_TYPES`` get their
            endpoints sorted first, so both orderings map to one id.
        source_id: Source neuron id.
        target_id: Target neuron id.

    Returns:
        An id of the form ``"<type>-<32 hex chars>"``, safe to inline into a
        SurrealDB record id.
    """
    if type in BIDIRECTIONAL_TYPES:
        source_id, target_id = sorted((source_id, target_id))
    digest = hashlib.sha256(f"{type.value}:{source_id}:{target_id}".encode()).hexdigest()
    return f"{type.value}-{digest[:_DIGEST_CHARS]}"

"""Per-fact supersession (U3 — pure engine helpers over storage).

When a newer fact replaces an older one, `supersede_typed_memory` marks the old
TypedMemory's validity window closed (authoritative, A side of the A↔C contract),
links the two with a SUPERSEDES synapse (new -> old), and stamps the old anchor
neuron's metadata (`_superseded`/`_superseded_by`/`_superseded_at`, the C side +
the signal the existing 0.25x demotion reads). Idempotent: re-superseding by the
same new fiber is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class SupersessionOutcome:
    """Result of a single supersede_typed_memory call."""

    old_fiber_id: str
    new_fiber_id: str
    superseded: bool  # True if a change was applied (False = nothing to do / idempotent no-op)


def _canonical_fiber_id(fiber_id: str) -> str:
    """Return the original dash-UUID form of a fiber id.

    Fiber ids are uuid4 (hex + dashes, never underscores); SurrealDB record-id
    sanitisation folds ``-`` -> ``_``, so a fiber loaded from the DB carries the
    underscore form (the deferred Fiber.id round-trip). We store ``superseded_by``
    in the canonical dash form regardless of which path produced ``new_fiber_id``
    (in-process dash id vs storage-loaded underscore id) so the value surfaced to
    callers is consistent. Lossless because UUIDs contain no underscores — mirrors
    the same reversal already done for synapse ids in ``_row_to_synapse``.
    """
    return fiber_id.replace("_", "-")


async def supersede_typed_memory(
    storage: NeuralStorage,
    old_fiber_id: str,
    new_fiber_id: str,
    new_anchor_id: str,
    old_anchor_id: str,
    reason: str = "",
    now: datetime | None = None,
) -> SupersessionOutcome:
    """Mark ``old_fiber_id`` superseded by ``new_fiber_id`` (idempotent).

    Boundary-preserving: once a fact's validity window is closed it STAYS closed.
    Re-superseding an already-superseded fact — by the same OR a different newer
    fact — is a no-op, because a newer fact supersedes the *current* fact, not a
    historical one. This protects the ``valid_until`` boundary that point-in-time
    (``valid_at``) recall depends on.
    """
    now = now or utcnow()
    canonical_new = _canonical_fiber_id(new_fiber_id)

    old_tm = await storage.get_typed_memory(old_fiber_id)
    if old_tm is None:
        return SupersessionOutcome(old_fiber_id, canonical_new, superseded=False)

    # Already closed → no-op (idempotent AND boundary-preserving against a different
    # successor). See docstring: the window's upper bound is immutable once set.
    if old_tm.valid_until is not None:
        return SupersessionOutcome(old_fiber_id, canonical_new, superseded=False)

    # A side (authoritative): close the validity window + record the successor.
    updated = old_tm.with_validity(valid_until=now, superseded_by=canonical_new)
    await storage.update_typed_memory(updated)

    # SUPERSEDES synapse new -> old (mirror _schema_evolve), created once.
    if not await _supersedes_synapse_exists(storage, new_anchor_id, old_anchor_id):
        synapse = Synapse.create(
            source_id=new_anchor_id,
            target_id=old_anchor_id,
            type=SynapseType.SUPERSEDES,
            weight=1.0,
            metadata={"reason": reason, "superseded_at": now.isoformat()},
        )
        await storage.add_synapse(synapse)

    # C side (metadata contract): stamp the old anchor neuron. This is also the
    # signal the retrieval 0.25x demotion (_deprioritize_disputed) reads.
    old_neuron = await storage.get_neuron(old_anchor_id)
    if old_neuron is not None and not old_neuron.metadata.get("_superseded"):
        stamped = old_neuron.with_metadata(
            _superseded=True,
            _superseded_by=canonical_new,
            _superseded_at=now.isoformat(),
        )
        await storage.update_neuron(stamped)

    return SupersessionOutcome(old_fiber_id, canonical_new, superseded=True)


async def _supersedes_synapse_exists(
    storage: NeuralStorage, source_id: str, target_id: str
) -> bool:
    try:
        synapses = await storage.get_synapses(source_id=source_id, target_id=target_id)
    except Exception:
        return False
    return any(s.type == SynapseType.SUPERSEDES for s in synapses)


async def resolve_fibers_for_neurons(
    storage: NeuralStorage, neuron_ids: list[str]
) -> dict[str, str]:
    """Map each neuron to its fiber, but ONLY for unambiguous neurons.

    A neuron that belongs to exactly one fiber maps to that fiber's id; neurons in
    zero or several fibers are skipped (the neuron->fiber relation is ambiguous, so
    supersession would be guessing).
    """
    mapping: dict[str, str] = {}
    for nid in neuron_ids:
        try:
            fibers = await storage.find_fibers(contains_neuron=nid, limit=2)
        except Exception:
            continue
        if len(fibers) == 1:
            mapping[nid] = fibers[0].id
    return mapping

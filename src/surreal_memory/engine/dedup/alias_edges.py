"""Idempotent creation of dedup ALIAS synapses.

An ALIAS edge means "this anchor is a duplicate of that canonical anchor".
That is a *set membership* fact, so exactly one edge per (source, target)
pair is meaningful — a second one carries no information.

Both dedup call sites used to express that intent like this::

    try:
        await storage.add_synapse(alias_synapse)
    except ValueError:
        logger.debug("ALIAS synapse already exists")

but the guard never fired. ``Synapse.create()`` mints a fresh UUID on every
call, so the only uniqueness either backend enforces — the synapse primary
key — could not collide: SQLite raises solely on PK/FK conflict, and the
SurrealDB ``INSERT RELATION`` path does not check at all. Neither backend has
a constraint on ``(source, target, type)``. The consolidation dedup pass
therefore re-inserted its entire alias edge set on *every* run, unbounded.

Measured on the live 11.5k-neuron brain: 144,565 alias rows backing only
2,375 distinct pairs — a 61x amplification, still growing by ~40k rows/day,
and 94% of the rows the prune pass has to load and walk.

This module makes the guard real, in two layers:

1. A pre-insert existence check — cheap and backend-agnostic.
2. A deterministic edge id derived from the pair, so a concurrent writer that
   slips past the check still collides on the primary key instead of
   producing a twin row.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.errors import is_duplicate_key_error

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

__all__ = [
    "AliasEdgeLedger",
    "AliasLinkOutcome",
    "AliasLinkResult",
    "ALIAS_EDGE_WEIGHT",
    "alias_edge_id",
    "ensure_alias_edge",
]


class AliasLinkOutcome(StrEnum):
    """What actually happened to one requested alias edge.

    ``ensure_alias_edge`` used to answer with ``Synapse | None``, which folded
    five disjoint outcomes into a single ``None``: the pair was degenerate, the
    edge was already there, a racing writer won, the existence check blew up,
    or the write blew up. The caller counted "not created" and the report
    printed ``new alias links: 0`` for all five — a healthy steady-state brain
    and a backend that had stopped answering looked identical.
    """

    CREATED = "created"
    """A new edge was written this call."""

    SKIPPED_INVALID = "skipped_invalid"
    """Empty endpoint or self-alias — nothing worth writing."""

    ALREADY_EXISTS = "already_exists"
    """Known present, from the ledger or from the targeted probe."""

    EXISTS_RACE = "exists_race"
    """Another writer inserted the same deterministic id first. Benign."""

    CHECK_FAILED = "check_failed"
    """The existence probe raised: the state is UNKNOWN and the write was
    skipped (fail-closed). Not the same as "it exists"."""

    WRITE_FAILED = "write_failed"
    """``add_synapse`` raised something that is not a duplicate-key error."""


@dataclass(frozen=True)
class AliasLinkResult:
    """Outcome of one ``ensure_alias_edge`` call."""

    outcome: AliasLinkOutcome
    synapse: Synapse | None = None
    """The written edge — non-None only for :attr:`AliasLinkOutcome.CREATED`."""

    error: BaseException | None = None
    """What went wrong, for CHECK_FAILED and WRITE_FAILED only.

    Carried rather than logged loudly in place: a dedup pass calls this helper
    once per candidate pair, so a backend outage would otherwise print one full
    traceback per pair for a single root cause. The helper logs the detail at
    DEBUG and hands the exception to the caller, which reports one aggregated
    WARNING for the whole pass.
    """

    @property
    def created(self) -> bool:
        """True when this call is the one that wrote the edge."""
        return self.outcome is AliasLinkOutcome.CREATED

    @property
    def failed(self) -> bool:
        """True when the pair's state is unknown or the write really failed.

        Deliberately excludes :attr:`AliasLinkOutcome.EXISTS_RACE`: losing a
        race means the edge exists, which is the goal, not a failure.
        """
        return self.outcome in (AliasLinkOutcome.CHECK_FAILED, AliasLinkOutcome.WRITE_FAILED)


# Both former call sites created ALIAS edges at 0.9; keep that so the change is
# purely about *how many* rows exist, not how strongly they pull in traversal.
ALIAS_EDGE_WEIGHT = 0.9

# How many alias rows the ledger is willing to materialise in one response.
#
# The dedup pass compares at most 2000 anchors and a healthy brain's alias slice
# is ~2.4k pairs, so this keeps the whole-slice fast path for every brain that is
# not already amplified — while refusing to ask a brain that *is* (137,871 rows
# live) for a response it cannot deliver. An unbounded `get_synapses` is only
# capped on SQLite (implicit LIMIT 10000); on SurrealDB it streams the entire
# slice in one HTTP body, which is how the LIFECYCLE pass earned its
# "[Errno 104] Connection reset by peer" before PR #97 bounded it.
_LEDGER_SCAN_LIMIT = 5000


def alias_edge_id(source_id: str, target_id: str) -> str:
    """Return the stable synapse id for the alias edge ``source -> target``.

    Neuron ids are UUIDs, so the digest is unique across brains without
    mixing brain_id in — which keeps the id stable if a neuron is ever
    transplanted. Truncated to 32 hex chars to match the UUID-shaped ids the
    rest of the synapse table uses.
    """
    digest = hashlib.sha256(f"alias:{source_id}:{target_id}".encode()).hexdigest()
    return f"alias-{digest[:32]}"


class AliasEdgeLedger:
    """Which (source, target) alias pairs already exist, loaded once.

    The consolidation dedup pass compares up to 2000 anchors pairwise, so a
    per-pair existence query would trade one write storm for a read storm.
    Loading the alias slice once and answering from memory keeps the pass at a
    single extra query.

    That cache is only worth having while the slice fits in one response, so
    the load is capped. A capped load can prove *presence* (every pair it holds
    really is in the brain) but not *absence*, hence ``is_complete``:
    ``ensure_alias_edge`` re-checks unknown pairs one at a time when the ledger
    is partial. Trusting a truncated ledger would mean re-creating every pair
    past the cap — the amplification this module exists to stop.

    Also records edges created through it, so repeated ``ensure`` calls for
    the same pair *within* one run are idempotent too.
    """

    def __init__(
        self,
        pairs: set[tuple[str, str]] | None = None,
        *,
        complete: bool = True,
    ) -> None:
        self._pairs: set[tuple[str, str]] = set(pairs or ())
        self._complete = complete

    @classmethod
    async def load(
        cls,
        storage: NeuralStorage,
        *,
        limit: int = _LEDGER_SCAN_LIMIT,
    ) -> AliasEdgeLedger:
        """Fetch this brain's ALIAS edges, up to ``limit`` rows."""
        try:
            existing = await storage.get_synapses(type=SynapseType.ALIAS, limit=limit)
        except Exception:
            # A ledger that cannot load must not silently degrade into
            # "nothing exists" — that is exactly the bug this module fixes.
            # Signal it so ensure_alias_edge falls back to per-pair queries.
            logger.debug("AliasEdgeLedger.load failed; falling back to per-pair checks")
            raise
        # A full page means there is very likely more behind it; the rows we did
        # get stay useful as positive hits, but absence is no longer provable.
        truncated = len(existing) >= limit
        if truncated:
            logger.debug(
                "alias slice exceeds %d rows; ledger is partial, falling back to "
                "per-pair checks for unknown pairs",
                limit,
            )
        return cls(
            {(s.source_id, s.target_id) for s in existing},
            complete=not truncated,
        )

    @property
    def is_complete(self) -> bool:
        """True when a miss in this ledger proves the pair does not exist."""
        return self._complete

    def __contains__(self, pair: tuple[str, str]) -> bool:
        return pair in self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def has(self, source_id: str, target_id: str) -> bool:
        return (source_id, target_id) in self._pairs

    def record(self, source_id: str, target_id: str) -> None:
        self._pairs.add((source_id, target_id))


async def _pair_exists(storage: NeuralStorage, source_id: str, target_id: str) -> bool:
    """Targeted existence check for a single pair.

    Used by the encode path, which handles one duplicate at a time and would
    be badly served by loading the whole alias slice, and by the dedup path
    whenever its ledger came back partial. Matches on the pair rather than on
    ``alias_edge_id`` because the rows already in the brain predate the
    deterministic id and carry random UUIDs.

    Raises:
        Exception: whatever the backend raised. This function deliberately does
            not decide what a failed check means — the caller applies the
            fail-closed policy *and* reports it, which is what keeps a broken
            backend from reading as "everything already linked".
    """
    hits = await storage.get_synapses(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.ALIAS,
        limit=1,
    )
    return bool(hits)


async def ensure_alias_edge(
    storage: NeuralStorage,
    source_id: str,
    target_id: str,
    *,
    weight: float = ALIAS_EDGE_WEIGHT,
    ledger: AliasEdgeLedger | None = None,
) -> AliasLinkResult:
    """Create the ``source -> target`` ALIAS edge unless it already exists.

    Args:
        storage: Brain-scoped storage.
        source_id: Duplicate anchor.
        target_id: Canonical anchor it aliases.
        weight: Edge weight (defaults to the historical 0.9).
        ledger: Preloaded pair set; pass one when calling in a loop. A partial
            ledger still short-circuits its known pairs, the rest fall back to
            the targeted per-pair check.

    Returns:
        An :class:`AliasLinkResult` naming exactly what happened. The write
        policy is unchanged — a failed existence check still skips the write,
        because a missing alias edge is recoverable on the next pass while a
        duplicate row is not — but the caller can now tell that skip apart from
        "already linked" and report it.
    """
    if not source_id or not target_id:
        logger.debug("alias edge skipped: empty endpoint (%r -> %r)", source_id, target_id)
        return AliasLinkResult(AliasLinkOutcome.SKIPPED_INVALID)
    if source_id == target_id:
        # A self-alias is a no-op relationship that PPR would traverse as a
        # weight-0.9 self-loop, quietly draining push mass from real edges.
        logger.debug("alias edge skipped: self-alias for %s", source_id)
        return AliasLinkResult(AliasLinkOutcome.SKIPPED_INVALID)

    if ledger is not None and ledger.has(source_id, target_id):
        return AliasLinkResult(AliasLinkOutcome.ALREADY_EXISTS)

    # A complete ledger already proves absence, so only probe when there is no
    # ledger at all or when the one we have is partial and a miss proves nothing.
    if ledger is None or not ledger.is_complete:
        try:
            exists = await _pair_exists(storage, source_id, target_id)
        except Exception as exc:
            # Fail closed: skip the write. Failing open would resurrect the
            # unbounded row growth this module exists to stop. The caller is
            # responsible for surfacing this — silently skipping is what made a
            # broken backend indistinguishable from a healthy, fully linked one.
            logger.debug(
                "alias edge existence check failed for %s -> %s; skipping the write "
                "(fail-closed) — the state of this pair is unknown",
                source_id,
                target_id,
                exc_info=True,
            )
            return AliasLinkResult(AliasLinkOutcome.CHECK_FAILED, error=exc)
        if exists:
            return AliasLinkResult(AliasLinkOutcome.ALREADY_EXISTS)

    synapse = Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.ALIAS,
        weight=weight,
        metadata={"_dedup": True},
        synapse_id=alias_edge_id(source_id, target_id),
    )
    try:
        await storage.add_synapse(synapse)
    except Exception as exc:
        if is_duplicate_key_error(exc):
            # Deterministic id means a racing writer collides on the primary key
            # here rather than creating a twin — losing the race is the correct
            # outcome, not an error. The edge exists, so record it: a later pair
            # in this same run must not pay for another probe.
            logger.debug(
                "ALIAS synapse %s -> %s already exists (lost write race)", source_id, target_id
            )
            if ledger is not None:
                ledger.record(source_id, target_id)
            return AliasLinkResult(AliasLinkOutcome.EXISTS_RACE)
        logger.debug(
            "ALIAS synapse write failed for %s -> %s (not a duplicate)",
            source_id,
            target_id,
            exc_info=True,
        )
        return AliasLinkResult(AliasLinkOutcome.WRITE_FAILED, error=exc)

    if ledger is not None:
        ledger.record(source_id, target_id)
    return AliasLinkResult(AliasLinkOutcome.CREATED, synapse=synapse)

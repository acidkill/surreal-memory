"""Dream engine — random exploration for hidden connections.

Simulates dream-like exploration by selecting random neurons,
running spreading activation, and creating weak RELATED_TO
synapses between co-activated pairs that lack direct connections.

Dream synapses decay Nx faster than normal during pruning,
so only repeatedly reinforced connections survive.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from surreal_memory.core.synapse import Synapse, SynapseType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.core.neuron import Neuron
    from surreal_memory.storage.base import NeuralStorage

_MAX_ACTIVATED = 500
_MAX_DREAM_PAIRS = 5_000
_MAX_NEW_SYNAPSES = 200
_PLAN_PAGE_SIZE = 100


@dataclass(frozen=True)
class DreamPlanCheckpoint:
    """Bounded, replayable planning state for one DREAM run."""

    seed: int
    activated_ids: tuple[str, ...]
    pair_cursor: int
    pair_receipts: tuple[str, ...]
    planned_synapses: tuple[Synapse, ...]
    pairs_explored: int
    complete: bool


@dataclass(frozen=True)
class DreamResult:
    """Result of a dream exploration session.

    Attributes:
        synapses_created: New RELATED_TO synapses discovered
        pairs_explored: Number of neuron pairs evaluated
    """

    synapses_created: list[Synapse] = field(default_factory=list)
    pairs_explored: int = 0


async def dream(
    storage: NeuralStorage,
    config: BrainConfig,
    seed: int | None = None,
    *,
    resume: DreamPlanCheckpoint | None = None,
    checkpoint: Callable[[DreamPlanCheckpoint], Awaitable[None]] | None = None,
) -> DreamResult:
    """Run dream exploration and optionally persist bounded plan pages.

    Related synapses are queried only for candidate pairs, never collected as a
    brain-wide list. A durable checkpoint records sorted activated IDs, the
    deterministic candidate cursor, page digests of observed pair existence,
    and the (at most 200) planned additions. On resume, committed pages are
    replay-read and their digests must still match before planning continues.
    """
    if resume is None:
        run_seed = seed if seed is not None else random.SystemRandom().getrandbits(64)
        rng = random.Random(run_seed)
        all_neurons: list[Neuron] = await storage.find_neurons(limit=10000, include_embedding=False)
        if len(all_neurons) < 2:
            return DreamResult()

        seed_neurons = rng.sample(all_neurons, min(config.dream_neuron_count, len(all_neurons)))
        from surreal_memory.engine.activation import SpreadingActivation

        activation_engine = SpreadingActivation(storage, config)
        activated_set: set[str] = set()
        for neuron in seed_neurons:
            if len(activated_set) >= _MAX_ACTIVATED:
                break
            try:
                results, _trace = await activation_engine.activate(
                    anchor_neurons=[neuron.id], max_hops=2
                )
                for result in results.values():
                    activated_set.add(result.neuron_id)
                    if len(activated_set) >= _MAX_ACTIVATED:
                        break
            except Exception:
                logger.debug("Dream activation failed for neuron %s", neuron.id, exc_info=True)
                continue

        state = DreamPlanCheckpoint(
            seed=run_seed,
            activated_ids=tuple(sorted(activated_set)),
            pair_cursor=0,
            pair_receipts=(),
            planned_synapses=(),
            pairs_explored=0,
            complete=False,
        )
        if checkpoint is not None:
            await checkpoint(state)
    else:
        state = resume
        if len(state.activated_ids) > _MAX_ACTIVATED:
            raise RuntimeError("dream checkpoint contains too many activated neurons")
        if tuple(sorted(set(state.activated_ids))) != state.activated_ids:
            raise RuntimeError("dream checkpoint activated IDs are not unique and stable")
        if len(state.planned_synapses) > _MAX_NEW_SYNAPSES:
            raise RuntimeError("dream checkpoint contains too many planned synapses")
        planned_pairs = [
            _canonical_pair(synapse.source_id, synapse.target_id)
            for synapse in state.planned_synapses
        ]
        if len(set(planned_pairs)) != len(planned_pairs):
            raise RuntimeError("dream checkpoint contains duplicate planned pairs")
        if any(
            synapse.type != SynapseType.RELATED_TO
            or synapse.metadata.get("_dream") is not True
            or synapse.source_id not in state.activated_ids
            or synapse.target_id not in state.activated_ids
            or synapse.source_id == synapse.target_id
            for synapse in state.planned_synapses
        ):
            raise RuntimeError("dream checkpoint contains an incompatible planned synapse")

    activated_ids = state.activated_ids
    candidates = _candidate_pairs(activated_ids)
    planned_by_pair = {
        _canonical_pair(synapse.source_id, synapse.target_id): synapse
        for synapse in state.planned_synapses
    }
    receipts = list(state.pair_receipts)
    cursor = state.pair_cursor
    total_pairs = len(activated_ids) * (len(activated_ids) - 1) // 2
    candidate_limit = min(total_pairs, _MAX_DREAM_PAIRS - 1)
    if cursor < 0 or cursor > candidate_limit:
        raise RuntimeError("dream checkpoint pair cursor is invalid")
    if (
        state.complete
        and cursor < candidate_limit
        and len(state.planned_synapses) < _MAX_NEW_SYNAPSES
    ):
        raise RuntimeError("completed dream checkpoint stops before a plan limit")
    expected_receipt_count = (cursor + _PLAN_PAGE_SIZE - 1) // _PLAN_PAGE_SIZE
    if len(receipts) != expected_receipt_count:
        raise RuntimeError("dream checkpoint pair receipts do not match its cursor")

    digest = hashlib.sha256()
    page_index = 0
    page_count = 0
    scanned = 0
    planned = list(state.planned_synapses)
    stopped_for_output_cap = len(planned) >= _MAX_NEW_SYNAPSES

    # The historical loop counts the cap-triggering pair but does not evaluate
    # it; only the first 4,999 candidates can be read when the 5,000 cap binds.
    for ordinal, (left, right) in enumerate(candidates):
        if ordinal >= _MAX_DREAM_PAIRS - 1:
            break
        if ordinal < cursor:
            observed = await _pair_signature(storage, left, right)
            expected_synapse = planned_by_pair.get(_canonical_pair(left, right))
            _digest_pair(digest, left, right, _normalize_signature(observed, expected_synapse))
            page_count += 1
            scanned += 1
            if page_count == _PLAN_PAGE_SIZE or scanned == cursor:
                if page_index >= len(receipts) or digest.hexdigest() != receipts[page_index]:
                    raise RuntimeError(
                        "dream related-to rows changed after the saved planning checkpoint"
                    )
                page_index += 1
                digest = hashlib.sha256()
                page_count = 0
            continue

        if state.complete:
            # A fully planned manifest still undergoes source revalidation above;
            # later candidates were intentionally outside its saved plan (pair or
            # output cap), so they must not be added during replay.
            break
        if stopped_for_output_cap:
            break

        observed = await _pair_signature(storage, left, right)
        existing = any(observed)
        _digest_pair(digest, left, right, observed)
        if not existing:
            synapse = Synapse.create(
                source_id=left,
                target_id=right,
                type=SynapseType.RELATED_TO,
                weight=0.1,
                metadata={"_dream": True},
            )
            planned.append(synapse)
            planned_by_pair[_canonical_pair(left, right)] = synapse
            stopped_for_output_cap = len(planned) >= _MAX_NEW_SYNAPSES

        cursor += 1
        scanned += 1
        page_count += 1
        if page_count == _PLAN_PAGE_SIZE:
            receipts.append(digest.hexdigest())
            state = DreamPlanCheckpoint(
                seed=state.seed,
                activated_ids=activated_ids,
                pair_cursor=cursor,
                pair_receipts=tuple(receipts),
                planned_synapses=tuple(planned),
                pairs_explored=cursor,
                complete=False,
            )
            if checkpoint is not None:
                await checkpoint(state)
            page_index += 1
            digest = hashlib.sha256()
            page_count = 0

    # Verify the last previously committed page, which can be shorter only for a
    # complete plan. Ordinary resumable cursors are page aligned.
    if scanned == cursor and page_count:
        if page_index < len(receipts):
            if digest.hexdigest() != receipts[page_index]:
                raise RuntimeError(
                    "dream related-to rows changed after the saved planning checkpoint"
                )
            page_index += 1
        elif not state.complete:
            # The final plan may end before a full page. Record that terminal
            # receipt so the completed plan can be revalidated on resume.
            receipts.append(digest.hexdigest())
            page_index += 1

    if page_index != len(receipts):
        raise RuntimeError("dream checkpoint contains orphaned pair receipts")

    if not state.complete:
        # Calculate the user-visible counter using the legacy cap behavior.
        pairs_explored = cursor
        if total_pairs >= _MAX_DREAM_PAIRS:
            pairs_explored = _MAX_DREAM_PAIRS
        elif stopped_for_output_cap and cursor < total_pairs:
            pairs_explored = cursor + 1
        complete_state = DreamPlanCheckpoint(
            seed=state.seed,
            activated_ids=activated_ids,
            pair_cursor=cursor,
            pair_receipts=tuple(receipts),
            planned_synapses=tuple(planned),
            pairs_explored=pairs_explored,
            complete=True,
        )
        if checkpoint is not None:
            await checkpoint(complete_state)
        state = complete_state

    return DreamResult(list(state.planned_synapses), state.pairs_explored)


def _candidate_pairs(activated_ids: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    for i, left in enumerate(activated_ids):
        for right in activated_ids[i + 1 :]:
            yield left, right


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


async def _pair_signature(storage: NeuralStorage, left: str, right: str) -> tuple[str, str]:
    forward = await storage.get_synapses(
        source_id=left, target_id=right, type=SynapseType.RELATED_TO, limit=1
    )
    reverse = await storage.get_synapses(
        source_id=right, target_id=left, type=SynapseType.RELATED_TO, limit=1
    )
    forward_ids = sorted(
        item.id
        for item in forward
        if item.source_id == left
        and item.target_id == right
        and item.type == SynapseType.RELATED_TO
    )
    reverse_ids = sorted(
        item.id
        for item in reverse
        if item.source_id == right
        and item.target_id == left
        and item.type == SynapseType.RELATED_TO
    )
    return (forward_ids[0] if forward_ids else "", reverse_ids[0] if reverse_ids else "")


def _normalize_signature(signature: tuple[str, str], planned: Synapse | None) -> tuple[str, str]:
    if planned is None:
        return signature
    return tuple("" if value == planned.id else value for value in signature)  # type: ignore[return-value]


def _digest_pair(digest: hashlib._Hash, left: str, right: str, signature: tuple[str, str]) -> None:
    digest.update(
        json.dumps([left, right, *signature], ensure_ascii=False, separators=(",", ":")).encode()
    )
    digest.update(b"\n")

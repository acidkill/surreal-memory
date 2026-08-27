"""Memory lifecycle management - decay, reinforcement, compression.

Implements the Ebbinghaus forgetting curve for natural memory decay
and reinforcement for frequently accessed memories.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from surreal_memory.core.neuron import NeuronState
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage


# Bucket edges for the weight distribution. Denser near zero because that is
# where the prune threshold sits and where a mis-tuned rate does its damage.
_WEIGHT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75)


def _weight_bucket(weight: float) -> str:
    """Label the bucket a weight falls into, e.g. "0.05-0.1"."""
    low = 0.0
    for edge in _WEIGHT_BUCKETS:
        if weight < edge:
            return f"{low:g}-{edge:g}"
        low = edge
    return f"{low:g}-1"


@dataclass
class DecayReport:
    """Report of decay operation results."""

    neurons_processed: int = 0
    neurons_decayed: int = 0
    neurons_pruned: int = 0
    synapses_processed: int = 0
    synapses_decayed: int = 0
    synapses_pruned: int = 0
    duration_ms: float = 0.0
    reference_time: datetime = field(default_factory=utcnow)

    # Why a processed synapse was not decayed. Without these, "processed" and
    # "decayed" differ by an unexplained number and no one can tell a healthy
    # pass (most edges simply not due yet) from a starved one (a gate stuck shut).
    synapses_skipped_pinned: int = 0
    synapses_skipped_idle_gate: int = 0
    synapses_skipped_bookmark: int = 0

    # Weight distribution across the synapses this pass actually decayed, before
    # and after. Buckets rather than rows: the question these answer is "is the
    # decay rate sane", which is a shape, not a per-edge history.
    weight_before: dict[str, int] = field(default_factory=dict)
    weight_after: dict[str, int] = field(default_factory=dict)

    # The knobs this pass ran with. A distribution is uninterpretable without
    # them — the same shape means different things at different decay rates.
    config_snapshot: dict[str, float] = field(default_factory=dict)

    def record_weights(self, before: float, after: float) -> None:
        """Bucket one decayed synapse's weight, before and after."""
        self.weight_before[_weight_bucket(before)] = (
            self.weight_before.get(_weight_bucket(before), 0) + 1
        )
        self.weight_after[_weight_bucket(after)] = (
            self.weight_after.get(_weight_bucket(after), 0) + 1
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Decay Report ({self.reference_time.strftime('%Y-%m-%d %H:%M')})",
            f"  Neurons: {self.neurons_decayed}/{self.neurons_processed} decayed, {self.neurons_pruned} pruned",
            f"  Synapses: {self.synapses_decayed}/{self.synapses_processed} decayed, {self.synapses_pruned} pruned",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        skipped = (
            self.synapses_skipped_pinned
            + self.synapses_skipped_idle_gate
            + self.synapses_skipped_bookmark
        )
        if skipped:
            lines.append(
                f"  Synapses skipped: {skipped} "
                f"({self.synapses_skipped_pinned} pinned, "
                f"{self.synapses_skipped_idle_gate} not idle long enough, "
                f"{self.synapses_skipped_bookmark} already charged)"
            )
        return "\n".join(lines)


class DecayManager:
    """Manage memory decay using Ebbinghaus forgetting curve.

    Decay formula: retention = e^(-decay_rate * days_since_access)

    Memories that haven't been accessed recently will have their
    activation levels reduced. Memories below the prune threshold
    can be marked as dormant or removed.
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        prune_threshold: float = 0.01,
        min_age_days: float = 1.0,
    ):
        """Initialize decay manager.

        Args:
            decay_rate: Rate of decay per day (0.1 = 10% per day)
            prune_threshold: Activation level below which to prune
            min_age_days: Minimum age before applying decay
        """
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.min_age_days = min_age_days

    async def apply_decay(
        self,
        storage: NeuralStorage,
        reference_time: datetime | None = None,
        dry_run: bool = False,
    ) -> DecayReport:
        """Apply decay to all neurons and synapses in storage.

        Args:
            storage: Storage instance to apply decay to
            reference_time: Reference time for decay calculation (default: now)
            dry_run: If True, calculate but don't save changes

        Returns:
            DecayReport with statistics
        """
        import time

        start_time = time.perf_counter()
        reference_time = reference_time or utcnow()
        report = DecayReport(reference_time=reference_time)

        # Preload pinned neuron IDs to skip during decay
        pinned_neuron_ids = await storage.get_pinned_neuron_ids()

        # Preload neuron→tier mapping for tier-aware decay
        from surreal_memory.core.memory_types import TIER_DECAY_FLOORS, TIER_DECAY_MULTIPLIERS

        neuron_tier_map: dict[str, str] = {}
        try:
            if hasattr(storage, "find_typed_memories"):
                # Collect fiber_id → tier mapping, then batch-resolve fibers
                fiber_tier_pairs: list[tuple[str, str]] = []
                for tier_val in ("hot", "cold"):
                    tier_mems = await storage.find_typed_memories(tier=tier_val, limit=1000)
                    for tm in tier_mems:
                        fiber_tier_pairs.append((tm.fiber_id, tier_val))

                # Enforce BOUNDARY invariant: boundary memories always get HOT floor
                # even if stored pre-Phase3 with default tier="warm"
                from surreal_memory.core.memory_types import MemoryType

                boundary_mems = await storage.find_typed_memories(
                    memory_type=MemoryType.BOUNDARY, limit=1000
                )
                for tm in boundary_mems:
                    fiber_tier_pairs.append((tm.fiber_id, "hot"))

                # Deduplicate fiber IDs and resolve fibers
                unique_fids = {fid for fid, _ in fiber_tier_pairs}
                fiber_cache: dict[str, Any] = {}
                for fid in unique_fids:
                    fiber_cache[fid] = await storage.get_fiber(fid)

                # Build neuron→tier map (boundary "hot" entries added last → override)
                for fid, tier_val in fiber_tier_pairs:
                    fiber = fiber_cache.get(fid)
                    if fiber:
                        for nid in fiber.neuron_ids:
                            neuron_tier_map[nid] = tier_val
        except (TypeError, AttributeError):
            logger.debug("Tier map build failed (non-critical)", exc_info=True)

        # Get all neuron states
        states = await storage.get_all_neuron_states()
        report.neurons_processed = len(states)

        for state in states:
            # Skip neurons belonging to pinned (KB) fibers
            if state.neuron_id in pinned_neuron_ids:
                continue
            # Use last_activated if available, otherwise fall back to created_at
            if state.last_activated is None:
                reference_activated = (
                    state.created_at if hasattr(state, "created_at") else reference_time
                )
            else:
                reference_activated = state.last_activated

            # Calculate time since last activation (or creation)
            time_diff = reference_time - reference_activated
            days_elapsed = time_diff.total_seconds() / 86400

            # Skip if too recent
            if days_elapsed < self.min_age_days:
                continue

            # Calculate decay using per-neuron rate (type-aware + tier-aware)
            neuron_tier = neuron_tier_map.get(state.neuron_id, "warm")
            tier_multiplier = TIER_DECAY_MULTIPLIERS.get(neuron_tier, 1.0)
            tier_floor = TIER_DECAY_FLOORS.get(neuron_tier, 0.0)
            effective_rate = state.decay_rate * tier_multiplier
            decay_factor = math.exp(-effective_rate * days_elapsed)
            new_level = max(tier_floor, state.activation_level * decay_factor)

            if new_level < state.activation_level:
                report.neurons_decayed += 1

                pruned = new_level < self.prune_threshold
                if pruned:
                    report.neurons_pruned += 1

                if not dry_run:
                    # Apply tier-aware level directly (skip state.decay() to avoid
                    # double-computing — we already have the correct new_level)
                    final_level = 0.0 if (pruned and tier_floor == 0.0) else new_level
                    decayed_state = dc_replace(
                        state,
                        activation_level=final_level,
                        last_activated=reference_time,
                    )
                    await storage.update_neuron_state(decayed_state)

        # Get all synapses and apply decay
        synapses = await storage.get_all_synapses()
        report.synapses_processed = len(synapses)

        for synapse in synapses:
            # Skip synapses connected to pinned neurons
            if synapse.source_id in pinned_neuron_ids or synapse.target_id in pinned_neuron_ids:
                report.synapses_skipped_pinned += 1
                continue

            # Eligibility gate: how long this synapse has been *idle*. Deliberately
            # still measured from last_activated/created_at rather than from the decay
            # bookmark below, because the two answer different questions. This one is
            # "is the memory unused enough to deserve decay at all", and a decay pass is
            # not usage. Gating on the bookmark instead would reset the clock every run,
            # so any schedule firing more often than min_age_days would starve the gate
            # and freeze decay permanently.
            if synapse.last_activated is None:
                idle_since = (
                    synapse.created_at if hasattr(synapse, "created_at") else reference_time
                )
            else:
                idle_since = synapse.last_activated
            idle_since = ensure_naive_utc(idle_since)

            idle_days = (reference_time - idle_since).total_seconds() / 86400

            if idle_days < self.min_age_days:
                report.synapses_skipped_idle_gate += 1
                continue

            # Charge only the stretch no earlier run has billed, starting from the decay
            # bookmark Synapse.decay writes.
            #
            # Incremental decay applies the same total decay as one pass from creation,
            # because the exponents telescope over any partition t0 < t1 < ... < tn:
            #   multiplying exp(-r * slice) across consecutive slices gives
            #   exp(-r * total of those slices), i.e. exp(-r * (tn - t0)).
            # So N runs each charging their own slice land on exactly the weight a single
            # run over the whole window produces, and how fast a memory fades per day of
            # wall-clock time is unchanged. (The emotional exponent below is a constant
            # per synapse, so raising each factor to it preserves the identity.)
            #
            # What this is NOT equal to is the code that shipped before the bookmark was
            # read: with no record of prior runs every pass re-measured from t0 and
            # multiplied an already-decayed weight again, giving exp(-r * Σ(ti - t0)) —
            # quadratic in run count, so ten daily runs over ten days applied ~55 days of
            # decay and rewrote all 57k rows each time. That over-decay is the defect the
            # bookmark was added to fix, not a semantic worth preserving; neuron decay
            # above has always been incremental (it stamps last_activated=reference_time
            # every run). Synapse weights therefore fade slower than the pre-fix build,
            # onto the Ebbinghaus curve this class documents — deliberate and stated, not
            # a silent change.
            #
            # Fallback: last_decayed is None on every existing row, leaving charge_from
            # at the `last_activated or created_at` resolved above — first pass on old
            # data behaves exactly as before, no migration needed. max() rather than the
            # bookmark alone because a synapse fired since its last decay must not be
            # charged for the stretch it spent being used. Deliberately not
            # Synapse.decay_reference_time, whose max() also folds in created_at: on a
            # record whose last_activated predates its creation (imports, merges) that
            # would start charging from the newer created_at and fade the row more slowly
            # than today. Repairing anomalous timestamps is not this fix's job.
            last_decayed = synapse.last_decayed
            charge_from = max(idle_since, last_decayed) if last_decayed else idle_since
            days_elapsed = (reference_time - charge_from).total_seconds() / 86400

            if days_elapsed <= 0:
                # Bookmark already at/after this reference time: nothing unbilled left
                # (re-run of the same window, or a backdated reference_time).
                report.synapses_skipped_bookmark += 1
                continue

            # Decay synapse weight
            decay_factor = math.exp(-self.decay_rate * days_elapsed)

            # Emotional synapses decay slower (emotional persistence)
            if synapse.type in (SynapseType.FELT, SynapseType.EVOKES):
                intensity = synapse.metadata.get("_intensity", 0.5)
                # High-intensity: decay^0.5 (much slower), low: decay^0.8 (slightly slower)
                emotional_factor = 0.5 + 0.3 * (1.0 - intensity)
                decay_factor = decay_factor**emotional_factor

            new_weight = synapse.weight * decay_factor

            if new_weight < synapse.weight:
                report.synapses_decayed += 1
                report.record_weights(synapse.weight, new_weight)

                if new_weight < self.prune_threshold:
                    report.synapses_pruned += 1
                    if not dry_run:
                        # Zero out weight for pruned synapses
                        pruned_synapse = synapse.decay(0.0, now=reference_time)
                        await storage.update_synapse(pruned_synapse)

                # The bookmark records the reference time this run charged up to, not
                # wall-clock now: the interval billed and the interval marked as billed
                # have to be the same one, or a run with an explicit reference_time
                # (backfill, test, catch-up) would leave a gap or a double-charge.
                elif not dry_run:
                    decayed_synapse = synapse.decay(decay_factor, now=reference_time)
                    await storage.update_synapse(decayed_synapse)

        report.config_snapshot = {
            "decay_rate": self.decay_rate,
            "prune_threshold": self.prune_threshold,
            "min_age_days": self.min_age_days,
        }
        report.duration_ms = (time.perf_counter() - start_time) * 1000
        await self._record_pass(storage, report, dry_run)
        return report

    async def _record_pass(
        self, storage: NeuralStorage, report: DecayReport, dry_run: bool
    ) -> None:
        """Persist one aggregate telemetry row, if telemetry is enabled.

        Opt-in and fail-soft, in that order. Telemetry that can break the decay
        pass it observes is worse than no telemetry, so every failure here is
        logged and swallowed — this is the one place where swallowing is the
        correct call, because the caller has nothing to do about it and the
        operation itself succeeded.
        """
        try:
            from surreal_memory.unified_config import get_config

            if not get_config().decay_telemetry.enabled:
                return
            if not hasattr(storage, "add_decay_pass"):
                return
            await storage.add_decay_pass(
                {
                    "ran_at": report.reference_time,
                    "duration_ms": report.duration_ms,
                    "dry_run": dry_run,
                    "counters": {
                        "neurons_processed": report.neurons_processed,
                        "neurons_decayed": report.neurons_decayed,
                        "neurons_pruned": report.neurons_pruned,
                        "synapses_processed": report.synapses_processed,
                        "synapses_decayed": report.synapses_decayed,
                        "synapses_pruned": report.synapses_pruned,
                        "synapses_skipped_pinned": report.synapses_skipped_pinned,
                        "synapses_skipped_idle_gate": report.synapses_skipped_idle_gate,
                        "synapses_skipped_bookmark": report.synapses_skipped_bookmark,
                    },
                    "weight_before": dict(report.weight_before),
                    "weight_after": dict(report.weight_after),
                    "config_snapshot": dict(report.config_snapshot),
                }
            )
        except Exception:
            logger.debug("Decay telemetry write skipped", exc_info=True)

    async def consolidate(
        self,
        storage: NeuralStorage,
        frequency_threshold: int = 5,
        boost_delta: float = 0.03,
    ) -> int:
        """Consolidate frequently-accessed memory paths.

        Boosts synapse weights for fibers that have been accessed
        at least `frequency_threshold` times, reinforcing well-trodden
        memory pathways into long-term structures.

        Args:
            storage: Storage instance containing fibers and synapses
            frequency_threshold: Minimum fiber frequency to consolidate
            boost_delta: Amount to boost each synapse weight

        Returns:
            Number of synapses consolidated (weight-boosted)
        """
        fibers = await storage.get_fibers(
            limit=100,
            order_by="frequency",
            descending=True,
        )

        consolidated = 0

        # Filter eligible fibers first
        eligible_fibers = [f for f in fibers if f.frequency >= frequency_threshold]

        # Collect ALL synapse IDs from ALL eligible fibers into one list
        all_synapse_ids: list[str] = []
        for fiber in eligible_fibers:
            all_synapse_ids.extend(fiber.synapse_ids)

        if not all_synapse_ids:
            return consolidated

        # Batch fetch: get all synapses for eligible fibers' neuron IDs
        # Since there's no get_synapses_batch(ids), use get_synapses_for_neurons
        # to fetch synapses connected to fiber neurons, then index by synapse ID.
        all_neuron_ids: list[str] = list(
            {nid for fiber in eligible_fibers for nid in fiber.neuron_ids}
        )
        outgoing = await storage.get_synapses_for_neurons(all_neuron_ids, direction="out")
        incoming = await storage.get_synapses_for_neurons(all_neuron_ids, direction="in")

        # Build synapse lookup by ID
        synapse_map: dict[str, Synapse] = {}
        for synapses_list in outgoing.values():
            for syn in synapses_list:
                synapse_map[syn.id] = syn
        for synapses_list in incoming.values():
            for syn in synapses_list:
                synapse_map[syn.id] = syn

        to_update: list[Synapse] = []
        for fiber in eligible_fibers:
            for synapse_id in fiber.synapse_ids:
                synapse = synapse_map.get(synapse_id)
                if synapse is None:
                    continue

                reinforced = synapse.reinforce(boost_delta)
                to_update.append(reinforced)
                consolidated += 1

        if to_update:
            await storage.update_synapses_batch(to_update)
        return consolidated


class ReinforcementManager:
    """Strengthen frequently accessed memory paths.

    When memories are accessed, their activation levels and
    synapse weights are increased (reinforced).
    """

    def __init__(
        self,
        reinforcement_delta: float = 0.05,
        max_activation: float = 1.0,
        max_weight: float = 1.0,
        rehearsal_neuron_limit: int = 15,
    ):
        """Initialize reinforcement manager.

        Args:
            reinforcement_delta: Amount to increase on each access
            max_activation: Maximum activation level
            max_weight: Maximum synapse weight
            rehearsal_neuron_limit: How many of the given neuron_ids feed
                maturation rehearsal (see BrainConfig.reinforcement_neuron_limit)
        """
        self.reinforcement_delta = reinforcement_delta
        self.max_activation = max_activation
        self.max_weight = max_weight
        self.rehearsal_neuron_limit = rehearsal_neuron_limit

    async def reinforce(
        self,
        storage: NeuralStorage,
        neuron_ids: list[str],
        synapse_ids: list[str] | None = None,
    ) -> int:
        """Reinforce accessed neurons and synapses.

        Args:
            storage: Storage instance
            neuron_ids: List of accessed neuron IDs
            synapse_ids: Optional list of accessed synapse IDs

        Returns:
            Number of items reinforced
        """
        reinforced = 0

        # Batch fetch all neuron states at once
        states_map = await storage.get_neuron_states_batch(neuron_ids)
        now = utcnow()

        to_update: list[NeuronState] = []
        for neuron_id in neuron_ids:
            state = states_map.get(neuron_id)
            if state:
                new_level = min(
                    state.activation_level + self.reinforcement_delta,
                    self.max_activation,
                )
                # Directly set activation level (bypass sigmoid for reinforcement)
                reinforced_state = dc_replace(
                    state,
                    activation_level=new_level,
                    access_frequency=state.access_frequency + 1,
                    last_activated=now,
                )
                to_update.append(reinforced_state)
                reinforced += 1

        if to_update:
            await storage.update_neuron_states_batch(to_update)

        # Rehearse maturation records for fibers connected to reinforced neurons.
        # This is required for EPISODIC → SEMANTIC transition (needs 3+ distinct days).
        # Every fiber find_fibers_batch surfaces gets rehearsed -- the only cap is
        # how many neurons feed the lookup (rehearsal_neuron_limit), so coverage
        # scales with what recall actually activated instead of a fixed count.
        if neuron_ids:
            try:
                from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage

                fibers = await storage.find_fibers_batch(
                    neuron_ids[: self.rehearsal_neuron_limit], limit_per_neuron=3
                )
                seen_fiber_ids: set[str] = set()
                for fiber in fibers:
                    if fiber.id in seen_fiber_ids:
                        continue
                    seen_fiber_ids.add(fiber.id)
                    record = await storage.get_maturation(fiber.id)
                    if record is None:
                        # A fiber with no maturation row can never advance a
                        # stage, and reinforcement was the one moment we knew it
                        # was alive -- yet the miss was silently ignored, so the
                        # fiber stayed invisible to maturation forever. Create
                        # the row at SHORT_TERM and count this recall as its
                        # first rehearsal.
                        brain_id = getattr(storage, "current_brain_id", "") or ""
                        record = MaturationRecord(
                            fiber_id=fiber.id,
                            brain_id=brain_id,
                            stage=MemoryStage.SHORT_TERM,
                            stage_entered_at=now,
                        )
                    updated = record.rehearse(now)
                    await storage.save_maturation(updated)
            except Exception:
                logger.debug("Maturation rehearsal skipped during reinforce", exc_info=True)

        if synapse_ids:
            # Batch fetch synapses via neuron-based lookup
            # Collect neuron IDs involved in synapse reinforcement from states
            all_neuron_ids = list(states_map.keys())
            if all_neuron_ids:
                outgoing = await storage.get_synapses_for_neurons(all_neuron_ids, direction="out")
                incoming = await storage.get_synapses_for_neurons(all_neuron_ids, direction="in")
                synapse_map_2: dict[str, Synapse] = {}
                for synapses_out in outgoing.values():
                    for syn in synapses_out:
                        synapse_map_2[syn.id] = syn
                for synapses_in in incoming.values():
                    for syn in synapses_in:
                        synapse_map_2[syn.id] = syn
            else:
                synapse_map_2 = {}

            synapse_updates: list[Synapse] = []
            for synapse_id in synapse_ids:
                synapse = synapse_map_2.get(synapse_id)
                if synapse is None:
                    # Fallback for synapses not connected to reinforced neurons
                    synapse = await storage.get_synapse(synapse_id)
                if synapse:
                    new_weight = min(
                        synapse.weight + self.reinforcement_delta,
                        self.max_weight,
                    )
                    reinforced_synapse = synapse.reinforce(new_weight - synapse.weight)
                    synapse_updates.append(reinforced_synapse)
                    reinforced += 1

            if synapse_updates:
                await storage.update_synapses_batch(synapse_updates)

        return reinforced

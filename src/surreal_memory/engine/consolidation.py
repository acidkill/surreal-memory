"""Memory consolidation engine — prune, merge, and summarize memories.

Provides automated memory maintenance:
- Prune: Remove dead synapses and orphan neurons
- Merge: Combine overlapping fibers
- Summarize: Create concept neurons for topic clusters
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.clustering import UnionFind
from surreal_memory.storage.errors import is_duplicate_key_error
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import TierConfig

logger = logging.getLogger(__name__)

# How many anchors one dedup pass compares pairwise. The loop is O(N^2), so the
# cap is what keeps a large brain from spending minutes here — but it also means
# the census only ever describes this many anchors, in scan order, which is why
# the pass reports the total and flags the truncation instead of hiding it.
_DEDUP_MAX_ANCHORS = 2000

# How many dormant neurons one dream cycle replays. Kept small on purpose: the
# point is a trickle of reactivation so nothing stays dormant forever, not a
# sweep of the whole dormant set (which is most of a mature brain).
_DORMANT_REPLAY_SAMPLE = 20


class ConsolidationStrategy(StrEnum):
    """Available consolidation strategies."""

    PRUNE = "prune"
    MERGE = "merge"
    SUMMARIZE = "summarize"
    MATURE = "mature"
    INFER = "infer"
    ENRICH = "enrich"
    DREAM = "dream"
    LEARN_HABITS = "learn_habits"
    DEDUP = "dedup"
    SEMANTIC_LINK = "semantic_link"
    COMPRESS = "compress"
    LIFECYCLE = "lifecycle"
    PROCESS_TOOL_EVENTS = "process_tool_events"
    PROCESS_REASONING_TRACES = "process_reasoning_traces"
    LEARN_REASONING = "learn_reasoning"
    ESSENCE_BACKFILL = "essence_backfill"
    REPLAY = "replay"  # Hippocampal replay: LTP/LTD on recent fibers
    SCHEMA = "schema"  # Schema assimilation: bottom-up knowledge organization
    INTERFERENCE = "interference"  # Interference forgetting: memory competition
    DETECT_DRIFT = "detect_drift"  # Tag-cooccurrence Jaccard clustering
    ALL = "all"


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for consolidation operations."""

    prune_weight_threshold: float = 0.05
    prune_min_inactive_days: float = 7.0
    prune_isolated_neurons: bool = True
    merge_overlap_threshold: float = 0.5
    merge_max_fiber_size: int = 50
    summarize_min_cluster_size: int = 3
    summarize_tag_overlap_threshold: float = 0.4
    infer_co_activation_threshold: int = 3
    infer_window_days: int = 7
    infer_max_per_run: int = 50
    # 600s per strategy, not 120s: on large brains (10k+ neurons) the heavy passes
    # — compress, lifecycle, essence backfill — legitimately need minutes, and a
    # 120s cap aborted them mid-run so consolidation never converged.
    strategy_timeout_seconds: float = 600.0
    # The total must stay well above the per-strategy cap, otherwise a single slow
    # strategy consumes the whole budget and every later strategy times out.
    total_timeout_seconds: float = 3600.0


@dataclass(frozen=True)
class MergeDetail:
    """Details of a single fiber merge operation."""

    original_fiber_ids: tuple[str, ...]
    merged_fiber_id: str
    neuron_count: int
    reason: str


@dataclass
class ConsolidationReport:
    """Report of consolidation operation results."""

    started_at: datetime = field(default_factory=utcnow)
    duration_ms: float = 0.0
    synapses_pruned: int = 0
    neurons_pruned: int = 0
    fibers_merged: int = 0
    fibers_removed: int = 0
    fibers_created: int = 0
    summaries_created: int = 0
    stages_advanced: int = 0
    patterns_extracted: int = 0
    synapses_inferred: int = 0
    co_activations_pruned: int = 0
    synapses_enriched: int = 0
    dream_synapses_created: int = 0
    habits_learned: int = 0
    query_patterns_learned: int = 0
    action_events_pruned: int = 0
    retrieval_traces_pruned: int = 0
    duplicates_found: int = 0
    new_alias_links: int = 0
    """ALIAS edges actually created this run.

    ``duplicates_found`` is a *census* -- it counts anchors that look like
    duplicates, whether or not anything was done about them. Reporting only the
    census made a steady-state brain look like it was failing to do work every
    single run. This counter is the work.
    """
    alias_links_existing: int = 0
    """Census pairs that already carried their ALIAS edge, so nothing was written.

    Together with the failure counters in ``extra`` this closes the accounting:
    for a single non-dry-run dedup pass over a fresh report,

        duplicates_found == new_alias_links + alias_links_existing
                            + alias_checks_failed + alias_writes_failed
                            + alias_pairs_skipped_invalid

    (the three ``extra`` keys are absent when zero). Without it, ``0 new links``
    could mean "everything was already linked" or "every attempt failed", and
    the report gave the reader no way to tell.
    """
    semantic_synapses_created: int = 0
    semantic_synapses_skipped: int = 0
    """Eligible pairs that already carried a synapse, so no edge was created."""
    drift_clusters_found: int = 0
    """Tag clusters (re)detected and persisted by DETECT_DRIFT this run."""
    memories_promoted: int = 0
    fibers_compressed: int = 0
    tokens_saved: int = 0
    neurons_reactivated: int = 0
    essences_generated: int = 0
    reasoning_traces_ingested: int = 0
    reasoning_patterns_learned: int = 0
    merge_details: list[MergeDetail] = field(default_factory=list)
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def _semantic_synapse_line(self) -> str:
        """Render the semantic-link counters so a capped run cannot be misread.

        Printing a bare number made a truncated run and a saturated brain look
        identical: the cap produced the same figure every time, which reads as
        "nothing is progressing" when the truth is "a backlog is draining one
        capped run at a time".
        """
        line = (
            f"{self.semantic_synapses_created} created, "
            f"{self.semantic_synapses_skipped} skipped (existing)"
        )
        if self.extra.get("semantic_link_truncated"):
            line += " [truncated at cap]"
        return line

    def _stage_transitions_suffix(self) -> str:
        """Render the per-hop breakdown so a flat total cannot hide "zero new semantic".

        stages_advanced sums stm->working, working->episodic, and
        episodic->semantic into one count; without this, "15 advanced" and
        "zero new semantic" print identically.
        """
        transitions = self.extra.get("stage_transitions")
        if not transitions:
            return ""
        parts = ", ".join(f"{key}: {count}" for key, count in transitions.items() if count)
        return f" ({parts})" if parts else ""

    def _alias_link_line(self) -> str:
        """Render the dedup counters so "nothing to do" cannot read like "all broken".

        The old line printed the census and the created count only, so a brain
        whose duplicates were all already linked and a brain whose every write
        was being suppressed both rendered as ``(new alias links: 0)``.
        """
        if self.dry_run:
            line = f"{self.duplicates_found} pairs (census only; links not checked in dry run)"
        else:
            line = (
                f"{self.duplicates_found} pairs (census), "
                f"{self.new_alias_links} new links, "
                f"{self.alias_links_existing} already linked"
            )
            problems = []
            checks_failed = int(self.extra.get("alias_checks_failed", 0))
            writes_failed = int(self.extra.get("alias_writes_failed", 0))
            skipped = int(self.extra.get("alias_pairs_skipped_invalid", 0))
            if checks_failed:
                problems.append(f"{checks_failed} checks FAILED (state unknown)")
            if writes_failed:
                problems.append(f"{writes_failed} writes FAILED")
            if skipped:
                problems.append(f"{skipped} pairs skipped as invalid")
            if problems:
                line += f" [{', '.join(problems)}]"
        if self.extra.get("dedup_anchors_truncated"):
            total = self.extra.get("dedup_anchors_total")
            scanned = self.extra.get("dedup_anchors_scanned")
            line += f" [census truncated at anchor cap: {scanned} of {total} anchors compared]"
        return line

    def summary(self) -> str:
        """Generate human-readable summary."""
        mode = " (dry run)" if self.dry_run else ""
        lines = [
            f"Consolidation Report{mode} ({self.started_at.strftime('%Y-%m-%d %H:%M')})",
            f"  Synapses pruned: {self.synapses_pruned}",
            f"  Neurons pruned: {self.neurons_pruned}",
            f"  Fibers merged: {self.fibers_merged} -> {self.fibers_created} new",
            f"  Fibers removed: {self.fibers_removed}",
            f"  Summaries created: {self.summaries_created}",
            f"  Synapses inferred: {self.synapses_inferred}",
            f"  Co-activations pruned: {self.co_activations_pruned}",
            f"  Synapses enriched: {self.synapses_enriched}",
            f"  Dream synapses created: {self.dream_synapses_created}",
            f"  Habits learned: {self.habits_learned}",
            f"  Query patterns learned: {self.query_patterns_learned}",
            f"  Action events pruned: {self.action_events_pruned}",
            f"  Duplicate anchors: {self._alias_link_line()}",
            f"  Semantic synapses: {self._semantic_synapse_line()}",
            f"  Memories promoted (type): {self.memories_promoted}",
            f"  Stages advanced: {self.stages_advanced}{self._stage_transitions_suffix()}",
            f"  Fibers compressed: {self.fibers_compressed}",
            f"  Tokens saved: {self.tokens_saved}",
            f"  Reasoning traces ingested: {self.reasoning_traces_ingested}",
            f"  Reasoning patterns learned: {self.reasoning_patterns_learned}",
            f"  Drift clusters found: {self.drift_clusters_found}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        if self.merge_details:
            lines.append("  Merge details:")
            for detail in self.merge_details:
                lines.append(
                    f"    {len(detail.original_fiber_ids)} fibers -> {detail.merged_fiber_id[:8]}... "
                    f"({detail.neuron_count} neurons, {detail.reason})"
                )

        # Zeros from a pass where stages died look exactly like zeros from a
        # pass where there was nothing to do. Say which it was — but only when
        # something actually went wrong, so a clean run stays quiet.
        failed = self.extra.get("failed_strategies")
        if failed:
            lines.append(f"  Stages failed: {', '.join(failed)}")
        timed_out = self.extra.get("timed_out_strategies")
        if timed_out:
            lines.append(f"  Stages timed out: {', '.join(timed_out)}")

        backfilled = self.extra.get("maturations_backfilled")
        if backfilled:
            lines.append(
                f"  Maturations backfilled from fiber age: {backfilled} "
                "(these reached their stage without earning it through recall)"
            )
        unreachable = self.extra.get("maturations_unreachable")
        if unreachable:
            lines.append(
                f"  Maturations still missing (outside backfill's fiber window): {unreachable}"
            )

        # Add eligibility hints when nothing happened
        hints = self._eligibility_hints()
        if hints:
            lines.append("")
            lines.append("  Why nothing changed:")
            for hint in hints:
                lines.append(f"    - {hint}")

        return "\n".join(lines)

    def _eligibility_hints(self) -> list[str]:
        """Explain why consolidation produced no changes."""
        hints: list[str] = []
        total_changes = (
            self.synapses_pruned
            + self.neurons_pruned
            + self.fibers_merged
            + self.fibers_removed
            + self.summaries_created
            + self.synapses_inferred
            + self.synapses_enriched
            + self.dream_synapses_created
            + self.habits_learned
            + self.query_patterns_learned
            + self.duplicates_found
            + self.semantic_synapses_created
            + self.fibers_compressed
            + self.stages_advanced
            + self.drift_clusters_found
        )
        if total_changes > 0:
            return hints

        hints.append("Prune: synapses must be inactive for 7+ days with weight below 0.05")
        hints.append("Merge: fibers need >50% neuron overlap (Jaccard) and <=50 neurons each")
        hints.append("Summarize: need 3+ fibers sharing >40% tag overlap to form a cluster")
        hints.append("Mature: memories advance stages over time through repeated recall")
        hints.append("Habits: need 3+ occurrences of the same action sequence within 30 days")
        hints.append(
            "Tip: store more memories and recall them over several days, then consolidate again"
        )
        return hints


class ConsolidationEngine:
    """Engine for memory consolidation operations.

    Supports strategies: prune, merge, summarize, mature, infer, enrich,
    dream, learn_habits, dedup.

    Strategies are grouped into dependency tiers and run in parallel
    within each tier sequentially (to avoid stale data).
    """

    # Dependency tiers — strategies within a tier are independent and
    # can safely run concurrently. Tiers execute sequentially because
    # later tiers depend on results from earlier ones.
    STRATEGY_TIERS: tuple[frozenset[ConsolidationStrategy], ...] = (
        frozenset(
            {
                ConsolidationStrategy.PRUNE,
                ConsolidationStrategy.LEARN_HABITS,
                ConsolidationStrategy.DEDUP,
                ConsolidationStrategy.PROCESS_TOOL_EVENTS,
                ConsolidationStrategy.PROCESS_REASONING_TRACES,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.MERGE,
                ConsolidationStrategy.INTERFERENCE,
                ConsolidationStrategy.MATURE,
                ConsolidationStrategy.COMPRESS,
                ConsolidationStrategy.LIFECYCLE,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.SUMMARIZE,
                ConsolidationStrategy.INFER,
                ConsolidationStrategy.SCHEMA,
                ConsolidationStrategy.ESSENCE_BACKFILL,
                ConsolidationStrategy.LEARN_REASONING,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.ENRICH,
                ConsolidationStrategy.DREAM,
                ConsolidationStrategy.REPLAY,
            }
        ),
        frozenset(
            {
                ConsolidationStrategy.SEMANTIC_LINK,
                ConsolidationStrategy.DETECT_DRIFT,
            }
        ),
    )

    def __init__(
        self,
        storage: NeuralStorage,
        config: ConsolidationConfig | None = None,
        dream_decay_multiplier: float = 10.0,
        tier_config: TierConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or ConsolidationConfig()
        self._dream_decay_multiplier = dream_decay_multiplier
        self._tier_config = tier_config

    async def _run_strategy(
        self,
        strategy: ConsolidationStrategy,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Dispatch a single strategy to its implementation method."""
        dispatch: dict[ConsolidationStrategy, Callable[[], Awaitable[None]]] = {
            ConsolidationStrategy.PRUNE: lambda: self._prune(report, reference_time, dry_run),
            ConsolidationStrategy.MERGE: lambda: self._merge(report, dry_run),
            ConsolidationStrategy.SUMMARIZE: lambda: self._summarize(report, dry_run),
            ConsolidationStrategy.MATURE: lambda: self._mature(report, reference_time, dry_run),
            ConsolidationStrategy.INFER: lambda: self._infer(report, reference_time, dry_run),
            ConsolidationStrategy.ENRICH: lambda: self._enrich(report, dry_run),
            ConsolidationStrategy.DREAM: lambda: self._dream(report, dry_run),
            ConsolidationStrategy.LEARN_HABITS: lambda: self._learn_habits(
                report, reference_time, dry_run
            ),
            ConsolidationStrategy.DEDUP: lambda: self._dedup(report, dry_run),
            ConsolidationStrategy.SEMANTIC_LINK: lambda: self._semantic_link(report, dry_run),
            ConsolidationStrategy.COMPRESS: lambda: self._compress(report, reference_time, dry_run),
            ConsolidationStrategy.LIFECYCLE: lambda: self._lifecycle(
                report, reference_time, dry_run
            ),
            ConsolidationStrategy.PROCESS_TOOL_EVENTS: lambda: self._process_tool_events(
                report, dry_run
            ),
            ConsolidationStrategy.PROCESS_REASONING_TRACES: lambda: self._process_reasoning_traces(
                report, dry_run
            ),
            ConsolidationStrategy.LEARN_REASONING: lambda: self._learn_reasoning(report, dry_run),
            ConsolidationStrategy.ESSENCE_BACKFILL: lambda: self._essence_backfill(report, dry_run),
            ConsolidationStrategy.REPLAY: lambda: self._replay(report, dry_run),
            ConsolidationStrategy.SCHEMA: lambda: self._schema(report, dry_run),
            ConsolidationStrategy.INTERFERENCE: lambda: self._interference(report, dry_run),
            ConsolidationStrategy.DETECT_DRIFT: lambda: self._detect_drift(report, dry_run),
        }
        handler = dispatch.get(strategy)
        if handler is not None:
            await handler()

    async def run(
        self,
        strategies: list[ConsolidationStrategy] | None = None,
        dry_run: bool = False,
        reference_time: datetime | None = None,
    ) -> ConsolidationReport:
        """Run consolidation with specified strategies.

        Strategies are grouped into dependency tiers and run in parallel
        within each tier. Tiers execute sequentially so that later
        strategies can depend on results from earlier ones.

        Each strategy has a per-strategy timeout (default 600s) and the
        entire consolidation has a total timeout (default 3600s) to prevent
        runaway execution.

        Args:
            strategies: List of strategies to run (default: all)
            dry_run: If True, calculate but don't apply changes
            reference_time: Reference time for age calculations

        Returns:
            ConsolidationReport with operation statistics
        """
        if strategies is None:
            strategies = [ConsolidationStrategy.ALL]

        reference_time = ensure_naive_utc(reference_time) if reference_time else utcnow()
        report = ConsolidationReport(started_at=reference_time, dry_run=dry_run)
        start = time.perf_counter()

        # Normalize string strategies to enum values (callers may pass raw strings)
        normalized: list[ConsolidationStrategy] = [
            s if isinstance(s, ConsolidationStrategy) else ConsolidationStrategy(s)
            for s in strategies
        ]

        run_all = ConsolidationStrategy.ALL in normalized
        requested: set[ConsolidationStrategy] = (
            {s for s in ConsolidationStrategy if s != ConsolidationStrategy.ALL}
            if run_all
            else set(normalized)
        )

        strategy_timeout = self._config.strategy_timeout_seconds
        total_timeout = self._config.total_timeout_seconds
        timed_out_strategies: list[str] = []
        # A strategy that raises anything other than TimeoutError used to escape
        # this loop and kill the whole pass, so the caller saw a traceback (or,
        # worse, a report full of zeros that looks exactly like "nothing to do").
        # Record the failure, keep going, and name it in the summary.
        failed_strategies: list[str] = []

        for tier in self.STRATEGY_TIERS:
            tier_strategies = tier & requested
            if not tier_strategies:
                continue
            # Run strategies sequentially within each tier to avoid
            # stale data snapshots and shared mutable report races
            for strategy in tier_strategies:
                elapsed = time.perf_counter() - start
                if elapsed > total_timeout:
                    remaining = [s.value for s in tier_strategies if s >= strategy]
                    timed_out_strategies.extend(remaining)
                    logger.warning(
                        "Consolidation total timeout (%.0fs) reached after %.1fs, "
                        "skipping remaining strategies: %s",
                        total_timeout,
                        elapsed,
                        remaining,
                    )
                    break

                logger.info("Consolidation: starting %s", strategy.value)
                strategy_start = time.perf_counter()
                try:
                    await asyncio.wait_for(
                        self._run_strategy(strategy, report, reference_time, dry_run),
                        timeout=strategy_timeout,
                    )
                except TimeoutError:
                    strategy_elapsed = time.perf_counter() - strategy_start
                    logger.warning(
                        "Consolidation: %s timed out after %.1fs (limit: %.0fs)",
                        strategy.value,
                        strategy_elapsed,
                        strategy_timeout,
                    )
                    timed_out_strategies.append(strategy.value)
                except Exception as exc:
                    strategy_elapsed = time.perf_counter() - strategy_start
                    logger.error(
                        "Consolidation: %s failed after %.1fs: %s",
                        strategy.value,
                        strategy_elapsed,
                        exc,
                        exc_info=True,
                    )
                    failed_strategies.append(f"{strategy.value} ({type(exc).__name__})")
                finally:
                    strategy_elapsed = time.perf_counter() - strategy_start
                    logger.info(
                        "Consolidation: %s finished in %.1fs",
                        strategy.value,
                        strategy_elapsed,
                    )
            else:
                continue
            break  # break outer loop if inner broke due to total timeout

        if timed_out_strategies:
            report.extra["timed_out_strategies"] = timed_out_strategies
        if failed_strategies:
            report.extra["failed_strategies"] = failed_strategies

        # Auto-tier promotion/demotion (Pro feature, runs after standard strategies)
        await self._run_auto_tier(report, dry_run)

        report.duration_ms = (time.perf_counter() - start) * 1000
        return report

    async def _run_auto_tier(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run auto-tier promotion/demotion if enabled (Pro feature).

        Runs after all standard consolidation strategies complete.
        Results are attached to report.extra["auto_tier"].
        """
        if self._tier_config is None or not self._tier_config.auto_enabled:
            return

        # Pro gate: auto-tier requires Pro license
        try:
            from surreal_memory.plugins import has_pro

            if not has_pro():
                return
        except ImportError:
            return

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return

        try:
            from surreal_memory.engine.tier_engine import TierEngine

            engine = TierEngine(self._storage, self._tier_config)
            tier_report = await engine.apply(brain_id, dry_run=dry_run)
            report.extra["auto_tier"] = tier_report.to_dict()
        except Exception as e:
            logger.error("Auto-tier failed during consolidation: %s", e, exc_info=True)
            report.extra["auto_tier"] = {"error": "auto-tier failed"}

    async def _prune(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Prune weak synapses and orphan neurons."""
        logger = logging.getLogger(__name__)

        # Ensure brain context is set
        if not self._storage.current_brain_id:
            return

        # Get all synapses
        all_synapses = await self._storage.get_synapses()
        pruned_synapse_ids: set[str] = set()

        # Preload pinned neuron IDs to protect from pruning
        pinned_neuron_ids = await self._storage.get_pinned_neuron_ids()

        # Build fiber salience cache for high-salience protection
        fibers_for_salience = await self._storage.get_fibers(limit=10000)
        fiber_salience_cache: dict[str, list[Fiber]] = {}
        for fib in fibers_for_salience:
            if fib.salience > 0.8:
                for nid in fib.neuron_ids:
                    fiber_salience_cache.setdefault(nid, []).append(fib)

        # Bridge detection (below) needs, per source neuron, its outgoing synapses.
        # We ALREADY hold every synapse for the brain in ``all_synapses``, so group
        # them in-memory instead of asking the storage layer. The SurrealDB backend
        # has no batched ``get_synapses_for_neurons``, so the previous call fanned out
        # into one query PER candidate source neuron — tens of thousands on a large
        # brain — which blew the per-strategy prune budget (120s timeout). Grouping the
        # already-loaded list is O(n) in Python and issues zero extra queries.
        neighbor_synapses_map: dict[str, list[Synapse]] = {}
        for s in all_synapses:
            neighbor_synapses_map.setdefault(s.source_id, []).append(s)

        for syn_idx, synapse in enumerate(all_synapses):
            if syn_idx % 500 == 0 and syn_idx > 0:
                await asyncio.sleep(0)  # Yield to event loop

            # Skip synapses connected to pinned (KB) neurons
            if synapse.source_id in pinned_neuron_ids or synapse.target_id in pinned_neuron_ids:
                continue

            # Apply time-based decay before checking weight threshold
            decayed = synapse.time_decay(reference_time=reference_time)

            # Inferred synapses with low reinforcement decay 2x faster
            is_inferred = synapse.metadata.get("_inferred", False)
            if is_inferred and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=0.5)

            # Dream synapses decay Nx faster (default 10x)
            is_dream = synapse.metadata.get("_dream", False)
            if is_dream and synapse.reinforced_count < 2:
                dream_factor = 1.0 / self._dream_decay_multiplier
                decayed = decayed.decay(factor=dream_factor)

            # Semantic discovery synapses decay 2x faster unless reinforced
            is_semantic = synapse.metadata.get("_semantic_discovery", False)
            if is_semantic and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=0.5)

            should_prune = decayed.weight < self._config.prune_weight_threshold

            # Check inactivity
            if synapse.last_activated is not None:
                days_inactive = (reference_time - synapse.last_activated).total_seconds() / 86400
                should_prune = (
                    should_prune and days_inactive >= self._config.prune_min_inactive_days
                )
            elif synapse.created_at is not None:
                days_since_creation = (reference_time - synapse.created_at).total_seconds() / 86400
                # Never-activated synapses use a shorter grace period
                grace_period = max(1.0, self._config.prune_min_inactive_days / 7)
                should_prune = should_prune and days_since_creation >= grace_period

            if should_prune:
                # High-salience fibers resist pruning
                source_fibers = fiber_salience_cache.get(synapse.source_id, [])
                for fib in source_fibers:
                    if fib.salience > 0.8:
                        should_prune = False
                        break

            if should_prune:
                # Protect bridge synapses (only connection between source and target)
                if synapse.weight >= 0.02:
                    out_synapses = neighbor_synapses_map.get(synapse.source_id, [])
                    neighbor_ids = {s.target_id for s in out_synapses}
                    if synapse.target_id in neighbor_ids and len(neighbor_ids) <= 1:
                        continue  # Bridge synapse — don't prune

                pruned_synapse_ids.add(synapse.id)
                report.synapses_pruned += 1

        # Batch delete all pruned synapses at once
        if pruned_synapse_ids and not dry_run:
            if hasattr(self._storage, "delete_synapses_batch"):
                await self._storage.delete_synapses_batch(pruned_synapse_ids)
            else:
                for sid in pruned_synapse_ids:
                    await self._storage.delete_synapse(sid)

        # Update fiber synapse_ids to remove pruned refs (only if synapses were pruned)
        fibers = fibers_for_salience
        if pruned_synapse_ids:
            # Build inverted index: synapse_id -> fiber indices (only for pruned IDs)
            synapse_to_fiber_idx: dict[str, list[int]] = {}
            for idx, fiber in enumerate(fibers):
                for sid in fiber.synapse_ids & pruned_synapse_ids:
                    synapse_to_fiber_idx.setdefault(sid, []).append(idx)

            # Only update fibers that reference pruned synapses
            affected_indices: set[int] = set()
            for indices in synapse_to_fiber_idx.values():
                affected_indices.update(indices)

            for idx in affected_indices:
                if not dry_run:
                    fiber = fibers[idx]
                    updated_fiber = dc_replace(
                        fiber,
                        synapse_ids=fiber.synapse_ids - pruned_synapse_ids,
                    )
                    await self._storage.update_fiber(updated_fiber)

        # Find orphan neurons (no synapses AND not in any fiber)
        if not self._config.prune_isolated_neurons:
            return

        # Derive remaining synapses from cached list instead of re-fetching
        connected_neuron_ids: set[str] = set()
        for syn in all_synapses:
            if syn.id not in pruned_synapse_ids:
                connected_neuron_ids.add(syn.source_id)
                connected_neuron_ids.add(syn.target_id)

        # Protect ALL neurons in fibers, not just anchors
        fiber_neuron_ids: set[str] = set()
        for fiber in fibers:
            fiber_neuron_ids.update(fiber.neuron_ids)

        # Dead neuron pruning: never-accessed + old enough + not pinned
        dead_neuron_days = getattr(self._config, "prune_dead_neuron_days", 14.0)

        # Paginate through all neurons in fixed-size batches. OMIT the embedding
        # vector: orphan/dead detection needs only id + created_at, so dragging the
        # 1024-float vector for tens of thousands of neurons (~100 MB per 5k-page on
        # a large brain, ~1.4 GB total) was the dominant cost that blew the 120s
        # prune budget after the synapse N+1 was removed.
        batch_size = 5000
        offset = 0
        orphan_ids: list[str] = []
        dead_ids: list[str] = []
        est_batches = (len(connected_neuron_ids | fiber_neuron_ids) // batch_size) + 1
        logger.info(
            "Prune: scanning neurons in %d-row batches (~%d+ batches; embedding "
            "vectors omitted from the scan)",
            batch_size,
            est_batches,
        )
        # Dead-neuron detection needs each neuron's access_frequency. Fetching states
        # per page (get_neuron_states_batch → a 5000-id `IN` query) cost ~10s/page on a
        # 67k-neuron brain — ~140s total, the dominant prune cost after the embedding
        # OMIT. One brain-wide fetch is a single filtered scan (~1.4s), so load all
        # states once and look them up in-memory across every page.
        try:
            states_by_id: dict[str, Any] = {
                s.neuron_id: s for s in await self._storage.get_all_neuron_states()
            }
            use_prefetched_states = True
        except Exception:
            logger.debug("get_all_neuron_states failed; per-page state fallback", exc_info=True)
            states_by_id = {}
            use_prefetched_states = False

        while True:
            batch = await self._storage.find_neurons(
                limit=batch_size, offset=offset, ephemeral=False, include_embedding=False
            )
            if not batch:
                break

            # Dead-neuron check reads access_frequency from the prefetched states; only
            # fall back to a per-page batch if the brain-wide fetch was unavailable.
            if use_prefetched_states:
                states = states_by_id
            else:
                states = await self._storage.get_neuron_states_batch([n.id for n in batch])

            for neuron in batch:
                # Never auto-prune pinned neurons, whether isolated (orphan) or
                # dead. The dead-neuron path already honored pinned, but the
                # orphan path short-circuited above it, so pinned isolated
                # neurons were permanently deleted. Hoist the guard so it
                # protects both paths.
                if neuron.id in pinned_neuron_ids:
                    continue

                is_orphan = (
                    neuron.id not in connected_neuron_ids and neuron.id not in fiber_neuron_ids
                )

                # Fiber members are never "dead neuron" candidates: reinforce()
                # (retrieval.py) only bumps access_frequency for the top-10
                # highest-activation neurons per recall, so most neurons that are
                # genuinely part of an actively-recalled fiber still read
                # access_frequency == 0 forever. Without this guard, "dead"
                # pruning deletes real memory content (measured live:
                # 57150/63380 neuron_states were fiber members with
                # access_frequency == 0 — nearly the whole brain was wrongly
                # eligible). A fiber member can never be an orphan either (that
                # requires NOT being in fiber_neuron_ids), so this skip is safe
                # for both branches.
                if neuron.id in fiber_neuron_ids:
                    continue

                # Same never-accessed + old-enough predicate now gates BOTH the
                # orphan branch and the dead-neuron branch (#113). Previously the
                # orphan branch short-circuited straight to pruning — a neuron
                # written moments ago, before consolidation had any chance to
                # link it, was immediately deleted just for being momentarily
                # unconnected. Hoisting this guard above the orphan short-circuit
                # mirrors how the pinned guard was hoisted above both branches in
                # #17: a merely-young or recently-accessed orphan is left alone
                # and re-evaluated on the next consolidation pass instead of
                # being deleted outright.
                state = states.get(neuron.id)
                freq = state.access_frequency if state else 0
                if freq > 0:
                    continue
                age_days = (reference_time - neuron.created_at).total_seconds() / 86400
                if age_days < dead_neuron_days:
                    continue

                report.neurons_pruned += 1
                if is_orphan:
                    orphan_ids.append(neuron.id)
                else:
                    dead_ids.append(neuron.id)

            offset += len(batch)
            if len(batch) < batch_size:
                break

        all_prune_ids = orphan_ids + dead_ids
        if dead_ids:
            logger.info(
                "Dead neuron prune: %d orphans + %d dead (never accessed, >%gd old)",
                len(orphan_ids),
                len(dead_ids),
                dead_neuron_days,
            )

        if not dry_run and all_prune_ids:
            # Use batch delete if available, else fall back to individual deletes
            if hasattr(self._storage, "delete_neurons_batch"):
                await self._storage.delete_neurons_batch(all_prune_ids)
            else:
                for nid in all_prune_ids:
                    await self._storage.delete_neuron(nid)

        # Prune old unpromoted entity refs (lazy entity promotion cleanup)
        if not dry_run and hasattr(self._storage, "prune_old_entity_refs"):
            prune_days = getattr(self._config, "lazy_entity_prune_days", 90)
            try:
                pruned_refs = await self._storage.prune_old_entity_refs(prune_days)
                if pruned_refs > 0:
                    logger.info("Pruned %d old unpromoted entity refs", pruned_refs)
            except Exception:
                logger.debug("Entity ref pruning skipped (table may not exist)")

        # Prune old retrieval traces (telemetry TTL + max-count cap) — U4. Runs even
        # when tracing is currently disabled so a re-disable still cleans up its
        # accumulated traces; on a never-traced brain this is a cheap empty DELETE.
        if not dry_run and hasattr(self._storage, "prune_retrieval_traces"):
            try:
                from surreal_memory.unified_config import get_config

                trace_cfg = get_config().trace
                pruned_traces = await self._storage.prune_retrieval_traces(
                    retention_days=trace_cfg.retention_days,
                    max_traces=trace_cfg.max_traces,
                )
                if pruned_traces > 0:
                    logger.info("Pruned %d old retrieval traces", pruned_traces)
                    report.retrieval_traces_pruned = pruned_traces
            except Exception:
                logger.debug("Retrieval trace pruning skipped", exc_info=True)

    async def _merge(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Merge overlapping fibers using inverted index for O(n*m) performance.

        Instead of O(n²) pairwise comparison, builds a neuron→fiber inverted
        index to find only fibers that actually share neurons.
        """
        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < 2:
            return

        fiber_list = list(fibers)
        n = len(fiber_list)

        # Build inverted index: neuron_id → set of fiber indices
        neuron_to_fibers: dict[str, set[int]] = {}
        for idx, fiber in enumerate(fiber_list):
            if len(fiber.neuron_ids) > self._config.merge_max_fiber_size:
                continue
            for nid in fiber.neuron_ids:
                neuron_to_fibers.setdefault(nid, set()).add(idx)

        # Find candidate pairs (fibers sharing at least one neuron)
        candidate_pairs: set[tuple[int, int]] = set()
        max_candidate_pairs = 50_000
        for indices in neuron_to_fibers.values():
            # Skip overly-shared neurons (e.g. entity appearing in 500+ fibers)
            if len(indices) > 100:
                continue
            indices_list = sorted(indices)
            for i_pos in range(len(indices_list)):
                for j_pos in range(i_pos + 1, len(indices_list)):
                    candidate_pairs.add((indices_list[i_pos], indices_list[j_pos]))
            if len(candidate_pairs) >= max_candidate_pairs:
                break

        # Union-Find clustering
        uf = UnionFind(n)

        # Only compute Jaccard for actual candidate pairs
        pairs_checked = 0
        for i, j in candidate_pairs:
            pairs_checked += 1
            if pairs_checked % 1000 == 0:
                await asyncio.sleep(0)  # yield so timeout can fire
            # Domain guard: never merge structured/verbatim fibers with non-structured
            fi_verbatim = fiber_list[i].metadata.get("_verbatim", False)
            fj_verbatim = fiber_list[j].metadata.get("_verbatim", False)
            if fi_verbatim != fj_verbatim:
                continue

            # Never merge a learned artifact away. The merged fiber REPLACES its
            # members' metadata wholesale, so every marker and every field the
            # feature reads is dropped, and the members are then deleted.
            #
            # `_habit_pattern`: `smem habits list` would silently lose the habit
            # and habits could never accumulate over time.
            #
            # `_reasoning_pattern`: category coverage is computed from
            # `_source_model` / `_reasoning_category` / `_reasoning_confidence`.
            # Learned patterns are the most exposed of all fibers here, because
            # every pattern in a category shares that category's concept neuron
            # -- exactly what the overlap check keys on -- so a whole mining
            # run collapses into one metadata-less fiber. Coverage then reads
            # zero with the traces already marked processed, which before the
            # reprocess path existed was unrecoverable.
            if any(
                f.metadata.get(marker)
                for f in (fiber_list[i], fiber_list[j])
                for marker in ("_habit_pattern", "_reasoning_pattern")
            ):
                continue

            set_a = fiber_list[i].neuron_ids
            set_b = fiber_list[j].neuron_ids
            intersection = len(set_a & set_b)
            union_size = len(set_a | set_b)

            if union_size > 0:
                jaccard = intersection / union_size
                # Lower threshold for temporally-close fibers
                if fiber_list[i].created_at and fiber_list[j].created_at:
                    time_diff = abs(
                        (fiber_list[i].created_at - fiber_list[j].created_at).total_seconds()
                    )
                else:
                    time_diff = float("inf")
                effective_threshold = (
                    self._config.merge_overlap_threshold * 0.6
                    if time_diff < 3600
                    else self._config.merge_overlap_threshold
                )
                if jaccard >= effective_threshold:
                    uf.union(i, j)

        # Group fibers by root
        groups = uf.groups()

        # Merge groups with more than 1 member
        for members in groups.values():
            if len(members) < 2:
                continue

            member_fibers = [fiber_list[i] for i in members]

            # Create merged fiber
            merged_neuron_ids: set[str] = set()
            merged_synapse_ids: set[str] = set()
            merged_tags: set[str] = set()
            max_salience = 0.0
            best_anchor = member_fibers[0].anchor_neuron_id
            best_frequency = 0

            for fiber in member_fibers:
                merged_neuron_ids |= fiber.neuron_ids
                merged_synapse_ids |= fiber.synapse_ids
                merged_tags |= fiber.tags
                if fiber.salience > max_salience:
                    max_salience = fiber.salience
                if fiber.frequency > best_frequency:
                    best_frequency = fiber.frequency
                    best_anchor = fiber.anchor_neuron_id

            merged_fiber_id = str(uuid4())
            # Merge auto_tags and agent_tags separately
            merged_auto_tags: set[str] = set()
            merged_agent_tags: set[str] = set()
            for fiber in member_fibers:
                merged_auto_tags |= fiber.auto_tags
                merged_agent_tags |= fiber.agent_tags
            merged_fiber = Fiber(
                id=merged_fiber_id,
                neuron_ids=merged_neuron_ids,
                synapse_ids=merged_synapse_ids,
                anchor_neuron_id=best_anchor,
                pathway=[best_anchor],
                salience=max_salience,
                frequency=best_frequency,
                auto_tags=merged_auto_tags,
                agent_tags=merged_agent_tags,
                summary=f"Merged from {len(member_fibers)} fibers",
                metadata={"merged_from": [f.id for f in member_fibers]},
                created_at=min(f.created_at for f in member_fibers),
            )

            report.fibers_merged += len(member_fibers)
            report.fibers_created += 1
            report.merge_details.append(
                MergeDetail(
                    original_fiber_ids=tuple(f.id for f in member_fibers),
                    merged_fiber_id=merged_fiber_id,
                    neuron_count=len(merged_neuron_ids),
                    reason="neuron_overlap",
                )
            )

            if not dry_run:
                from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage

                # Read source maturation BEFORE delete_fiber. delete_fiber has no
                # cascade to the maturation table (confirmed: it only deletes the
                # fiber row), so this ordering is defensive, not load-bearing --
                # but reading after would still be wrong to *rely* on that. A
                # concurrent reinforce() landing between this read and the
                # delete below is a narrow, self-healing loss (the row becomes
                # an orphan cleaned up by cleanup_orphaned_maturations, and the
                # fiber gets rehearsed again on its next recall) -- consistent
                # with this codebase's non-transactional storage model
                # elsewhere (e.g. save_maturation's own delete-race comment).
                _stage_order = list(MemoryStage)
                source_maturations = []
                for fiber in member_fibers:
                    try:
                        record = await self._storage.get_maturation(fiber.id)
                    except Exception:
                        # A persistent storage fault here must not abort the
                        # whole merge pass -- this group simply starts the
                        # merged fiber without inherited maturation, same as
                        # if none of its sources had a maturation row at all.
                        logger.debug(
                            "Could not read maturation for fiber %s during merge",
                            fiber.id,
                            exc_info=True,
                        )
                        continue
                    if record is not None:
                        source_maturations.append(record)

                for fiber in member_fibers:
                    await self._storage.delete_fiber(fiber.id)
                    report.fibers_removed += 1
                await self._storage.add_fiber(merged_fiber)

                if source_maturations:
                    # Merging must never cost a fiber the maturation progress it
                    # already earned: highest stage of any source, the union of
                    # reinforcement timestamps (those, not a count, decide
                    # distinct_reinforcement_days), rehearsal_count derived
                    # from that same union (not summed independently -- every
                    # organically-created record keeps rehearsal_count ==
                    # len(reinforcement_timestamps), and summing counts while
                    # deduplicating timestamps could violate that invariant on
                    # a timestamp collision), and the oldest stage_entered_at
                    # so the merged fiber keeps whatever dwell time a source
                    # already accrued instead of resetting the clock.
                    inherited_stage = max(
                        source_maturations, key=lambda r: _stage_order.index(r.stage)
                    ).stage
                    merged_timestamps = tuple(
                        sorted(
                            {
                                ts
                                for record in source_maturations
                                for ts in record.reinforcement_timestamps
                            }
                        )
                    )
                    await self._storage.save_maturation(
                        MaturationRecord(
                            fiber_id=merged_fiber_id,
                            brain_id=source_maturations[0].brain_id,
                            stage=inherited_stage,
                            stage_entered_at=min(r.stage_entered_at for r in source_maturations),
                            rehearsal_count=len(merged_timestamps),
                            reinforcement_timestamps=merged_timestamps,
                        )
                    )

    async def _summarize(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Create concept neurons for tag-based clusters using inverted index."""
        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < self._config.summarize_min_cluster_size:
            return

        fiber_list = [f for f in fibers if f.tags]

        # Cap fiber count for O(N²) pair comparison — keep highest-salience
        max_fibers_for_clustering = 1000
        if len(fiber_list) > max_fibers_for_clustering:
            fiber_list = sorted(fiber_list, key=lambda f: f.salience, reverse=True)[
                :max_fibers_for_clustering
            ]
        if len(fiber_list) < self._config.summarize_min_cluster_size:
            return

        n = len(fiber_list)

        # Build inverted index: tag → set of fiber indices
        tag_to_fibers: dict[str, set[int]] = {}
        for idx, fiber in enumerate(fiber_list):
            for tag in fiber.tags:
                tag_to_fibers.setdefault(tag, set()).add(idx)

        # Find candidate pairs (fibers sharing at least one tag)
        # Skip overly common tags (>100 fibers) to avoid O(N²) explosion
        max_pairs = 50_000
        candidate_pairs: set[tuple[int, int]] = set()
        for indices in tag_to_fibers.values():
            if len(indices) > 100:
                continue  # Tag too common — skip to avoid combinatorial blowup
            indices_list = sorted(indices)
            for i_pos in range(len(indices_list)):
                if len(candidate_pairs) >= max_pairs:
                    break
                for j_pos in range(i_pos + 1, len(indices_list)):
                    if len(candidate_pairs) >= max_pairs:
                        break
                    candidate_pairs.add((indices_list[i_pos], indices_list[j_pos]))
            if len(candidate_pairs) >= max_pairs:
                break

        # Union-Find for tag clustering
        parent: dict[int, int] = {i: i for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for pair_idx, (i, j) in enumerate(candidate_pairs):
            if pair_idx % 1000 == 0 and pair_idx > 0:
                await asyncio.sleep(0)  # Yield to event loop
            tags_a = fiber_list[i].tags
            tags_b = fiber_list[j].tags
            intersection = len(tags_a & tags_b)
            union_size = len(tags_a | tags_b)
            if (
                union_size > 0
                and intersection / union_size >= self._config.summarize_tag_overlap_threshold
            ):
                union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        for members in groups.values():
            if len(members) < self._config.summarize_min_cluster_size:
                continue

            cluster_fibers = [fiber_list[i] for i in members]

            summaries = [f.summary for f in cluster_fibers if f.summary]
            all_tags: set[str] = set()
            for f in cluster_fibers:
                all_tags |= f.tags

            summary_content = (
                "; ".join(summaries[:10])
                if summaries
                else f"Cluster of {len(cluster_fibers)} memories"
            )
            tag_label = ", ".join(sorted(all_tags)[:5])
            concept_content = f"[{tag_label}] {summary_content[:200]}"

            if dry_run:
                report.summaries_created += 1
                continue

            concept_neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=concept_content,
                metadata={
                    "_consolidation": "summary",
                    "cluster_size": len(cluster_fibers),
                    "tags": sorted(all_tags),
                },
            )
            await self._storage.add_neuron(concept_neuron)

            anchor_ids: set[str] = set()
            for fiber in cluster_fibers:
                anchor_ids.add(fiber.anchor_neuron_id)

            # Filter out anchor neurons that were pruned in earlier tiers
            valid_anchor_ids: set[str] = set()
            for aid in anchor_ids:
                anchor_neuron = await self._storage.get_neuron(aid)
                if anchor_neuron is not None:
                    valid_anchor_ids.add(aid)
            anchor_ids = valid_anchor_ids

            synapse_ids: set[str] = set()
            for anchor_id in list(anchor_ids)[:10]:
                synapse = Synapse.create(
                    source_id=concept_neuron.id,
                    target_id=anchor_id,
                    type=SynapseType.RELATED_TO,
                    weight=0.6,
                )
                await self._storage.add_synapse(synapse)
                synapse_ids.add(synapse.id)

            summary_fiber = Fiber.create(
                neuron_ids={concept_neuron.id} | anchor_ids,
                synapse_ids=synapse_ids,
                anchor_neuron_id=concept_neuron.id,
                summary=concept_content,
                tags=all_tags,
                metadata={
                    "_consolidation": "summary_fiber",
                    "source_fibers": [f.id for f in cluster_fibers],
                },
            )
            await self._storage.add_fiber(summary_fiber)
            report.summaries_created += 1

    async def _mature(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Advance memory maturation stages, auto-promote types, extract patterns.

        0. Auto-promote frequently-recalled context memories to fact
        1. Advance all maturation records through stage transitions
        2. Extract patterns from episodic memories ready for semantic promotion
        """
        import logging

        from surreal_memory.core.memory_types import MemoryType
        from surreal_memory.engine.memory_stages import (
            MemoryStage,
            compute_stage_transition,
        )
        from surreal_memory.engine.pattern_extraction import extract_patterns

        _logger = logging.getLogger(__name__)

        # stages_advanced sums all three hops into one counter, so "15 advanced"
        # and "zero new semantic" were indistinguishable without reading raw
        # maturation rows. compute_stage_transition only ever moves one hop per
        # call, so every transition it produces is one of exactly these three.
        _hop_keys = {
            (MemoryStage.SHORT_TERM, MemoryStage.WORKING): "stm_to_working",
            (MemoryStage.WORKING, MemoryStage.EPISODIC): "working_to_episodic",
            (MemoryStage.EPISODIC, MemoryStage.SEMANTIC): "episodic_to_semantic",
        }

        # Phase 0: Auto-promote context→fact for frequently-recalled memories
        # Must run before prune to prevent promotion candidates from expiring.
        # Graduated: frequency >= 5 triggers promotion to fact (no expiry).
        if not dry_run:
            try:
                candidates = await self._storage.get_promotion_candidates(
                    min_frequency=5,
                    source_type="context",
                )
                for candidate in candidates:
                    fiber_id = candidate["fiber_id"]
                    meta = candidate.get("metadata", {})
                    # Skip already-promoted memories
                    if meta.get("auto_promoted"):
                        continue
                    promoted = await self._storage.promote_memory_type(
                        fiber_id=fiber_id,
                        new_type=MemoryType.FACT,
                        new_expires_at=None,  # Facts don't expire
                    )
                    if promoted:
                        report.memories_promoted += 1
                if report.memories_promoted > 0:
                    _logger.info(
                        "Auto-promoted %d context memories to fact",
                        report.memories_promoted,
                    )
            except Exception:
                _logger.warning("Auto-promote failed (non-critical)", exc_info=True)

        # Clean up orphaned maturation records (fibers deleted without CASCADE)
        cleaned = await self._storage.cleanup_orphaned_maturations()
        if cleaned > 0:
            _logger.info("Cleaned up %d orphaned maturation records", cleaned)

        # Phase 0.5: give fibers that predate the maturation subsystem a row.
        # Without one a fiber can never advance a stage, so it is invisible to
        # every consolidation metric forever -- measured at 874 of 2819 fibers
        # on the live brain. Guarded on the counts so a healthy brain pays only
        # two cheap counts, and skipped entirely on a dry run.
        if not dry_run:
            try:
                fiber_count = await self._storage.get_total_fiber_count()
                maturation_count = len(await self._storage.find_maturations())
                if fiber_count > maturation_count:
                    backfilled = await self._storage.backfill_maturations()
                    # "skipped" counts fibers that already had a row -- adding it
                    # in would report the whole brain as freshly backfilled.
                    total = sum(v for k, v in backfilled.items() if k != "skipped")
                    if total:
                        report.extra["maturations_backfilled"] = total
                        _logger.info(
                            "Backfilled %d maturation rows (%d fibers, %d rows before)",
                            total,
                            fiber_count,
                            maturation_count,
                        )
                    elif backfilled:
                        # A deficit that the backfill could not close: its fiber
                        # scan is windowed, so on a very large brain the oldest
                        # fibers -- exactly the ones predating maturation -- stay
                        # out of reach. Surface it instead of silently retrying
                        # the same no-op on every consolidation run.
                        deficit = fiber_count - maturation_count
                        report.extra["maturations_unreachable"] = deficit
                        _logger.warning(
                            "Maturation backfill created nothing while %d fibers "
                            "still lack a row; they are outside the backfill's "
                            "fiber window",
                            deficit,
                        )
            except Exception:
                # A backend without maturation, or a backfill that failed, must
                # not take the whole maturation phase down with it.
                _logger.debug("Maturation backfill skipped", exc_info=True)

        # Get all maturation records
        all_maturations = await self._storage.find_maturations()

        # Phase 1: Advance stages
        for record in all_maturations:
            advanced = compute_stage_transition(record, now=reference_time)
            if advanced.stage != record.stage:
                report.stages_advanced += 1
                hop_key = _hop_keys.get((record.stage, advanced.stage))
                if hop_key:
                    transitions = report.extra.setdefault("stage_transitions", {})
                    transitions[hop_key] = transitions.get(hop_key, 0) + 1
                if not dry_run:
                    try:
                        await self._storage.save_maturation(advanced)
                    except Exception as exc:
                        if "FOREIGN KEY" in str(exc):
                            _logger.warning(
                                "Skipping orphaned maturation for fiber %s",
                                record.fiber_id,
                            )
                            continue
                        raise

        # Phase 2: Extract patterns from mature episodic fibers
        if dry_run:
            return

        # Re-fetch after stage updates
        maturations = await self._storage.find_maturations()
        maturation_map = {m.fiber_id: m for m in maturations}

        fibers = await self._storage.get_fibers(limit=10000)
        patterns, extraction_report = extract_patterns(
            fibers=fibers,
            maturations=maturation_map,
            min_cluster_size=self._config.summarize_min_cluster_size,
            tag_overlap_threshold=self._config.summarize_tag_overlap_threshold,
        )

        report.patterns_extracted = extraction_report.patterns_extracted

        for pattern in patterns:
            await self._storage.add_neuron(pattern.concept_neuron)
            for synapse in pattern.synapses:
                await self._storage.add_synapse(synapse)

        # Phase 3: Generate essence for fibers that have content but no essence
        await self._essence_backfill(report, dry_run)

    async def _essence_backfill(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Generate essence for fibers missing it, or upgrade extractive → LLM.

        Uses configured essence_generator strategy from BrainConfig:
        - "extractive" (default): sentence-level scoring, fast and free
        - "llm": LLM abstractive with cost guard (priority < 3 skipped)

        Paginates in batches of 500 to avoid the storage cap.
        """
        from surreal_memory.engine.fidelity import get_essence_generator

        # Resolve generator strategy from brain config
        strategy = "extractive"
        try:
            brain_id = self._storage._get_brain_id()
            brain = await self._storage.get_brain(brain_id)
            if brain and brain.config:
                strategy = getattr(brain.config, "essence_generator", "extractive")
        except Exception:
            logger.debug("Could not read brain config for essence strategy", exc_info=True)

        generator = get_essence_generator(strategy)

        max_backfill = 2000  # Safety cap to avoid runaway

        # Fetch fibers in one batch (get_fibers has no offset param; limit=1000 is storage cap)
        fibers = await self._storage.get_fibers(limit=1000)
        candidates = [f for f in fibers if not f.essence]

        backfilled = 0
        for idx, fiber in enumerate(candidates):
            if backfilled >= max_backfill:
                break
            if idx % 50 == 0 and idx > 0:
                await asyncio.sleep(0)  # Yield to event loop

            anchor = await self._storage.get_neuron(fiber.anchor_neuron_id)
            if not anchor or not anchor.content:
                continue

            # Get priority from typed memory for cost guard
            priority = 5
            try:
                typed_mem = await self._storage.get_typed_memory(fiber.id)
                if (
                    typed_mem
                    and hasattr(typed_mem, "priority")
                    and isinstance(typed_mem.priority, (int, float))
                ):
                    priority = int(typed_mem.priority)
            except Exception:
                pass

            essence = await generator.generate(anchor.content, priority=priority)
            if not essence:
                continue

            if dry_run:
                backfilled += 1
                continue

            updated = fiber.with_essence(essence)
            await self._storage.update_fiber(updated)
            backfilled += 1

        if backfilled > 0:
            logger.info("Essence backfill: %d fibers updated", backfilled)
        report.essences_generated += backfilled

    async def _infer(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Run associative inference from co-activation data.

        1. Query co-activation counts within the time window
        2. Identify new + reinforcement candidates
        3. Create CO_OCCURS synapses for new candidates
        4. Reinforce existing synapses for reinforce candidates
        5. Generate + apply associative tags
        6. Prune old co-activation events
        """
        import logging

        from surreal_memory.engine.associative_inference import (
            InferenceConfig,
            create_inferred_synapse,
            generate_associative_tags,
            identify_candidates,
        )
        from surreal_memory.utils.tag_normalizer import TagNormalizer

        logger = logging.getLogger(__name__)

        config = InferenceConfig(
            co_activation_threshold=self._config.infer_co_activation_threshold,
            co_activation_window_days=self._config.infer_window_days,
            max_inferences_per_run=self._config.infer_max_per_run,
        )

        # 1. Query co-activation counts within time window
        from datetime import timedelta

        window_start = reference_time - timedelta(days=config.co_activation_window_days)
        counts = await self._storage.get_co_activation_counts(
            since=window_start,
            min_count=config.co_activation_threshold,
        )

        if not counts:
            return

        # 2. Build existing synapse pairs set + lookup for reinforcement
        # Need all types: existing_pairs prevents duplicate creation, synapse_by_pair enables reinforcement
        all_synapses = await self._storage.get_synapses()
        existing_pairs: set[tuple[str, str]] = set()
        synapse_by_pair: dict[tuple[str, str], Synapse] = {}
        for syn in all_synapses:
            existing_pairs.add((syn.source_id, syn.target_id))
            existing_pairs.add((syn.target_id, syn.source_id))
            synapse_by_pair[(syn.source_id, syn.target_id)] = syn

        new_candidates, reinforce_candidates = identify_candidates(counts, existing_pairs, config)

        if dry_run:
            report.synapses_inferred = len(new_candidates) + len(reinforce_candidates)
            return

        # 3. Create CO_OCCURS synapses for new candidates
        for candidate in new_candidates:
            synapse = create_inferred_synapse(candidate)
            try:
                await self._storage.add_synapse(synapse)
                report.synapses_inferred += 1
            except ValueError:
                logger.debug("Inferred synapse already exists, skipping")

        # 4. Reinforce existing synapses for reinforce candidates
        #    Use cached synapse_by_pair lookup instead of N+1 queries
        for candidate in reinforce_candidates:
            a, b = candidate.neuron_a, candidate.neuron_b
            existing_synapse = synapse_by_pair.get((a, b)) or synapse_by_pair.get((b, a))

            if existing_synapse:
                reinforced = existing_synapse.reinforce(delta=0.05)
                try:
                    await self._storage.update_synapse(reinforced)
                    report.synapses_inferred += 1
                except ValueError:
                    logger.debug("Synapse reinforcement failed")

        # 5. Generate and apply associative tags
        all_candidates = new_candidates + reinforce_candidates
        if all_candidates:
            neuron_ids = set()
            for c in all_candidates:
                neuron_ids.add(c.neuron_a)
                neuron_ids.add(c.neuron_b)

            neurons = await self._storage.get_neurons_batch(list(neuron_ids))
            content_map = {nid: n.content for nid, n in neurons.items()}

            fibers = await self._storage.get_fibers(limit=10000)
            existing_tags: set[str] = set()
            for f in fibers:
                existing_tags |= f.tags

            assoc_tags = generate_associative_tags(all_candidates, content_map, existing_tags)

            normalizer = TagNormalizer()

            # Build inverted index: neuron_id -> fiber indices
            neuron_to_fiber_idx: dict[str, set[int]] = {}
            for idx, fiber in enumerate(fibers):
                for nid in fiber.neuron_ids:
                    neuron_to_fiber_idx.setdefault(nid, set()).add(idx)

            # Accumulate all new tags per fiber, then write once
            fiber_new_tags: dict[int, set[str]] = {}
            for atag in assoc_tags:
                normalized_tag = normalizer.normalize(atag.tag)
                # Find affected fibers via inverted index
                affected: set[int] = set()
                for nid in atag.source_neuron_ids:
                    if nid in neuron_to_fiber_idx:
                        affected |= neuron_to_fiber_idx[nid]
                for idx in affected:
                    fiber_new_tags.setdefault(idx, set()).add(normalized_tag)

            # Write accumulated tags in a single pass
            for idx, new_tags in fiber_new_tags.items():
                fiber = fibers[idx]
                updated_auto_tags = fiber.auto_tags | new_tags
                if updated_auto_tags != fiber.auto_tags:
                    updated_fiber = dc_replace(fiber, auto_tags=updated_auto_tags)
                    try:
                        await self._storage.update_fiber(updated_fiber)
                    except Exception:
                        logger.debug("Associative tag update failed", exc_info=True)

            # Log drift detection
            drift_reports = normalizer.detect_drift(existing_tags)
            for dr in drift_reports:
                logger.info("Tag drift detected: %s → %s", dr.variants, dr.canonical)

        # 6. Prune old co-activation events
        pruned = await self._storage.prune_co_activations(older_than=window_start)
        report.co_activations_pruned = pruned

    async def _enrich(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run enrichment: transitive closure + cross-cluster linking."""
        import logging

        from surreal_memory.engine.enrichment import enrich

        logger = logging.getLogger(__name__)

        result = await enrich(self._storage)

        all_synapses = result.transitive_synapses + result.cross_cluster_synapses
        if dry_run:
            report.synapses_enriched = len(all_synapses)
            return

        for synapse in all_synapses:
            try:
                await self._storage.add_synapse(synapse)
                report.synapses_enriched += 1
            except ValueError:
                logger.debug("Enriched synapse already exists, skipping")

        # Reactivate dormant neurons (access_frequency=0) to prevent permanent dormancy
        await self._reactivate_dormant(report, dry_run)

    async def _reactivate_dormant(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Bump dormant neurons with minimal activation to simulate memory replay."""
        from dataclasses import replace as dc_replace

        try:
            # Filter and sample in storage: the dormant set is most of a mature
            # brain, so pulling every state here just to keep 20 of them made a
            # dream cycle scan the whole neuron_state table.
            sample = await self._storage.get_dormant_neuron_states(limit=_DORMANT_REPLAY_SAMPLE)
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to get neuron states for dream cycle", exc_info=True
            )
            return

        if not sample:
            return

        if dry_run:
            report.neurons_reactivated = len(sample)
            return

        now = utcnow()
        for state in sample:
            reactivated = dc_replace(
                state,
                activation_level=min(state.activation_level + 0.05, 1.0),
                access_frequency=1,
                last_activated=now,
            )
            await self._storage.update_neuron_state(reactivated)
            report.neurons_reactivated += 1

    async def _dream(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run dream exploration for hidden connections."""
        import logging

        from surreal_memory.engine.dream import dream

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await dream(self._storage, brain.config)

        if dry_run:
            report.dream_synapses_created = len(result.synapses_created)
            return

        for synapse in result.synapses_created:
            try:
                await self._storage.add_synapse(synapse)
                report.dream_synapses_created += 1
            except ValueError:
                logger.debug("Dream synapse already exists, skipping")

    async def _replay(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run hippocampal replay — LTP/LTD on recent fibers."""
        from surreal_memory.engine.hippocampal_replay import hippocampal_replay

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await hippocampal_replay(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["replay_episodes"] = result.episodes_replayed
        report.extra["replay_ltp"] = result.synapses_strengthened
        report.extra["replay_ltd"] = result.synapses_weakened

    async def _schema(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run schema assimilation — create/update schemas from tag clusters."""
        from surreal_memory.engine.schema_assimilation import batch_schema_assimilation

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        schemas_created = await batch_schema_assimilation(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["schemas_created"] = schemas_created

    async def _interference(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run interference scan — detect fan effects across tag clusters."""
        from surreal_memory.engine.interference import batch_interference_scan

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await batch_interference_scan(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["interference_fan_effects"] = result.fan_effects_flagged

    async def _learn_habits(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Learn habits from action event sequences."""
        import logging

        from surreal_memory.engine.sequence_mining import learn_habits

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        if dry_run:
            return

        try:
            learned, habit_report = await learn_habits(self._storage, brain.config, reference_time)
            report.habits_learned = habit_report.habits_learned
            report.action_events_pruned = habit_report.action_events_pruned
        except Exception:
            logger.debug("Habit learning failed (non-critical)", exc_info=True)

        # Also learn query topic patterns (same substrate, different metadata)
        try:
            from surreal_memory.engine.query_pattern_mining import learn_query_patterns

            qp_report = await learn_query_patterns(self._storage, brain.config, reference_time)
            # Query patterns are CONCEPT-neuron/synapse strengthenings, NOT listable
            # `_habit_pattern` workflow fibers. Counting them under habits_learned made
            # `consolidate` report "Habits learned: N" while `smem habits list` (which
            # lists habit fibers) stayed empty. Keep them as a distinct metric.
            report.query_patterns_learned += qp_report.patterns_learned
        except Exception:
            logger.debug("Query pattern learning failed (non-critical)", exc_info=True)

        # Also learn tool-usage habits from the tool_events buffer (Read → Edit →
        # Bash, …). Unlike query patterns these ARE listable `_habit_pattern`
        # fibers, so they count toward habits_learned.
        try:
            from surreal_memory.engine.sequence_mining import learn_tool_habits

            _, tool_report = await learn_tool_habits(self._storage, brain.config, reference_time)
            report.habits_learned += tool_report.habits_learned
        except Exception:
            logger.debug("Tool-usage habit learning failed (non-critical)", exc_info=True)

    async def _dedup(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Link near-duplicate anchor neurons via ALIAS edges.

        Scans anchor neurons and finds near-duplicates by SimHash Hamming
        distance, then records each pair with an ALIAS synapse pointing at the
        canonical anchor.

        Nothing is merged and no fiber is redirected -- the previous wording
        claimed both. ``duplicates_found`` is therefore a **census** of pairs
        that look alike, which is why it stays high on a steady-state brain;
        ``new_alias_links`` is the work actually performed this run.
        """
        import logging

        from surreal_memory.engine.dedup.alias_edges import (
            AliasEdgeLedger,
            AliasLinkOutcome,
            ensure_alias_edge,
        )
        from surreal_memory.utils.simhash import is_near_duplicate

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return

        # Paginate through all neurons to collect anchors (avoid OOM)
        batch_size = 5000
        offset = 0
        anchors: list[Neuron] = []
        while True:
            # Anchors are selected on metadata alone, so skip the embedding vector
            # — it is ~4-8 KB/row and only inflates the response.
            batch = await self._storage.find_neurons(
                limit=batch_size, offset=offset, ephemeral=False, include_embedding=False
            )
            if not batch:
                break
            anchors.extend(n for n in batch if n.metadata.get("is_anchor", False))
            offset += len(batch)
            if len(batch) < batch_size:
                break

        # Report the census *input*, always. Without it the pair count reads as
        # "duplicates in this brain" when past the cap it only ever means
        # "duplicates among the first _DEDUP_MAX_ANCHORS anchors, in scan order".
        anchors_total = len(anchors)
        report.extra["dedup_anchors_total"] = anchors_total

        if anchors_total < 2:
            report.extra["dedup_anchors_scanned"] = anchors_total
            return

        # Cap anchors to prevent O(N^2) blowup (N=2000 → 2M comparisons)
        if anchors_total > _DEDUP_MAX_ANCHORS:
            anchors = anchors[:_DEDUP_MAX_ANCHORS]
            report.extra["dedup_anchors_truncated"] = True
            # INFO, not WARNING: on a brain that has simply outgrown the cap this
            # is a steady state, and every run would raise the same alarm until
            # operators learned to ignore dedup warnings — burying the real
            # failures below. The report field and the summary line carry it.
            logger.info(
                "Dedup census truncated: %d anchors present, only the first %d are compared. "
                "The reported duplicate count covers that prefix, not the whole brain.",
                anchors_total,
                _DEDUP_MAX_ANCHORS,
            )
        report.extra["dedup_anchors_scanned"] = len(anchors)

        # This pass re-derives the *same* duplicate pairs on every run, so without
        # a memory of what already exists it re-inserts its whole alias edge set
        # each time. Preload the alias slice once rather than probing per pair —
        # 2000 anchors would otherwise turn one write storm into a read storm.
        ledger: AliasEdgeLedger | None = None
        if not dry_run:
            try:
                ledger = await AliasEdgeLedger.load(self._storage)
            except Exception:
                # Per-pair checks are slower but still correct. Treating a failed
                # preload as "nothing exists" is what re-opens the growth bug, so
                # never substitute an empty ledger here.
                report.extra["alias_ledger_load_failed"] = True
                logger.warning(
                    "Alias edge preload failed; falling back to per-pair checks", exc_info=True
                )
            else:
                report.extra["alias_ledger_pairs"] = len(ledger)
                report.extra["alias_ledger_complete"] = ledger.is_complete
                if not ledger.is_complete:
                    # Also INFO: a brain whose alias slice exceeds the scan limit
                    # falls back to per-pair checks by design. Correct, just slower.
                    logger.info(
                        "Alias ledger is partial (%d pairs loaded); unknown pairs fall back to "
                        "per-pair existence checks",
                        len(ledger),
                    )

        # Group duplicates by SimHash proximity
        seen: set[str] = set()
        created_links = 0
        existing_links = 0
        checks_failed = 0
        writes_failed = 0
        skipped_invalid = 0
        first_failure: BaseException | None = None
        for i, anchor_a in enumerate(anchors):
            if anchor_a.id in seen:
                continue
            # Yield to event loop every 50 outer iterations so timeout can fire
            if i % 50 == 0:
                await asyncio.sleep(0)
            if anchor_a.content_hash is None or anchor_a.content_hash == 0:
                continue

            for anchor_b in anchors[i + 1 :]:
                if anchor_b.id in seen:
                    continue
                if anchor_b.content_hash is None or anchor_b.content_hash == 0:
                    continue

                if is_near_duplicate(anchor_a.content_hash, anchor_b.content_hash):
                    report.duplicates_found += 1
                    seen.add(anchor_b.id)

                    if dry_run:
                        continue

                    # ALIAS synapse from newer to older (canonical). The edge id
                    # is derived from the pair, so a second run re-writes the
                    # same row instead of adding another one for the same fact.
                    # The outcome separates the census from the work actually
                    # done — and, just as importantly, from the work that could
                    # not be done because the backend refused to answer.
                    result = await ensure_alias_edge(
                        self._storage,
                        anchor_b.id,
                        anchor_a.id,
                        ledger=ledger,
                    )
                    if result.outcome is AliasLinkOutcome.CREATED:
                        created_links += 1
                    elif result.outcome in (
                        AliasLinkOutcome.ALREADY_EXISTS,
                        AliasLinkOutcome.EXISTS_RACE,
                    ):
                        # A lost write race means the edge is there, which is the
                        # goal. Counting it as a failure would invent an incident.
                        existing_links += 1
                    elif result.outcome is AliasLinkOutcome.CHECK_FAILED:
                        checks_failed += 1
                        first_failure = first_failure or result.error
                    elif result.outcome is AliasLinkOutcome.WRITE_FAILED:
                        writes_failed += 1
                        first_failure = first_failure or result.error
                    else:
                        skipped_invalid += 1

        if dry_run:
            return

        report.new_alias_links += created_links
        report.alias_links_existing += existing_links
        # Failure keys appear only when something failed, so a healthy report
        # stays free of zero-valued noise (same rule as semantic_link_failures).
        for key, value in (
            ("alias_checks_failed", checks_failed),
            ("alias_writes_failed", writes_failed),
            ("alias_pairs_skipped_invalid", skipped_invalid),
        ):
            if value:
                report.extra[key] = int(report.extra.get(key, 0)) + value

        if checks_failed or writes_failed:
            # One WARNING for the whole pass, with one traceback: an outage
            # produces the same root cause once per pair, and printing it that
            # many times hides rather than reveals it.
            logger.warning(
                "Dedup alias linking degraded: %d existence checks failed (state unknown, "
                "writes skipped) and %d writes failed out of %d duplicate pairs. "
                "First failure below.",
                checks_failed,
                writes_failed,
                report.duplicates_found,
                exc_info=first_failure,
            )

    async def _semantic_link(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Discover and create SIMILAR_TO synapses via embedding similarity.

        Optional — silently skips if embeddings are not available.
        Created synapses decay 2x faster during pruning unless reinforced.
        """
        import logging

        from surreal_memory.engine.semantic_discovery import discover_semantic_synapses

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        result = await discover_semantic_synapses(self._storage, brain.config)

        # Discovery already refused every pair that carried a synapse of any
        # type; surfacing that number is what turns a flat "2000" into an
        # honest "N created, K skipped".
        report.semantic_synapses_skipped += result.skipped_existing
        if result.truncated:
            report.extra["semantic_link_truncated"] = True

        if dry_run:
            report.semantic_synapses_created = result.synapses_created
            return

        for synapse in result.synapses:
            try:
                await self._storage.add_synapse(synapse)
                report.semantic_synapses_created += 1
            except ValueError:
                # SQLite raises this when an endpoint vanished between
                # discovery and insert. The pair is not linkable, not linked.
                report.semantic_synapses_skipped += 1
                logger.debug("Semantic synapse endpoint missing, skipping")
            except Exception as exc:
                # Only a primary-key collision means "the edge already exists":
                # the id is derived from the sorted pair, so a writer that got
                # there first wins and losing that race is correct. Anything
                # else -- a dropped connection, a malformed row -- is a real
                # failure, and folding it into "skipped (existing)" would be
                # exactly the dishonest counter this release exists to remove.
                if is_duplicate_key_error(exc):
                    report.semantic_synapses_skipped += 1
                    logger.debug("Semantic synapse already exists, skipping")
                else:
                    report.extra["semantic_link_failures"] = (
                        int(report.extra.get("semantic_link_failures", 0)) + 1
                    )
                    logger.warning("Semantic synapse write failed (not a duplicate)", exc_info=True)

    async def _detect_drift(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Recompute semantic drift clusters from accumulated tag co-occurrence.

        Native SurrealDB port of the SQLite-only feature removed in 3524066d.
        A pure read-detect-persist step: it never touches neurons, synapses or
        fibers, only the tag_cooccurrence/drift_clusters tables.

        A dry run still detects — it reports how many clusters it WOULD have
        saved and skips only the writes, matching ``_dedup``'s census-always /
        write-never-on-dry-run shape. Returning 0 without looking would make a
        preview of a clean brain indistinguishable from a preview of a drifting
        one, which is the exact ambiguity this feature exists to remove.
        """
        from surreal_memory.engine.drift_clusters import refresh_drift_clusters

        try:
            report.drift_clusters_found = await refresh_drift_clusters(
                self._storage, persist=not dry_run
            )
        except Exception:
            logger.warning("Drift cluster detection failed", exc_info=True)

    async def _compress(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Run tiered memory compression on all eligible fibers.

        Creates a CompressionEngine with default config and runs it for the
        current brain context.  Results are merged into *report*.
        """
        import logging as _logging

        from surreal_memory.engine.compression import CompressionEngine

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("COMPRESS skipped: no brain context")
            return

        engine = CompressionEngine(self._storage)
        # Leave headroom under the per-strategy timeout for the initial
        # get_fibers() scan and report assembly — compressing right up to the
        # wire risks the outer asyncio.wait_for cancelling mid-fiber instead of
        # returning a clean, resumable report.
        time_budget = self._config.strategy_timeout_seconds * 0.8
        compression_report = await engine.run(
            reference_time=reference_time,
            dry_run=dry_run,
            time_budget_seconds=time_budget,
        )

        report.fibers_compressed += compression_report.fibers_compressed
        report.tokens_saved += compression_report.tokens_saved
        if compression_report.fibers_deferred:
            report.extra["compress_fibers_deferred"] = compression_report.fibers_deferred

    async def _lifecycle(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Calculate heat scores and update lifecycle_state for all neurons.

        Fetches all neurons in the current brain, computes heat scores from
        access frequency and recency, then updates the lifecycle_state column.
        Each state maps to a compression tier range:
          ACTIVE → < 7d or hot (heat > threshold)
          WARM   → 7-30d or accessed in last 14d
          COOL   → 30-90d
          COMPRESSED → 90-180d
          ARCHIVED → 180d+

        Args:
            report: ConsolidationReport to update.
            reference_time: UTC reference time for age/recency calculations.
            dry_run: If True, calculate but do not apply changes.
        """
        import logging as _logging

        from surreal_memory.engine.compression import (
            CompressionConfig,
            calculate_heat_score,
            determine_lifecycle_state,
        )

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("LIFECYCLE skipped: no brain context")
            return

        # Page through the neurons instead of pulling 10k rows in one response, and
        # drop the embedding vector: this pass only reads neuron.metadata
        # (last_accessed_at / access_frequency / priority). A single
        # `limit=10000` with embeddings meant a multi-hundred-MB HTTP body that
        # SurrealDB dropped mid-transfer — "[Errno 104] Connection reset by peer",
        # which aborted the whole LIFECYCLE pass.
        neurons: list[Neuron] = []
        batch_size = 500
        offset = 0
        try:
            while len(neurons) < 10000:
                batch = await self._storage.find_neurons(
                    limit=batch_size,
                    offset=offset,
                    ephemeral=False,
                    include_embedding=False,
                )
                if not batch:
                    break
                neurons.extend(batch)
                offset += len(batch)
                if len(batch) < batch_size:
                    break
        except Exception:
            _logger.error("LIFECYCLE failed to fetch neurons", exc_info=True)
            return

        config = CompressionConfig()
        states_updated = 0

        for neuron in neurons:
            # Retrieve last_accessed_at and access_frequency from neuron metadata
            # (access_frequency is stored in neuron_states, not neurons directly)
            last_accessed_raw: str | None = neuron.metadata.get("last_accessed_at")
            last_accessed_at: datetime | None = None
            if last_accessed_raw:
                try:
                    last_accessed_at = datetime.fromisoformat(last_accessed_raw)
                except ValueError:
                    pass

            access_count: int = int(neuron.metadata.get("access_frequency", 0))
            priority: int = int(neuron.metadata.get("priority", 5))

            heat = calculate_heat_score(
                last_accessed_at=last_accessed_at,
                access_count=access_count,
                priority=priority,
                reference_time=reference_time,
                config=config,
            )

            age_days = (reference_time - neuron.created_at).total_seconds() / 86400.0
            new_state = determine_lifecycle_state(age_days, heat, config)

            current_state: str = neuron.metadata.get("lifecycle_state", "active")
            if current_state == str(new_state):
                continue

            if not dry_run:
                try:
                    await self._storage.update_neuron_lifecycle(neuron.id, str(new_state))
                    states_updated += 1
                except Exception:
                    _logger.error(
                        "Failed to update lifecycle_state for neuron %s", neuron.id, exc_info=True
                    )
            else:
                states_updated += 1

        if states_updated:
            report.extra["lifecycle_states_updated"] = (
                report.extra.get("lifecycle_states_updated", 0) + states_updated
            )
        _logger.info("LIFECYCLE: updated %d neuron lifecycle states", states_updated)

    async def _process_tool_events(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Process buffered tool events into neurons and synapses.

        Reads the JSONL buffer, ingests into tool_events table, then runs
        pattern detection. Only executes if tool_memory.enabled in config.
        """
        import logging as _logging

        from surreal_memory.unified_config import UnifiedConfig

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("PROCESS_TOOL_EVENTS skipped: no brain context")
            return

        try:
            config = UnifiedConfig.load()
        except Exception:
            _logger.debug("PROCESS_TOOL_EVENTS skipped: config load failed", exc_info=True)
            return

        if not config.tool_memory.enabled:
            return

        if dry_run:
            _logger.debug("PROCESS_TOOL_EVENTS skipped: dry_run mode")
            return

        from surreal_memory.engine.tool_memory import ingest_buffer, process_events

        # Ingest JSONL buffer
        buffer_path = config.data_dir / "tool_events.jsonl"
        ingest_result = await ingest_buffer(
            self._storage,  # type: ignore[arg-type]
            brain_id,
            buffer_path,
            config.tool_memory.max_buffer_lines,
        )
        if ingest_result.events_ingested > 0:
            _logger.debug(
                "PROCESS_TOOL_EVENTS: ingested %d events from buffer",
                ingest_result.events_ingested,
            )

        # Process events into neurons/synapses
        result = await process_events(self._storage, brain_id, config.tool_memory)  # type: ignore[arg-type]
        if result.events_processed > 0:
            _logger.debug(
                "PROCESS_TOOL_EVENTS: processed %d events, created %d neurons, %d synapses",
                result.events_processed,
                result.neurons_created,
                result.synapses_created,
            )

    async def _process_reasoning_traces(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Mine reasoning traces from transcripts into the staging table.

        Scans ~/.claude transcripts for model thinking blocks and stages them
        (redacted) via reasoning_miner. Only runs when
        reasoning_training.mining_enabled is set (opt-in for privacy). Mirrors
        PROCESS_TOOL_EVENTS and is likewise excluded from scheduled defaults.
        """
        import logging as _logging

        from surreal_memory.unified_config import UnifiedConfig

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("PROCESS_REASONING_TRACES skipped: no brain context")
            return

        try:
            config = UnifiedConfig.load()
        except Exception:
            _logger.debug("PROCESS_REASONING_TRACES skipped: config load failed", exc_info=True)
            return

        if not config.reasoning_training.mining_enabled:
            return

        if dry_run:
            _logger.debug("PROCESS_REASONING_TRACES skipped: dry_run mode")
            return

        from surreal_memory.engine.reasoning_miner import ingest_reasoning_traces

        try:
            result = await ingest_reasoning_traces(self._storage, brain_id, config)
        except Exception:
            _logger.debug("PROCESS_REASONING_TRACES: ingest failed (non-critical)", exc_info=True)
            return
        report.reasoning_traces_ingested = result.traces_ingested
        if result.traces_ingested > 0:
            _logger.debug(
                "PROCESS_REASONING_TRACES: ingested %d reasoning traces",
                result.traces_ingested,
            )

    async def _learn_reasoning(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Distill staged reasoning traces into ReasoningBank pattern fibers.

        Runs in the SUMMARIZE tier (after PROCESS_REASONING_TRACES ingest). Only
        runs when reasoning_training.mining_enabled is set; excluded from
        scheduled defaults (like PROCESS_REASONING_TRACES / LEARN_HABITS).
        """
        import logging as _logging

        from surreal_memory.unified_config import UnifiedConfig

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("LEARN_REASONING skipped: no brain context")
            return

        try:
            config = UnifiedConfig.load()
        except Exception:
            _logger.debug("LEARN_REASONING skipped: config load failed", exc_info=True)
            return

        if not config.reasoning_training.mining_enabled:
            return

        if dry_run:
            _logger.debug("LEARN_REASONING skipped: dry_run mode")
            return

        from surreal_memory.engine.reasoning_distiller import distill_reasoning_patterns

        try:
            result = await distill_reasoning_patterns(self._storage, brain_id, config)
        except Exception:
            _logger.debug("LEARN_REASONING: distillation failed (non-critical)", exc_info=True)
            return
        report.reasoning_patterns_learned = result.patterns_learned
        if result.patterns_learned > 0:
            _logger.debug(
                "LEARN_REASONING: learned %d reasoning patterns from %d traces",
                result.patterns_learned,
                result.traces_processed,
            )

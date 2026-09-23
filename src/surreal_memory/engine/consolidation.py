"""Memory consolidation engine — prune, merge, and summarize memories.

Provides automated memory maintenance:
- Prune: Remove dead synapses and orphan neurons
- Merge: Combine overlapping fibers
- Summarize: Create concept neurons for topic clusters
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from surreal_memory.core.constants import GRAPH_ONLY_PLACEHOLDER
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.clustering import UnionFind
from surreal_memory.engine.consolidation_progress import (
    ConsolidationLeaseLostError,
    ConsolidationPausedError,
    ConsolidationProgressError,
    ConsolidationProgressSession,
    options_fingerprint,
    supports_persistent_progress,
    supports_storage_methods,
)
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

# How many superseded change-log rows one consolidation pass may remove. A log
# that has never been pruned can hold millions; deleting them all in one pass
# would turn a routine consolidation into a multi-minute stall. The pass is
# idempotent, so a backlog drains over consecutive runs — and hitting this cap
# is reported, not smoothed over, so a draining backlog cannot be mistaken for a
# finished one.
_CHANGE_LOG_COLLAPSE_CAP = 200_000


def _encode_prune_synapse_cursor(synapse: Synapse) -> str:
    """Serialize the stable (created_at, id) prune cursor into progress state."""
    return json.dumps(
        {
            "version": 1,
            "created_at": synapse.created_at.isoformat(),
            "id": synapse.id,
        },
        separators=(",", ":"),
    )


def _decode_prune_synapse_cursor(cursor: str | None) -> tuple[str | None, datetime | None]:
    """Validate and decode the versioned prune cursor, failing closed on old shapes."""
    if cursor is None:
        return None, None
    try:
        payload = json.loads(cursor)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(payload.get("id"), str)
            or not isinstance(payload.get("created_at"), str)
        ):
            raise ValueError("unexpected cursor fields or version")
        created_at = datetime.fromisoformat(payload["created_at"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConsolidationProgressError(
            "Cannot resume prune: saved synapse cursor is incompatible with "
            "the current timestamp-keyset format; progress was preserved."
        ) from exc
    return payload["id"], created_at


def _summary_cluster_key_from_ids(fiber_ids: Iterable[str]) -> str:
    """Stable identity of a summary cluster: a hash of its sorted source fiber ids.

    Sorted so member order never changes the key, hashed so the value stays short
    enough to live in metadata and be compared cheaply.
    """
    joined = "|".join(sorted(fiber_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def _summary_cluster_key(cluster_fibers: Sequence[Fiber]) -> str:
    """Cluster key for the fibers a summary is about."""
    return _summary_cluster_key_from_ids(f.id for f in cluster_fibers)


_MERGED_SUMMARY_MAX_CHARS = 500


def _merged_summary(member_fibers: Sequence[Fiber]) -> str:
    """Summary for a merged fiber, built from its sources.

    The previous constant string ("Merged from N fibers") is what recall surfaces
    return for the merged fiber, so every merge used to erase the only human-readable
    trace of what the memory was about.
    """
    parts = [f.summary.strip() for f in member_fibers if f.summary and f.summary.strip()]
    if not parts:
        return f"Merged from {len(member_fibers)} fibers"
    joined = "; ".join(dict.fromkeys(parts))
    if len(joined) <= _MERGED_SUMMARY_MAX_CHARS:
        return joined
    return joined[: _MERGED_SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def _merged_metadata(member_fibers: Sequence[Fiber]) -> dict[str, Any]:
    """Union of the sources' metadata plus the merge provenance.

    Replacing metadata wholesale dropped essence, conductivity, coherence,
    compression_tier and _verbatim. Sources are applied in ascending salience so the
    most salient fiber wins a key collision.
    """
    merged: dict[str, Any] = {}
    for fiber in sorted(member_fibers, key=lambda f: f.salience):
        for key, value in fiber.metadata.items():
            if key == "merged_from":
                continue
            merged[key] = value
    merged["merged_from"] = [f.id for f in member_fibers]
    return merged


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
    # How many anchors one dedup census compares pairwise. Configurable because the
    # pass walks a rotating window: with a hardcoded value the only way to widen
    # coverage on a large brain was to edit the source.
    dedup_max_anchors: int = _DEDUP_MAX_ANCHORS
    # Hamming distance below which two SimHash fingerprints count as near-duplicates.
    # Mirrors DedupConfig.simhash_threshold — the census used to fall back to the
    # looser library default (10), so it counted pairs the project's own policy no
    # longer considers duplicates.
    dedup_simhash_threshold: int = 7
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
    decay_passes_pruned: int = 0
    change_log_collapsed: int = 0
    """Superseded pending change-log updates removed this run.

    The change log is the one table with no upper bound on growth: its only
    retention path required rows to have been synced, so on a brain whose sync
    never completed nothing was ever removed. Reporting the collapse here is
    what makes that growth visible at all -- the dashboard card that would have
    shown it was itself too slow to load while the table was large.
    """
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
    # Detected and persisted are separate: the counter used to return the SAVED count
    # while the report called it "found", so a failing write looked like a smaller brain.
    drift_clusters_persisted: int = 0
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
        seen_twice = int(self.extra.get("semantic_pairs_seen_twice", 0))
        if seen_twice:
            # These were created by THIS pass and reached a second time from the other
            # end of the neighbourhood. Folding them into "existing" overstated how much
            # of the graph was already linked.
            line += f", {seen_twice} reached twice this run"
        if self.extra.get("semantic_link_truncated"):
            line += " [truncated at cap]"
        total = self.extra.get("semantic_candidates_total")
        scanned = self.extra.get("semantic_candidates_scanned")
        if total and scanned:
            line += (
                f" [candidates resampled: {scanned} of {total} — counters are NOT "
                "comparable with the previous run]"
            )
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

    def _drift_cluster_line(self) -> str:
        """Drift line that distinguishes detection from persistence.

        The old wording printed the SAVED count under the word "found", so a run whose
        writes failed looked like a run that simply found less.
        """
        line = f"  Drift clusters found: {self.drift_clusters_found}"
        failed = self.drift_clusters_found - self.drift_clusters_persisted
        if failed > 0:
            line += f" ({self.drift_clusters_persisted} persisted, {failed} FAILED to persist)"
        return line

    def _semantic_link_failure_line(self) -> str | None:
        """Surface write failures that were previously recorded and never shown."""
        failures = int(self.extra.get("semantic_link_failures", 0))
        if failures <= 0:
            return None
        return f"  Semantic synapse writes FAILED: {failures}"

    def _merge_delete_failure_line(self) -> str | None:
        """Surface merge deletes that did not confirm."""
        failures = int(self.extra.get("merge_delete_failures", 0))
        if failures <= 0:
            return None
        return f"  Merge source deletes FAILED: {failures}"

    def _summaries_skipped_line(self) -> str | None:
        """Show idempotence at work: clusters whose summary already existed."""
        skipped = int(self.extra.get("summaries_skipped_existing", 0))
        total = self.extra.get("summarize_fibers_total")
        scanned = self.extra.get("summarize_fibers_scanned")
        parts = []
        if skipped > 0:
            parts.append(f"  Summaries skipped (already exist): {skipped}")
        if total and scanned:
            parts.append(
                f"  [summarize input truncated: {scanned} of {total} tagged fibers clustered]"
            )
        return "\n".join(parts) if parts else None

    def _silent_strategy_lines(self) -> list[str]:
        """Work done by strategies that never had a line of their own.

        replay, interference, schema and tiering all wrote their results into
        ``extra`` and nothing rendered them, so a run that reorganised the graph
        looked identical to a run that did nothing.
        """
        lines: list[str] = []
        ltp = int(self.extra.get("replay_ltp", 0))
        ltd = int(self.extra.get("replay_ltd", 0))
        episodes = int(self.extra.get("replay_episodes", 0))
        if episodes or ltp or ltd:
            lines.append(
                f"  Hippocampal replay: {episodes} episodes ({ltp} strengthened, {ltd} weakened)"
            )
        fan_effects = int(self.extra.get("interference_fan_effects", 0))
        if fan_effects:
            lines.append(f"  Interference fan effects: {fan_effects}")
        schemas = int(self.extra.get("schemas_created", 0))
        if schemas:
            lines.append(f"  Schemas created: {schemas}")
        auto_tier = self.extra.get("auto_tier")
        if auto_tier:
            lines.append(f"  Auto-tiering: {auto_tier}")
        ingested = int(self.extra.get("tool_events_ingested", 0))
        processed = int(self.extra.get("tool_events_processed", 0))
        if ingested or processed:
            lines.append(f"  Tool events: {ingested} ingested, {processed} processed")
        states = int(self.extra.get("lifecycle_states_updated", 0))
        if states:
            lines.append(f"  Lifecycle states updated: {states}")
        return lines

    def _compress_deferred_line(self) -> str | None:
        """Show work postponed by the compression budget instead of hiding it as zero."""
        deferred = int(self.extra.get("compress_fibers_deferred", 0))
        if deferred <= 0:
            return None
        return f"  Fibers deferred (time budget): {deferred}"

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
            line += self._dedup_window_note()
        return line

    def _change_log_collapse_line(self) -> str:
        """Render the change-log collapse, and say so when the pass was cut short.

        A bounded pass that hit its ceiling has NOT finished the job, and a
        backlog draining over several runs looks identical to a completed run
        unless the truncation is stated. Same reasoning as the dedup census
        line: a number alone cannot distinguish "done" from "gave up here".
        """
        line = f"{self.change_log_collapsed} superseded updates removed"
        remaining = int(self.extra.get("change_log_pending_after", 0) or 0)
        if self.extra.get("change_log_collapse_truncated"):
            line += f" [pass capped; {remaining} pending rows remain, next run continues]"
        return line

    def _dedup_window_note(self) -> str:
        """Describe the census WINDOW, not just the cap.

        "truncated at anchor cap: 2000 of 3859" reads exactly like the frozen-prefix
        behaviour this pass was rebuilt to end: it reports that work was cut and says
        nothing about the window advancing, so a rotating window and a permanent blind
        spot render identically. Naming the slice and how many runs make a full pass is
        what tells them apart at a glance.
        """
        total = int(self.extra.get("dedup_anchors_total") or 0)
        scanned = int(self.extra.get("dedup_anchors_scanned") or 0)
        start = int(self.extra.get("dedup_window_start") or 0)
        if not total or not scanned:
            return ""
        end = start + scanned
        runs_for_full_pass = -(-total // scanned)  # ceil division
        wrapped = ", wraps" if end > total else ""
        return (
            f" [census window {start}-{min(end, total)} of {total} anchors{wrapped};"
            f" rotates per run, full pass every {runs_for_full_pass} runs]"
        )

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
            f"  Decay telemetry pruned: {self.decay_passes_pruned}",
            f"  Change log collapsed: {self._change_log_collapse_line()}",
            f"  Duplicate anchors: {self._alias_link_line()}",
            f"  Semantic synapses: {self._semantic_synapse_line()}",
            f"  Memories promoted (type): {self.memories_promoted}",
            f"  Stages advanced: {self.stages_advanced}{self._stage_transitions_suffix()}",
            f"  Fibers compressed: {self.fibers_compressed}",
            f"  Tokens saved: {self.tokens_saved}",
            f"  Reasoning traces ingested: {self.reasoning_traces_ingested}",
            f"  Reasoning patterns learned: {self.reasoning_patterns_learned}",
            self._drift_cluster_line(),
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        if self.merge_details:
            lines.append("  Merge details:")
            for detail in self.merge_details:
                lines.append(
                    f"    {len(detail.original_fiber_ids)} fibers -> {detail.merged_fiber_id[:8]}... "
                    f"({detail.neuron_count} neurons, {detail.reason})"
                )

        # Counters that were recorded but never rendered: a run whose writes failed
        # used to be indistinguishable from a clean one. Each line appears only when
        # non-zero, so a healthy run stays quiet.
        for optional_line in (
            self._summaries_skipped_line(),
            self._merge_delete_failure_line(),
            self._semantic_link_failure_line(),
            self._compress_deferred_line(),
        ):
            if optional_line:
                lines.append(optional_line)
        lines.extend(self._silent_strategy_lines())
        dedup_resume = self.extra.get("dedup_resumed_checkpoint")
        if dedup_resume:
            lines.append(f"  Dedup resumed from saved checkpoint: {dedup_resume}")
        last_checkpoint = self.extra.get("last_checkpoint")
        if last_checkpoint:
            lines.append(f"  Last committed checkpoint: {last_checkpoint}")
        for message in self.extra.get("consolidation_progress_messages", []):
            lines.append(f"  {message}")
        status = self.extra.get("consolidation_status")
        if status in {"paused", "blocked", "failed", "lease_lost", "timed_out"}:
            lines.append(f"  Consolidation status: {status}")

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
        self._config = config or self._config_from_settings()
        self._dream_decay_multiplier = dream_decay_multiplier
        self._tier_config = tier_config
        self._progress_session: ConsolidationProgressSession | None = None
        self._active_strategy: ConsolidationStrategy | None = None
        self._strategy_deadline: float | None = None
        self._total_deadline: float | None = None

    def _strategy_progress_state(self) -> dict[str, Any]:
        """Return this strategy's persisted state, if a SurrealDB run is active."""
        if self._progress_session is None or self._active_strategy is None:
            return {}
        return self._progress_session.strategy_state(self._active_strategy.value)

    def _strategy_resume_cursor(self) -> str | None:
        """Return the most recently committed cursor for the active strategy."""
        cursor = self._strategy_progress_state().get("cursor")
        return str(cursor) if cursor is not None else None

    async def _check_progress_budget(self) -> None:
        """Stop before another work unit when the lease or time budget is gone."""
        progress_session = getattr(self, "_progress_session", None)
        if progress_session is None:
            return
        if getattr(progress_session, "lease_lost", False):
            raise ConsolidationLeaseLostError(
                f"lease for brain {progress_session.brain_id!r} was lost; "
                "stopping before another work unit"
            )
        deadlines = [
            deadline
            for deadline in (self._strategy_deadline, self._total_deadline)
            if deadline is not None
        ]
        if not deadlines:
            return
        remaining = min(deadlines) - time.perf_counter()
        headroom = min(10.0, max(0.05, self._config.strategy_timeout_seconds * 0.02))
        if remaining <= headroom:
            raise ConsolidationPausedError(
                f"budget is down to {max(0.0, remaining):.2f}s; the next unit was not started"
            )

    async def _checkpoint_progress(
        self,
        phase: str,
        *,
        cursor: str | None = None,
        pending: Sequence[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        """Persist one committed unit boundary for the active strategy."""
        if self._progress_session is None or self._active_strategy is None:
            return
        await self._progress_session.checkpoint(
            self._active_strategy.value,
            phase,
            cursor=cursor,
            pending=pending,
            counters=counters,
        )
        await self._check_progress_budget()

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
        on_progress: Callable[[str], None] | None = None,
    ) -> ConsolidationReport:
        """Run consolidation, resuming a compatible durable run when available."""
        if strategies is None:
            strategies = [ConsolidationStrategy.ALL]

        explicit_reference_time = reference_time
        reference_time = ensure_naive_utc(reference_time) if reference_time else utcnow()
        report = ConsolidationReport(started_at=reference_time, dry_run=dry_run)
        start = time.perf_counter()

        normalized: list[ConsolidationStrategy] = [
            strategy
            if isinstance(strategy, ConsolidationStrategy)
            else ConsolidationStrategy(strategy)
            for strategy in strategies
        ]
        run_all = ConsolidationStrategy.ALL in normalized
        requested: set[ConsolidationStrategy] = (
            {
                strategy
                for strategy in ConsolidationStrategy
                if strategy != ConsolidationStrategy.ALL
            }
            if run_all
            else set(normalized)
        )
        requested_names = sorted(strategy.value for strategy in requested)
        ordered_strategies = [
            strategy
            for tier in self.STRATEGY_TIERS
            for strategy in sorted(tier & requested, key=lambda item: item.value)
        ]

        strategy_timeout = self._config.strategy_timeout_seconds
        total_timeout = self._config.total_timeout_seconds
        timed_out_strategies: list[str] = []
        failed_strategies: list[str] = []
        progress_messages: list[str] = []
        resume_message: str | None = None

        progress_session: ConsolidationProgressSession | None = None
        if not dry_run and supports_persistent_progress(self._storage):
            try:
                tier_config = self._tier_config
                if tier_config is None:
                    tier_config_value = None
                elif hasattr(tier_config, "model_dump"):
                    tier_config_value = tier_config.model_dump()
                elif hasattr(tier_config, "dict"):
                    tier_config_value = tier_config.dict()
                else:
                    tier_config_value = repr(tier_config)
                fingerprint = options_fingerprint(
                    {
                        "requested_strategies": requested_names,
                        "config": asdict(self._config),
                        "dream_decay_multiplier": self._dream_decay_multiplier,
                        "tier_config": tier_config_value,
                    }
                )
                progress_session = await ConsolidationProgressSession.open(
                    self._storage,
                    requested_names,
                    fingerprint,
                    reference_time,
                    explicit_reference_time=explicit_reference_time,
                )
            except ConsolidationProgressError as exc:
                report.extra["consolidation_status"] = "blocked"
                report.extra["failed_strategies"] = [
                    f"progress coordination ({type(exc).__name__})"
                ]
                report.extra["consolidation_progress_messages"] = [
                    f"Consolidation stopped safely: {exc}"
                ]
                report.duration_ms = (time.perf_counter() - start) * 1000
                return report
            except Exception as exc:
                logger.exception("Consolidation could not initialize durable progress")
                report.extra["consolidation_status"] = "blocked"
                report.extra["failed_strategies"] = [
                    f"progress initialization ({type(exc).__name__})"
                ]
                report.extra["consolidation_progress_messages"] = [
                    f"Consolidation stopped safely because durable progress is unavailable: {exc}"
                ]
                report.duration_ms = (time.perf_counter() - start) * 1000
                return report

        if progress_session is not None:
            self._progress_session = progress_session
            if progress_session.resumed:
                reference_time = progress_session.reference_time
                report.started_at = reference_time
                completed = progress_session.completed_strategies
                resume_strategy = next(
                    (
                        strategy.value
                        for strategy in ordered_strategies
                        if strategy.value not in completed
                    ),
                    "unknown",
                )
                resume_message = (
                    f"Consolidation: {resume_strategy} progress state found - resuming..."
                )
                progress_messages.append(resume_message)

        self._total_deadline = time.perf_counter() + total_timeout
        paused = False
        lease_lost = False

        try:
            if resume_message is not None and on_progress is not None:
                on_progress(resume_message)
            for tier in self.STRATEGY_TIERS:
                tier_strategies = sorted(tier & requested, key=lambda item: item.value)
                for strategy in tier_strategies:
                    if (
                        progress_session is not None
                        and strategy.value in progress_session.completed_strategies
                    ):
                        continue

                    elapsed = time.perf_counter() - start
                    remaining_total = total_timeout - elapsed
                    if remaining_total <= 0:
                        timed_out_strategies.extend(
                            item.value
                            for item in ordered_strategies
                            if progress_session is None
                            or item.value not in progress_session.completed_strategies
                        )
                        logger.warning(
                            "Consolidation total timeout (%.0fs) reached after %.1fs",
                            total_timeout,
                            elapsed,
                        )
                        paused = progress_session is not None
                        break

                    self._active_strategy = strategy
                    strategy_start = time.perf_counter()
                    strategy_budget = min(strategy_timeout, remaining_total)
                    self._strategy_deadline = strategy_start + strategy_budget

                    logger.info("Consolidation: starting %s", strategy.value)
                    try:
                        if progress_session is not None:
                            state = progress_session.strategy_state(strategy.value)
                            if not state:
                                await progress_session.checkpoint(strategy.value, "starting")
                        await self._check_progress_budget()
                        await asyncio.wait_for(
                            self._run_strategy(strategy, report, reference_time, dry_run),
                            timeout=strategy_budget,
                        )
                        if progress_session is not None:
                            await progress_session.complete_strategy(strategy.value)
                    except ConsolidationPausedError as exc:
                        timed_out_strategies.append(strategy.value)
                        if progress_session is not None:
                            try:
                                await progress_session.pause()
                                progress_messages.append(
                                    "Consolidation paused: "
                                    f"{progress_session.last_checkpoint}; {exc}. "
                                    "The next smem consolidate will resume this checkpoint."
                                )
                                report.extra["last_checkpoint"] = progress_session.last_checkpoint
                            except ConsolidationProgressError as lease_exc:
                                lease_lost = True
                                progress_messages.append(
                                    f"Consolidation stopped after losing its lease: {lease_exc}"
                                )
                        paused = True
                        break
                    except TimeoutError:
                        strategy_elapsed = time.perf_counter() - strategy_start
                        logger.warning(
                            "Consolidation: %s reached its %.0fs limit after %.1fs",
                            strategy.value,
                            strategy_timeout,
                            strategy_elapsed,
                        )
                        timed_out_strategies.append(strategy.value)
                        if progress_session is not None:
                            try:
                                await progress_session.pause()
                                progress_messages.append(
                                    "Consolidation paused: "
                                    f"{progress_session.last_checkpoint}; the strategy limit "
                                    "was reached. The next smem consolidate will resume "
                                    "the last committed checkpoint."
                                )
                                report.extra["last_checkpoint"] = progress_session.last_checkpoint
                            except ConsolidationProgressError as lease_exc:
                                lease_lost = True
                                progress_messages.append(
                                    f"Consolidation stopped after losing its lease: {lease_exc}"
                                )
                            paused = True
                            break
                    except ConsolidationProgressError as exc:
                        lease_lost = True
                        failed_strategies.append(
                            f"{strategy.value} (progress coordination: {type(exc).__name__})"
                        )
                        progress_messages.append(
                            f"Consolidation stopped safely in {strategy.value}: {exc}"
                        )
                        break
                    except Exception as exc:
                        logger.error(
                            "Consolidation: %s failed after %.1fs: %s",
                            strategy.value,
                            time.perf_counter() - strategy_start,
                            exc,
                            exc_info=True,
                        )
                        failed_strategies.append(f"{strategy.value} ({type(exc).__name__})")
                        if progress_session is not None:
                            previous = progress_session.strategy_state(strategy.value)
                            try:
                                await progress_session.checkpoint(
                                    strategy.value,
                                    "failed",
                                    cursor=previous.get("cursor"),
                                    pending=previous.get("pending"),
                                )
                            except ConsolidationProgressError as progress_exc:
                                lease_lost = True
                                progress_messages.append(
                                    "Consolidation stopped safely after a checkpoint "
                                    f"could not be saved: {progress_exc}"
                                )
                                break
                    finally:
                        logger.info(
                            "Consolidation: %s finished in %.1fs",
                            strategy.value,
                            time.perf_counter() - strategy_start,
                        )
                        self._active_strategy = None
                        self._strategy_deadline = None

                if paused or lease_lost:
                    break

            if timed_out_strategies:
                report.extra["timed_out_strategies"] = list(dict.fromkeys(timed_out_strategies))
            if failed_strategies:
                report.extra["failed_strategies"] = failed_strategies

            all_complete = progress_session is None or set(requested_names).issubset(
                progress_session.completed_strategies
            )
            if (
                not paused
                and not lease_lost
                and not failed_strategies
                and not timed_out_strategies
                and all_complete
            ):
                try:
                    await self._run_auto_tier(report, dry_run)
                except Exception as exc:
                    failed_strategies.append(f"auto_tier ({type(exc).__name__})")
                    report.extra["failed_strategies"] = failed_strategies

            if progress_session is not None and not lease_lost:
                if paused:
                    report.extra["consolidation_status"] = "paused"
                elif failed_strategies:
                    failed_names = [
                        item.split(" ", 1)[0]
                        for item in failed_strategies
                        if item.split(" ", 1)[0] in requested_names
                    ]
                    if failed_names:
                        failed_name = failed_names[0]
                        failed_state = progress_session.strategy_state(failed_name)
                        await progress_session.checkpoint(
                            failed_name,
                            "failed",
                            cursor=failed_state.get("cursor"),
                            pending=failed_state.get("pending"),
                        )
                    await progress_session.fail("; ".join(failed_strategies))
                    report.extra["consolidation_status"] = "failed"
                elif set(requested_names).issubset(progress_session.completed_strategies):
                    await progress_session.complete()
                    report.extra["consolidation_status"] = "completed"
                else:
                    await progress_session.pause()
                    report.extra["consolidation_status"] = "paused"
                    progress_messages.append(
                        "Consolidation paused with unfinished strategies; "
                        "the next smem consolidate will continue."
                    )
            elif lease_lost:
                report.extra["consolidation_status"] = "lease_lost"
            elif dry_run:
                report.extra["consolidation_status"] = "dry_run"
            elif timed_out_strategies:
                report.extra["consolidation_status"] = "timed_out"
            elif failed_strategies:
                report.extra["consolidation_status"] = "failed"
            else:
                report.extra["consolidation_status"] = "completed"

            if progress_messages:
                report.extra["consolidation_progress_messages"] = progress_messages

            report.duration_ms = (time.perf_counter() - start) * 1000
            return report
        except asyncio.CancelledError:
            if progress_session is not None:
                try:
                    await asyncio.shield(progress_session.pause())
                except Exception:
                    logger.exception("Failed to pause consolidation after task cancellation")
            raise
        finally:
            self._active_strategy = None
            self._strategy_deadline = None
            self._total_deadline = None
            self._progress_session = None
            if progress_session is not None:
                await progress_session.close()

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
        """Prune graph records in bounded, checkpointed units."""
        logger = logging.getLogger(__name__)
        if not self._storage.current_brain_id:
            return

        storage: Any = self._storage
        required_methods = (
            "get_synapse_prune_page",
            "get_synapses_by_ids",
            "get_synapse_target_counts_for_sources",
            "get_fiber_neuron_ids_for",
            "remove_synapse_refs_from_fibers",
            "find_neurons_after_id",
            "find_neurons_by_ids",
            "get_neuron_states_batch",
            "delete_synapses_batch",
            "delete_neurons_batch",
        )
        if not supports_storage_methods(storage, required_methods):
            # Keep other storage adapters working until they implement the same bounded
            # keyset operations; SurrealDB takes the optimized path.
            return await self._prune_legacy(report, reference_time, dry_run)

        progress_state = self._strategy_progress_state()
        phase = str(progress_state.get("phase") or "")
        saved_cursor = progress_state.get("cursor")
        cursor = str(saved_cursor) if saved_cursor is not None else None
        pending = [str(value) for value in (progress_state.get("pending") or [])]
        counters = dict(progress_state.get("counters") or {})
        synapses_pruned = int(counters.get("synapses_pruned", 0))
        neurons_pruned = int(counters.get("neurons_pruned", 0))
        synapse_pages = int(counters.get("synapse_pages", 0))
        neuron_pages = int(counters.get("neuron_pages", 0))
        report.synapses_pruned = synapses_pruned
        report.neurons_pruned = neurons_pruned

        async def _save_checkpoint(phase: str, **kwargs: Any) -> None:
            if not dry_run:
                await self._checkpoint_progress(phase, **kwargs)

        pinned_neuron_ids = await storage.get_pinned_neuron_ids()
        synapse_page_size = 2000
        neuron_page_size = 1000
        dead_neuron_days = getattr(self._config, "prune_dead_neuron_days", 14.0)

        def _base_synapse_candidate(synapse: Synapse) -> bool:
            if synapse.source_id in pinned_neuron_ids or synapse.target_id in pinned_neuron_ids:
                return False
            if synapse.created_at is not None and synapse.created_at > reference_time:
                return False

            decayed = synapse.time_decay(reference_time=reference_time)
            if synapse.metadata.get("_inferred", False) and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=0.5)
            if synapse.metadata.get("_dream", False) and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=1.0 / self._dream_decay_multiplier)
            if synapse.metadata.get("_semantic_discovery", False) and synapse.reinforced_count < 2:
                decayed = decayed.decay(factor=0.5)

            should_prune = decayed.weight < self._config.prune_weight_threshold
            if synapse.last_activated is not None:
                days_inactive = (reference_time - synapse.last_activated).total_seconds() / 86400
                should_prune = (
                    should_prune and days_inactive >= self._config.prune_min_inactive_days
                )
            elif synapse.created_at is not None:
                days_since_creation = (reference_time - synapse.created_at).total_seconds() / 86400
                grace_period = max(1.0, self._config.prune_min_inactive_days / 7)
                should_prune = should_prune and days_since_creation >= grace_period
            return should_prune

        async def _synapse_candidates(synapses: list[Synapse]) -> list[str]:
            base_candidates = [synapse for synapse in synapses if _base_synapse_candidate(synapse)]
            if not base_candidates:
                return []

            source_ids = {synapse.source_id for synapse in base_candidates}
            high_salience_ids = await storage.get_fiber_neuron_ids_for(
                list(source_ids), min_salience=0.8
            )
            bridge_sources = {
                synapse.source_id
                for synapse in base_candidates
                if synapse.weight >= 0.02 and synapse.source_id not in high_salience_ids
            }
            target_counts_by_source = (
                await storage.get_synapse_target_counts_for_sources(list(bridge_sources))
                if bridge_sources
                else {}
            )

            candidates: list[str] = []
            for synapse in base_candidates:
                if synapse.source_id in high_salience_ids:
                    continue
                if (
                    synapse.weight >= 0.02
                    and target_counts_by_source.get(synapse.source_id, 0) <= 1
                ):
                    continue
                candidates.append(synapse.id)
            return candidates

        async def _finish_synapse_pending(ids: list[str], page_cursor: str | None) -> None:
            nonlocal synapses_pruned
            current = await storage.get_synapses_by_ids(ids)
            eligible_ids = set(await _synapse_candidates(current))
            if eligible_ids:
                await storage.delete_synapses_batch(eligible_ids)

            remaining = await storage.get_synapses_by_ids(ids)
            remaining_ids = {synapse.id for synapse in remaining}
            removed_ids = set(ids) - remaining_ids
            if removed_ids:
                await storage.remove_synapse_refs_from_fibers(removed_ids)
            synapses_pruned += len(removed_ids)
            report.synapses_pruned = synapses_pruned
            counters.update(
                synapses_pruned=synapses_pruned,
                neurons_pruned=neurons_pruned,
                synapse_pages=synapse_pages,
                neuron_pages=neuron_pages,
            )
            await _save_checkpoint(
                "synapse_scan",
                cursor=page_cursor,
                pending=[],
                counters=counters,
            )

        resume_neurons = phase.startswith(("neuron_", "retention_"))
        if not resume_neurons:
            synapse_cursor = cursor if phase.startswith("synapse_") else None
            synapse_cursor_id, synapse_cursor_created_at = _decode_prune_synapse_cursor(
                synapse_cursor
            )
            if phase == "synapse_pending" and pending:
                await _finish_synapse_pending(pending, synapse_cursor)
                synapse_cursor = cursor

            while True:
                if (
                    synapse_cursor_created_at is not None
                    and synapse_cursor_created_at > reference_time
                ):
                    break
                page = await storage.get_synapse_prune_page(
                    synapse_cursor_created_at,
                    synapse_cursor_id,
                    limit=synapse_page_size,
                )
                if not page:
                    break
                page_crossed_reference = page[-1].created_at > reference_time
                await asyncio.sleep(0)

                candidates = await _synapse_candidates(page)
                next_cursor = _encode_prune_synapse_cursor(page[-1])
                synapse_pages += 1
                counters.update(
                    synapses_pruned=synapses_pruned,
                    neurons_pruned=neurons_pruned,
                    synapse_pages=synapse_pages,
                    neuron_pages=neuron_pages,
                )

                if candidates and not dry_run:
                    await _save_checkpoint(
                        "synapse_pending",
                        cursor=next_cursor,
                        pending=candidates,
                        counters=counters,
                    )
                    await _finish_synapse_pending(candidates, next_cursor)
                else:
                    if dry_run:
                        synapses_pruned += len(candidates)
                        report.synapses_pruned = synapses_pruned
                    counters["synapses_pruned"] = synapses_pruned
                    await _save_checkpoint(
                        "synapse_scan",
                        cursor=next_cursor,
                        pending=[],
                        counters=counters,
                    )
                synapse_cursor = next_cursor
                synapse_cursor_id = page[-1].id
                synapse_cursor_created_at = page[-1].created_at
                if page_crossed_reference or len(page) < synapse_page_size:
                    break

        if not self._config.prune_isolated_neurons:
            return

        if not resume_neurons:
            await _save_checkpoint(
                "neuron_scan",
                cursor=None,
                pending=[],
                counters={
                    "synapses_pruned": synapses_pruned,
                    "neurons_pruned": neurons_pruned,
                    "synapse_pages": synapse_pages,
                    "neuron_pages": neuron_pages,
                },
            )
            neuron_cursor = None
        else:
            neuron_cursor = cursor if phase.startswith("neuron_") else None
            if phase == "neuron_pending" and pending:
                neuron_cursor = cursor

        async def _neuron_candidates(neurons: list[Neuron]) -> list[str]:
            if not neurons:
                return []
            neuron_ids = [neuron.id for neuron in neurons]
            pinned = await storage.get_pinned_neuron_ids()
            fiber_members = await storage.get_fiber_neuron_ids_for(neuron_ids)
            states = await storage.get_neuron_states_batch(neuron_ids)
            eligible: list[str] = []
            for neuron in neurons:
                if neuron.id in pinned or neuron.id in fiber_members or neuron.ephemeral:
                    continue
                state = states.get(neuron.id)
                access_frequency = state.access_frequency if state else 0
                if access_frequency > 0:
                    continue
                age_days = (reference_time - neuron.created_at).total_seconds() / 86400
                if age_days < dead_neuron_days:
                    continue
                eligible.append(neuron.id)
            return eligible

        async def _finish_neuron_pending(ids: list[str], page_cursor: str | None) -> None:
            nonlocal neurons_pruned
            current = await storage.find_neurons_by_ids(ids, include_embedding=False)
            eligible_ids = set(await _neuron_candidates(current))
            if eligible_ids:
                await storage.delete_neurons_batch(sorted(eligible_ids))

            remaining = await storage.find_neurons_by_ids(ids, include_embedding=False)
            remaining_ids = {neuron.id for neuron in remaining}
            removed_ids = set(ids) - remaining_ids
            still_eligible = set(await _neuron_candidates(remaining))
            if still_eligible:
                raise RuntimeError(
                    "prune could not finish a neuron batch; durable pending IDs are preserved"
                )
            neurons_pruned += len(removed_ids)
            report.neurons_pruned = neurons_pruned
            counters.update(
                synapses_pruned=synapses_pruned,
                neurons_pruned=neurons_pruned,
                synapse_pages=synapse_pages,
                neuron_pages=neuron_pages,
            )
            await _save_checkpoint(
                "neuron_scan",
                cursor=page_cursor,
                pending=[],
                counters=counters,
            )

        if phase == "neuron_pending" and pending:
            await _finish_neuron_pending(pending, neuron_cursor)

        while not phase.startswith("retention_"):
            page = await storage.find_neurons_after_id(
                neuron_cursor,
                limit=neuron_page_size,
                created_before=reference_time,
                ephemeral=False,
                include_embedding=False,
            )
            if not page:
                break
            await asyncio.sleep(0)

            candidates = await _neuron_candidates(page)
            next_cursor = page[-1].id
            neuron_pages += 1
            counters.update(
                synapses_pruned=synapses_pruned,
                neurons_pruned=neurons_pruned,
                synapse_pages=synapse_pages,
                neuron_pages=neuron_pages,
            )
            if candidates and not dry_run:
                await _save_checkpoint(
                    "neuron_pending",
                    cursor=next_cursor,
                    pending=candidates,
                    counters=counters,
                )
                await _finish_neuron_pending(candidates, next_cursor)
            else:
                if dry_run:
                    neurons_pruned += len(candidates)
                    report.neurons_pruned = neurons_pruned
                counters["neurons_pruned"] = neurons_pruned
                await _save_checkpoint(
                    "neuron_scan",
                    cursor=next_cursor,
                    pending=[],
                    counters=counters,
                )
            neuron_cursor = next_cursor
            if len(page) < neuron_page_size:
                break

        await _save_checkpoint(
            "retention_start",
            cursor=None,
            pending=[],
            counters={
                "synapses_pruned": synapses_pruned,
                "neurons_pruned": neurons_pruned,
                "synapse_pages": synapse_pages,
                "neuron_pages": neuron_pages,
            },
        )

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

        # Prune decay telemetry. Runs even when telemetry is currently disabled,
        # so turning it off still cleans up what it accumulated — same reasoning
        # as the retrieval-trace prune above.
        if not dry_run and hasattr(self._storage, "prune_decay_passes"):
            try:
                from surreal_memory.unified_config import get_config

                dt_cfg = get_config().decay_telemetry
                pruned_passes = await self._storage.prune_decay_passes(
                    retention_days=dt_cfg.retention_days,
                    max_records=dt_cfg.max_records,
                )
                if pruned_passes > 0:
                    logger.info("Pruned %d decay telemetry rows", pruned_passes)
                    report.decay_passes_pruned = pruned_passes
            except Exception:
                logger.debug("Decay telemetry pruning skipped", exc_info=True)

        # Collapse superseded pending change-log updates.
        #
        # This is the ONLY retention path for a brain whose sync never completes.
        # prune_synced_changes can only remove rows that were successfully synced,
        # so where sync is configured but never lands, nothing ever removed a row
        # and the table grew without bound -- and the dashboard card that would
        # have revealed it was itself too slow to load at that size, so the defect
        # hid its own symptom. Collapsing is lossless: replication converges on the
        # newest payload per entity, so superseded updates cannot change the
        # outcome for any peer, at any sync position.
        if not dry_run and hasattr(self._storage, "collapse_pending_updates"):
            try:
                collapsed = await self._storage.collapse_pending_updates(
                    max_rows=_CHANGE_LOG_COLLAPSE_CAP
                )
                report.change_log_collapsed = collapsed
                if collapsed:
                    logger.info("Collapsed %d superseded change-log updates", collapsed)
                if collapsed >= _CHANGE_LOG_COLLAPSE_CAP:
                    report.extra["change_log_collapse_truncated"] = True
                    try:
                        stats = await self._storage.get_change_log_stats()
                        report.extra["change_log_pending_after"] = int(stats.get("pending", 0) or 0)
                    except Exception:
                        logger.debug("Change-log stats unavailable after collapse", exc_info=True)
            except Exception:
                logger.debug("Change-log collapse skipped", exc_info=True)

    async def _prune_legacy(
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
        all_synapses = await self._all_synapses_paged()
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

        # Prune decay telemetry. Runs even when telemetry is currently disabled,
        # so turning it off still cleans up what it accumulated — same reasoning
        # as the retrieval-trace prune above.
        if not dry_run and hasattr(self._storage, "prune_decay_passes"):
            try:
                from surreal_memory.unified_config import get_config

                dt_cfg = get_config().decay_telemetry
                pruned_passes = await self._storage.prune_decay_passes(
                    retention_days=dt_cfg.retention_days,
                    max_records=dt_cfg.max_records,
                )
                if pruned_passes > 0:
                    logger.info("Pruned %d decay telemetry rows", pruned_passes)
                    report.decay_passes_pruned = pruned_passes
            except Exception:
                logger.debug("Decay telemetry pruning skipped", exc_info=True)

        # Collapse superseded pending change-log updates.
        #
        # This is the ONLY retention path for a brain whose sync never completes.
        # prune_synced_changes can only remove rows that were successfully synced,
        # so where sync is configured but never lands, nothing ever removed a row
        # and the table grew without bound -- and the dashboard card that would
        # have revealed it was itself too slow to load at that size, so the defect
        # hid its own symptom. Collapsing is lossless: replication converges on the
        # newest payload per entity, so superseded updates cannot change the
        # outcome for any peer, at any sync position.
        if not dry_run and hasattr(self._storage, "collapse_pending_updates"):
            try:
                collapsed = await self._storage.collapse_pending_updates(
                    max_rows=_CHANGE_LOG_COLLAPSE_CAP
                )
                report.change_log_collapsed = collapsed
                if collapsed:
                    logger.info("Collapsed %d superseded change-log updates", collapsed)
                if collapsed >= _CHANGE_LOG_COLLAPSE_CAP:
                    report.extra["change_log_collapse_truncated"] = True
                    try:
                        stats = await self._storage.get_change_log_stats()
                        report.extra["change_log_pending_after"] = int(stats.get("pending", 0) or 0)
                    except Exception:
                        logger.debug("Change-log stats unavailable after collapse", exc_info=True)
            except Exception:
                logger.debug("Change-log collapse skipped", exc_info=True)

    async def _merge(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Merge overlapping fibers in stable, checkpointed work units.

        Each unit is one connected component. Its source signatures and successor ID
        are saved before any write, so a restart can distinguish an unchanged unit
        from data that changed while it was paused. The source fibers are not removed
        until the successor, typed-memory row, and maturation row are durable.
        """
        import json

        from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage

        progress_state = self._strategy_progress_state() if not dry_run else {}
        phase = str(progress_state.get("phase") or "")
        saved_counters = dict(progress_state.get("counters") or {})
        fibers_merged = int(saved_counters.get("fibers_merged", 0))
        fibers_created = int(saved_counters.get("fibers_created", 0))
        fibers_removed = int(saved_counters.get("fibers_removed", 0))
        delete_failures = int(saved_counters.get("merge_delete_failures", 0))
        if not dry_run:
            report.fibers_merged = fibers_merged
            report.fibers_created = fibers_created
            report.fibers_removed = fibers_removed
            if delete_failures:
                report.extra["merge_delete_failures"] = delete_failures

        def _normalize(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
                return value.value
            if callable(getattr(value, "to_dict", None)):
                return _normalize(value.to_dict())
            if hasattr(value, "__dataclass_fields__"):
                return {
                    name: _normalize(getattr(value, name))
                    for name in value.__dataclass_fields__
                    if not name.startswith("_")
                }
            if isinstance(value, dict):
                return {str(key): _normalize(item) for key, item in value.items()}
            if isinstance(value, (set, frozenset)):
                items = [_normalize(item) for item in value]
                return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
            if isinstance(value, (tuple, list)):
                return [_normalize(item) for item in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return repr(value)

        def _fingerprint(value: Any) -> str:
            encoded = json.dumps(
                _normalize(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        def _fiber_fingerprint(fiber: Fiber) -> str:
            return _fingerprint(
                {
                    "id": fiber.id,
                    "neuron_ids": fiber.neuron_ids,
                    "synapse_ids": fiber.synapse_ids,
                    "anchor_neuron_id": fiber.anchor_neuron_id,
                    "pathway": fiber.pathway,
                    "conductivity": fiber.conductivity,
                    "last_conducted": fiber.last_conducted,
                    "time_start": fiber.time_start,
                    "time_end": fiber.time_end,
                    "coherence": fiber.coherence,
                    "salience": fiber.salience,
                    "frequency": fiber.frequency,
                    "summary": fiber.summary,
                    "essence": fiber.essence,
                    "last_ghost_shown_at": fiber.last_ghost_shown_at,
                    "auto_tags": fiber.auto_tags,
                    "agent_tags": fiber.agent_tags,
                    "metadata": fiber.metadata,
                    "compression_tier": fiber.compression_tier,
                    "pinned": fiber.pinned,
                    "created_at": fiber.created_at,
                }
            )

        def _snapshot_signature_map(records: dict[str, Any]) -> dict[str, str]:
            return {
                fiber_id: _fingerprint(record)
                for fiber_id, record in records.items()
                if record is not None
            }

        def _decode_descriptor(raw_cursor: Any) -> dict[str, Any]:
            if not isinstance(raw_cursor, str):
                raise RuntimeError("merge checkpoint has no resumable work-unit cursor")
            try:
                descriptor = json.loads(raw_cursor)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("merge checkpoint cursor is malformed") from exc
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("version") != 1
                or not isinstance(descriptor.get("source_ids"), list)
                or not isinstance(descriptor.get("fiber_signatures"), list)
                or not isinstance(descriptor.get("unit_id"), str)
                or not isinstance(descriptor.get("merged_id"), str)
                or not isinstance(descriptor.get("removed_base"), int)
            ):
                raise RuntimeError("merge checkpoint cursor is incompatible")
            source_ids = descriptor["source_ids"]
            if (
                len(source_ids) < 2
                or source_ids != sorted(set(source_ids))
                or len(source_ids) != len(descriptor["fiber_signatures"])
            ):
                raise RuntimeError("merge checkpoint source identities are invalid")
            if descriptor["merged_id"] != f"merge-{descriptor['unit_id']}":
                raise RuntimeError("merge checkpoint successor identity is invalid")
            return descriptor

        def _encode_descriptor(descriptor: dict[str, Any]) -> str:
            return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), default=str)

        def _counter_values() -> dict[str, int | float]:
            return {
                "fibers_merged": fibers_merged,
                "fibers_created": fibers_created,
                "fibers_removed": fibers_removed,
                "merge_delete_failures": delete_failures,
            }

        async def _checkpoint(
            phase_name: str,
            descriptor: dict[str, Any] | None,
            pending: Sequence[str] = (),
        ) -> None:
            if dry_run:
                return
            await self._checkpoint_progress(
                phase_name,
                cursor=_encode_descriptor(descriptor) if descriptor is not None else None,
                pending=pending,
                counters=_counter_values(),
            )

        async def _read_source_typed(source_ids: list[str]) -> dict[str, Any]:
            rows = await self._storage.get_typed_memories_batch(source_ids)
            if not isinstance(rows, dict):
                raise RuntimeError("merge could not read the typed-memory layer safely")
            return {fiber_id: rows.get(fiber_id) for fiber_id in source_ids}

        async def _read_source_maturations(source_ids: list[str]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for fiber_id in source_ids:
                try:
                    result[fiber_id] = await self._storage.get_maturation(fiber_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"merge could not read maturation for source {fiber_id!r}"
                    ) from exc
            return result

        async def _read_typed_target(merged_id: str) -> Any:
            rows = await self._storage.get_typed_memories_batch([merged_id])
            if not isinstance(rows, dict):
                raise RuntimeError("merge could not verify the successor typed-memory row")
            return rows.get(merged_id)

        async def _validate_source_fibers(
            descriptor: dict[str, Any],
            source_ids: list[str],
            *,
            allow_missing: bool,
        ) -> dict[str, Fiber]:
            expected = dict(zip(source_ids, descriptor["fiber_signatures"], strict=True))
            current: dict[str, Fiber] = {}
            for fiber_id in source_ids:
                fiber = await self._storage.get_fiber(fiber_id)
                if fiber is None:
                    if allow_missing:
                        continue
                    raise RuntimeError(
                        f"merge source {fiber_id!r} disappeared before successor data was safe"
                    )
                if _fiber_fingerprint(fiber) != expected[fiber_id]:
                    raise RuntimeError(
                        f"merge source {fiber_id!r} changed after its checkpoint; "
                        "the pending unit was left untouched"
                    )
                current[fiber_id] = fiber
            return current

        async def _validate_typed_source_state(
            descriptor: dict[str, Any],
            source_ids: list[str],
            *,
            allow_partial_transfer: bool,
        ) -> tuple[dict[str, Any], Any]:
            expected = dict(descriptor.get("typed_signatures") or {})
            current = await _read_source_typed(source_ids)
            target = await _read_typed_target(descriptor["merged_id"])
            for fiber_id in source_ids:
                record = current[fiber_id]
                signature = expected.get(fiber_id)
                if record is not None and signature is None:
                    raise RuntimeError(
                        f"typed memory for source {fiber_id!r} appeared after the merge checkpoint"
                    )
                if record is not None and _fingerprint(record) != signature:
                    raise RuntimeError(
                        f"typed memory for source {fiber_id!r} changed after its checkpoint"
                    )
                if signature is not None and record is None and target is None:
                    raise RuntimeError(
                        f"typed memory for source {fiber_id!r} is missing before reassignment"
                    )
                if signature is not None and record is None and not allow_partial_transfer:
                    raise RuntimeError(
                        f"typed memory for source {fiber_id!r} disappeared before reassignment"
                    )
            if target is not None and not expected:
                raise RuntimeError("merge successor has an unexpected typed-memory row")
            return current, target

        async def _validate_maturation_source_state(
            descriptor: dict[str, Any],
            source_ids: list[str],
            live_source_ids: set[str],
            *,
            transfer_complete: bool,
        ) -> dict[str, Any]:
            expected = dict(descriptor.get("maturation_signatures") or {})
            current = await _read_source_maturations(source_ids)
            for fiber_id in source_ids:
                record = current[fiber_id]
                signature = expected.get(fiber_id)
                if fiber_id in live_source_ids:
                    if record is not None and signature is None:
                        raise RuntimeError(
                            f"maturation for source {fiber_id!r} appeared after its checkpoint"
                        )
                    if record is None and signature is not None:
                        raise RuntimeError(
                            f"maturation for source {fiber_id!r} disappeared before transfer"
                        )
                    if record is not None and _fingerprint(record) != signature:
                        raise RuntimeError(
                            f"maturation for source {fiber_id!r} changed after its checkpoint"
                        )
            target = await self._storage.get_maturation(descriptor["merged_id"])
            if transfer_complete:
                if expected and target is None:
                    raise RuntimeError("merge successor is missing inherited maturation")
                if not expected and target is not None:
                    raise RuntimeError("merge successor has unexpected maturation")
            return current

        def _make_merged_fiber(member_fibers: list[Fiber], merged_id: str) -> Fiber:
            merged_neuron_ids: set[str] = set()
            merged_synapse_ids: set[str] = set()
            max_salience = 0.0
            best_anchor = member_fibers[0].anchor_neuron_id
            best_frequency = 0
            merged_auto_tags: set[str] = set()
            merged_agent_tags: set[str] = set()
            for fiber in member_fibers:
                merged_neuron_ids |= fiber.neuron_ids
                merged_synapse_ids |= fiber.synapse_ids
                merged_auto_tags |= fiber.auto_tags
                merged_agent_tags |= fiber.agent_tags
                if fiber.salience > max_salience:
                    max_salience = fiber.salience
                if fiber.frequency > best_frequency:
                    best_frequency = fiber.frequency
                    best_anchor = fiber.anchor_neuron_id
            return Fiber(
                id=merged_id,
                neuron_ids=merged_neuron_ids,
                synapse_ids=merged_synapse_ids,
                anchor_neuron_id=best_anchor,
                pathway=[best_anchor],
                salience=max_salience,
                frequency=best_frequency,
                auto_tags=merged_auto_tags,
                agent_tags=merged_agent_tags,
                summary=_merged_summary(member_fibers),
                metadata=_merged_metadata(member_fibers),
                created_at=min(fiber.created_at for fiber in member_fibers),
            )

        async def _make_descriptor(member_fibers: list[Fiber]) -> dict[str, Any]:
            source_ids = sorted(fiber.id for fiber in member_fibers)
            ordered_fibers = sorted(member_fibers, key=lambda item: item.id)
            typed = await _read_source_typed(source_ids)
            maturations = await _read_source_maturations(source_ids)
            fiber_signatures = [_fiber_fingerprint(fiber) for fiber in ordered_fibers]
            typed_signatures = _snapshot_signature_map(typed)
            maturation_signatures = _snapshot_signature_map(maturations)
            identity = _fingerprint(
                {
                    "source_ids": source_ids,
                    "fiber_signatures": fiber_signatures,
                    "typed_signatures": typed_signatures,
                    "maturation_signatures": maturation_signatures,
                }
            )
            return {
                "version": 1,
                "unit_id": identity[:32],
                "source_ids": source_ids,
                "fiber_signatures": fiber_signatures,
                "typed_signatures": typed_signatures,
                "maturation_signatures": maturation_signatures,
                "merged_id": f"merge-{identity[:32]}",
                "removed_base": fibers_removed,
            }

        async def _add_or_validate_successor(
            descriptor: dict[str, Any],
            member_fibers: list[Fiber],
        ) -> None:
            merged = _make_merged_fiber(member_fibers, descriptor["merged_id"])
            existing = await self._storage.get_fiber(descriptor["merged_id"])
            if existing is None:
                try:
                    await self._storage.add_fiber(merged)
                except Exception as exc:
                    if not is_duplicate_key_error(exc):
                        raise
                existing = await self._storage.get_fiber(descriptor["merged_id"])
            if existing is None:
                raise RuntimeError("merge successor write was not visible after add_fiber")
            if _fiber_fingerprint(existing) != _fiber_fingerprint(merged):
                raise RuntimeError("merge successor ID is occupied by different fiber data")

        async def _transfer_typed_memory(
            descriptor: dict[str, Any],
            member_fibers: list[Fiber],
        ) -> None:
            source_ids = descriptor["source_ids"]
            expected = dict(descriptor.get("typed_signatures") or {})
            source_rows, target_row = await _validate_typed_source_state(
                descriptor,
                source_ids,
                allow_partial_transfer=True,
            )
            if expected and target_row is None:
                # The target write is first in _reassign_typed_memory; only a complete
                # target write permits the source rows to be cleaned up on retries.
                if any(source_rows[fiber_id] is None for fiber_id in expected):
                    raise RuntimeError("typed-memory sources are incomplete before reassignment")
                await self._reassign_typed_memory(
                    source_rows,
                    member_fibers,
                    descriptor["merged_id"],
                )
                source_rows, target_row = await _validate_typed_source_state(
                    descriptor,
                    source_ids,
                    allow_partial_transfer=True,
                )
                if target_row is None:
                    raise RuntimeError("typed-memory reassignment did not create its successor row")
            elif not expected and target_row is not None:
                raise RuntimeError("merge successor has unexpected typed memory")

            # _reassign_typed_memory is deliberately target-first. If a prior
            # attempt completed that UPSERT but stopped while deleting source rows,
            # the target proves the complete combined row is durable; clean up only
            # unchanged source rows and never recompute the target from a partial set.
            for fiber_id, record in source_rows.items():
                if record is not None:
                    if _fingerprint(record) != expected.get(fiber_id):
                        raise RuntimeError(
                            f"typed memory for source {fiber_id!r} changed before cleanup"
                        )
                    await self._storage.delete_typed_memory(fiber_id)
            remaining = await _read_source_typed(source_ids)
            if any(record is not None for record in remaining.values()):
                raise RuntimeError("typed-memory source cleanup did not complete")

        async def _transfer_maturation(
            descriptor: dict[str, Any],
            source_records: dict[str, Any],
        ) -> None:
            expected = dict(descriptor.get("maturation_signatures") or {})
            records = [source_records[fiber_id] for fiber_id in descriptor["source_ids"]]
            records = [record for record in records if record is not None]
            if not expected:
                existing = await self._storage.get_maturation(descriptor["merged_id"])
                if existing is not None:
                    raise RuntimeError("merge successor has unexpected maturation")
                return
            stage_order = list(MemoryStage)
            inherited_stage = max(records, key=lambda record: stage_order.index(record.stage)).stage
            timestamps = tuple(
                sorted(
                    {
                        timestamp
                        for record in records
                        for timestamp in record.reinforcement_timestamps
                    }
                )
            )
            await self._storage.save_maturation(
                MaturationRecord(
                    fiber_id=descriptor["merged_id"],
                    brain_id=records[0].brain_id,
                    stage=inherited_stage,
                    stage_entered_at=min(record.stage_entered_at for record in records),
                    rehearsal_count=len(timestamps),
                    reinforcement_timestamps=timestamps,
                )
            )
            persisted = await self._storage.get_maturation(descriptor["merged_id"])
            if persisted is None or _fingerprint(persisted) != _fingerprint(
                MaturationRecord(
                    fiber_id=descriptor["merged_id"],
                    brain_id=records[0].brain_id,
                    stage=inherited_stage,
                    stage_entered_at=min(record.stage_entered_at for record in records),
                    rehearsal_count=len(timestamps),
                    reinforcement_timestamps=timestamps,
                )
            ):
                raise RuntimeError("maturation inheritance was not durable on the merge successor")

        async def _finish_unit(
            descriptor: dict[str, Any],
            unit_phase: str,
            pending_ids: list[str],
        ) -> None:
            nonlocal fibers_merged, fibers_created, fibers_removed, delete_failures

            source_ids = list(descriptor["source_ids"])
            removed_base = int(descriptor["removed_base"])
            active_phase = unit_phase
            members_by_id: dict[str, Fiber] = {}
            if active_phase != "merge_deleting_sources":
                members_by_id = await _validate_source_fibers(
                    descriptor, source_ids, allow_missing=False
                )
                member_fibers = [members_by_id[fiber_id] for fiber_id in source_ids]
                if active_phase == "merge_pending":
                    await _add_or_validate_successor(descriptor, member_fibers)
                    active_phase = "merge_fiber_created"
                    await _checkpoint(active_phase, descriptor, source_ids)

                if active_phase == "merge_fiber_created":
                    await _validate_source_fibers(descriptor, source_ids, allow_missing=False)
                    await _transfer_typed_memory(descriptor, member_fibers)
                    active_phase = "merge_typed_reassigned"
                    await _checkpoint(active_phase, descriptor, source_ids)

                if active_phase == "merge_typed_reassigned":
                    await _validate_source_fibers(descriptor, source_ids, allow_missing=False)
                    current_typed = await _read_source_typed(source_ids)
                    if any(record is not None for record in current_typed.values()):
                        raise RuntimeError(
                            "typed-memory source rows remain before maturation transfer"
                        )
                    source_maturations = await _validate_maturation_source_state(
                        descriptor,
                        source_ids,
                        set(source_ids),
                        transfer_complete=False,
                    )
                    await _transfer_maturation(descriptor, source_maturations)
                    active_phase = "merge_maturation_transferred"
                    await _checkpoint(active_phase, descriptor, source_ids)

                if active_phase == "merge_maturation_transferred":
                    await _validate_source_fibers(descriptor, source_ids, allow_missing=False)
                    await _validate_typed_source_state(
                        descriptor,
                        source_ids,
                        allow_partial_transfer=True,
                    )
                    await _validate_maturation_source_state(
                        descriptor,
                        source_ids,
                        set(source_ids),
                        transfer_complete=True,
                    )
                    active_phase = "merge_deleting_sources"
                    pending_ids = list(source_ids)
                    await _checkpoint(active_phase, descriptor, pending_ids)

            elif active_phase != "merge_deleting_sources":
                raise RuntimeError(f"unsupported merge checkpoint phase {active_phase!r}")

            # The successor already contains the full unit before source removal
            # begins. Validate every still-live source before deleting any of them.
            successor = await self._storage.get_fiber(descriptor["merged_id"])
            if successor is None:
                raise RuntimeError("merge successor disappeared before source deletion")
            if sorted(successor.metadata.get("merged_from", [])) != source_ids:
                raise RuntimeError("merge successor provenance no longer matches its work unit")

            remaining_fibers = await _validate_source_fibers(
                descriptor,
                source_ids,
                allow_missing=True,
            )
            pending_set = set(pending_ids)
            reappeared = (set(source_ids) - pending_set) & set(remaining_fibers)
            if reappeared:
                raise RuntimeError("a previously deleted merge source reappeared")
            live_pending = [fiber_id for fiber_id in pending_ids if fiber_id in remaining_fibers]
            unexpected_live = set(remaining_fibers) - pending_set
            if unexpected_live:
                raise RuntimeError("a merge source reappeared outside the durable pending list")

            current_typed, target_typed = await _validate_typed_source_state(
                descriptor,
                source_ids,
                allow_partial_transfer=True,
            )
            expected_typed = dict(descriptor.get("typed_signatures") or {})
            if expected_typed and target_typed is None:
                raise RuntimeError("merge successor lost typed memory before source deletion")
            for fiber_id, record in current_typed.items():
                if record is not None:
                    if _fingerprint(record) != expected_typed.get(fiber_id):
                        raise RuntimeError(
                            f"typed memory for source {fiber_id!r} changed before deletion"
                        )
                    await self._storage.delete_typed_memory(fiber_id)
            if any(
                record is not None for record in (await _read_source_typed(source_ids)).values()
            ):
                raise RuntimeError("typed-memory cleanup is incomplete before source deletion")

            live_ids = set(remaining_fibers)
            await _validate_maturation_source_state(
                descriptor,
                source_ids,
                live_ids,
                transfer_complete=True,
            )

            for fiber_id in live_pending:
                # Revalidate just before each destructive call. A different member
                # changing does not get silently folded into this frozen snapshot.
                current = await self._storage.get_fiber(fiber_id)
                expected = dict(zip(source_ids, descriptor["fiber_signatures"], strict=True))
                if current is None:
                    continue
                if _fiber_fingerprint(current) != expected[fiber_id]:
                    raise RuntimeError(
                        f"merge source {fiber_id!r} changed immediately before deletion"
                    )
                deleted = await self._storage.delete_fiber(fiber_id)
                after_delete = await self._storage.get_fiber(fiber_id)
                if after_delete is not None:
                    delete_failures += 1
                    report.extra["merge_delete_failures"] = delete_failures
                    remaining_ids = live_pending[live_pending.index(fiber_id) :]
                    await _checkpoint("merge_deleting_sources", descriptor, remaining_ids)
                    if self._progress_session is not None:
                        raise RuntimeError(f"merge could not delete source fiber {fiber_id!r}")
                    continue
                if not deleted:
                    # The row may have been deleted by a prior attempt between
                    # the read and delete; absence after the call confirms completion.
                    pass
                live_pending = [item for item in live_pending if item != fiber_id]
                fibers_removed = removed_base + len(source_ids) - len(live_pending)
                report.fibers_removed = fibers_removed
                await _checkpoint("merge_deleting_sources", descriptor, live_pending)

            # Count only confirmed absences. A non-persistent legacy adapter may
            # report failed deletes while still allowing independent source deletes.
            live_after = [
                fiber_id
                for fiber_id in source_ids
                if await self._storage.get_fiber(fiber_id) is not None
            ]
            fibers_removed = removed_base + len(source_ids) - len(live_after)
            fibers_merged += len(source_ids)
            fibers_created += 1
            report.fibers_merged = fibers_merged
            report.fibers_created = fibers_created
            report.fibers_removed = fibers_removed
            await _checkpoint("merge_scan", None, ())
            report.merge_details.append(
                MergeDetail(
                    original_fiber_ids=tuple(source_ids),
                    merged_fiber_id=descriptor["merged_id"],
                    neuron_count=len(successor.neuron_ids),
                    reason="neuron_overlap",
                )
            )

        async def _candidate_groups(fibers: list[Fiber]) -> list[list[Fiber]]:
            fiber_list = sorted(fibers, key=lambda item: item.id)
            if len(fiber_list) < 2:
                return []
            neuron_to_fibers: dict[str, set[int]] = {}
            for idx, fiber in enumerate(fiber_list):
                if len(fiber.neuron_ids) > self._config.merge_max_fiber_size:
                    continue
                for neuron_id in fiber.neuron_ids:
                    neuron_to_fibers.setdefault(neuron_id, set()).add(idx)

            candidate_pairs: set[tuple[int, int]] = set()
            for indices in neuron_to_fibers.values():
                if len(indices) > 100:
                    continue
                indices_list = sorted(indices)
                for i_pos in range(len(indices_list)):
                    for j_pos in range(i_pos + 1, len(indices_list)):
                        candidate_pairs.add((indices_list[i_pos], indices_list[j_pos]))
                if len(candidate_pairs) >= 50_000:
                    break

            uf = UnionFind(len(fiber_list))
            for pair_number, (left, right) in enumerate(sorted(candidate_pairs), start=1):
                if pair_number % 1000 == 0:
                    # Let the strategy deadline and lease heartbeat run during large scans.
                    await asyncio.sleep(0)
                first = fiber_list[left]
                second = fiber_list[right]
                if first.metadata.get("_verbatim", False) != second.metadata.get(
                    "_verbatim", False
                ):
                    continue
                if any(
                    fiber.metadata.get(marker)
                    for fiber in (first, second)
                    for marker in ("_habit_pattern", "_reasoning_pattern")
                ):
                    continue
                if first.pinned or second.pinned:
                    continue
                union_size = len(first.neuron_ids | second.neuron_ids)
                if union_size == 0:
                    continue
                jaccard = len(first.neuron_ids & second.neuron_ids) / union_size
                if first.created_at and second.created_at:
                    time_diff = abs((first.created_at - second.created_at).total_seconds())
                else:
                    time_diff = float("inf")
                threshold = (
                    self._config.merge_overlap_threshold * 0.6
                    if time_diff < 3600
                    else self._config.merge_overlap_threshold
                )
                if jaccard >= threshold:
                    uf.union(left, right)

            groups: list[list[Fiber]] = []
            for members in uf.groups().values():
                if len(members) >= 2:
                    groups.append(
                        sorted((fiber_list[index] for index in members), key=lambda item: item.id)
                    )
            groups.sort(key=lambda group: tuple(fiber.id for fiber in group))
            return groups

        resumable_phases = {
            "merge_pending",
            "merge_fiber_created",
            "merge_typed_reassigned",
            "merge_maturation_transferred",
            "merge_deleting_sources",
        }
        if not dry_run and phase in resumable_phases:
            descriptor = _decode_descriptor(progress_state.get("cursor"))
            pending_ids = [str(value) for value in (progress_state.get("pending") or [])]
            if not pending_ids and phase != "merge_deleting_sources":
                pending_ids = list(descriptor["source_ids"])
            await _finish_unit(descriptor, phase, pending_ids)

        fibers = await self._storage.get_fibers(limit=10000)
        groups = await _candidate_groups(list(fibers))
        for member_fibers in groups:
            if dry_run:
                source_ids = sorted(fiber.id for fiber in member_fibers)
                descriptor_id = _fingerprint(source_ids)[:32]
                merged = _make_merged_fiber(member_fibers, f"merge-{descriptor_id}")
                report.fibers_merged += len(member_fibers)
                report.fibers_created += 1
                report.merge_details.append(
                    MergeDetail(
                        original_fiber_ids=tuple(source_ids),
                        merged_fiber_id=merged.id,
                        neuron_count=len(merged.neuron_ids),
                        reason="neuron_overlap",
                    )
                )
                continue

            descriptor = await _make_descriptor(member_fibers)
            source_ids = list(descriptor["source_ids"])
            await _checkpoint("merge_pending", descriptor, source_ids)
            await _finish_unit(descriptor, "merge_pending", source_ids)

        await _checkpoint("merge_complete", None, ())

    _DEDUP_CURSOR_KEY = "_dedup_anchor_cursor"
    _SYNAPSE_PAGE_SIZE = 5000

    @staticmethod
    def _config_from_settings() -> ConsolidationConfig:
        """Default config, with the dedup knobs taken from the user's [dedup] section.

        Without this the census cap and threshold were only reachable by editing the
        source — the settings existed but nothing read them.
        """
        try:
            from surreal_memory.unified_config import UnifiedConfig

            dedup = UnifiedConfig.load().dedup
            return ConsolidationConfig(
                dedup_max_anchors=int(dedup.consolidation_max_anchors),
                dedup_simhash_threshold=int(dedup.simhash_threshold),
            )
        except Exception:
            # Config is optional here: consolidation must still run on a bare install.
            return ConsolidationConfig()

    async def _all_synapses_paged(self) -> list[Synapse]:
        """Every synapse in the brain, fetched in pages instead of one giant read.

        Measured on a real brain: an unbounded ``get_synapses()`` pulls the entire
        table — six figures of rows — over a single response. That is the shape that
        produced ``[Errno 104] Connection reset by peer`` elsewhere in this engine, and
        it is why semantic_discovery already pages. Same total work, but no single
        oversized response and a yield point between pages.
        """
        collected: list[Synapse] = []
        offset = 0
        while True:
            page = await self._storage.get_synapses(limit=self._SYNAPSE_PAGE_SIZE, offset=offset)
            if not page:
                break
            collected.extend(page)
            if len(page) < self._SYNAPSE_PAGE_SIZE:
                break
            offset += len(page)
            # Nothing else in these passes awaits, so without this the connection's
            # keepalive never runs during a long scan.
            await asyncio.sleep(0)
        return collected

    async def _dedup_cursor(self, anchors_total: int) -> int:
        """Where this run's dedup window starts.

        Persisted in Brain.metadata (SCHEMALESS, same place the quality badge lives)
        rather than a new table — no migration, and the state is naturally per-brain.
        Read-modify-write is not atomic, which is acceptable because consolidations do
        not run concurrently; the worst case is a window repeated once.
        """
        try:
            brain = await self._storage.get_brain(self._storage.current_brain_id or "")
        except Exception:
            logger.debug("Could not read dedup cursor", exc_info=True)
            return 0
        if brain is None:
            return 0
        raw = brain.metadata.get(self._DEDUP_CURSOR_KEY, 0)
        try:
            cursor = int(raw)
        except (TypeError, ValueError):
            return 0
        # A shrunken brain must not leave the cursor past the end.
        return cursor % anchors_total if anchors_total else 0

    async def _advance_dedup_cursor(self, cursor: int, cap: int, anchors_total: int) -> bool:
        """Move the window forward only after its comparison pass completed."""
        if anchors_total <= 0:
            return True
        next_cursor = (cursor + cap) % anchors_total
        try:
            brain_id = self._storage.current_brain_id or ""
            brain = await self._storage.get_brain(brain_id)
            if brain is None:
                return False
            brain.metadata[self._DEDUP_CURSOR_KEY] = next_cursor
            await self._storage.save_brain(brain)
            return True
        except Exception:
            logger.debug("Could not persist dedup cursor", exc_info=True)
            return False

    async def _collect_source_typed_memories(
        self, member_fibers: list[Fiber]
    ) -> dict[str, Any] | None:
        """Typed-memory rows of the fibers about to be merged, read before any delete.

        Returns ``None`` when the layer cannot be read, so the caller can tell "no rows
        existed" apart from "the lookup failed" and skip the reassignment rather than
        destroying rows it never saw.
        """
        try:
            return await self._storage.get_typed_memories_batch([f.id for f in member_fibers])
        except Exception:
            logger.debug("Could not read typed memories during merge", exc_info=True)
            return None

    async def _reassign_typed_memory(
        self,
        source_typed: dict[str, Any],
        member_fibers: list[Fiber],
        merged_fiber_id: str,
    ) -> None:
        """Move the merged fibers' typed-memory layer onto the surviving fiber.

        ``add_typed_memory`` is an UPSERT keyed by fiber_id, so writing the merged row
        first and deleting the sources afterwards never leaves the merge without a
        typed-memory row. Conflicts resolve deterministically: strongest priority wins,
        then the widest validity window, so merging can only preserve reach, never
        silently shorten how long a memory stays valid.
        """
        records = [tm for tm in source_typed.values() if tm is not None]
        if not records:
            return

        def _priority_of(record: Any) -> tuple[int, float]:
            # created_at as the tie-break keeps the choice stable across runs.
            return (int(record.priority), -record.created_at.timestamp())

        winner = max(records, key=_priority_of)
        trust_scores = [r.trust_score for r in records if r.trust_score is not None]
        valid_froms = [r.valid_from for r in records if r.valid_from is not None]
        expiries = [r.expires_at for r in records if r.expires_at is not None]

        merged_tags: set[str] = set()
        for record in records:
            merged_tags |= set(record.tags)

        merged_record = dc_replace(
            winner,
            fiber_id=merged_fiber_id,
            trust_score=max(trust_scores) if trust_scores else winner.trust_score,
            valid_from=min(valid_froms) if valid_froms else winner.valid_from,
            # All sources expiring -> keep the latest; any source open-ended -> stay
            # open-ended, because a merge must not invent an expiry.
            expires_at=(max(expiries) if len(expiries) == len(records) and expiries else None),
            tags=frozenset(merged_tags),
        )

        try:
            await self._storage.add_typed_memory(merged_record)
        except Exception:
            logger.warning("Could not write merged typed memory", exc_info=True)
            return

        for fiber in member_fibers:
            if fiber.id == merged_fiber_id:
                continue
            if fiber.id in source_typed:
                try:
                    await self._storage.delete_typed_memory(fiber.id)
                except Exception:
                    logger.debug("Could not delete source typed memory", exc_info=True)

    @staticmethod
    def _summarize_input_fibers(fibers: list[Fiber]) -> list[Fiber]:
        """Fibers eligible as clustering INPUT.

        Excludes this pass's own output: a summary fiber carries the union of its
        sources' tags, so leaving it in would let each run cluster the previous run's
        summaries and grow the brain without bound.
        """
        return [f for f in fibers if f.tags and f.metadata.get("_consolidation") != "summary_fiber"]

    @staticmethod
    def _existing_summary_cluster_keys(fibers: list[Fiber]) -> set[str]:
        """Cluster keys of summaries that already exist.

        Derived from the same fiber list the pass already fetched rather than a second
        query: ``find_fibers`` has no offset parameter, so a separate lookup could not
        be paged and would silently see only its first page.
        """
        keys: set[str] = set()
        for fiber in fibers:
            if fiber.metadata.get("_consolidation") != "summary_fiber":
                continue
            key = fiber.metadata.get("_cluster_key")
            if key:
                keys.add(str(key))
                continue
            # Summaries written before the key existed: derive it from the recorded
            # sources so historical rows still suppress duplicates.
            sources = fiber.metadata.get("source_fibers")
            if isinstance(sources, list) and sources:
                keys.add(_summary_cluster_key_from_ids(str(s) for s in sources))
        return keys

    async def _summarize(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Create concept neurons for tag-based clusters using inverted index.

        Durable runs process clusters in stable order. Each cluster's exact
        output and artifact IDs are saved as the pending work unit before graph
        writes, so replay can finish partial writes without duplicating them.
        """
        import json

        fibers = await self._storage.get_fibers(limit=10000)
        if len(fibers) < self._config.summarize_min_cluster_size:
            return

        # Stable order is important both for pair enumeration and for resuming
        # the same cluster sequence after a process interruption.
        source_fibers = sorted(self._summarize_input_fibers(fibers), key=lambda f: f.id)

        max_fibers_for_clustering = 1000
        if len(source_fibers) > max_fibers_for_clustering:
            report.extra["summarize_fibers_total"] = len(source_fibers)
            report.extra["summarize_fibers_scanned"] = max_fibers_for_clustering
            source_fibers = sorted(
                source_fibers,
                key=lambda f: (-f.salience, f.id),
            )[:max_fibers_for_clustering]
            source_fibers.sort(key=lambda f: f.id)
        if len(source_fibers) < self._config.summarize_min_cluster_size:
            return

        snapshot_fingerprint = options_fingerprint(
            {
                "algorithm": "summarize-v1",
                "min_cluster_size": self._config.summarize_min_cluster_size,
                "tag_overlap_threshold": self._config.summarize_tag_overlap_threshold,
                "max_fibers": max_fibers_for_clustering,
                "max_pairs": 50_000,
                "source_fibers": [
                    {
                        "id": fiber.id,
                        "anchor_neuron_id": fiber.anchor_neuron_id,
                        "salience": fiber.salience,
                        "summary": fiber.summary,
                        "tags": sorted(fiber.tags),
                    }
                    for fiber in source_fibers
                ],
            }
        )

        n = len(source_fibers)
        tag_to_fibers: dict[str, set[int]] = {}
        for idx, fiber in enumerate(source_fibers):
            for tag in sorted(fiber.tags):
                tag_to_fibers.setdefault(tag, set()).add(idx)

        max_pairs = 50_000
        candidate_pairs: set[tuple[int, int]] = set()
        for tag in sorted(tag_to_fibers):
            indices = tag_to_fibers[tag]
            if len(indices) > 100:
                continue
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

        for pair_idx, (i, j) in enumerate(sorted(candidate_pairs)):
            if pair_idx % 1000 == 0 and pair_idx > 0:
                await asyncio.sleep(0)
            tags_a = source_fibers[i].tags
            tags_b = source_fibers[j].tags
            intersection = len(tags_a & tags_b)
            union_size = len(tags_a | tags_b)
            if (
                union_size > 0
                and intersection / union_size >= self._config.summarize_tag_overlap_threshold
            ):
                union(i, j)

        grouped_members: dict[int, list[int]] = {}
        for i in range(n):
            grouped_members.setdefault(find(i), []).append(i)

        cluster_work: list[tuple[str, list[Fiber]]] = []
        for members in grouped_members.values():
            if len(members) < self._config.summarize_min_cluster_size:
                continue
            cluster_fibers = [source_fibers[i] for i in members]
            cluster_work.append((_summary_cluster_key(cluster_fibers), cluster_fibers))
        cluster_work.sort(key=lambda item: item[0])

        existing_cluster_keys = self._existing_summary_cluster_keys(fibers)
        resumable = self._progress_session is not None and not dry_run
        last_cluster_key = ""
        created_count = 0
        skipped_count = 0

        def cursor_for(cluster_key: str) -> str:
            return f"summarize-v1|{snapshot_fingerprint}|{cluster_key}"

        if resumable:
            state = self._strategy_progress_state()
            cursor_value = state.get("cursor")
            pending = state.get("pending")
            if cursor_value is None:
                if pending:
                    raise ConsolidationProgressError(
                        "summarize pending checkpoint has no compatible cursor"
                    )
                await self._checkpoint_progress(
                    "summarize_scan",
                    cursor=cursor_for(""),
                    pending=[],
                    counters={
                        "summaries_created": 0,
                        "summaries_skipped_existing": 0,
                    },
                )
                state = self._strategy_progress_state()
                cursor_value = state.get("cursor")
                pending = state.get("pending")

            if not isinstance(cursor_value, str):
                raise ConsolidationProgressError("summarize cursor is malformed")
            cursor_parts = cursor_value.split("|", 2)
            if (
                len(cursor_parts) != 3
                or cursor_parts[0] != "summarize-v1"
                or cursor_parts[1] != snapshot_fingerprint
            ):
                raise ConsolidationProgressError(
                    "summarize inputs or algorithm changed; refusing to skip its checkpoint"
                )
            last_cluster_key = cursor_parts[2]

            raw_counters = state.get("counters")
            counters = raw_counters if isinstance(raw_counters, dict) else {}
            created_count = int(counters.get("summaries_created", 0))
            skipped_count = int(counters.get("summaries_skipped_existing", 0))
            report.summaries_created += created_count
            if skipped_count:
                report.extra["summaries_skipped_existing"] = skipped_count

            async def apply_snapshot(snapshot: dict[str, Any]) -> None:
                concept_id = str(snapshot["concept_neuron_id"])
                concept = await self._storage.get_neuron(concept_id)
                if concept is None:
                    await self._check_progress_budget()
                    concept = Neuron.create(
                        type=NeuronType.CONCEPT,
                        content=str(snapshot["concept_content"]),
                        neuron_id=concept_id,
                        metadata={
                            "_consolidation": "summary",
                            "_cluster_key": str(snapshot["cluster_key"]),
                            "cluster_size": len(snapshot["source_fiber_ids"]),
                            "tags": list(snapshot["tags"]),
                        },
                    )
                    await self._storage.add_neuron(concept)

                synapse_ids: set[str] = set()
                for edge in snapshot["synapses"]:
                    synapse_id = str(edge["id"])
                    existing_synapse = await self._storage.get_synapse(synapse_id)
                    if existing_synapse is None:
                        await self._check_progress_budget()
                        synapse = Synapse.create(
                            source_id=concept_id,
                            target_id=str(edge["anchor_id"]),
                            type=SynapseType.RELATED_TO,
                            weight=0.6,
                            synapse_id=synapse_id,
                        )
                        await self._storage.add_synapse(synapse)
                    synapse_ids.add(synapse_id)

                summary_fiber_id = str(snapshot["summary_fiber_id"])
                summary_fiber = await self._storage.get_fiber(summary_fiber_id)
                if summary_fiber is None:
                    await self._check_progress_budget()
                    anchor_ids = {str(anchor_id) for anchor_id in snapshot["anchor_ids"]}
                    summary_fiber = Fiber.create(
                        neuron_ids={concept_id} | anchor_ids,
                        synapse_ids=synapse_ids,
                        anchor_neuron_id=concept_id,
                        summary=str(snapshot["concept_content"]),
                        tags={str(tag) for tag in snapshot["tags"]},
                        metadata={
                            "_consolidation": "summary_fiber",
                            "_cluster_key": str(snapshot["cluster_key"]),
                            "source_fibers": list(snapshot["source_fiber_ids"]),
                        },
                        fiber_id=summary_fiber_id,
                    )
                    await self._storage.add_fiber(summary_fiber)

            async def replay_pending(serialized: str) -> str:
                try:
                    snapshot = json.loads(serialized)
                    if (
                        snapshot["kind"] != "summary_cluster"
                        or snapshot["version"] != 1
                        or snapshot["fingerprint"] != snapshot_fingerprint
                    ):
                        raise ValueError("incompatible summary snapshot")
                    cluster_key = str(snapshot["cluster_key"])
                    if not any(key == cluster_key for key, _ in cluster_work):
                        raise ValueError("summary cluster is not in the current input snapshot")
                except (TypeError, ValueError, KeyError) as exc:
                    raise ConsolidationProgressError(
                        "summarize pending checkpoint is malformed or incompatible"
                    ) from exc
                await self._check_progress_budget()
                await apply_snapshot(snapshot)
                return cluster_key

            if pending:
                if not isinstance(pending, (list, tuple)) or len(pending) != 1:
                    raise ConsolidationProgressError("summarize pending checkpoint is malformed")
                pending_cluster_key = await replay_pending(str(pending[0]))
                if pending_cluster_key <= last_cluster_key:
                    raise ConsolidationProgressError(
                        "summarize pending unit is not after its committed cursor"
                    )
                created_count += 1
                last_cluster_key = pending_cluster_key
                existing_cluster_keys.add(pending_cluster_key)
                await self._checkpoint_progress(
                    "summarize_cluster",
                    cursor=cursor_for(last_cluster_key),
                    pending=[],
                    counters={
                        "summaries_created": created_count,
                        "summaries_skipped_existing": skipped_count,
                    },
                )
                report.summaries_created += 1

        for cluster_key, cluster_fibers in cluster_work:
            if resumable and cluster_key <= last_cluster_key:
                continue
            if len(cluster_fibers) < self._config.summarize_min_cluster_size:
                continue

            if cluster_key in existing_cluster_keys:
                if resumable:
                    skipped_count += 1
                    await self._checkpoint_progress(
                        "summarize_cluster",
                        cursor=cursor_for(cluster_key),
                        pending=[],
                        counters={
                            "summaries_created": created_count,
                            "summaries_skipped_existing": skipped_count,
                        },
                    )
                    last_cluster_key = cluster_key
                    report.extra["summaries_skipped_existing"] = skipped_count
                else:
                    report.extra["summaries_skipped_existing"] = (
                        int(report.extra.get("summaries_skipped_existing", 0)) + 1
                    )
                continue

            summaries = [fiber.summary for fiber in cluster_fibers if fiber.summary]
            all_tags: set[str] = set()
            for fiber in cluster_fibers:
                all_tags |= fiber.tags

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

            if not resumable:
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

                anchor_ids = {fiber.anchor_neuron_id for fiber in cluster_fibers}
                valid_anchor_ids: set[str] = set()
                for anchor_id in sorted(anchor_ids):
                    anchor_neuron = await self._storage.get_neuron(anchor_id)
                    if anchor_neuron is not None:
                        valid_anchor_ids.add(anchor_id)
                anchor_ids = valid_anchor_ids

                synapse_ids: set[str] = set()
                for anchor_id in sorted(anchor_ids)[:10]:
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
                        "_cluster_key": cluster_key,
                        "source_fibers": [fiber.id for fiber in cluster_fibers],
                    },
                )
                await self._storage.add_fiber(summary_fiber)
                report.summaries_created += 1
                continue

            await self._check_progress_budget()
            cluster_anchor_ids = sorted({fiber.anchor_neuron_id for fiber in cluster_fibers})
            snapshot_anchor_ids: list[str] = []
            for anchor_id in cluster_anchor_ids:
                await self._check_progress_budget()
                anchor_neuron = await self._storage.get_neuron(anchor_id)
                if anchor_neuron is not None:
                    snapshot_anchor_ids.append(anchor_id)

            snapshot = {
                "kind": "summary_cluster",
                "version": 1,
                "fingerprint": snapshot_fingerprint,
                "cluster_key": cluster_key,
                "source_fiber_ids": [fiber.id for fiber in cluster_fibers],
                "concept_content": concept_content,
                "tags": sorted(all_tags),
                "concept_neuron_id": str(uuid4()),
                "anchor_ids": snapshot_anchor_ids,
                "synapses": [
                    {"anchor_id": anchor_id, "id": str(uuid4())}
                    for anchor_id in snapshot_anchor_ids[:10]
                ],
                "summary_fiber_id": str(uuid4()),
            }
            serialized = json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            await self._checkpoint_progress(
                "summarize_pending",
                cursor=cursor_for(last_cluster_key),
                pending=[serialized],
                counters={
                    "summaries_created": created_count,
                    "summaries_skipped_existing": skipped_count,
                },
            )
            await apply_snapshot(snapshot)
            created_count += 1
            last_cluster_key = cluster_key
            existing_cluster_keys.add(cluster_key)
            await self._checkpoint_progress(
                "summarize_cluster",
                cursor=cursor_for(last_cluster_key),
                pending=[],
                counters={
                    "summaries_created": created_count,
                    "summaries_skipped_existing": skipped_count,
                },
            )
            report.summaries_created += 1

    async def _mature(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Advance memory maturation stages, auto-promote types, extract patterns."""
        import hashlib
        import json
        import logging

        from surreal_memory.core.memory_types import MemoryType
        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.core.synapse import Direction, Synapse, SynapseType
        from surreal_memory.engine.consolidation_progress import ConsolidationProgressError
        from surreal_memory.engine.memory_stages import (
            MemoryStage,
            compute_stage_transition,
        )
        from surreal_memory.engine.pattern_extraction import extract_patterns

        _logger = logging.getLogger(__name__)
        resumable = self._progress_session is not None and not dry_run
        state = self._strategy_progress_state() if resumable else {}
        phase = str(state.get("phase") or "")

        def phase_rank(value: str) -> int:
            if value.startswith("mature_promotion"):
                return 0
            if value.startswith("mature_cleanup"):
                return 1
            if value.startswith("mature_backfill"):
                return 2
            if value.startswith("mature_stage"):
                return 3
            if value.startswith("mature_pattern"):
                return 4
            if value.startswith("essence_"):
                return 5
            if value == "mature_complete":
                return 6
            return -1

        def encode_synapse(synapse: Synapse) -> dict[str, Any]:
            return {
                "id": synapse.id,
                "source_id": synapse.source_id,
                "target_id": synapse.target_id,
                "type": synapse.type.value,
                "weight": synapse.weight,
                "direction": synapse.direction.value,
                "metadata": synapse.metadata,
                "reinforced_count": synapse.reinforced_count,
                "last_activated": (
                    synapse.last_activated.isoformat() if synapse.last_activated else None
                ),
                "created_at": synapse.created_at.isoformat(),
            }

        def decode_synapse(payload: dict[str, Any]) -> Synapse:
            return Synapse(
                id=str(payload["id"]),
                source_id=str(payload["source_id"]),
                target_id=str(payload["target_id"]),
                type=SynapseType(payload["type"]),
                weight=float(payload["weight"]),
                direction=Direction(payload["direction"]),
                metadata=dict(payload.get("metadata") or {}),
                reinforced_count=int(payload.get("reinforced_count", 0)),
                last_activated=(
                    datetime.fromisoformat(payload["last_activated"])
                    if payload.get("last_activated")
                    else None
                ),
                created_at=datetime.fromisoformat(payload["created_at"]),
            )

        def encode_neuron(neuron: Neuron) -> dict[str, Any]:
            return {
                "id": neuron.id,
                "type": neuron.type.value,
                "content": neuron.content,
                "metadata": neuron.metadata,
                "content_hash": neuron.content_hash,
                "created_at": neuron.created_at.isoformat(),
                "ephemeral": neuron.ephemeral,
            }

        def decode_neuron(payload: dict[str, Any]) -> Neuron:
            return Neuron(
                id=str(payload["id"]),
                type=NeuronType(payload["type"]),
                content=str(payload["content"]),
                metadata=dict(payload.get("metadata") or {}),
                content_hash=int(payload.get("content_hash", 0)),
                created_at=datetime.fromisoformat(payload["created_at"]),
                ephemeral=bool(payload.get("ephemeral", False)),
            )

        async def pattern_source_fingerprint() -> str:
            source_fibers = await self._storage.get_fibers(limit=10000)
            source_maturations = await self._storage.find_maturations()
            payload = {
                "fibers": [
                    {
                        "id": fiber.id,
                        "neuron_ids": sorted(fiber.neuron_ids),
                        "tags": sorted(fiber.tags),
                    }
                    for fiber in sorted(source_fibers, key=lambda item: item.id)
                ],
                "maturations": [
                    {
                        "fiber_id": record.fiber_id,
                        "brain_id": record.brain_id,
                        "stage": record.stage.value,
                        "stage_entered_at": record.stage_entered_at.isoformat(),
                        "rehearsal_count": record.rehearsal_count,
                        "reinforcement_timestamps": list(record.reinforcement_timestamps),
                    }
                    for record in sorted(source_maturations, key=lambda item: item.fiber_id)
                ],
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        _hop_keys = {
            (MemoryStage.SHORT_TERM, MemoryStage.WORKING): "stm_to_working",
            (MemoryStage.WORKING, MemoryStage.EPISODIC): "working_to_episodic",
            (MemoryStage.EPISODIC, MemoryStage.SEMANTIC): "episodic_to_semantic",
        }

        # Phase 0: Auto-promote context→fact for frequently-recalled memories.
        # A pending promotion is safe to replay: the candidate disappears from the
        # query once promoted, so an absent pending candidate means the write landed.
        if not dry_run and (not resumable or phase_rank(phase) <= 0):
            candidates = await self._storage.get_promotion_candidates(
                min_frequency=5,
                source_type="context",
            )
            candidates = sorted(candidates, key=lambda candidate: str(candidate["fiber_id"]))
            cursor_value = state.get("cursor") if phase.startswith("mature_promotion") else None
            cursor = str(cursor_value) if cursor_value is not None else None
            counters = state.get("counters")
            promoted_count = (
                int(counters.get("memories_promoted", 0))
                if phase.startswith("mature_promotion") and isinstance(counters, dict)
                else 0
            )
            pending = state.get("pending") if phase == "mature_promotion_pending" else []
            if pending:
                if not isinstance(pending, (list, tuple)) or len(pending) != 1:
                    raise ConsolidationProgressError("invalid maturation promotion checkpoint")
                try:
                    snapshot = json.loads(pending[0])
                    fiber_id = str(snapshot["fiber_id"])
                    expected = str(snapshot["candidate_fingerprint"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ConsolidationProgressError(
                        "invalid maturation promotion snapshot"
                    ) from exc
                candidate = next(
                    (item for item in candidates if str(item["fiber_id"]) == fiber_id),
                    None,
                )
                if candidate is None:
                    # Promotion committed before the previous process saved its cursor.
                    promoted_count += 1
                else:
                    actual = json.dumps(
                        candidate, sort_keys=True, separators=(",", ":"), default=str
                    )
                    if actual != expected:
                        raise ConsolidationProgressError(
                            "maturation promotion candidate changed after its checkpoint"
                        )
                    await self._check_progress_budget()
                    promoted = await self._storage.promote_memory_type(
                        fiber_id=fiber_id,
                        new_type=MemoryType.FACT,
                        new_expires_at=None,
                    )
                    promoted_count += int(bool(promoted))
                cursor = fiber_id
                counters = {"memories_promoted": promoted_count}
                await self._checkpoint_progress(
                    "mature_promotion",
                    cursor=cursor,
                    pending=[],
                    counters=counters,
                )

            for candidate in candidates:
                fiber_id = str(candidate["fiber_id"])
                if cursor is not None and fiber_id <= cursor:
                    continue
                meta = candidate.get("metadata", {})
                if meta.get("auto_promoted"):
                    cursor = fiber_id
                    if resumable:
                        await self._checkpoint_progress(
                            "mature_promotion",
                            cursor=cursor,
                            pending=[],
                            counters={"memories_promoted": promoted_count},
                        )
                    continue
                if resumable:
                    await self._check_progress_budget()
                    serialized = json.dumps(
                        {
                            "fiber_id": fiber_id,
                            "candidate_fingerprint": json.dumps(
                                candidate,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    await self._checkpoint_progress(
                        "mature_promotion_pending",
                        cursor=cursor,
                        pending=[serialized],
                        counters={"memories_promoted": promoted_count},
                    )
                promoted = await self._storage.promote_memory_type(
                    fiber_id=fiber_id,
                    new_type=MemoryType.FACT,
                    new_expires_at=None,
                )
                promoted_count += int(bool(promoted))
                cursor = fiber_id
                if resumable:
                    await self._checkpoint_progress(
                        "mature_promotion",
                        cursor=cursor,
                        pending=[],
                        counters={"memories_promoted": promoted_count},
                    )
            if resumable:
                await self._checkpoint_progress(
                    "mature_promotion_complete",
                    cursor=cursor,
                    pending=[],
                    counters={"memories_promoted": promoted_count},
                )
            report.memories_promoted += promoted_count
            if promoted_count > 0:
                _logger.info("Auto-promoted %d context memories to fact", promoted_count)

        # Clean up orphaned maturation records. This operation is idempotent and is
        # deliberately omitted during dry-run, which must not mutate storage.
        if not dry_run and (not resumable or phase_rank(phase) <= 1):
            if resumable:
                await self._check_progress_budget()
                await self._checkpoint_progress(
                    "mature_cleanup_pending",
                    cursor=None,
                    pending=[],
                )
            cleaned = await self._storage.cleanup_orphaned_maturations()
            if cleaned > 0:
                _logger.info("Cleaned up %d orphaned maturation records", cleaned)
            if resumable:
                await self._checkpoint_progress(
                    "mature_cleanup_complete",
                    cursor=None,
                    pending=[],
                )

        # Give fibers that predate the maturation subsystem a row. Backfill is
        # idempotent, so a crash after it commits can safely repeat this single unit.
        if not dry_run and (not resumable or phase_rank(phase) <= 2):
            if resumable:
                await self._check_progress_budget()
                await self._checkpoint_progress(
                    "mature_backfill_pending",
                    cursor=None,
                    pending=[],
                )
            try:
                fiber_count = await self._storage.get_total_fiber_count()
                maturation_count = len(await self._storage.find_maturations())
                if fiber_count > maturation_count:
                    backfilled = await self._storage.backfill_maturations()
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
                        deficit = fiber_count - maturation_count
                        report.extra["maturations_unreachable"] = deficit
                        _logger.warning(
                            "Maturation backfill created nothing while %d fibers still lack a row",
                            deficit,
                        )
            except Exception:
                _logger.debug("Maturation backfill skipped", exc_info=True)
            if resumable:
                await self._checkpoint_progress(
                    "mature_backfill_complete",
                    cursor=None,
                    pending=[],
                )

        # Advance stage records in stable fiber-ID order. A save is an upsert, so
        # replaying the row after a write/acknowledgment gap is idempotent.
        stage_done = resumable and phase_rank(phase) > 3
        patterns_done = resumable and phase_rank(phase) > 4
        stage_cursor: str | None = None
        stage_counts: dict[str, int | float] = dict.fromkeys(_hop_keys.values(), 0)
        if resumable and phase.startswith("mature_stage"):
            raw_cursor = state.get("cursor")
            stage_cursor = str(raw_cursor) if raw_cursor is not None else None
            raw_counts = state.get("counters")
            if isinstance(raw_counts, dict):
                stage_counts.update(
                    {key: int(value) for key, value in raw_counts.items() if key in stage_counts}
                )

        if not stage_done:
            all_maturations = await self._storage.find_maturations()
            all_maturations.sort(key=lambda record: record.fiber_id)
            for record in all_maturations:
                if resumable and stage_cursor is not None and record.fiber_id <= stage_cursor:
                    continue
                if resumable:
                    await self._check_progress_budget()
                advanced = compute_stage_transition(record, now=reference_time)
                if advanced.stage != record.stage:
                    report.stages_advanced += 1
                    hop_key = _hop_keys.get((record.stage, advanced.stage))
                    if hop_key:
                        transitions = report.extra.setdefault("stage_transitions", {})
                        transitions[hop_key] = transitions.get(hop_key, 0) + 1
                        stage_counts[hop_key] += 1
                    if not dry_run:
                        try:
                            await self._storage.save_maturation(advanced)
                        except Exception as exc:
                            if "FOREIGN KEY" in str(exc):
                                _logger.warning(
                                    "Skipping orphaned maturation for fiber %s",
                                    record.fiber_id,
                                )
                                if resumable:
                                    stage_cursor = record.fiber_id
                                    await self._checkpoint_progress(
                                        "mature_stage",
                                        cursor=stage_cursor,
                                        pending=[],
                                        counters=stage_counts,
                                    )
                                continue
                            raise
                if resumable:
                    stage_cursor = record.fiber_id
                    await self._checkpoint_progress(
                        "mature_stage",
                        cursor=stage_cursor,
                        pending=[],
                        counters=stage_counts,
                    )
            if resumable:
                await self._checkpoint_progress(
                    "mature_stage_complete",
                    cursor=stage_cursor,
                    pending=[],
                    counters=stage_counts,
                )
                report.stages_advanced = sum(int(value) for value in stage_counts.values())
                if any(stage_counts.values()):
                    report.extra["stage_transitions"] = {
                        key: value for key, value in stage_counts.items() if value
                    }

        if dry_run:
            return

        # Resume pattern extraction from its saved per-pattern snapshot. The source
        # fingerprint prevents applying output derived from fibers/maturation data
        # that changed while this strategy was paused.
        if patterns_done:
            await self._essence_backfill(report, dry_run)
            await self._checkpoint_progress(
                "mature_complete",
                cursor=str(state.get("cursor")) if state.get("cursor") is not None else None,
                pending=[],
            )
            return

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

        pattern_cursor: str | None = None
        if resumable and phase.startswith("mature_pattern"):
            raw_cursor = state.get("cursor")
            pattern_cursor = str(raw_cursor) if raw_cursor is not None else None
        if resumable:
            pattern_pending = state.get("pending") if phase == "mature_pattern_pending" else []
            source_fingerprint = await pattern_source_fingerprint()
            if pattern_pending:
                if not isinstance(pattern_pending, (list, tuple)) or len(pattern_pending) != 1:
                    raise ConsolidationProgressError("invalid maturation pattern checkpoint")
                serialized = pattern_pending[0]
                try:
                    manifest = json.loads(serialized)
                    if manifest.get("kind") != "mature_pattern" or manifest.get("version") != 1:
                        raise ValueError("unsupported pattern manifest")
                    if manifest.get("source_fingerprint") != source_fingerprint:
                        raise ConsolidationProgressError(
                            "maturation pattern source data changed after its checkpoint"
                        )
                    source_ids = manifest["source_fiber_ids"]
                    if not isinstance(source_ids, list) or not source_ids:
                        raise ValueError("source fiber IDs are malformed")
                    pattern_key = json.dumps(
                        sorted(str(item) for item in source_ids), separators=(",", ":")
                    )
                    concept = decode_neuron(manifest["concept_neuron"])
                    synapses = [decode_synapse(item) for item in manifest["synapses"]]
                except ConsolidationProgressError:
                    raise
                except (KeyError, TypeError, ValueError) as exc:
                    raise ConsolidationProgressError("invalid maturation pattern snapshot") from exc

                await self._check_progress_budget()
                existing = await self._storage.get_neuron(concept.id)
                if existing is None:
                    await self._storage.add_neuron(concept)
                elif existing != concept:
                    raise ConsolidationProgressError(
                        "maturation concept changed after its pending checkpoint"
                    )
                for synapse in synapses:
                    existing_synapse = await self._storage.get_synapse(synapse.id)
                    if existing_synapse is None:
                        await self._storage.add_synapse(synapse)
                    elif existing_synapse != synapse:
                        raise ConsolidationProgressError(
                            "maturation pattern synapse changed after its pending checkpoint"
                        )
                pattern_cursor = pattern_key
                await self._checkpoint_progress(
                    "mature_pattern",
                    cursor=pattern_cursor,
                    pending=[],
                    counters={"patterns_extracted": extraction_report.patterns_extracted},
                )

            def pattern_key_for(pattern: Any) -> str:
                return json.dumps(
                    sorted(pattern.source_fiber_ids),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            for pattern in sorted(patterns, key=pattern_key_for):
                pattern_key = pattern_key_for(pattern)
                if pattern_cursor is not None and pattern_key <= pattern_cursor:
                    continue
                await self._check_progress_budget()
                manifest = {
                    "kind": "mature_pattern",
                    "version": 1,
                    "source_fingerprint": source_fingerprint,
                    "source_fiber_ids": list(pattern.source_fiber_ids),
                    "concept_neuron": encode_neuron(pattern.concept_neuron),
                    "synapses": [encode_synapse(edge) for edge in pattern.synapses],
                }
                serialized = json.dumps(
                    manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                await self._checkpoint_progress(
                    "mature_pattern_pending",
                    cursor=pattern_cursor,
                    pending=[serialized],
                    counters={"patterns_extracted": extraction_report.patterns_extracted},
                )
                existing = await self._storage.get_neuron(pattern.concept_neuron.id)
                if existing is None:
                    await self._storage.add_neuron(pattern.concept_neuron)
                elif existing != pattern.concept_neuron:
                    raise ConsolidationProgressError(
                        "maturation concept ID has conflicting contents"
                    )
                for synapse in pattern.synapses:
                    existing_synapse = await self._storage.get_synapse(synapse.id)
                    if existing_synapse is None:
                        await self._storage.add_synapse(synapse)
                    elif existing_synapse != synapse:
                        raise ConsolidationProgressError(
                            "maturation pattern synapse ID has conflicting contents"
                        )
                pattern_cursor = pattern_key
                await self._checkpoint_progress(
                    "mature_pattern",
                    cursor=pattern_cursor,
                    pending=[],
                    counters={"patterns_extracted": extraction_report.patterns_extracted},
                )
            await self._checkpoint_progress(
                "mature_patterns_complete",
                cursor=pattern_cursor,
                pending=[],
                counters={"patterns_extracted": extraction_report.patterns_extracted},
            )
        else:
            for pattern in patterns:
                await self._storage.add_neuron(pattern.concept_neuron)
                for synapse in pattern.synapses:
                    await self._storage.add_synapse(synapse)

        await self._essence_backfill(report, dry_run)
        if resumable:
            await self._checkpoint_progress(
                "mature_complete",
                cursor=pattern_cursor,
                pending=[],
                counters={"patterns_extracted": extraction_report.patterns_extracted},
            )

    async def _essence_backfill(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Generate essence for fibers missing it, or upgrade extractive → LLM.

        Uses configured essence_generator strategy from BrainConfig:
        - "extractive" (default): sentence-level scoring, fast and free
        - "llm": LLM abstractive with cost guard (priority < 3 skipped)

        In a durable progress session, fibers are processed in stable ID order.
        The generated result is checkpointed before the idempotent DB update so
        a resume can apply it without another LLM call. A crash before that
        result checkpoint can still repeat the external LLM call.
        """
        import json

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

        # Keep the legacy adapter path simple: non-SurrealDB backends and dry
        # runs do not write progress state. Dry-run must never mutate storage.
        if self._progress_session is None or dry_run:
            fibers = await self._storage.get_fibers(limit=1000)
            candidates = [fiber for fiber in fibers if not fiber.essence]

            backfilled = 0
            for idx, fiber in enumerate(candidates):
                if backfilled >= max_backfill:
                    break
                if idx % 50 == 0 and idx > 0:
                    await asyncio.sleep(0)

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

                await self._storage.update_fiber(fiber.with_essence(essence))
                backfilled += 1

            if backfilled > 0:
                logger.info("Essence backfill: %d fibers updated", backfilled)
            report.essences_generated += backfilled
            return

        # get_fibers is bounded by the storage API; sorting the returned window
        # by the immutable fiber ID gives this strategy a stable work-unit order.
        fibers = await self._storage.get_fibers(limit=1000)
        fibers_by_id = {fiber.id: fiber for fiber in fibers}
        candidates = sorted(
            (fiber for fiber in fibers if not fiber.essence),
            key=lambda fiber: fiber.id,
        )

        state = self._strategy_progress_state()
        cursor_value = state.get("cursor")
        cursor = str(cursor_value) if cursor_value is not None else None
        raw_counters = state.get("counters")
        counters = raw_counters if isinstance(raw_counters, dict) else {}
        backfilled = int(counters.get("essences_generated", 0))

        # A single generated result is the pending unit. Its JSON string is
        # stored in the existing pending:string[] field, avoiding a schema
        # change while preserving arbitrary essence text exactly.
        pending = state.get("pending")
        if pending:
            if not isinstance(pending, (list, tuple)) or len(pending) != 1:
                raise RuntimeError("invalid essence-backfill pending checkpoint")
            try:
                pending_result = json.loads(pending[0])
                pending_fiber_id = str(pending_result["fiber_id"])
                pending_essence = str(pending_result["essence"])
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("invalid essence-backfill pending result") from exc

            await self._check_progress_budget()
            pending_fiber = fibers_by_id.get(pending_fiber_id)
            if pending_fiber is None:
                # Do not discard a generated result if its fiber has fallen
                # outside the storage window; the caller can safely retry after
                # the storage view is corrected.
                raise RuntimeError(
                    f"pending essence fiber {pending_fiber_id!r} is not in the current fiber window"
                )

            # update_fiber writes the same value for this stable ID and essence.
            # If the previous process applied it before crashing, skip the write.
            if not pending_fiber.essence:
                await self._storage.update_fiber(pending_fiber.with_essence(pending_essence))
            backfilled += 1
            cursor = pending_fiber_id
            counters = {"essences_generated": backfilled}
            await self._checkpoint_progress(
                "essence_scan",
                cursor=cursor,
                pending=[],
                counters=counters,
            )

        for idx, fiber in enumerate(candidates):
            if backfilled >= max_backfill:
                break
            if cursor is not None and fiber.id <= cursor:
                continue
            if idx % 50 == 0 and idx > 0:
                await asyncio.sleep(0)
            await self._check_progress_budget()

            anchor = await self._storage.get_neuron(fiber.anchor_neuron_id)
            if not anchor or not anchor.content:
                cursor = fiber.id
                await self._checkpoint_progress(
                    "essence_scan",
                    cursor=cursor,
                    pending=[],
                    counters={"essences_generated": backfilled},
                )
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

            # If the process dies before this checkpoint, the external LLM call
            # may repeat on resume; after it succeeds, resume uses the saved text.
            essence = await generator.generate(anchor.content, priority=priority)
            if essence:
                result = json.dumps(
                    {"fiber_id": fiber.id, "essence": essence},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                await self._checkpoint_progress(
                    "essence_pending",
                    cursor=cursor,
                    pending=[result],
                    counters={"essences_generated": backfilled},
                )
                # A crash between this write and the committed checkpoint is
                # replay-safe: resume observes the saved text and avoids a
                # duplicate write when the fiber already contains that essence.
                current_fiber = fibers_by_id[fiber.id]
                if not current_fiber.essence:
                    await self._storage.update_fiber(current_fiber.with_essence(essence))
                backfilled += 1

            # Every fiber is a committed work unit, including skips and empty
            # generator results, so the cursor can resume at the next stable ID.
            cursor = fiber.id
            await self._checkpoint_progress(
                "essence_scan",
                cursor=cursor,
                pending=[],
                counters={"essences_generated": backfilled},
            )

        await self._checkpoint_progress(
            "essence_complete",
            cursor=cursor,
            pending=[],
            counters={"essences_generated": backfilled},
        )
        if backfilled > 0:
            logger.info("Essence backfill: %d fibers updated", backfilled)
        report.essences_generated += backfilled

    async def _infer(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Run associative inference with replay-safe durable work-unit checkpoints."""
        import json
        import logging
        from datetime import timedelta

        from surreal_memory.core.synapse import Direction
        from surreal_memory.engine.associative_inference import (
            InferenceConfig,
            create_inferred_synapse,
            generate_associative_tags,
            identify_candidates,
        )
        from surreal_memory.utils.tag_normalizer import TagNormalizer

        logger = logging.getLogger(__name__)
        strategy_state = self._strategy_progress_state() if not dry_run else {}
        persisted_counters = strategy_state.get("counters", {})
        if not isinstance(persisted_counters, dict):
            persisted_counters = {}
        counters: dict[str, int | float] = {
            "synapses_inferred": int(persisted_counters.get("synapses_inferred", 0)),
            "co_activations_pruned": int(persisted_counters.get("co_activations_pruned", 0)),
        }
        report.synapses_inferred = int(counters["synapses_inferred"])
        report.co_activations_pruned = int(counters["co_activations_pruned"])

        def pair_key(source_id: str, target_id: str) -> str:
            return json.dumps(sorted((source_id, target_id)), separators=(",", ":"))

        def decode_pending(raw: str) -> dict[str, Any] | None:
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None

        def dump_synapse(synapse: Synapse) -> dict[str, Any]:
            payload = asdict(synapse)

            def encode(value: Any) -> Any:
                if isinstance(value, (SynapseType, Direction)):
                    return value.value
                if isinstance(value, datetime):
                    return value.isoformat()
                raise TypeError(f"unsupported synapse snapshot value: {type(value).__name__}")

            decoded = json.loads(json.dumps(payload, default=encode, sort_keys=True))
            if not isinstance(decoded, dict):
                raise TypeError("serialized Synapse snapshot is not a mapping")
            return decoded

        def load_synapse(payload: dict[str, Any]) -> Synapse:
            values = dict(payload)
            values["type"] = SynapseType(values["type"])
            values["direction"] = Direction(values["direction"])
            if values.get("last_activated") is not None:
                values["last_activated"] = datetime.fromisoformat(values["last_activated"])
            values["created_at"] = datetime.fromisoformat(values["created_at"])
            return Synapse(**values)

        manifest: dict[str, Any] = {
            "kind": "infer_manifest",
            "created_pairs": [],
            "completed_add": [],
            "completed_reinforce": [],
        }
        pending_operation: dict[str, Any] | None = None
        for raw in strategy_state.get("pending", []):
            if not isinstance(raw, str):
                continue
            value = decode_pending(raw)
            if value is None:
                continue
            if value.get("kind") == "infer_manifest":
                manifest = {
                    "kind": "infer_manifest",
                    "created_pairs": sorted(set(value.get("created_pairs", []))),
                    "completed_add": sorted(set(value.get("completed_add", []))),
                    "completed_reinforce": sorted(set(value.get("completed_reinforce", []))),
                }
            elif value.get("kind") in {"infer_add_pending", "infer_reinforce_pending"}:
                pending_operation = value

        def serialized_pending(operation: dict[str, Any] | None = None) -> list[str]:
            items = [json.dumps(manifest, sort_keys=True, separators=(",", ":"))]
            if operation is not None:
                items.append(json.dumps(operation, sort_keys=True, separators=(",", ":")))
            return items

        async def checkpoint(
            phase: str,
            cursor: str | None,
            operation: dict[str, Any] | None = None,
        ) -> None:
            if not dry_run:
                await self._checkpoint_progress(
                    phase,
                    cursor=cursor,
                    pending=serialized_pending(operation),
                    counters=counters,
                )

        async def current_synapses() -> list[Synapse]:
            return await self._all_synapses_paged()

        def find_pair(synapses: list[Synapse], key: str) -> list[Synapse]:
            return [syn for syn in synapses if pair_key(syn.source_id, syn.target_id) == key]

        # Resolve a saved operation before looking at fresh candidates. A pending
        # reinforcement is a full replacement snapshot, never a newly computed delta.
        if pending_operation is not None and not dry_run:
            operation_kind = str(pending_operation["kind"])
            key = str(pending_operation["pair_key"])
            snapshot = load_synapse(dict(pending_operation["synapse"]))
            did_apply = False
            if operation_kind == "infer_add_pending":
                matches = find_pair(await current_synapses(), key)
                exact_already_applied = any(syn.id == snapshot.id for syn in matches)
                if not matches:
                    try:
                        await self._storage.add_synapse(snapshot)
                        did_apply = True
                    except ValueError:
                        refreshed = find_pair(await current_synapses(), key)
                        exact_already_applied = any(syn.id == snapshot.id for syn in refreshed)
                        if not exact_already_applied and not refreshed:
                            raise
                if exact_already_applied:
                    did_apply = True
                if did_apply:
                    manifest["created_pairs"] = sorted(set(manifest["created_pairs"]) | {key})
                    counters["synapses_inferred"] = int(counters["synapses_inferred"]) + 1
                manifest["completed_add"] = sorted(set(manifest["completed_add"]) | {key})
                next_phase = "infer_add"
            else:
                try:
                    await self._storage.update_synapse(snapshot)
                    did_apply = True
                except ValueError:
                    logger.debug("Synapse reinforcement failed while replaying checkpoint")
                manifest["completed_reinforce"] = sorted(
                    set(manifest["completed_reinforce"]) | {key}
                )
                if did_apply:
                    counters["synapses_inferred"] = int(counters["synapses_inferred"]) + 1
                next_phase = "infer_reinforce"
            report.synapses_inferred = int(counters["synapses_inferred"])
            await checkpoint(next_phase, key)

        config = InferenceConfig(
            co_activation_threshold=self._config.infer_co_activation_threshold,
            co_activation_window_days=self._config.infer_window_days,
            max_inferences_per_run=self._config.infer_max_per_run,
        )
        window_start = reference_time - timedelta(days=config.co_activation_window_days)
        counts = await self._storage.get_co_activation_counts(
            since=window_start,
            min_count=config.co_activation_threshold,
        )
        if not counts:
            return

        # Normalize pair orientation before candidate selection so both tie
        # ordering and inferred Synapse identity are stable across retries.
        normalized_counts: dict[str, tuple[str, str, int, float]] = {}
        for a, b, count, strength in counts:
            left, right = sorted((a, b))
            key = pair_key(left, right)
            current = normalized_counts.get(key)
            count_row = (left, right, count, strength)
            if current is None or (count, strength) > (current[2], current[3]):
                normalized_counts[key] = count_row
        stable_counts = sorted(
            normalized_counts.values(),
            key=lambda item: (-item[2], pair_key(item[0], item[1])),
        )

        all_synapses = await current_synapses()
        existing_pairs: set[tuple[str, str]] = set()
        synapse_by_pair: dict[tuple[str, str], Synapse] = {}
        for synapse in all_synapses:
            existing_pairs.add((synapse.source_id, synapse.target_id))
            existing_pairs.add((synapse.target_id, synapse.source_id))
            synapse_by_pair[(synapse.source_id, synapse.target_id)] = synapse

        new_candidates, reinforce_candidates = identify_candidates(
            stable_counts, existing_pairs, config
        )

        def candidate_order(candidate: Any) -> tuple[int, str]:
            return (
                -candidate.co_activation_count,
                pair_key(candidate.neuron_a, candidate.neuron_b),
            )

        new_candidates.sort(key=candidate_order)
        reinforce_candidates.sort(key=candidate_order)

        if dry_run:
            report.synapses_inferred = len(new_candidates) + len(reinforce_candidates)
            return

        if strategy_state.get("phase") not in {
            "infer_add_pending",
            "infer_reinforce_pending",
            "infer_add",
            "infer_reinforce",
            "infer_tags",
            "infer_prune",
        }:
            await checkpoint("infer_scan", None)

        created_pairs = set(manifest["created_pairs"])
        completed_add = set(manifest["completed_add"])
        completed_reinforce = set(manifest["completed_reinforce"])
        if (
            strategy_state.get("phase") == "infer_prune"
            and strategy_state.get("cursor") == "completed"
        ):
            return

        for candidate in new_candidates:
            key = pair_key(candidate.neuron_a, candidate.neuron_b)
            if key in completed_add:
                continue
            snapshot = create_inferred_synapse(candidate)
            operation = {
                "kind": "infer_add_pending",
                "pair_key": key,
                "synapse": dump_synapse(snapshot),
            }
            await checkpoint("infer_add_pending", key, operation)
            matches = find_pair(all_synapses, key)
            exact_already_applied = any(synapse.id == snapshot.id for synapse in matches)
            did_apply = False
            if not matches:
                try:
                    await self._storage.add_synapse(snapshot)
                    did_apply = True
                    all_synapses.append(snapshot)
                except ValueError:
                    all_synapses = await current_synapses()
                    matches = find_pair(all_synapses, key)
                    exact_already_applied = any(synapse.id == snapshot.id for synapse in matches)
                    if not exact_already_applied and not matches:
                        raise
            if exact_already_applied:
                did_apply = True
            completed_add.add(key)
            manifest["completed_add"] = sorted(completed_add)
            if did_apply:
                created_pairs.add(key)
                manifest["created_pairs"] = sorted(created_pairs)
                counters["synapses_inferred"] = int(counters["synapses_inferred"]) + 1
                report.synapses_inferred = int(counters["synapses_inferred"])
            await checkpoint("infer_add", key)

        # A pair created earlier in this active inference run is still included
        # in tag generation, but must not be reinforced due to recomputation.
        for candidate in reinforce_candidates:
            key = pair_key(candidate.neuron_a, candidate.neuron_b)
            if key in created_pairs or key in completed_reinforce:
                continue
            existing_synapse = synapse_by_pair.get(
                (candidate.neuron_a, candidate.neuron_b)
            ) or synapse_by_pair.get((candidate.neuron_b, candidate.neuron_a))
            if existing_synapse is None:
                continue
            reinforced = existing_synapse.reinforce(delta=0.05)
            operation = {
                "kind": "infer_reinforce_pending",
                "pair_key": key,
                "synapse": dump_synapse(reinforced),
            }
            await checkpoint("infer_reinforce_pending", key, operation)
            did_apply = False
            try:
                await self._storage.update_synapse(reinforced)
                did_apply = True
                synapse_by_pair[(reinforced.source_id, reinforced.target_id)] = reinforced
            except ValueError:
                logger.debug("Synapse reinforcement failed")
            completed_reinforce.add(key)
            manifest["completed_reinforce"] = sorted(completed_reinforce)
            if did_apply:
                counters["synapses_inferred"] = int(counters["synapses_inferred"]) + 1
                report.synapses_inferred = int(counters["synapses_inferred"])
            await checkpoint("infer_reinforce", key)

        # Tag generation intentionally sees newly added pairs as well as the
        # usual reinforcement candidates, including those protected above.
        all_candidates = new_candidates + reinforce_candidates
        if all_candidates:
            neuron_ids = {
                neuron_id
                for candidate in all_candidates
                for neuron_id in (candidate.neuron_a, candidate.neuron_b)
            }
            neurons = await self._storage.get_neurons_batch(sorted(neuron_ids))
            content_map = {neuron_id: neuron.content for neuron_id, neuron in neurons.items()}
            fibers = sorted(await self._storage.get_fibers(limit=10000), key=lambda f: f.id)
            existing_tags: set[str] = set()
            for fiber in fibers:
                existing_tags |= fiber.tags
            assoc_tags = generate_associative_tags(all_candidates, content_map, existing_tags)
            normalizer = TagNormalizer()
            neuron_to_fiber_idx: dict[str, set[int]] = {}
            for index, fiber in enumerate(fibers):
                for neuron_id in fiber.neuron_ids:
                    neuron_to_fiber_idx.setdefault(neuron_id, set()).add(index)
            fiber_new_tags: dict[int, set[str]] = {}
            for assoc_tag in assoc_tags:
                normalized_tag = normalizer.normalize(assoc_tag.tag)
                affected: set[int] = set()
                for neuron_id in assoc_tag.source_neuron_ids:
                    affected |= neuron_to_fiber_idx.get(neuron_id, set())
                for index in affected:
                    fiber_new_tags.setdefault(index, set()).add(normalized_tag)
            for index, new_tags in sorted(fiber_new_tags.items()):
                fiber = fibers[index]
                if strategy_state.get("phase") == "infer_tags":
                    tag_cursor = strategy_state.get("cursor")
                    if tag_cursor is not None and fiber.id <= str(tag_cursor):
                        continue
                updated_auto_tags = fiber.auto_tags | new_tags
                if updated_auto_tags != fiber.auto_tags:
                    try:
                        await self._storage.update_fiber(
                            dc_replace(fiber, auto_tags=updated_auto_tags)
                        )
                    except Exception:
                        logger.debug("Associative tag update failed", exc_info=True)
                await checkpoint("infer_tags", fiber.id)

            for drift_report in normalizer.detect_drift(existing_tags):
                logger.info(
                    "Tag drift detected: %s → %s",
                    drift_report.variants,
                    drift_report.canonical,
                )

        if strategy_state.get("phase") != "infer_prune":
            await checkpoint("infer_prune", None)
        pruned = await self._storage.prune_co_activations(older_than=window_start)
        counters["co_activations_pruned"] = pruned
        report.co_activations_pruned = pruned
        await checkpoint("infer_prune", "completed")

    async def _enrich(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run enrichment: transitive closure + cross-cluster linking."""
        import hashlib
        import json
        import logging

        from surreal_memory.core.synapse import Direction, Synapse, SynapseType
        from surreal_memory.engine.consolidation_progress import ConsolidationProgressError
        from surreal_memory.engine.enrichment import enrich

        logger = logging.getLogger(__name__)

        def encode_synapse(synapse: Synapse) -> dict[str, Any]:
            return {
                "id": synapse.id,
                "source_id": synapse.source_id,
                "target_id": synapse.target_id,
                "type": synapse.type.value,
                "weight": synapse.weight,
                "direction": synapse.direction.value,
                "metadata": synapse.metadata,
                "reinforced_count": synapse.reinforced_count,
                "last_activated": (
                    synapse.last_activated.isoformat() if synapse.last_activated else None
                ),
                "created_at": synapse.created_at.isoformat(),
            }

        def decode_synapse(payload: dict[str, Any]) -> Synapse:
            return Synapse(
                id=str(payload["id"]),
                source_id=str(payload["source_id"]),
                target_id=str(payload["target_id"]),
                type=SynapseType(payload["type"]),
                weight=float(payload["weight"]),
                direction=Direction(payload["direction"]),
                metadata=dict(payload.get("metadata") or {}),
                reinforced_count=int(payload.get("reinforced_count", 0)),
                last_activated=(
                    datetime.fromisoformat(payload["last_activated"])
                    if payload.get("last_activated")
                    else None
                ),
                created_at=datetime.fromisoformat(payload["created_at"]),
            )

        async def source_fingerprint(excluded_ids: set[str]) -> str:
            causal = await self._storage.get_synapses(type=SynapseType.CAUSED_BY)
            related = await self._storage.get_synapses_paged(type=SynapseType.RELATED_TO)
            fibers = await self._storage.get_fibers(limit=10000)
            payload = {
                "causal": [
                    encode_synapse(item)
                    for item in sorted(causal, key=lambda edge: edge.id)
                    if item.id not in excluded_ids
                ],
                "related": [
                    encode_synapse(item)
                    for item in sorted(related, key=lambda edge: edge.id)
                    if item.id not in excluded_ids
                ],
                "fibers": [
                    {
                        "id": fiber.id,
                        "anchor_neuron_id": fiber.anchor_neuron_id,
                        "neuron_ids": sorted(fiber.neuron_ids),
                        "tags": sorted(fiber.tags),
                        "salience": fiber.salience,
                    }
                    for fiber in sorted(fibers, key=lambda item: item.id)
                ],
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        result = await enrich(self._storage)
        all_synapses = sorted(
            result.transitive_synapses + result.cross_cluster_synapses,
            key=lambda synapse: synapse.id,
        )
        if dry_run:
            report.synapses_enriched = len(all_synapses)
            return

        # Preserve the simple adapter path for non-durable runs.
        if self._progress_session is None:
            for synapse in all_synapses:
                try:
                    await self._storage.add_synapse(synapse)
                    report.synapses_enriched += 1
                except ValueError:
                    logger.debug("Enriched synapse already exists, skipping")
            await self._reactivate_dormant(report, dry_run)
            return

        state = self._strategy_progress_state()
        cursor_value = state.get("cursor")
        cursor = (
            str(cursor_value)
            if cursor_value is not None and state.get("phase") != "enrich_complete"
            else None
        )
        pending = state.get("pending")
        if pending:
            if not isinstance(pending, (list, tuple)) or len(pending) != 1:
                raise ConsolidationProgressError("invalid enrich pending checkpoint")
            try:
                manifest = json.loads(pending[0])
                if manifest.get("kind") != "enrich_synapses" or manifest.get("version") != 1:
                    raise ValueError("unsupported pending manifest")
                raw_synapses = manifest["synapses"]
                if not isinstance(raw_synapses, list):
                    raise ValueError("synapses is not a list")
                synapses = [decode_synapse(item) for item in raw_synapses]
                excluded_ids = {synapse.id for synapse in synapses}
                current_fingerprint = await source_fingerprint(excluded_ids)
                if current_fingerprint != manifest.get("source_fingerprint"):
                    raise ConsolidationProgressError(
                        "enrich source data changed after its pending checkpoint"
                    )
            except ConsolidationProgressError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ConsolidationProgressError("invalid enrich pending snapshot") from exc
            serialized = pending[0]
        else:
            excluded_ids = {synapse.id for synapse in all_synapses}
            manifest = {
                "kind": "enrich_synapses",
                "version": 1,
                "source_fingerprint": await source_fingerprint(excluded_ids),
                "synapses": [encode_synapse(synapse) for synapse in all_synapses],
            }
            serialized = json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            synapses = all_synapses
            await self._checkpoint_progress(
                "enrich_pending",
                cursor=cursor,
                pending=[serialized],
                counters={"synapses_enriched": 0},
            )

        raw_counters = state.get("counters")
        counters = raw_counters if isinstance(raw_counters, dict) else {}
        enriched = int(counters.get("synapses_enriched", 0))
        for synapse in synapses:
            if cursor is not None and synapse.id <= cursor:
                continue
            await self._check_progress_budget()
            existing = await self._storage.get_synapse(synapse.id)
            if existing is not None:
                if existing != synapse:
                    raise ConsolidationProgressError(
                        f"enrich synapse {synapse.id!r} changed after its checkpoint"
                    )
                # The write may have committed before its acknowledgment/checkpoint.
                enriched += 1
            else:
                try:
                    await self._storage.add_synapse(synapse)
                    enriched += 1
                except ValueError:
                    logger.debug("Enriched synapse already exists, skipping")
            cursor = synapse.id
            counters = {"synapses_enriched": enriched}
            await self._checkpoint_progress(
                "enrich_apply",
                cursor=cursor,
                pending=[serialized],
                counters=counters,
            )

        await self._checkpoint_progress(
            "enrich_complete",
            cursor=cursor,
            pending=[],
            counters={"synapses_enriched": enriched},
        )
        report.synapses_enriched = enriched
        # Reactivate dormant neurons (access_frequency=0) to prevent permanent dormancy.
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
        """Run dream exploration with replayable synapse-add snapshots."""
        import json
        from dataclasses import asdict
        from datetime import datetime
        from enum import Enum

        from surreal_memory.core.synapse import Direction, SynapseType
        from surreal_memory.engine.dream import dream

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        state = self._strategy_progress_state() if not dry_run else {}
        saved_counters = state.get("counters") or {}
        counters: dict[str, int | float] = {
            "dream_synapses_created": int(saved_counters.get("dream_synapses_created", 0))
        }
        report.dream_synapses_created = int(counters["dream_synapses_created"])
        if state.get("phase") == "completed":
            return

        def encode(value: object) -> str:
            if isinstance(value, Enum):
                return str(value.value)
            if isinstance(value, datetime):
                return value.isoformat()
            raise TypeError(f"unsupported dream checkpoint value: {type(value).__name__}")

        def dump_synapse(synapse: Synapse) -> str:
            return json.dumps(
                asdict(synapse), default=encode, sort_keys=True, separators=(",", ":")
            )

        def load_synapse(payload: str) -> Synapse:
            values = json.loads(payload)
            values["type"] = SynapseType(values["type"])
            values["direction"] = Direction(values["direction"])
            if values.get("last_activated") is not None:
                values["last_activated"] = datetime.fromisoformat(values["last_activated"])
            values["created_at"] = datetime.fromisoformat(values["created_at"])
            return Synapse(**values)

        def manifest_from(raw_pending: object) -> dict[str, object] | None:
            if not isinstance(raw_pending, list):
                return None
            for raw in raw_pending:
                if not isinstance(raw, str):
                    continue
                try:
                    item = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(item, dict) and item.get("kind") == "dream_manifest":
                    return item
            return None

        manifest = manifest_from(state.get("pending"))
        pending_op: dict[str, object] | None = None
        for raw in state.get("pending", []) if isinstance(state.get("pending"), list) else []:
            if not isinstance(raw, str):
                continue
            try:
                item = json.loads(raw)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("kind") == "dream_add":
                pending_op = item

        if dry_run:
            result = await dream(self._storage, brain.config)
            report.dream_synapses_created = len(result.synapses_created)
            return

        if manifest is None:
            await self._check_progress_budget()
            result = await dream(self._storage, brain.config)
            manifest = {
                "kind": "dream_manifest",
                "synapses": [dump_synapse(synapse) for synapse in result.synapses_created],
                "completed": [],
            }
            await self._checkpoint_progress(
                "dream_apply",
                pending=[json.dumps(manifest, sort_keys=True, separators=(",", ":"))],
                counters=counters,
            )

        raw_completed = manifest.get("completed", [])
        completed = (
            {str(item) for item in raw_completed} if isinstance(raw_completed, list) else set()
        )
        raw_synapses = manifest.get("synapses", [])
        synapse_payloads = (
            [str(item) for item in raw_synapses] if isinstance(raw_synapses, list) else []
        )

        async def commit(synapse: Synapse) -> bool:
            try:
                await self._storage.add_synapse(synapse)
                return True
            except ValueError:
                existing = await self._storage.get_synapses(source_id=synapse.source_id)
                return any(item.id == synapse.id for item in existing)

        async def checkpoint(
            phase: str, cursor: str | None, op: dict[str, str] | None = None
        ) -> None:
            pending = [json.dumps(manifest, sort_keys=True, separators=(",", ":"))]
            if op is not None:
                pending.append(json.dumps(op, sort_keys=True, separators=(",", ":")))
            await self._checkpoint_progress(
                phase,
                cursor=cursor,
                pending=pending,
                counters=counters,
            )

        if pending_op is not None:
            snapshot = load_synapse(str(pending_op["synapse"]))
            did_apply = await commit(snapshot)
            if snapshot.id not in completed:
                completed.add(snapshot.id)
                manifest["completed"] = sorted(completed)
                if did_apply:
                    counters["dream_synapses_created"] = int(counters["dream_synapses_created"]) + 1
                    report.dream_synapses_created = int(counters["dream_synapses_created"])
            await checkpoint("dream_apply", snapshot.id)

        for payload in synapse_payloads:
            synapse = load_synapse(payload)
            if synapse.id in completed:
                continue
            await self._check_progress_budget()
            op = {"kind": "dream_add", "synapse": payload}
            await checkpoint("dream_pending", synapse.id, op)
            did_apply = await commit(synapse)
            completed.add(synapse.id)
            manifest["completed"] = sorted(completed)
            if did_apply:
                counters["dream_synapses_created"] = int(counters["dream_synapses_created"]) + 1
                report.dream_synapses_created = int(counters["dream_synapses_created"])
            await checkpoint("dream_apply", synapse.id)

        await self._checkpoint_progress(
            "completed",
            cursor="completed",
            pending=[],
            counters=counters,
        )

    async def _replay(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Plan replay writes first, then apply exact snapshots with progress receipts."""
        import json
        from dataclasses import asdict
        from datetime import datetime
        from enum import Enum
        from typing import cast

        from surreal_memory.core.synapse import Direction, SynapseType
        from surreal_memory.engine.hippocampal_replay import hippocampal_replay

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        state = self._strategy_progress_state() if not dry_run else {}
        saved_counters = state.get("counters") or {}
        counters: dict[str, int | float] = {
            "replay_episodes": int(saved_counters.get("replay_episodes", 0)),
            "replay_ltp": int(saved_counters.get("replay_ltp", 0)),
            "replay_ltd": int(saved_counters.get("replay_ltd", 0)),
        }
        report.extra["replay_episodes"] = int(counters["replay_episodes"])
        report.extra["replay_ltp"] = int(counters["replay_ltp"])
        report.extra["replay_ltd"] = int(counters["replay_ltd"])
        if not dry_run and state.get("phase") == "completed":
            return

        def encode(value: object) -> str:
            if isinstance(value, Enum):
                return str(value.value)
            if isinstance(value, datetime):
                return value.isoformat()
            raise TypeError(f"unsupported replay checkpoint value: {type(value).__name__}")

        def dump_synapse(synapse: Synapse) -> str:
            return json.dumps(
                asdict(synapse), default=encode, sort_keys=True, separators=(",", ":")
            )

        def load_synapse(payload: str) -> Synapse:
            values = json.loads(payload)
            values["type"] = SynapseType(values["type"])
            values["direction"] = Direction(values["direction"])
            if values.get("last_activated") is not None:
                values["last_activated"] = datetime.fromisoformat(values["last_activated"])
            values["created_at"] = datetime.fromisoformat(values["created_at"])
            return Synapse(**values)

        def pending_manifest(raw_pending: object) -> dict[str, Any] | None:
            if not isinstance(raw_pending, list):
                return None
            for raw in raw_pending:
                if not isinstance(raw, str):
                    continue
                try:
                    value = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(value, dict) and value.get("kind") == "replay_manifest":
                    return value
            return None

        manifest = pending_manifest(state.get("pending"))
        if dry_run:
            result = await hippocampal_replay(self._storage, brain.config, dry_run=True)
            report.extra["replay_episodes"] = result.episodes_replayed
            report.extra["replay_ltp"] = result.synapses_strengthened
            report.extra["replay_ltd"] = result.synapses_weakened
            return

        if manifest is None:
            await self._check_progress_budget()

            class _ReplayPlanner:
                """Collect replay replacements without allowing the helper to write."""

                def __init__(self, storage: Any) -> None:
                    self._storage = storage
                    self.before: dict[str, Synapse] = {}
                    self.virtual: dict[str, Synapse] = {}
                    self.operations: list[dict[str, str]] = []
                    self.active_episode_id = ""

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._storage, name)

                async def find_fibers(self, **kwargs: Any) -> list[Any]:
                    fibers = await self._storage.find_fibers(**kwargs)
                    planner = self

                    class _FiberView:
                        def __init__(self, fiber: Any) -> None:
                            self._fiber = fiber

                        def __getattr__(self, name: str) -> Any:
                            return getattr(self._fiber, name)

                        @property
                        def neuron_ids(self) -> Any:
                            planner.active_episode_id = str(self._fiber.id)
                            return self._fiber.neuron_ids

                    return [_FiberView(fiber) for fiber in fibers]

                async def get_synapses(self, **kwargs: Any) -> list[Synapse]:
                    synapses = cast("list[Synapse]", await self._storage.get_synapses(**kwargs))
                    for item in synapses:
                        self.before.setdefault(item.id, item)
                    return [self.virtual.get(item.id, item) for item in synapses]

                async def update_synapse(self, synapse: Synapse) -> None:
                    original = self.virtual.get(synapse.id) or self.before.get(synapse.id)
                    if original is None:
                        rows = cast(
                            "list[Synapse]",
                            await self._storage.get_synapses(source_id=synapse.source_id),
                        )
                        original = next((item for item in rows if item.id == synapse.id), None)
                    if original is None:
                        return
                    self.operations.append(
                        {
                            "before": dump_synapse(original),
                            "after": dump_synapse(synapse),
                            "kind": "ltp" if synapse.weight > original.weight else "ltd",
                            "episode_id": self.active_episode_id,
                        }
                    )
                    self.virtual[synapse.id] = synapse

            planner = _ReplayPlanner(self._storage)
            await hippocampal_replay(cast("Any", planner), brain.config, dry_run=False)
            manifest = {
                "kind": "replay_manifest",
                "operations": planner.operations,
                "completed": [],
                "episodes": [],
            }
            await self._checkpoint_progress(
                "replay_apply",
                pending=[json.dumps(manifest, sort_keys=True, separators=(",", ":"))],
                counters=counters,
            )

        completed = {str(item) for item in manifest.get("completed", [])}
        replayed_episodes = {str(item) for item in manifest.get("episodes", [])}

        async def checkpoint(
            phase: str, cursor: str | None, op: dict[str, str] | None = None
        ) -> None:
            pending = [json.dumps(manifest, sort_keys=True, separators=(",", ":"))]
            if op is not None:
                pending.append(json.dumps(op, sort_keys=True, separators=(",", ":")))
            await self._checkpoint_progress(
                phase,
                cursor=cursor,
                pending=pending,
                counters=counters,
            )

        for operation_index, op in enumerate(manifest.get("operations", [])):
            if not isinstance(op, dict):
                continue
            before = load_synapse(str(op["before"]))
            after = load_synapse(str(op["after"]))
            operation_id = f"{operation_index}:{before.id}"
            if operation_id in completed:
                continue

            await self._check_progress_budget()
            await checkpoint("replay_pending", operation_id, op)
            rows = await self._storage.get_synapses(source_id=before.source_id)
            current = next((item for item in rows if item.id == before.id), None)
            applied = False
            if current is not None and dump_synapse(current) == dump_synapse(after):
                applied = True
            elif current is not None and dump_synapse(current) == dump_synapse(before):
                try:
                    await self._storage.update_synapse(after)
                    applied = True
                except Exception:
                    # Replay has historically treated individual weight-update
                    # failures as non-critical; leave this operation uncounted.
                    applied = False

            completed.add(operation_id)
            manifest["completed"] = sorted(completed)
            if applied:
                if op.get("kind") == "ltp":
                    counters["replay_ltp"] = int(counters["replay_ltp"]) + 1
                else:
                    counters["replay_ltd"] = int(counters["replay_ltd"]) + 1
                episode_id = str(op.get("episode_id") or before.source_id)
                replayed_episodes.add(episode_id)
                manifest["episodes"] = sorted(replayed_episodes)
            report.extra["replay_episodes"] = len(replayed_episodes)
            report.extra["replay_ltp"] = int(counters["replay_ltp"])
            report.extra["replay_ltd"] = int(counters["replay_ltd"])
            await checkpoint("replay_apply", operation_id)

        counters["replay_episodes"] = len(replayed_episodes)
        await self._checkpoint_progress(
            "completed",
            cursor="completed",
            pending=[],
            counters=counters,
        )

    async def _schema(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Plan schema writes, then apply exact records with durable receipts."""
        import json
        from dataclasses import asdict
        from datetime import datetime
        from enum import Enum
        from typing import cast

        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.core.synapse import Direction, Synapse, SynapseType
        from surreal_memory.engine.schema_assimilation import batch_schema_assimilation

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        state = self._strategy_progress_state() if not dry_run else {}
        saved_counters = state.get("counters") or {}
        counters: dict[str, int | float] = {
            "schemas_created": int(saved_counters.get("schemas_created", 0))
        }
        report.extra["schemas_created"] = int(counters["schemas_created"])
        if not dry_run and state.get("phase") == "completed":
            return

        def encode(value: object) -> str:
            if isinstance(value, Enum):
                return str(value.value)
            if isinstance(value, datetime):
                return value.isoformat()
            raise TypeError(f"unsupported schema checkpoint value: {type(value).__name__}")

        def dump_record(record: Neuron | Synapse) -> str:
            return json.dumps(asdict(record), default=encode, sort_keys=True, separators=(",", ":"))

        def load_record(kind: str, payload: str) -> Neuron | Synapse:
            values = json.loads(payload)
            if kind == "neuron":
                values["type"] = NeuronType(values["type"])
                if values.get("created_at") is not None:
                    values["created_at"] = datetime.fromisoformat(values["created_at"])
                return Neuron(**values)
            values["type"] = SynapseType(values["type"])
            values["direction"] = Direction(values["direction"])
            if values.get("last_activated") is not None:
                values["last_activated"] = datetime.fromisoformat(values["last_activated"])
            values["created_at"] = datetime.fromisoformat(values["created_at"])
            return Synapse(**values)

        def read_manifest(raw_pending: object) -> dict[str, Any] | None:
            if not isinstance(raw_pending, list):
                return None
            for raw in raw_pending:
                if not isinstance(raw, str):
                    continue
                try:
                    value = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(value, dict) and value.get("kind") == "schema_manifest":
                    return value
            return None

        manifest = read_manifest(state.get("pending"))
        if dry_run:
            count = await batch_schema_assimilation(self._storage, brain.config, dry_run=True)
            report.extra["schemas_created"] = count
            return

        if manifest is None:
            await self._check_progress_budget()

            class _SchemaPlanner:
                """Capture creates while preserving the helper's read-only scan."""

                def __init__(self, storage: Any) -> None:
                    self._storage = storage
                    self.operations: list[dict[str, str]] = []

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._storage, name)

                async def add_neuron(self, neuron: Neuron) -> str:
                    self.operations.append({"kind": "neuron", "record": dump_record(neuron)})
                    return neuron.id

                async def add_synapse(self, synapse: Synapse) -> str:
                    self.operations.append({"kind": "synapse", "record": dump_record(synapse)})
                    return synapse.id

            planner = _SchemaPlanner(self._storage)
            planned_count = await batch_schema_assimilation(
                cast("Any", planner), brain.config, dry_run=False
            )
            manifest = {
                "kind": "schema_manifest",
                "operations": planner.operations,
                "completed": [],
                "schemas_created": planned_count,
                "counted_schema_ids": [],
            }
            await self._checkpoint_progress(
                "schema_apply",
                pending=[json.dumps(manifest, sort_keys=True, separators=(",", ":"))],
                counters=counters,
            )

        completed = {str(item) for item in manifest.get("completed", [])}
        counted_schema_ids = {str(item) for item in manifest.get("counted_schema_ids", [])}

        async def checkpoint(
            phase: str, cursor: str | None, op: dict[str, str] | None = None
        ) -> None:
            pending = [json.dumps(manifest, sort_keys=True, separators=(",", ":"))]
            if op is not None:
                pending.append(json.dumps(op, sort_keys=True, separators=(",", ":")))
            await self._checkpoint_progress(
                phase,
                cursor=cursor,
                pending=pending,
                counters=counters,
            )

        for op in manifest.get("operations", []):
            if not isinstance(op, dict):
                continue
            kind = str(op["kind"])
            record = load_record(kind, str(op["record"]))
            key = record.id
            if key in completed:
                continue

            await self._check_progress_budget()
            await checkpoint("schema_pending", key, op)
            applied = False
            if isinstance(record, Neuron):
                existing = await self._storage.find_neurons(type=NeuronType.SCHEMA, limit=200)
                if any(item.id == record.id for item in existing):
                    applied = True
                else:
                    try:
                        await self._storage.add_neuron(record)
                        applied = True
                    except ValueError:
                        refreshed = await self._storage.find_neurons(
                            type=NeuronType.SCHEMA, limit=200
                        )
                        applied = any(item.id == record.id for item in refreshed)
                if applied and key not in counted_schema_ids:
                    counted_schema_ids.add(key)
                    counters["schemas_created"] = int(counters["schemas_created"]) + 1
                    manifest["counted_schema_ids"] = sorted(counted_schema_ids)
            else:
                synapse_rows = await self._storage.get_synapses(source_id=record.source_id)
                if any(
                    item.id == record.id
                    or (item.source_id == record.source_id and item.target_id == record.target_id)
                    for item in synapse_rows
                ):
                    applied = True
                else:
                    try:
                        await self._storage.add_synapse(record)
                        applied = True
                    except ValueError:
                        refreshed_synapses = await self._storage.get_synapses(
                            source_id=record.source_id
                        )
                        applied = any(
                            item.id == record.id
                            or (
                                item.source_id == record.source_id
                                and item.target_id == record.target_id
                            )
                            for item in refreshed_synapses
                        )

            completed.add(key)
            manifest["completed"] = sorted(completed)
            report.extra["schemas_created"] = int(counters["schemas_created"])
            await checkpoint("schema_apply", key)

        await self._checkpoint_progress(
            "completed",
            cursor="completed",
            pending=[],
            counters=counters,
        )

    async def _interference(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Run the read-only interference scan with a durable result checkpoint."""
        from surreal_memory.engine.interference import batch_interference_scan

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        state = self._strategy_progress_state() if not dry_run else {}
        counters = state.get("counters") or {}
        if not dry_run and state.get("phase") == "completed":
            report.extra["interference_fan_effects"] = int(
                counters.get("interference_fan_effects", 0)
            )
            return

        if not dry_run:
            await self._check_progress_budget()
        result = await batch_interference_scan(
            self._storage,
            brain.config,
            dry_run=dry_run,
        )
        report.extra["interference_fan_effects"] = result.fan_effects_flagged
        if not dry_run:
            await self._checkpoint_progress(
                "completed",
                cursor="completed",
                pending=[],
                counters={"interference_fan_effects": result.fan_effects_flagged},
            )

    async def _learn_habits(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Learn habits with resumable candidate plans and atomic effect receipts."""
        import json
        import logging

        from surreal_memory.engine.sequence_mining import learn_habits

        logger = logging.getLogger(__name__)

        class _HabitCheckpointError(RuntimeError):
            pass

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain or dry_run:
            return

        progress_session = self._progress_session
        raw_run_id = progress_session.state.get("run_id") if progress_session else None
        run_id = str(raw_run_id) if raw_run_id else None
        state = self._strategy_progress_state()
        persisted = state.get("counters") or {}
        phase = str(state.get("phase") or "")
        action_count = int(
            persisted.get(
                "action_habits_learned",
                persisted.get("habits_learned", 0)
                if phase
                in {
                    "habits_action_completed",
                    "habits_query_pending",
                    "habits_query_completed",
                    "habits_tool_pending",
                    "completed",
                }
                else 0,
            )
        )
        tool_count = int(persisted.get("tool_habits_learned", 0))
        counters: dict[str, int | float] = {
            "habits_learned": action_count + tool_count,
            "action_habits_learned": action_count,
            "tool_habits_learned": tool_count,
            "action_events_pruned": int(persisted.get("action_events_pruned", 0)),
            "query_patterns_learned": int(persisted.get("query_patterns_learned", 0)),
        }
        report.habits_learned = int(counters["habits_learned"])
        report.action_events_pruned = int(counters["action_events_pruned"])
        report.query_patterns_learned = int(counters["query_patterns_learned"])

        async def checkpoint_manifest(
            stage: str,
            manifest: list[dict[str, object]],
            cursor: int,
        ) -> None:
            if run_id is None or self._progress_session is None:
                return
            phase_name = f"habits_{stage}_pending"
            payload = {
                "version": 1,
                "run_id": run_id,
                "stage": stage,
                "candidates": manifest,
            }
            pending = [json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)]
            if stage == "action":
                counters["action_habits_learned"] = cursor
                counters["habits_learned"] = int(counters["action_habits_learned"]) + int(
                    counters["tool_habits_learned"]
                )
            elif stage == "tool":
                counters["tool_habits_learned"] = cursor
                counters["habits_learned"] = int(counters["action_habits_learned"]) + cursor
            elif stage == "query":
                counters["query_patterns_learned"] = cursor
            report.habits_learned = int(counters["habits_learned"])
            report.query_patterns_learned = int(counters["query_patterns_learned"])
            try:
                await self._checkpoint_progress(
                    phase_name,
                    cursor=f"candidate:{cursor}",
                    pending=pending,
                    counters=counters,
                )
            except ConsolidationPausedError:
                raise
            except Exception as error:
                raise _HabitCheckpointError from error

        def resume_plan(stage: str) -> tuple[list[dict[str, object]] | None, int]:
            current = self._strategy_progress_state()
            if current.get("phase") != f"habits_{stage}_pending":
                return None, 0
            pending = current.get("pending") or []
            if not pending:
                return None, 0
            try:
                payload = json.loads(str(pending[0]))
                if (
                    payload.get("version") != 1
                    or payload.get("run_id") != run_id
                    or payload.get("stage") != stage
                ):
                    return None, 0
                candidates = payload.get("candidates")
                if not isinstance(candidates, list):
                    return None, 0
                cursor_value = str(current.get("cursor") or "candidate:0")
                cursor = int(cursor_value.partition(":")[2])
                return candidates, max(0, min(cursor, len(candidates)))
            except (TypeError, ValueError, json.JSONDecodeError):
                return None, 0

        async def run_stage(stage: str, runner: Callable[..., Awaitable[Any]]) -> Any:
            await self._check_progress_budget()
            manifest, cursor = resume_plan(stage)
            if manifest is None:
                await self._checkpoint_progress(
                    f"habits_{stage}_pending",
                    cursor="candidate:0",
                    pending=[],
                    counters=counters,
                )

            async def save_candidate_progress(
                candidate_manifest: list[dict[str, object]],
                next_cursor: int,
            ) -> None:
                await checkpoint_manifest(stage, candidate_manifest, next_cursor)

            return await runner(
                self._storage,
                brain.config,
                reference_time,
                run_id=run_id,
                resume_manifest=manifest,
                resume_cursor=cursor,
                on_progress=save_candidate_progress if run_id else None,
            )

        if phase == "completed":
            return

        if phase not in {
            "habits_action_completed",
            "habits_query_pending",
            "habits_query_completed",
            "habits_tool_pending",
            "completed",
        }:
            try:
                _, habit_report = await run_stage("action", learn_habits)
                manifest, cursor = resume_plan("action")
                action_done = (
                    len(manifest) if manifest is not None else int(habit_report.habits_learned)
                )
                counters["action_habits_learned"] = max(
                    int(counters["action_habits_learned"]), action_done
                )
                counters["habits_learned"] = int(counters["action_habits_learned"]) + int(
                    counters["tool_habits_learned"]
                )
                counters["action_events_pruned"] = int(habit_report.action_events_pruned)
                report.habits_learned = int(counters["habits_learned"])
                report.action_events_pruned = int(counters["action_events_pruned"])
            except (ConsolidationPausedError, _HabitCheckpointError):
                raise
            except Exception:
                logger.debug("Habit learning failed (non-critical)", exc_info=True)
            await self._checkpoint_progress(
                "habits_action_completed",
                cursor="completed",
                pending=[],
                counters=counters,
            )
            phase = "habits_action_completed"

        if phase not in {"habits_query_completed", "habits_tool_pending", "completed"}:
            try:
                from surreal_memory.engine.query_pattern_mining import learn_query_patterns

                qp_report = await run_stage("query", learn_query_patterns)
                manifest, _ = resume_plan("query")
                counters["query_patterns_learned"] = max(
                    int(counters["query_patterns_learned"]),
                    len(manifest) if manifest is not None else int(qp_report.patterns_learned),
                )
                report.query_patterns_learned = int(counters["query_patterns_learned"])
            except (ConsolidationPausedError, _HabitCheckpointError):
                raise
            except Exception:
                logger.debug("Query pattern learning failed (non-critical)", exc_info=True)
            await self._checkpoint_progress(
                "habits_query_completed",
                cursor="completed",
                pending=[],
                counters=counters,
            )
            phase = "habits_query_completed"

        if phase != "completed":
            try:
                from surreal_memory.engine.sequence_mining import learn_tool_habits

                _, tool_report = await run_stage("tool", learn_tool_habits)
                manifest, _ = resume_plan("tool")
                counters["tool_habits_learned"] = max(
                    int(counters["tool_habits_learned"]),
                    len(manifest) if manifest is not None else int(tool_report.habits_learned),
                )
                counters["habits_learned"] = int(counters["action_habits_learned"]) + int(
                    counters["tool_habits_learned"]
                )
                report.habits_learned = int(counters["habits_learned"])
            except (ConsolidationPausedError, _HabitCheckpointError):
                raise
            except Exception:
                logger.debug("Tool-usage habit learning failed (non-critical)", exc_info=True)

        await self._checkpoint_progress(
            "completed",
            cursor="completed",
            pending=[],
            counters=counters,
        )

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
        import json
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

        # Paginate through neurons to collect the bounded anchor window.
        batch_size = 5000
        offset = 0
        all_anchors: list[Neuron] = []
        while True:
            # Anchors are selected on metadata alone, so skip the embedding vector
            # — it is ~4-8 KB/row and only inflates the response.
            batch = await self._storage.find_neurons(
                limit=batch_size, offset=offset, ephemeral=False, include_embedding=False
            )
            if not batch:
                break
            all_anchors.extend(n for n in batch if n.metadata.get("is_anchor", False))
            offset += len(batch)
            if len(batch) < batch_size:
                break

        # Report the census input, even when this pass resumes a bounded window.
        anchors_total = len(all_anchors)
        report.extra["dedup_anchors_total"] = anchors_total

        if anchors_total < 2:
            report.extra["dedup_anchors_scanned"] = anchors_total
            return

        progress_state = self._strategy_progress_state()
        progress_phase = progress_state.get("phase")
        saved_counters = progress_state.get("counters") or {}
        resumable_phase = progress_phase in {"dedup_pairs", "dedup_window_complete"}
        if resumable_phase:
            report.duplicates_found = int(
                saved_counters.get("duplicates_found", report.duplicates_found)
            )
            report.new_alias_links = int(
                saved_counters.get("new_alias_links", report.new_alias_links)
            )
            report.alias_links_existing = int(
                saved_counters.get("alias_links_existing", report.alias_links_existing)
            )
            for key in (
                "alias_checks_failed",
                "alias_writes_failed",
                "alias_pairs_skipped_invalid",
            ):
                value = int(saved_counters.get(key, report.extra.get(key, 0)))
                if value:
                    report.extra[key] = value

        if progress_phase == "dedup_window_complete":
            report.extra["dedup_resumed_checkpoint"] = "dedup_window_complete"
            return

        # The cursor describes a stable anchor-ID window and the next unprocessed
        # outer anchor. This bounds replay after cancellation while alias edge IDs
        # make re-applying the last uncommitted batch safe.
        cap = max(2, int(self._config.dedup_max_anchors))
        cursor = 0
        next_outer_index = 0
        cursor_payload: dict[str, Any]
        if progress_phase == "dedup_pairs":
            try:
                cursor_payload = json.loads(str(progress_state.get("cursor") or "{}"))
                cursor = int(cursor_payload["window_start"])
                next_outer_index = int(cursor_payload["next_i"])
                saved_pending = [str(value) for value in (progress_state.get("pending") or [])]
                window_anchor_ids = [
                    value.removeprefix("window:")
                    for value in saved_pending
                    if value.startswith("window:")
                ]
                resumed_seen_ids = [
                    value.removeprefix("seen:")
                    for value in saved_pending
                    if value.startswith("seen:")
                ]
                if not window_anchor_ids or not 0 <= next_outer_index <= len(window_anchor_ids):
                    raise ValueError("dedup outer cursor is outside the saved anchor window")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "dedup progress checkpoint is malformed; refusing to skip its window"
                ) from exc
            anchors_by_id = {str(anchor.id): anchor for anchor in all_anchors}
            anchors: list[Neuron | None] = [
                anchors_by_id.get(anchor_id) for anchor_id in window_anchor_ids
            ]
        else:
            if anchors_total > cap:
                cursor = await self._dedup_cursor(anchors_total)
                window = all_anchors[cursor : cursor + cap]
                if len(window) < cap:
                    # Wrap around so the window keeps its size at the end of the list.
                    window += all_anchors[: cap - len(window)]
                all_anchors = window
                report.extra["dedup_anchors_truncated"] = True
                report.extra["dedup_window_start"] = cursor
                # INFO, not WARNING: truncation is steady state for a brain above
                # the configured cap; the report carries that limitation.
                logger.info(
                    "Dedup census truncated: %d anchors present, only the first %d are compared. "
                    "The reported duplicate count covers that window, not the whole brain.",
                    anchors_total,
                    cap,
                )
            window_anchor_ids = [str(anchor.id) for anchor in all_anchors]
            anchors = list(all_anchors)
            cursor_payload = {"window_start": cursor, "next_i": 0}
            if not dry_run:
                await self._checkpoint_progress(
                    "dedup_pairs",
                    cursor=json.dumps(cursor_payload, separators=(",", ":")),
                    pending=[f"window:{anchor_id}" for anchor_id in window_anchor_ids],
                    counters={
                        "duplicates_found": report.duplicates_found,
                        "new_alias_links": report.new_alias_links,
                        "alias_links_existing": report.alias_links_existing,
                    },
                )

        report.extra["dedup_anchors_scanned"] = sum(anchor is not None for anchor in anchors)

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
        seen: set[str] = set(resumed_seen_ids) if progress_phase == "dedup_pairs" else set()
        created_links = report.new_alias_links
        existing_links = report.alias_links_existing
        checks_failed = int(report.extra.get("alias_checks_failed", 0))
        writes_failed = int(report.extra.get("alias_writes_failed", 0))
        skipped_invalid = int(report.extra.get("alias_pairs_skipped_invalid", 0))
        first_failure: BaseException | None = None
        for i, anchor_a in enumerate(anchors):
            if i < next_outer_index:
                continue
            if not dry_run and i > next_outer_index and i % 10 == 0:
                cursor_payload["next_i"] = i
                try:
                    await self._checkpoint_progress(
                        "dedup_pairs",
                        cursor=json.dumps(cursor_payload, separators=(",", ":")),
                        pending=[f"window:{anchor_id}" for anchor_id in window_anchor_ids]
                        + [f"seen:{anchor_id}" for anchor_id in sorted(seen)],
                        counters={
                            "duplicates_found": report.duplicates_found,
                            "new_alias_links": created_links,
                            "alias_links_existing": existing_links,
                            "alias_checks_failed": checks_failed,
                            "alias_writes_failed": writes_failed,
                            "alias_pairs_skipped_invalid": skipped_invalid,
                        },
                    )
                except ConsolidationPausedError:
                    report.new_alias_links = created_links
                    report.alias_links_existing = existing_links
                    raise
            # Yield to event loop every 50 outer iterations so timeout can fire
            if i % 50 == 0:
                await asyncio.sleep(0)
            if anchor_a is None or anchor_a.id in seen:
                continue
            if anchor_a.content_hash is None or anchor_a.content_hash == 0:
                continue
            # GRAPH_ONLY tombstones written before the hash sentinel existed
            # still carry the SimHash of their deleted original text - a
            # fingerprint of content that is no longer there, which can
            # near-duplicate-match a genuine memory and persist a false ALIAS
            # edge. Guard on content like every other fingerprint consumer;
            # tombstones stamped with the sentinel are already skipped above.
            if anchor_a.content == GRAPH_ONLY_PLACEHOLDER:
                continue

            for anchor_b in anchors[i + 1 :]:
                if anchor_b is None:
                    continue
                if anchor_b.id in seen:
                    continue
                if anchor_b.content_hash is None or anchor_b.content_hash == 0:
                    continue
                if anchor_b.content == GRAPH_ONLY_PLACEHOLDER:
                    continue

                if is_near_duplicate(
                    anchor_a.content_hash,
                    anchor_b.content_hash,
                    threshold=self._config.dedup_simhash_threshold,
                ):
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
                        report.new_alias_links = created_links
                    elif result.outcome in (
                        AliasLinkOutcome.ALREADY_EXISTS,
                        AliasLinkOutcome.EXISTS_RACE,
                    ):
                        # A lost write race means the edge is there, which is the
                        # goal. Counting it as a failure would invent an incident.
                        existing_links += 1
                        report.alias_links_existing = existing_links
                    elif result.outcome is AliasLinkOutcome.CHECK_FAILED:
                        checks_failed += 1
                        report.extra["alias_checks_failed"] = checks_failed
                        first_failure = first_failure or result.error
                    elif result.outcome is AliasLinkOutcome.WRITE_FAILED:
                        writes_failed += 1
                        report.extra["alias_writes_failed"] = writes_failed
                        first_failure = first_failure or result.error
                    else:
                        skipped_invalid += 1
                        report.extra["alias_pairs_skipped_invalid"] = skipped_invalid

        if dry_run:
            return

        report.new_alias_links = created_links
        report.alias_links_existing = existing_links
        # Failure keys appear only when something failed, so a healthy report
        # stays free of zero-valued noise (same rule as semantic_link_failures).
        for key, value in (
            ("alias_checks_failed", checks_failed),
            ("alias_writes_failed", writes_failed),
            ("alias_pairs_skipped_invalid", skipped_invalid),
        ):
            if value:
                report.extra[key] = value

        if anchors_total > cap and not await self._advance_dedup_cursor(cursor, cap, anchors_total):
            raise RuntimeError(
                "dedup completed its window but failed to persist the rotation cursor"
            )

        cursor_payload["next_i"] = len(anchors)
        await self._checkpoint_progress(
            "dedup_window_complete",
            cursor=json.dumps(cursor_payload, separators=(",", ":")),
            pending=[],
            counters={
                "duplicates_found": report.duplicates_found,
                "new_alias_links": created_links,
                "alias_links_existing": existing_links,
                "alias_checks_failed": checks_failed,
                "alias_writes_failed": writes_failed,
                "alias_pairs_skipped_invalid": skipped_invalid,
            },
        )

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
        import hashlib
        import json
        import logging

        from surreal_memory.core.neuron import NeuronType
        from surreal_memory.core.synapse import Direction, Synapse, SynapseType
        from surreal_memory.engine.consolidation_progress import ConsolidationProgressError
        from surreal_memory.engine.semantic_discovery import discover_semantic_synapses

        logger = logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            return
        brain = await self._storage.get_brain(brain_id)
        if not brain:
            return

        def encode_synapse(synapse: Synapse) -> dict[str, Any]:
            return {
                "id": synapse.id,
                "source_id": synapse.source_id,
                "target_id": synapse.target_id,
                "type": synapse.type.value,
                "weight": synapse.weight,
                "direction": synapse.direction.value,
                "metadata": synapse.metadata,
                "reinforced_count": synapse.reinforced_count,
                "last_activated": (
                    synapse.last_activated.isoformat() if synapse.last_activated else None
                ),
                "created_at": synapse.created_at.isoformat(),
            }

        def decode_synapse(payload: dict[str, Any]) -> Synapse:
            return Synapse(
                id=str(payload["id"]),
                source_id=str(payload["source_id"]),
                target_id=str(payload["target_id"]),
                type=SynapseType(payload["type"]),
                weight=float(payload["weight"]),
                direction=Direction(payload["direction"]),
                metadata=dict(payload.get("metadata") or {}),
                reinforced_count=int(payload.get("reinforced_count", 0)),
                last_activated=(
                    datetime.fromisoformat(payload["last_activated"])
                    if payload.get("last_activated")
                    else None
                ),
                created_at=datetime.fromisoformat(payload["created_at"]),
            )

        async def source_fingerprint(excluded_ids: set[str]) -> str:
            eligible: list[dict[str, Any]] = []
            for neuron_type in (NeuronType.CONCEPT, NeuronType.ENTITY):
                offset = 0
                while True:
                    batch = await self._storage.find_neurons(
                        type=neuron_type, limit=1000, offset=offset
                    )
                    if not batch:
                        break
                    eligible.extend(
                        {
                            "id": neuron.id,
                            "type": neuron.type.value,
                            "content": neuron.content,
                            "embedding": neuron.metadata.get("_embedding"),
                        }
                        for neuron in batch
                    )
                    offset += len(batch)
                    if len(batch) < 1000:
                        break

            edges: list[dict[str, Any]] = []
            offset = 0
            while True:
                edge_batch = await self._storage.get_synapses(limit=1000, offset=offset)
                if not edge_batch:
                    break
                edges.extend(
                    {
                        "id": edge.id,
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "type": edge.type.value,
                    }
                    for edge in edge_batch
                    if edge.id not in excluded_ids
                )
                offset += len(edge_batch)
                if len(edge_batch) < 1000:
                    break

            payload = {
                "brain_id": brain_id,
                "config": repr(brain.config),
                "neurons": sorted(eligible, key=lambda item: item["id"]),
                "synapses": sorted(edges, key=lambda item: item["id"]),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        state = self._strategy_progress_state()
        if dry_run:
            result = await discover_semantic_synapses(self._storage, brain.config)
            report.semantic_synapses_skipped += result.skipped_existing
            if result.skipped_created_this_run:
                report.extra["semantic_pairs_seen_twice"] = (
                    int(report.extra.get("semantic_pairs_seen_twice", 0))
                    + result.skipped_created_this_run
                )
            if result.eligible_total:
                report.extra["semantic_candidates_total"] = result.eligible_total
                report.extra["semantic_candidates_scanned"] = result.neurons_embedded
            if result.truncated:
                report.extra["semantic_link_truncated"] = True
            report.semantic_synapses_created = result.synapses_created
            return

        # Keep the legacy adapter path free of durable snapshot queries.
        if self._progress_session is None:
            result = await discover_semantic_synapses(self._storage, brain.config)
            report.semantic_synapses_skipped += result.skipped_existing
            if result.skipped_created_this_run:
                report.extra["semantic_pairs_seen_twice"] = (
                    int(report.extra.get("semantic_pairs_seen_twice", 0))
                    + result.skipped_created_this_run
                )
            if result.eligible_total:
                report.extra["semantic_candidates_total"] = result.eligible_total
                report.extra["semantic_candidates_scanned"] = result.neurons_embedded
            if result.truncated:
                report.extra["semantic_link_truncated"] = True
            for synapse in result.synapses:
                try:
                    await self._storage.add_synapse(synapse)
                    report.semantic_synapses_created += 1
                except ValueError:
                    report.semantic_synapses_skipped += 1
                    logger.debug("Semantic synapse endpoint missing, skipping")
                except Exception as exc:
                    if is_duplicate_key_error(exc):
                        report.semantic_synapses_skipped += 1
                        logger.debug("Semantic synapse already exists, skipping")
                    else:
                        report.extra["semantic_link_failures"] = (
                            int(report.extra.get("semantic_link_failures", 0)) + 1
                        )
                        logger.warning(
                            "Semantic synapse write failed (not a duplicate)", exc_info=True
                        )
            return

        cursor_value = state.get("cursor")
        cursor = (
            str(cursor_value)
            if cursor_value is not None and state.get("phase") != "semantic_link_complete"
            else None
        )
        pending = state.get("pending")
        if pending:
            if not isinstance(pending, (list, tuple)) or len(pending) != 1:
                raise ConsolidationProgressError("invalid semantic-link pending checkpoint")
            try:
                manifest = json.loads(pending[0])
                if manifest.get("kind") != "semantic_link_synapses" or manifest.get("version") != 1:
                    raise ValueError("unsupported pending manifest")
                raw_synapses = manifest["synapses"]
                if not isinstance(raw_synapses, list):
                    raise ValueError("synapses is not a list")
                synapses = [decode_synapse(item) for item in raw_synapses]
                current_fingerprint = await source_fingerprint({synapse.id for synapse in synapses})
                if current_fingerprint != manifest.get("source_fingerprint"):
                    raise ConsolidationProgressError(
                        "semantic-link source data changed after its pending checkpoint"
                    )
                metrics = manifest["metrics"]
                if not isinstance(metrics, dict):
                    raise ValueError("metrics is not an object")
            except ConsolidationProgressError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ConsolidationProgressError("invalid semantic-link pending snapshot") from exc
            serialized = pending[0]
        else:
            result = await discover_semantic_synapses(self._storage, brain.config)
            synapses = sorted(result.synapses, key=lambda synapse: synapse.id)
            metrics = {
                "skipped_existing": result.skipped_existing,
                "skipped_created_this_run": result.skipped_created_this_run,
                "eligible_total": result.eligible_total,
                "neurons_embedded": result.neurons_embedded,
                "truncated": result.truncated,
            }
            manifest = {
                "kind": "semantic_link_synapses",
                "version": 1,
                "source_fingerprint": await source_fingerprint(
                    {synapse.id for synapse in synapses}
                ),
                "metrics": metrics,
                "synapses": [encode_synapse(synapse) for synapse in synapses],
            }
            serialized = json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            await self._checkpoint_progress(
                "semantic_link_pending",
                cursor=cursor,
                pending=[serialized],
                counters={
                    "semantic_synapses_created": 0,
                    "semantic_synapses_skipped": int(metrics["skipped_existing"]),
                    "semantic_link_failures": 0,
                },
            )

        report.semantic_synapses_skipped += int(metrics.get("skipped_existing", 0))
        if metrics.get("skipped_created_this_run"):
            report.extra["semantic_pairs_seen_twice"] = int(
                report.extra.get("semantic_pairs_seen_twice", 0)
            ) + int(metrics["skipped_created_this_run"] or 0)
        if metrics.get("eligible_total"):
            report.extra["semantic_candidates_total"] = int(metrics["eligible_total"])
            report.extra["semantic_candidates_scanned"] = int(metrics["neurons_embedded"])
        if metrics.get("truncated"):
            report.extra["semantic_link_truncated"] = True

        raw_counters = state.get("counters")
        counters = raw_counters if isinstance(raw_counters, dict) else {}
        created = int(counters.get("semantic_synapses_created", 0))
        raw_skipped = counters.get("semantic_synapses_skipped")
        if raw_skipped is None:
            raw_skipped = metrics.get("skipped_existing")
        skipped = int(raw_skipped) if isinstance(raw_skipped, (int, float)) else 0
        failures = int(counters.get("semantic_link_failures", 0))
        for synapse in synapses:
            if cursor is not None and synapse.id <= cursor:
                continue
            await self._check_progress_budget()
            existing = await self._storage.get_synapse(synapse.id)
            if existing is not None:
                if existing != synapse:
                    raise ConsolidationProgressError(
                        f"semantic-link synapse {synapse.id!r} changed after its checkpoint"
                    )
                # The write may have committed before the previous checkpoint.
                created += 1
            else:
                try:
                    await self._storage.add_synapse(synapse)
                    created += 1
                except ValueError:
                    skipped += 1
                    logger.debug("Semantic synapse endpoint missing, skipping")
                except Exception as exc:
                    if is_duplicate_key_error(exc):
                        skipped += 1
                        logger.debug("Semantic synapse already exists, skipping")
                    else:
                        failures += 1
                        logger.warning(
                            "Semantic synapse write failed (not a duplicate)", exc_info=True
                        )
            cursor = synapse.id
            await self._checkpoint_progress(
                "semantic_link_apply",
                cursor=cursor,
                pending=[serialized],
                counters={
                    "semantic_synapses_created": created,
                    "semantic_synapses_skipped": skipped,
                    "semantic_link_failures": failures,
                },
            )

        await self._checkpoint_progress(
            "semantic_link_complete",
            cursor=cursor,
            pending=[],
            counters={
                "semantic_synapses_created": created,
                "semantic_synapses_skipped": skipped,
                "semantic_link_failures": failures,
            },
        )
        report.semantic_synapses_created = created
        # On resume, skipped includes the saved discovery count plus write collisions.
        report.semantic_synapses_skipped += max(
            0, skipped - int(metrics.get("skipped_existing", 0))
        )
        if failures:
            report.extra["semantic_link_failures"] = (
                int(report.extra.get("semantic_link_failures", 0)) + failures
            )

    async def _detect_drift(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Recompute semantic drift clusters and checkpoint each persisted cluster.

        Cluster IDs are deterministic, so the lexically ordered ID is a stable
        cursor. Saving the same cluster again after a crash is an idempotent
        upsert; the cursor advances only after its row has been saved.
        """
        import logging as _logging

        from surreal_memory.engine.drift_clusters import MIN_COOCCURRENCE_COUNT, detect_clusters

        _logger = _logging.getLogger(__name__)

        try:
            cooccurrences = await self._storage.get_tag_cooccurrence(
                min_count=MIN_COOCCURRENCE_COUNT
            )
            tag_fiber_counts = await self._storage.get_tag_fiber_counts()
        except Exception:
            _logger.warning("Failed to read tag data for drift detection", exc_info=True)
            report.drift_clusters_found = 0
            report.drift_clusters_persisted = 0
            return

        clusters = sorted(
            detect_clusters(cooccurrences, tag_fiber_counts),
            key=lambda item: item.cluster_id,
        )
        report.drift_clusters_found = len(clusters)

        if dry_run:
            # A preview still performs the census, but never mutates the durable
            # strategy cursor or drift-cluster table.
            report.drift_clusters_persisted = len(clusters)
            return

        progress = self._strategy_progress_state()
        counters = progress.get("counters")
        raw_persisted = counters.get("persisted", 0) if isinstance(counters, dict) else 0
        previous_persisted = int(raw_persisted) if isinstance(raw_persisted, (int, float)) else 0
        persisted = previous_persisted
        cursor = self._strategy_resume_cursor()

        for cluster in clusters:
            if cursor is not None and cluster.cluster_id <= cursor:
                continue
            await self._check_progress_budget()
            try:
                await self._storage.save_drift_cluster(
                    cluster_id=cluster.cluster_id,
                    canonical=cluster.cluster.canonical,
                    members=sorted(cluster.cluster.members),
                    confidence=cluster.cluster.confidence,
                    status="detected",
                )
            except Exception:
                _logger.warning(
                    "Failed to persist drift cluster %s", cluster.cluster_id, exc_info=True
                )
                # Keep the durable cursor on the preceding committed unit so a
                # resumed strategy retries this deterministic upsert.
                raise
            persisted += 1
            cursor = cluster.cluster_id
            await self._checkpoint_progress(
                "drift_clusters",
                cursor=cursor,
                counters={"detected": len(clusters), "persisted": persisted},
            )

        report.drift_clusters_persisted = persisted

    async def _compress(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Compress fibers in stable, individually checkpointed work units."""
        import logging as _logging
        import time as _time

        from surreal_memory.engine.compression import CompressionEngine, CompressionTier

        _logger = _logging.getLogger(__name__)

        brain_id = self._storage.current_brain_id
        if not brain_id:
            _logger.debug("COMPRESS skipped: no brain context")
            return

        engine = CompressionEngine(self._storage)
        start = _time.perf_counter()
        time_budget = self._config.strategy_timeout_seconds * 0.8
        fibers = sorted(await self._storage.get_fibers(limit=10000), key=lambda item: str(item.id))

        try:
            brain = await self._storage.get_brain(brain_id)
        except Exception:
            # Keep the CompressionEngine's fail-soft behavior: its refresh helper
            # can fall back to per-fiber brain lookups.
            _logger.warning(
                "Brain pre-fetch failed; falling back to a per-fiber lookup", exc_info=True
            )
            brain = None

        progress = self._strategy_progress_state() if not dry_run else {}
        counters = progress.get("counters")
        raw_compressed = counters.get("fibers_compressed", 0) if isinstance(counters, dict) else 0
        total_compressed = int(raw_compressed) if isinstance(raw_compressed, (int, float)) else 0
        raw_tokens_saved = counters.get("tokens_saved", 0) if isinstance(counters, dict) else 0
        total_tokens_saved = (
            int(raw_tokens_saved) if isinstance(raw_tokens_saved, (int, float)) else 0
        )
        cursor = self._strategy_resume_cursor() if not dry_run else None
        raw_run_id = self._progress_session.state.get("run_id") if self._progress_session else None
        run_id = str(raw_run_id) if raw_run_id else None
        pending_fibers = [fiber for fiber in fibers if cursor is None or str(fiber.id) > cursor]

        for index, fiber in enumerate(pending_fibers):
            await self._check_progress_budget()
            fiber_cursor = str(fiber.id)

            pending = fiber.metadata.get("_compression_pending") if not dry_run else None
            receipt = fiber.metadata.get("_compression_receipt") if not dry_run else None
            if (
                pending is None
                and run_id
                and isinstance(receipt, dict)
                and receipt.get("run_id") == run_id
                and receipt.get("target_tier") == fiber.compression_tier
            ):
                total_compressed += 1
                total_tokens_saved += int(receipt["tokens_saved"])
                await self._checkpoint_progress(
                    "compress_fibers",
                    cursor=fiber_cursor,
                    counters={
                        "fibers_compressed": total_compressed,
                        "tokens_saved": total_tokens_saved,
                    },
                )
                continue
            if (fiber.pinned or fiber.metadata.get("_verbatim")) and pending is None:
                if not dry_run:
                    await self._checkpoint_progress(
                        "compress_fibers",
                        cursor=fiber_cursor,
                        counters={
                            "fibers_compressed": total_compressed,
                            "tokens_saved": total_tokens_saved,
                        },
                    )
                continue

            arousal = fiber.metadata.get("_arousal", 0.0) if fiber.metadata else 0.0
            arousal_heat = float(arousal) * 0.3 if isinstance(arousal, (int, float)) else 0.0
            target_tier = (
                CompressionTier(int(pending["target_tier"]))
                if isinstance(pending, dict)
                else engine.determine_target_tier(
                    fiber,
                    reference_time,
                    heat_score=arousal_heat,
                )
            )
            if pending is None and int(target_tier) <= fiber.compression_tier:
                if not dry_run:
                    await self._checkpoint_progress(
                        "compress_fibers",
                        cursor=fiber_cursor,
                        counters={
                            "fibers_compressed": total_compressed,
                            "tokens_saved": total_tokens_saved,
                        },
                    )
                continue

            if _time.perf_counter() - start > time_budget:
                report.extra["compress_fibers_deferred"] = len(pending_fibers) - index
                report.fibers_compressed += total_compressed
                report.tokens_saved += total_tokens_saved
                raise ConsolidationPausedError(
                    f"compression deferred {len(pending_fibers) - index} fibers "
                    "at its time budget; the last committed fiber checkpoint is retained"
                )

            try:
                result = await engine.compress_fiber(
                    fiber,
                    CompressionTier(target_tier),
                    dry_run=dry_run,
                    brain=brain,
                    **({"run_id": run_id} if run_id else {}),
                )
            except Exception:
                _logger.error("Compression failed for fiber %s", fiber.id, exc_info=True)
                # Do not move the durable cursor past a failed fiber. The next run
                # must retry it instead of silently treating it as completed.
                raise

            if not result.skipped:
                total_compressed += 1
                total_tokens_saved += result.tokens_saved
            if not dry_run:
                await self._checkpoint_progress(
                    "compress_fibers",
                    cursor=fiber_cursor,
                    counters={
                        "fibers_compressed": total_compressed,
                        "tokens_saved": total_tokens_saved,
                    },
                )

        report.fibers_compressed += total_compressed
        report.tokens_saved += total_tokens_saved

    async def _lifecycle(
        self,
        report: ConsolidationReport,
        reference_time: datetime,
        dry_run: bool,
    ) -> None:
        """Calculate heat scores and update lifecycle_state for all neurons.

        Lifecycle scans use a stable record-ID keyset on stores that expose it.
        Durable progress is saved only after each neuron has been handled, so a
        resumed scan starts strictly after the last committed neuron. The
        supplied reference_time remains fixed across the scan and any resume.
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

        config = CompressionConfig()
        progress_session = getattr(self, "_progress_session", None)
        active_strategy = getattr(self, "_active_strategy", None)
        if progress_session is not None and active_strategy is not None:
            strategy_state = progress_session.strategy_state(active_strategy.value)
        else:
            strategy_state = {}
        cursor = strategy_state.get("cursor") if strategy_state.get("phase") == "scan" else None
        counters = strategy_state.get("counters") or {}
        states_updated = int(counters.get("lifecycle_states_updated", 0))
        batch_size = 500
        keyset_fetch: Any = getattr(self._storage, "find_neurons_after_id", None)
        has_keyset = supports_storage_methods(self._storage, ("find_neurons_after_id",))
        fetch_by_ids: Any = getattr(self._storage, "find_neurons_by_ids", None)
        offset = 0
        scanned = 0
        pending: list[str] = list(strategy_state.get("pending") or [])

        prefetched_states: dict[str, Any] | None = None
        attempted_state_prefetch = False

        async def _states_for(neurons: list[Neuron]) -> dict[str, Any]:
            nonlocal prefetched_states, attempted_state_prefetch
            if not attempted_state_prefetch:
                attempted_state_prefetch = True
                try:
                    prefetched_states = {
                        state.neuron_id: state
                        for state in await self._storage.get_all_neuron_states()
                    }
                except Exception:
                    _logger.debug(
                        "LIFECYCLE: get_all_neuron_states failed; falling back to a batch fetch",
                        exc_info=True,
                    )
            if prefetched_states is not None:
                return {
                    neuron.id: prefetched_states[neuron.id]
                    for neuron in neurons
                    if neuron.id in prefetched_states
                }
            return await self._storage.get_neuron_states_batch([neuron.id for neuron in neurons])

        async def _process(neuron: Neuron, state_map: dict[str, Any]) -> bool:
            nonlocal states_updated
            # Refresh the candidate before applying a decision based on a page
            # that may have become stale while earlier records were processed.
            if callable(fetch_by_ids) and not dry_run:
                fresh = await fetch_by_ids([neuron.id], include_embedding=False)
                if not fresh:
                    return True
                neuron = fresh[0]
            neuron_state = state_map.get(neuron.id)
            last_accessed_at: datetime | None = (
                neuron_state.last_activated if neuron_state is not None else None
            )
            access_count = neuron_state.access_frequency if neuron_state is not None else 0
            priority = int(neuron.metadata.get("priority", 5))
            heat = calculate_heat_score(
                last_accessed_at=last_accessed_at,
                access_count=access_count,
                priority=priority,
                reference_time=reference_time,
                config=config,
            )
            age_days = (reference_time - neuron.created_at).total_seconds() / 86400.0
            new_state = determine_lifecycle_state(age_days, heat, config)
            current_state = str(neuron.metadata.get("lifecycle_state", "active"))
            if current_state == str(new_state):
                return True
            if dry_run:
                states_updated += 1
                return True
            try:
                await self._storage.update_neuron_lifecycle(neuron.id, str(new_state))
                states_updated += 1
                return True
            except Exception:
                _logger.error(
                    "Failed to update lifecycle_state for neuron %s", neuron.id, exc_info=True
                )
                return False

        try:
            if has_keyset:
                # Failed updates from a previous attempt are revisited before the
                # cursor scan; these IDs are explicit work, not offset progress.
                if pending:
                    if not callable(fetch_by_ids):
                        pending_neurons = []
                    else:
                        pending_neurons = await fetch_by_ids(pending, include_embedding=False)
                    state_map = await _states_for(pending_neurons)
                    still_pending: list[str] = []
                    for neuron in sorted(pending_neurons, key=lambda item: item.id):
                        ok = await _process(neuron, state_map)
                        if not ok:
                            still_pending.append(neuron.id)
                        elif not dry_run:
                            await self._checkpoint_progress(
                                "scan",
                                cursor=str(cursor) if cursor is not None else None,
                                pending=still_pending,
                                counters={"lifecycle_states_updated": states_updated},
                            )
                    pending = still_pending

                while scanned < 10000:
                    batch = await keyset_fetch(
                        str(cursor) if cursor is not None else None,
                        limit=min(batch_size, 10000 - scanned),
                        created_before=reference_time,
                        ephemeral=False,
                        include_embedding=False,
                    )
                    if not batch:
                        break
                    state_map = await _states_for(batch)
                    for neuron in batch:
                        ok = await _process(neuron, state_map)
                        cursor = neuron.id
                        scanned += 1
                        if not ok:
                            pending.append(neuron.id)
                        if not dry_run:
                            await self._checkpoint_progress(
                                "scan",
                                cursor=str(cursor),
                                pending=pending,
                                counters={"lifecycle_states_updated": states_updated},
                            )
                    if len(batch) < batch_size:
                        break
            else:
                # Legacy in-memory/test backends have no keyset API and cannot
                # participate in durable resume; retain their existing paging.
                neurons: list[Neuron] = []
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
                state_map = await _states_for(neurons)
                for neuron in neurons:
                    await _process(neuron, state_map)
        except Exception:
            _logger.error("LIFECYCLE failed to fetch or process neurons", exc_info=True)
            raise

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
            # Into the report, not only the debug log: this strategy could ingest
            # thousands of events and the report looked identical to it being disabled.
            report.extra["tool_events_ingested"] = (
                int(report.extra.get("tool_events_ingested", 0)) + ingest_result.events_ingested
            )
            _logger.debug(
                "PROCESS_TOOL_EVENTS: ingested %d events from buffer",
                ingest_result.events_ingested,
            )

        # Each process_events call commits graph mutations and marks its event batch
        # processed before returning. Persist the matching run checkpoint only after
        # that commit, then fetch the next unprocessed batch. The database event flag
        # is the authoritative resume cursor; last_event_id is the durable batch receipt.
        progress_state = self._strategy_progress_state()
        saved_counters = progress_state.get("counters") or {}
        processed_total = int(saved_counters.get("events_processed", 0))
        ingested_total = (
            int(saved_counters.get("events_ingested", 0)) + ingest_result.events_ingested
        )
        while True:
            await self._check_progress_budget()
            result = await process_events(self._storage, brain_id, config.tool_memory)  # type: ignore[arg-type]
            if result.events_processed <= 0:
                break

            processed_total += result.events_processed
            report.extra["tool_events_processed"] = (
                int(report.extra.get("tool_events_processed", 0)) + result.events_processed
            )
            _logger.debug(
                "PROCESS_TOOL_EVENTS: processed %d events, created %d neurons, %d synapses",
                result.events_processed,
                result.neurons_created,
                result.synapses_created,
            )
            await self._checkpoint_progress(
                "batch_committed",
                cursor=result.last_event_id,
                counters={
                    "events_processed": processed_total,
                    "events_ingested": ingested_total,
                },
            )

    async def _process_reasoning_traces(
        self,
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Mine transcript files with their existing scan-state plus strategy checkpoints."""
        import asyncio
        import hashlib
        import logging as _logging
        import time as _time

        from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
        from surreal_memory.engine.reasoning_miner import (
            _STATE_SAVE_EVERY,
            _discover_all_transcripts,
            _resolve_transcript_roots,
            ingest_reasoning_traces,
        )
        from surreal_memory.engine.reasoning_progress import MiningProgress
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

        progress_enabled = self._progress_session is not None
        file_cursors = (
            [
                f"file:{hashlib.sha256(str(path).encode()).hexdigest()}"
                for path, _project in _discover_all_transcripts(
                    _resolve_transcript_roots(config.reasoning_training, claude_dir=None)
                )
            ]
            if progress_enabled
            else []
        )
        state = self._strategy_progress_state() if progress_enabled else {}
        previous_counters = state.get("counters")
        previous_ingested = (
            int(previous_counters.get("traces_ingested", 0))
            if isinstance(previous_counters, dict)
            else 0
        )

        checkpoint_tasks: list[asyncio.Task[None]] = []
        checkpoint_tail: asyncio.Task[None] | None = None
        last_scheduled_files = 0

        def _budget_is_low() -> bool:
            if self._progress_session is None:
                return False
            deadlines = [
                deadline
                for deadline in (self._strategy_deadline, self._total_deadline)
                if deadline is not None
            ]
            if not deadlines:
                return False
            remaining = min(deadlines) - _time.perf_counter()
            headroom = min(10.0, max(0.05, self._config.strategy_timeout_seconds * 0.02))
            return remaining <= headroom

        def _queue_checkpoint(
            files_scanned: int,
            traces_ingested: int,
            traces_scanned: int,
        ) -> None:
            nonlocal checkpoint_tail, last_scheduled_files
            if not progress_enabled or files_scanned <= last_scheduled_files:
                return
            if files_scanned <= len(file_cursors):
                cursor = file_cursors[files_scanned - 1]
            else:
                cursor = f"files_scanned:{files_scanned}"
            predecessor = checkpoint_tail

            async def _persist() -> None:
                if predecessor is not None:
                    await predecessor
                await self._checkpoint_progress(
                    "reasoning_ingest",
                    cursor=cursor,
                    counters={
                        "files_scanned": files_scanned,
                        "traces_scanned": traces_scanned,
                        "traces_ingested": previous_ingested + traces_ingested,
                    },
                )

            checkpoint_tail = asyncio.create_task(_persist())
            checkpoint_tasks.append(checkpoint_tail)
            last_scheduled_files = files_scanned

        def _on_progress(progress: MiningProgress) -> None:
            if progress.phase != "ingesting" or progress.files_scanned <= 0:
                return
            save_every = _STATE_SAVE_EVERY
            if (
                progress.files_scanned % save_every == 0
                or progress.files_scanned >= progress.files_total
            ):
                _queue_checkpoint(
                    progress.files_scanned,
                    progress.traces_ingested,
                    progress.traces_found,
                )
            elif _budget_is_low():
                # ingest_reasoning_traces persists its scan state in its finally
                # block. Queue this exact file boundary first, then stop before
                # another transcript is opened.
                _queue_checkpoint(
                    progress.files_scanned,
                    progress.traces_ingested,
                    progress.traces_found,
                )
                raise ConsolidationPausedError(
                    "reasoning-ingest budget is low; paused after a committed transcript"
                )

        async def _flush_checkpoints() -> None:
            if not checkpoint_tasks:
                return
            outcomes = await asyncio.gather(*checkpoint_tasks, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, ConsolidationPausedError):
                    raise outcome
                if isinstance(outcome, Exception):
                    raise outcome

        try:
            result = await ingest_reasoning_traces(
                self._storage,
                brain_id,
                config,
                progress=_on_progress if progress_enabled else None,
            )
        except ConsolidationPausedError:
            raise
        except Exception:
            _logger.debug("PROCESS_REASONING_TRACES: ingest failed (non-critical)", exc_info=True)
            return
        finally:
            await _flush_checkpoints()

        if progress_enabled and result.files_scanned > last_scheduled_files:
            _queue_checkpoint(
                result.files_scanned,
                result.traces_ingested,
                result.traces_scanned,
            )
            await _flush_checkpoints()

        report.reasoning_traces_ingested += previous_ingested + result.traces_ingested
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
        """Distill staged traces in model-batch units with a durable model cursor."""
        import asyncio
        import json
        import logging as _logging
        import time as _time

        from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
        from surreal_memory.engine.reasoning_distiller import distill_reasoning_patterns
        from surreal_memory.engine.reasoning_progress import MiningProgress
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

        progress_state = (
            self._strategy_progress_state() if self._progress_session is not None else {}
        )
        previous_counters = progress_state.get("counters")
        previous_patterns = (
            int(previous_counters.get("patterns_learned", 0))
            if isinstance(previous_counters, dict)
            else 0
        )
        previous_traces = (
            int(previous_counters.get("traces_processed", 0))
            if isinstance(previous_counters, dict)
            else 0
        )
        pending_patterns: list[dict[str, Any]] = []
        raw_pending = progress_state.get("pending")
        if isinstance(raw_pending, list):
            for encoded in raw_pending:
                try:
                    pattern = json.loads(encoded)
                except (TypeError, json.JSONDecodeError):
                    _logger.error("LEARN_REASONING: invalid durable pending pattern")
                    return
                if not isinstance(pattern, dict):
                    _logger.error("LEARN_REASONING: invalid durable pending pattern")
                    return
                pending_patterns.append(pattern)

        checkpoint_tasks: list[asyncio.Task[None]] = []
        checkpoint_tail: asyncio.Task[None] | None = None
        last_model: str | None = None

        async def _checkpoint_patterns(patterns: Sequence[dict[str, Any]]) -> None:
            nonlocal checkpoint_tail
            predecessor = checkpoint_tail
            if predecessor is not None:
                await predecessor
                checkpoint_tail = None
            await self._checkpoint_progress(
                "reasoning_pattern_pending",
                cursor=self._strategy_resume_cursor(),
                pending=[
                    json.dumps(pattern, sort_keys=True, separators=(",", ":"))
                    for pattern in patterns
                ],
                counters={
                    "models_done": int(previous_counters.get("models_done", 0))
                    if isinstance(previous_counters, dict)
                    else 0,
                    "models_total": int(previous_counters.get("models_total", 0))
                    if isinstance(previous_counters, dict)
                    else 0,
                    "traces_processed": previous_traces,
                    "patterns_learned": previous_patterns,
                },
            )

        def _budget_is_low() -> bool:
            if self._progress_session is None:
                return False
            deadlines = [
                deadline
                for deadline in (self._strategy_deadline, self._total_deadline)
                if deadline is not None
            ]
            if not deadlines:
                return False
            remaining = min(deadlines) - _time.perf_counter()
            headroom = min(10.0, max(0.05, self._config.strategy_timeout_seconds * 0.02))
            return remaining <= headroom

        def _on_progress(progress: MiningProgress) -> None:
            nonlocal checkpoint_tail, last_model
            model = progress.current_model
            if self._progress_session is None or model is None:
                return
            # distill_reasoning_patterns emits after a model's batch has been
            # materialized and its consumed traces have been marked processed.
            if model == last_model:
                return
            predecessor = checkpoint_tail

            async def _persist() -> None:
                if predecessor is not None:
                    await predecessor
                await self._checkpoint_progress(
                    "reasoning_distill",
                    cursor=model,
                    counters={
                        "models_done": progress.models_done,
                        "models_total": progress.models_total,
                        "traces_processed": previous_traces + progress.traces_processed,
                        "patterns_learned": previous_patterns + progress.patterns_learned,
                    },
                )

            checkpoint_tail = asyncio.create_task(_persist())
            checkpoint_tasks.append(checkpoint_tail)
            last_model = model
            if _budget_is_low():
                raise ConsolidationPausedError(
                    "reasoning-distillation budget is low; paused after a committed model batch"
                )

        async def _flush_checkpoints() -> None:
            if not checkpoint_tasks:
                return
            outcomes = await asyncio.gather(*checkpoint_tasks, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, ConsolidationPausedError):
                    raise outcome
                if isinstance(outcome, Exception):
                    raise outcome

        try:
            result = await distill_reasoning_patterns(
                self._storage,
                brain_id,
                config,
                progress=_on_progress if self._progress_session is not None else None,
                pending_patterns=pending_patterns,
                pattern_checkpoint=(
                    _checkpoint_patterns if self._progress_session is not None else None
                ),
            )
        except ConsolidationPausedError:
            raise
        except Exception:
            _logger.debug("LEARN_REASONING: distillation failed (non-critical)", exc_info=True)
            return
        finally:
            await _flush_checkpoints()

        report.reasoning_patterns_learned += previous_patterns + result.patterns_learned
        if result.patterns_learned > 0:
            _logger.debug(
                "LEARN_REASONING: learned %d reasoning patterns from %d traces",
                result.patterns_learned,
                result.traces_processed,
            )

"""Brain health diagnostics and quality analysis.

Computes composite purity score, individual metrics, and
actionable warnings from the neural graph structure.
Supports both MCP and CLI exposure.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.memory_stages import MemoryStage, classify_episodic_blocker
from surreal_memory.utils.tag_normalizer import TagNormalizer
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage


# ── Data structures ──────────────────────────────────────────────


class WarningSeverity(StrEnum):
    """Severity level for diagnostic warnings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DiagnosticWarning:
    """A single diagnostic warning with severity and context."""

    severity: WarningSeverity
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityBadge:
    """Quality badge for marketplace eligibility.

    Computed from brain health diagnostics.
    Stored in Brain.metadata["_quality_badge"] when computed.
    """

    grade: str
    purity_score: float
    marketplace_eligible: bool
    badge_label: str
    computed_at: datetime
    component_summary: dict[str, float]


@dataclass(frozen=True)
class PenaltyFactor:
    """A ranked penalty factor explaining why health score is low.

    Attributes:
        component: Name of the health component (e.g. "connectivity")
        current_score: Current score for this component (0.0-1.0)
        weight: Weight of this component in purity calculation
        penalty_points: Points lost due to this component (0-100 scale)
        estimated_gain: Points gained if component improved to 0.8
        action: Suggested action to improve this component
    """

    component: str
    current_score: float
    weight: float
    penalty_points: float
    estimated_gain: float
    action: str


# Component weights and improvement actions (must match purity formula)
_COMPONENT_WEIGHTS: dict[str, tuple[float, str]] = {
    "connectivity": (
        0.25,
        "Store memories with context (causes, effects, relationships) to build connections.",
    ),
    "diversity": (
        0.20,
        "Use varied language: 'X caused Y', 'after A then B', 'X is related to Y'.",
    ),
    "freshness": (
        0.15,
        "Recall or store memories regularly — brain needs activity within the last 7 days.",
    ),
    "consolidation_ratio": (
        0.15,
        "Recall or store these memories again spread across 3+ different days "
        "(or reach 15+ rehearsals across 5+ time windows) — memories advance to "
        "semantic via spaced repetition, not by running consolidate.",
    ),
    "orphan_rate": (
        0.10,
        "Run: smem consolidate --strategy prune — removes neurons with no synapses or fiber links.",
    ),
    "activation_efficiency": (
        0.10,
        "Recall stored memories by topic to activate them: smem_recall 'your topic'.",
    ),
    "recall_confidence": (
        0.05,
        "Recall memories multiple times to strengthen synapse weights.",
    ),
}


def _build_dynamic_action(
    component: str,
    static_action: str,
    metrics: dict[str, Any] | None,
) -> str:
    """Build a concrete action string with actual metrics.

    Falls back to static action if metrics are unavailable.
    """
    if not metrics:
        return static_action

    neuron_count = metrics.get("neuron_count", 0)
    synapse_count = metrics.get("synapse_count", 0)
    fiber_count = metrics.get("fiber_count", 0)
    freshness = metrics.get("freshness", 0.0)
    orphan_rate = metrics.get("orphan_rate", 0.0)
    activation_efficiency = metrics.get("activation_efficiency", 0.0)
    recall_confidence = metrics.get("recall_confidence", 0.0)
    consolidation_ratio = metrics.get("consolidation_ratio", 0.0)
    types_used = metrics.get("types_used", 0)

    if component == "connectivity" and neuron_count > 0:
        # Quote the SEMANTIC count: connectivity is scored on that graph only, so
        # citing the total (alias/audit rows included) would advertise a ratio the
        # score never saw. Naming both units keeps the 0-1 score from being read
        # as if it were the synapses/neuron target.
        semantic_count = int(metrics.get("semantic_synapse_count", synapse_count))
        ratio = semantic_count / max(neuron_count, 1)
        gap = max(0, int(3.0 * neuron_count) - semantic_count)
        return (
            f"Store memories with context to add ~{gap} semantic connections "
            f"(now {ratio:.1f} semantic synapses/neuron; the 0-1 connectivity score "
            "reaches 0.5 at 3/neuron and saturates near 1.0 at 8/neuron)."
        )
    elif component == "diversity":
        # Diversity is Shannon entropy over synapse types (distribution balance),
        # normalised against _EXPECTED_SYNAPSE_TYPES common types. A brain can use
        # MORE than that baseline yet still score low when the mix is skewed, so
        # never claim "{used} of {expected}" when used >= expected (that read as
        # the contradictory "16 of 8 expected synapse types used").
        expected = DiagnosticsEngine._EXPECTED_SYNAPSE_TYPES
        if types_used < expected:
            return (
                f"Use varied memory types — only {types_used} of {expected} common "
                "synapse types used. The 0-1 diversity score measures how evenly "
                "that mix is spread, not how many types exist. Try: 'X caused Y', "
                "'after A then B', 'X is related to Y'."
            )
        return (
            f"Balance your synapse types — {types_used} types are in use but "
            "unevenly distributed, so a few relations dominate and the 0-1 "
            "diversity score stays low. Vary phrasing: 'X caused Y', "
            "'after A then B', 'X is related to Y'."
        )
    elif component == "freshness":
        fresh_count = int(freshness * max(fiber_count, 1))
        target_per_week = max(5, fiber_count // 10)
        return (
            f"Recall or store {target_per_week}+ memories this week "
            f"(current: {fresh_count} active in last 7 days)."
        )
    elif component == "consolidation_ratio":
        episodic_pct = int((1.0 - consolidation_ratio) * 100)
        return (
            f"{episodic_pct}% of fibers are still episodic (target: 50%+ semantic). "
            "They advance to semantic only through spaced recall — reinforcement "
            "spread across 3+ distinct days (or 15+ rehearsals across 5+ time "
            "windows). Running `smem consolidate` will not move this on its own."
        )
    elif component == "orphan_rate" and neuron_count > 0:
        orphan_count = int(orphan_rate * neuron_count)
        return (
            f"{orphan_count} neurons have no synapses or fiber links. "
            "Recall related topics to build connections, "
            "or run `smem consolidate --strategy prune` to clean up."
        )
    elif component == "activation_efficiency":
        never_accessed_pct = int((1.0 - activation_efficiency) * 100)
        return (
            f"Recall memories by topic — {never_accessed_pct}% of neurons never accessed. "
            "Try: `smem_recall 'topic'` for 5+ different topics."
        )
    elif component == "recall_confidence":
        return (
            f"Recall existing memories to reinforce connections "
            f"(avg synapse weight: {recall_confidence:.2f}, target: 0.50+)."
        )
    return static_action


def _rank_penalty_factors(
    scores: dict[str, float],
    *,
    top_n: int = 3,
    target: float = 0.8,
    metrics: dict[str, Any] | None = None,
) -> tuple[PenaltyFactor, ...]:
    """Rank health components by their penalty contribution.

    For each component, penalty = (1.0 - effective_score) * weight * 100.
    Estimated gain = (min(target, 1.0) - effective_score) * weight * 100 (clamped >= 0).

    Args:
        scores: Mapping of component name to current score (0.0-1.0).
        top_n: Number of top factors to return.
        target: Target score for estimated gain calculation.
        metrics: Optional dict with actual counts for dynamic action strings.

    Returns:
        Top penalty factors sorted by penalty_points descending.
    """
    factors: list[PenaltyFactor] = []
    for component, (weight, static_action) in _COMPONENT_WEIGHTS.items():
        score = scores.get(component, 0.0)
        # orphan_rate is inverted in purity formula: (1.0 - orphan_rate) * weight
        effective_score = (1.0 - score) if component == "orphan_rate" else score
        penalty = (1.0 - effective_score) * weight * 100
        gain = max(0.0, (min(target, 1.0) - effective_score) * weight * 100)
        action = _build_dynamic_action(component, static_action, metrics)
        factors.append(
            PenaltyFactor(
                component=component,
                current_score=round(score, 4),
                weight=weight,
                penalty_points=round(penalty, 1),
                estimated_gain=round(gain, 1),
                action=action,
            )
        )
    factors.sort(key=lambda f: f.penalty_points, reverse=True)
    return tuple(factors[:top_n])


@dataclass(frozen=True)
class BrainHealthReport:
    """Complete brain health diagnostics report.

    All component scores are normalized to [0.0, 1.0].
    Purity score is a weighted composite in [0, 100].
    """

    # Overall health
    purity_score: float
    grade: str

    # Component scores (0.0-1.0)
    connectivity: float
    diversity: float
    freshness: float
    consolidation_ratio: float
    orphan_rate: float
    activation_efficiency: float
    recall_confidence: float

    # Raw counts
    neuron_count: int
    synapse_count: int
    fiber_count: int

    # Diagnostics
    warnings: tuple[DiagnosticWarning, ...]
    recommendations: tuple[str, ...]

    # Penalty breakdown (top factors hurting the score)
    top_penalties: tuple[PenaltyFactor, ...] = ()

    # Conflict health (U6) — already computed for the purity penalty; surfaced here
    # for the dashboard. Does NOT change the grade/purity formula.
    contradiction_count: int = 0
    conflict_rate: float = 0.0

    # Synapses left after dropping structural/dedup buckets — this is the count
    # `connectivity` is actually scored on, whereas `synapse_count` above is the
    # raw table total. Kept separate so a dashboard can show both without guessing.
    semantic_synapse_count: int = 0

    # Structural (code-index) neuron count and the organic remainder used as the
    # connectivity denominator (run 013). A dashboard needs both to explain why
    # `connectivity` moved independently of `synapse_count`: the denominator
    # changed, not the graph.
    structural_neuron_count: int = 0
    connectivity_neuron_count: int = 0

    # Maturation visibility (run 010 / D2): consolidation_ratio alone cannot
    # explain itself -- these two answer "what does the stage breakdown look
    # like" and "of the fibers not yet semantic, which gate is blocking them".
    # None on backends without maturation support rather than an empty dict,
    # so a caller can distinguish "no episodic fibers" from "not supported".
    stage_distribution: dict[str, int] | None = None
    semantic_gate_blockers: dict[str, int] | None = None


# ── Report serialization ─────────────────────────────────────────


def build_health_payload(report: BrainHealthReport, *, brain: str) -> dict[str, Any]:
    """Serialize a report into the payload every user-facing health surface returns.

    `smem_health` (MCP) and `smem health --json` (CLI) show the same report to the
    same operator, and each used to build its dict by hand, field by field. That is
    the mechanism behind this whole fix: two hand-maintained lists of keys drift the
    moment the report grows one. They had drifted in both directions at once —
    `top_penalties` reached only the MCP payload, and the maturation fields reached
    neither. One builder, one shape; a field added to the report and to this function
    is on both surfaces or on neither. Callers add their own surface-specific keys
    (roadmap, embedding probe, deep analysis) on top of what this returns.

    `stage_distribution` and `semantic_gate_blockers` are omitted, not nulled, when
    the report carries `None` — that means "this backend cannot answer", which is
    not an empty distribution, and it matches how `smem_evolution` already
    serializes the same two fields.
    """
    payload: dict[str, Any] = {
        "brain": brain,
        "grade": report.grade,
        "purity_score": report.purity_score,
        "connectivity": report.connectivity,
        "diversity": report.diversity,
        "freshness": report.freshness,
        "consolidation_ratio": report.consolidation_ratio,
        "orphan_rate": report.orphan_rate,
        "activation_efficiency": report.activation_efficiency,
        "recall_confidence": report.recall_confidence,
        "neuron_count": report.neuron_count,
        "synapse_count": report.synapse_count,
        "fiber_count": report.fiber_count,
        # run 013: connectivity is scored on the organic subgraph. Exposing both
        # counts lets a dashboard explain a purity move that the raw counts can't.
        "structural_neuron_count": report.structural_neuron_count,
        "connectivity_neuron_count": report.connectivity_neuron_count,
        "contradiction_count": report.contradiction_count,
        "conflict_rate": report.conflict_rate,
        "warnings": [
            {"severity": w.severity.value, "code": w.code, "message": w.message}
            for w in report.warnings
        ],
        "recommendations": list(report.recommendations),
        # The remedy text lives in `action` — without it a caller sees which
        # component is costing points and nothing about how to move it.
        "top_penalties": [
            {
                "component": p.component,
                "current_score": p.current_score,
                "weight": p.weight,
                "penalty_points": p.penalty_points,
                "estimated_gain": p.estimated_gain,
                "action": p.action,
            }
            for p in report.top_penalties
        ],
    }

    # Maturation view: which stages fibers sit at, and what blocks the episodic
    # ones from SEMANTIC. These two exist to explain the consolidation_ratio above.
    if report.stage_distribution is not None:
        payload["stage_distribution"] = dict(report.stage_distribution)
    if report.semantic_gate_blockers is not None:
        payload["semantic_gate_blockers"] = dict(report.semantic_gate_blockers)

    return payload


# ── Grade mapping ────────────────────────────────────────────────


def _score_to_grade(score: float) -> str:
    """Map purity score (0-100) to letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# ── Diagnostics engine ──────────────────────────────────────────


class DiagnosticsEngine:
    """Brain health diagnostics and quality analysis.

    Computes composite purity score, individual metrics, and
    actionable warnings from the neural graph structure.
    """

    # Number of defined synapse types (for diversity normalization)
    _TOTAL_SYNAPSE_TYPES = len(SynapseType)

    # Synapse types that are graph plumbing rather than remembered meaning. They are
    # subtracted before scoring connectivity, so the metric describes the semantic
    # graph a recall actually traverses:
    #   alias — dedup pointer (weight 0.0, metadata {"_dedup": true}) written once per
    #     repeated mention, so it scales with write volume, not with what is known. On
    #     a real brain 137871 of 154252 synapses were alias rows (89%): they pinned the
    #     sigmoid at its 1.0 ceiling while the semantic graph held 1.4 edges/neuron —
    #     the metric was hiding the very weakness it exists to expose.
    #   stored_by / verified_at / approved_by / source_of — audit and provenance edges
    #     pointing at an agent, user or source record, not at another memory.
    # similar_to is deliberately NOT structural. It is embedding-derived rather than
    # authored, but it joins two content neurons, carries a real weight and is walked
    # during retrieval; dropping it would understate a graph that genuinely recalls.
    # in_row / in_column stay for the same reason — table structure links content to
    # content. The test is "does traversing this edge return meaning?", not "did a
    # human type it?".
    #
    # NOTE (run 013): the code-index synapse TYPES (contains / co_occurs / is_a /
    # related_to) are NOT added here. They are shared with the organic encoder —
    # consolidation, dream, enrichment, db_knowledge and others all emit them
    # between organic neurons — so excluding by type would drop real semantic
    # edges. The code-index edges are filtered instead by ENDPOINT: a synapse is
    # structural when either endpoint is an indexed neuron. See
    # _count_organic_synapses / connectivity_neuron_count (K2/K3) and
    # DECISIONS.md variant (c).
    _STRUCTURAL_SYNAPSE_TYPES: frozenset[str] = frozenset(
        {
            SynapseType.ALIAS.value,
            SynapseType.STORED_BY.value,
            SynapseType.VERIFIED_AT.value,
            SynapseType.APPROVED_BY.value,
            SynapseType.SOURCE_OF.value,
        }
    )

    @staticmethod
    def _is_structural_neuron(metadata: dict[str, Any] | None) -> bool:
        """Central predicate: is this neuron structural (code-index plumbing)?

        The single place to extend when a new class of non-memory neuron lands.
        Today the only structural neurons are code-index leaves emitted by
        ``CodebaseIndexer`` (``metadata["indexed"] = True``). Connectivity and
        activation score against the organic remainder
        (``connectivity_neuron_count = neuron_count - structural_neuron_count``);
        the synapse-side filter is endpoint-based (variant c, DECISIONS.md):
        a synapse counts only when BOTH endpoints are organic.
        """
        if not metadata:
            return False
        return bool(metadata.get("indexed"))

    def __init__(self, storage: NeuralStorage) -> None:
        self._storage = storage

    async def analyze(self, brain_id: str) -> BrainHealthReport:
        """Run full brain diagnostics.

        Args:
            brain_id: ID of the brain to analyze

        Returns:
            BrainHealthReport with scores, warnings, and recommendations
        """
        # Pin storage to the analyzed brain before any brain-scoped reads.
        # The SurrealDB backend uses a single shared storage singleton with a
        # mutable current-brain pointer; without this, a concurrent multi-brain
        # dashboard call (e.g. /stats analyzing both brains) leaves the pointer
        # on a different brain, and get_all_synapses()/get_fibers() then return
        # the WRONG brain's data — producing a false orphan rate. Harmless on
        # backends that scope storage per brain (e.g. SQLite per-file).
        if hasattr(self._storage, "set_brain"):
            self._storage.set_brain(brain_id)

        # Gather base data. Skip the neuron-type breakdown — it is the priciest
        # query on a large brain (~2.6 s / 64k neurons) and none of the health
        # metrics below read it. Backends whose get_enhanced_stats predates the
        # flag fall back to the full call.
        try:
            enhanced = await self._storage.get_enhanced_stats(brain_id, include_neuron_types=False)
        except TypeError:
            enhanced = await self._storage.get_enhanced_stats(brain_id)
        neuron_count: int = enhanced.get("neuron_count", 0)
        synapse_count: int = enhanced.get("synapse_count", 0)
        fiber_count: int = enhanced.get("fiber_count", 0)
        # Structural (code-index) neuron count — maintained as a live indexed
        # aggregate by the storage backend (see get_stats). Used to score
        # connectivity on the organic subgraph: a code-index neuron is a leaf
        # in a per-file tree, never the endpoint of a memory the user cares
        # about, so including it in the denominator deflates connectivity.
        # See DECISIONS.md (run 013, K1) — variant (c): both endpoints organic.
        structural_neuron_count: int = enhanced.get("structural_neuron_count", 0)
        connectivity_neuron_count = max(0, neuron_count - structural_neuron_count)

        # Early return for empty brain
        if neuron_count == 0 or fiber_count == 0:
            return self._empty_brain_report(neuron_count, synapse_count, fiber_count)

        # Compute individual metrics
        synapse_stats = enhanced.get("synapse_stats", {})

        # The remaining DB-bound reads are independent, so run them concurrently
        # rather than serially (measured ~1.6x on the shared HTTP connection).
        # orphan-rate needs both fibers and the connected-neuron set, so fetch
        # the set here and hand it in (avoids a second scan inside the compute).
        get_connected = getattr(self._storage, "get_connected_neuron_ids", None)

        async def _connected_or_none() -> set[str] | None:
            return await get_connected() if get_connected is not None else None

        async def _stage_distribution_or_none() -> dict[str, int] | None:
            try:
                return await self._storage.get_fiber_stage_counts(brain_id)
            except Exception:
                return None

        async def _semantic_gate_blockers_or_none() -> dict[str, int] | None:
            try:
                episodic_records = await self._storage.find_maturations(stage=MemoryStage.EPISODIC)
            except Exception:
                return None
            if not episodic_records:
                return None
            now = utcnow()
            counts = {"time_gate": 0, "spacing_gate": 0, "ready": 0}
            for record in episodic_records:
                counts[classify_episodic_blocker(record, now=now)] += 1
            return counts

        (
            fibers,
            activation_efficiency,
            connected,
            stage_distribution,
            semantic_gate_blockers,
        ) = await asyncio.gather(
            self._storage.get_fibers(limit=10000),
            # run 013: activation efficiency is scored against the ORGANIC brain
            # — indexed neurons are never recalled (access_frequency stays 0), so
            # including them deflates the ratio on code-heavy brains.
            self._compute_activation_efficiency(connectivity_neuron_count),
            _connected_or_none(),
            _stage_distribution_or_none(),
            _semantic_gate_blockers_or_none(),
        )

        # run 013: consolidation ratio over the ORGANIC fiber count. Computed
        # after the gather because it depends on `fibers` (the code_index tag
        # filter), which the gather just returned. Code-index fibers never
        # mature, so including them deflated the ratio on code-heavy brains.
        organic_fiber_count = sum(1 for f in fibers if "code_index" not in (f.tags or set()))
        consolidation_ratio = await self._compute_consolidation_ratio(
            fiber_count, organic_fiber_count
        )

        # Score connectivity on the semantic graph only — see _STRUCTURAL_SYNAPSE_TYPES.
        semantic_synapse_count = self._count_semantic_synapses(synapse_count, synapse_stats)
        # K2/K3 (run 013): connectivity is scored on the ORGANIC subgraph.
        # Denominator = organic neuron count (K2). Numerator = organic synapse
        # count: both endpoints non-indexed AND type not structural (variant c,
        # DECISIONS.md). The endpoint filter is what makes this honest — the
        # code-index synapse types (contains/co_occurs/is_a/related_to) are
        # shared with the organic encoder, so a type-only filter would drop real
        # semantic edges. Falls back to the type-filtered count when the backend
        # cannot answer the endpoint join (older backends / mocks).
        organic_synapse_count = int(synapse_stats.get("organic_synapse_count", -1))
        if organic_synapse_count < 0:
            organic_synapse_count = semantic_synapse_count
        connectivity = self._compute_connectivity(organic_synapse_count, connectivity_neuron_count)
        diversity = self._compute_diversity(synapse_stats)
        # run 013: freshness on the organic fiber set only.
        freshness = self._compute_freshness(fibers, organic_only=True)
        orphan_rate = await self._compute_orphan_rate(neuron_count, fibers, connected=connected)
        recall_confidence = self._compute_recall_confidence(synapse_stats)

        # Compute purity score
        purity = (
            connectivity * 0.25
            + diversity * 0.20
            + freshness * 0.15
            + consolidation_ratio * 0.15
            + (1.0 - orphan_rate) * 0.10
            + activation_efficiency * 0.10
            + recall_confidence * 0.05
        ) * 100

        # Apply penalty for unresolved CONTRADICTS synapses
        by_type = synapse_stats.get("by_type", {})
        contradicts_entry = by_type.get(SynapseType.CONTRADICTS, {})
        contradicts_count: int = (
            (
                contradicts_entry["count"]
                if isinstance(contradicts_entry, dict)
                else contradicts_entry
            )
            if contradicts_entry
            else 0
        )
        conflict_rate = contradicts_count / max(neuron_count, 1)
        conflict_penalty = min(10.0, conflict_rate * 50.0)  # Max 10 point penalty
        purity = max(0.0, purity - conflict_penalty)

        grade = _score_to_grade(purity)

        # Generate warnings and recommendations. The ratio quoted to the user must be
        # the one the score was computed from, otherwise the text and the bar disagree.
        # Use the organic numerator AND denominator so the quoted ratio matches the
        # 0-1 score, which is also computed on the organic subgraph (run 013).
        raw_connectivity = organic_synapse_count / max(connectivity_neuron_count, 1)
        warnings, recommendations = self._generate_diagnostics(
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            fiber_count=fiber_count,
            raw_connectivity=raw_connectivity,
            semantic_synapse_count=semantic_synapse_count,
            connectivity_score=connectivity,
            synapse_stats=synapse_stats,
            orphan_rate=orphan_rate,
            consolidation_ratio=consolidation_ratio,
            freshness=freshness,
            fibers=fibers,
            contradicts_count=contradicts_count,
        )

        # Rank penalty factors with actual metrics for dynamic action strings
        component_scores = {
            "connectivity": connectivity,
            "diversity": diversity,
            "freshness": freshness,
            "consolidation_ratio": consolidation_ratio,
            "orphan_rate": orphan_rate,
            "activation_efficiency": activation_efficiency,
            "recall_confidence": recall_confidence,
        }
        by_type = synapse_stats.get("by_type", {})
        penalty_metrics = {
            "neuron_count": neuron_count,
            "structural_neuron_count": structural_neuron_count,
            "connectivity_neuron_count": connectivity_neuron_count,
            "synapse_count": synapse_count,
            "semantic_synapse_count": semantic_synapse_count,
            "fiber_count": fiber_count,
            "freshness": freshness,
            "orphan_rate": orphan_rate,
            "activation_efficiency": activation_efficiency,
            "recall_confidence": recall_confidence,
            "consolidation_ratio": consolidation_ratio,
            "types_used": len(by_type),
        }
        top_penalties = _rank_penalty_factors(component_scores, metrics=penalty_metrics)

        return BrainHealthReport(
            purity_score=round(purity, 1),
            grade=grade,
            connectivity=round(connectivity, 4),
            diversity=round(diversity, 4),
            freshness=round(freshness, 4),
            consolidation_ratio=round(consolidation_ratio, 4),
            orphan_rate=round(orphan_rate, 4),
            activation_efficiency=round(activation_efficiency, 4),
            recall_confidence=round(recall_confidence, 4),
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            fiber_count=fiber_count,
            warnings=tuple(warnings),
            recommendations=tuple(recommendations),
            top_penalties=top_penalties,
            contradiction_count=contradicts_count,
            conflict_rate=round(conflict_rate, 4),
            semantic_synapse_count=semantic_synapse_count,
            structural_neuron_count=structural_neuron_count,
            connectivity_neuron_count=connectivity_neuron_count,
            stage_distribution=stage_distribution,
            semantic_gate_blockers=semantic_gate_blockers,
        )

    # ── Metric computations ──────────────────────────────────────

    @classmethod
    def _count_semantic_synapses(cls, synapse_count: int, synapse_stats: dict[str, Any]) -> int:
        """Total synapses minus the structural/dedup buckets.

        Falls back to the raw total when a backend reports no by_type breakdown —
        a possibly-inflated number beats silently claiming zero connectivity.
        """
        by_type = synapse_stats.get("by_type", {})
        if not by_type:
            return synapse_count

        structural = 0
        for stype, entry in by_type.items():
            # Backends key by SynapseType value ("alias"); some fixtures use the
            # member name ("ALIAS"), so match case-insensitively.
            if str(stype).lower() in cls._STRUCTURAL_SYNAPSE_TYPES:
                structural += int(entry["count"] if isinstance(entry, dict) else entry)
        return max(0, synapse_count - structural)

    @staticmethod
    def _compute_connectivity(semantic_synapse_count: int, neuron_count: int) -> float:
        """Normalize semantic synapses/neuron to a saturating 0-1 score.

        Takes the SEMANTIC synapse count (see _count_semantic_synapses), not the
        table total. sigmoid(-1.5 * (x - 3)) over that ratio: x=0 -> ~0.01,
        x=3 -> 0.5, x=8 -> ~1.0. Note 1.0 is a CEILING, not a ratio — anything
        from ~8 semantic synapses/neuron upward reports 1.0, so this score must
        never be presented against the raw "3-8 per neuron" target.
        """
        if neuron_count == 0:
            return 0.0
        raw = semantic_synapse_count / neuron_count
        return 1.0 / (1.0 + math.exp(-1.5 * (raw - 3.0)))

    # Synapse types realistically expected in typical usage.
    # Spatial/semantic types only appear with specialized content.
    _EXPECTED_SYNAPSE_TYPES = 8

    @staticmethod
    def _compute_diversity(synapse_stats: dict[str, Any]) -> float:
        """Compute synapse type diversity via Shannon entropy.

        Normalized against log(expected_types) rather than all defined types,
        since most brains won't use spatial/semantic types without specialized
        content. Using all 20 types as baseline unfairly penalizes typical usage.
        """
        by_type = synapse_stats.get("by_type", {})
        if not by_type:
            return 0.0

        type_counts = [
            entry["count"] if isinstance(entry, dict) else entry for entry in by_type.values()
        ]
        total = sum(type_counts)
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in type_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        expected_types = DiagnosticsEngine._EXPECTED_SYNAPSE_TYPES
        max_entropy = math.log(expected_types) if expected_types > 1 else 1.0
        return min(1.0, entropy / max_entropy)

    @staticmethod
    def _compute_freshness(fibers: list[Any], organic_only: bool = False) -> float:
        """Compute fraction of fibers accessed/created in last 7 days.

        run 013: when ``organic_only`` is set, code-index fibers are excluded —
        a bulk re-index touches every source file and stamps them all recent,
        which otherwise reports a brain with 20k indexed fibers as maximally
        fresh even when no real memory was stored in a week.
        """
        if organic_only:
            fibers = [f for f in fibers if "code_index" not in (f.tags or set())]
        if not fibers:
            return 0.0

        now = utcnow()
        cutoff = now - timedelta(days=7)
        fresh_count = sum(1 for f in fibers if (f.last_conducted or f.created_at) >= cutoff)
        return fresh_count / len(fibers)

    async def _compute_consolidation_ratio(
        self, fiber_count: int, organic_fiber_count: int | None = None
    ) -> float:
        """Compute fraction of fibers that reached SEMANTIC stage.

        run 013: scores against the ORGANIC fiber count when available —
        code-index fibers never mature past their initial stage, so including
        them in the denominator deflates the ratio on code-heavy brains.
        Falls back to the total when the organic count is unknown.
        """
        denom = organic_fiber_count if organic_fiber_count is not None else fiber_count
        if denom == 0:
            return 0.0
        semantic_records = await self._storage.find_maturations(
            stage=MemoryStage.SEMANTIC,
        )
        return len(semantic_records) / denom

    async def _compute_orphan_rate(
        self,
        neuron_count: int,
        fibers: list[Any] | None = None,
        connected: set[str] | None = None,
    ) -> float:
        """Compute fraction of neurons with no synapses AND no fiber membership.

        A neuron is considered "connected" if it appears in at least one
        synapse (source or target) OR belongs to at least one fiber.
        Previous implementation only checked synapses, inflating orphan
        counts for spatial/temporal neurons that are fiber-linked but
        have no direct synapses.
        """
        if neuron_count == 0:
            return 0.0

        # The caller may hand in the connected-neuron set (computed concurrently
        # with the other reads). Otherwise fetch it: prefer the DB-side
        # distinct-endpoints aggregate over loading ~185k Synapse objects (that
        # scan was seconds of the dashboard's slowness).
        if connected is None:
            get_connected = getattr(self._storage, "get_connected_neuron_ids", None)
            if get_connected is not None:
                connected = set(await get_connected())
            else:
                connected = set()
                for s in await self._storage.get_all_synapses():
                    connected.add(s.source_id)
                    connected.add(s.target_id)
        else:
            connected = set(connected)

        # Also count neurons that belong to fibers as connected
        if fibers:
            for fiber in fibers:
                connected.update(fiber.neuron_ids)

        orphan_count = max(0, neuron_count - len(connected))
        return orphan_count / neuron_count

    async def _compute_activation_efficiency(self, neuron_count: int) -> float:
        """Compute fraction of neurons that have been activated at least once.

        Proxy metric: neurons with access_frequency > 0 indicate the brain
        is actively utilizing its neural graph during retrieval.
        """
        if neuron_count == 0:
            return 0.0

        # DB aggregate instead of loading every neuron_state (~64k rows).
        count_active = getattr(self._storage, "count_activated_neuron_states", None)
        if count_active is not None:
            activated_count = int(await count_active())
        else:
            states = await self._storage.get_all_neuron_states()
            activated_count = sum(1 for s in states if s.access_frequency > 0)
        return activated_count / max(neuron_count, 1)

    @staticmethod
    def _compute_recall_confidence(synapse_stats: dict[str, Any]) -> float:
        """Compute recall confidence from average synapse weight.

        Higher average weight indicates stronger recall pathways.
        """
        avg_weight: float = synapse_stats.get("avg_weight", 0.0)
        return min(1.0, max(0.0, avg_weight))

    # ── Warning and recommendation generation ────────────────────

    def _generate_diagnostics(
        self,
        *,
        neuron_count: int,
        synapse_count: int,
        fiber_count: int,
        raw_connectivity: float,
        synapse_stats: dict[str, Any],
        orphan_rate: float,
        consolidation_ratio: float,
        freshness: float,
        fibers: list[Any],
        contradicts_count: int = 0,
        semantic_synapse_count: int | None = None,
        connectivity_score: float | None = None,
    ) -> tuple[list[DiagnosticWarning], list[str]]:
        """Generate warnings and recommendations from metrics.

        `raw_connectivity` is semantic synapses per neuron; `connectivity_score` is
        the 0-1 sigmoid of it. Both are accepted so the text can name each number's
        unit instead of mixing them.
        """
        warnings: list[DiagnosticWarning] = []
        recommendations: list[str] = []

        # Stale brain
        if fiber_count > 0 and freshness == 0.0:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.CRITICAL,
                    code="STALE_BRAIN",
                    message="No fibers accessed or created in the last 7 days.",
                )
            )
            recommendations.append(
                f"Brain has {fiber_count} memories but none accessed recently. "
                "Try: smem_recall with a topic you're currently working on "
                "to reactivate relevant memories."
            )

        # Low connectivity. Every number here is in synapses/neuron, the same unit as
        # the 3-8 target; the 0-1 score rides along in details so a renderer showing
        # both never implies "score 1.0 is below the 3-8 target".
        if raw_connectivity < 2.0 and neuron_count > 0:
            semantic_count = (
                synapse_count if semantic_synapse_count is None else semantic_synapse_count
            )
            gap = max(0, int(3.0 * neuron_count) - semantic_count)
            details: dict[str, Any] = {
                "semantic_synapses_per_neuron": round(raw_connectivity, 2),
                "semantic_synapse_count": semantic_count,
                "structural_synapse_count": max(0, synapse_count - semantic_count),
            }
            if connectivity_score is not None:
                details["connectivity_score"] = round(connectivity_score, 4)
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="LOW_CONNECTIVITY",
                    message=(
                        f"Low connectivity: {raw_connectivity:.1f} semantic synapses/neuron "
                        "(healthy: 3-8 per neuron; alias and audit edges excluded)."
                    ),
                    details=details,
                )
            )
            recommendations.append(
                f"Low connectivity ({raw_connectivity:.1f} semantic synapses/neuron, "
                f"healthy 3-8 per neuron). ~{gap} more connections needed — alias and "
                "audit edges do not count. Store memories with context like "
                "'X because Y' or 'after doing A, I learned B' to build richer links."
            )

        # Low diversity. Fires on type COVERAGE (fewer than 3 distinct types in the
        # by_type breakdown), which is a different question from the 0-1 diversity
        # score — that one is entropy over the same breakdown and can sit low while
        # many types are in use but one dominates. A brain using 17 types is not
        # under-covered, so this warning stays silent there by design.
        by_type = synapse_stats.get("by_type", {})
        types_used = len(by_type)
        expected = DiagnosticsEngine._EXPECTED_SYNAPSE_TYPES
        if types_used < 3 and synapse_count > 0:
            used_names = sorted(by_type.keys()) if by_type else []
            missing_hint = ""
            common_types = {"caused_by", "leads_to", "related_to", "co_occurs"}
            missing = common_types - set(used_names)
            if missing:
                missing_hint = f" Missing types: {', '.join(sorted(missing))}."
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="LOW_DIVERSITY",
                    message=(
                        f"Low synapse diversity: {types_used} of {expected} common "
                        "synapse types in use."
                    ),
                    details={"types_used": types_used, "types_expected": expected},
                )
            )
            recommendations.append(
                f"Only {types_used}/{expected} synapse types used ({', '.join(used_names) or 'none'}).{missing_hint} "
                "Store memories describing causes, sequences, and relationships."
            )

        # High orphan rate
        if orphan_rate > 0.20:
            orphan_count = int(orphan_rate * neuron_count) if neuron_count > 0 else 0
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="HIGH_ORPHAN_RATE",
                    message=f"High orphan rate: {orphan_rate:.0%} of neurons have no synapses or fiber links.",
                )
            )
            recommendations.append(
                f"{orphan_count} neurons ({orphan_rate:.0%}) have no synapses or fiber links. "
                "Run: smem consolidate --strategy prune to remove orphans, "
                "or recall related topics to build connections."
            )

        # No consolidation
        if consolidation_ratio == 0.0 and fiber_count > 0:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="NO_CONSOLIDATION",
                    message="No memories have reached SEMANTIC stage.",
                )
            )
            recommendations.append(
                f"All {fiber_count} memories are still episodic (not consolidated). "
                "They advance to semantic only through spaced recall — reinforcement "
                "spread across 3+ distinct days (or 15+ rehearsals across 5+ time "
                "windows). Running `smem consolidate` will not move this on its own."
            )

        # Tag drift detection
        all_tags: set[str] = set()
        for fiber in fibers:
            all_tags |= fiber.tags
        if all_tags:
            normalizer = TagNormalizer()
            drift_reports = normalizer.detect_drift(all_tags)
            for drift in drift_reports:
                warnings.append(
                    DiagnosticWarning(
                        severity=WarningSeverity.INFO,
                        code="TAG_DRIFT",
                        message=f"Tag drift: {drift.variants} -> '{drift.canonical}'",
                        details={
                            "canonical": drift.canonical,
                            "variants": drift.variants,
                        },
                    )
                )
            if drift_reports:
                recommendations.append("Normalize tags to reduce semantic drift.")

        # High conflict count (unresolved CONTRADICTS synapses)
        if contradicts_count > 5:
            warnings.append(
                DiagnosticWarning(
                    severity=WarningSeverity.WARNING,
                    code="HIGH_CONFLICT_COUNT",
                    message=f"{contradicts_count} unresolved memory conflicts detected.",
                    details={"count": contradicts_count},
                )
            )
            recommendations.append(
                "Run `smem_conflicts` to review and resolve memory contradictions."
            )

        return warnings, recommendations

    # ── Quality badge ────────────────────────────────────────────

    async def compute_quality_badge(self, brain_id: str) -> QualityBadge:
        """Compute a quality badge for the brain.

        Runs full diagnostics and maps the result to a marketplace-ready badge.

        Args:
            brain_id: ID of the brain to evaluate

        Returns:
            QualityBadge with grade, purity score, and eligibility
        """
        report = await self.analyze(brain_id)

        badge_labels = {
            "A": "A - Excellent",
            "B": "B - Good",
            "C": "C - Fair",
            "D": "D - Poor",
            "F": "F - Failing",
        }

        return QualityBadge(
            grade=report.grade,
            purity_score=report.purity_score,
            marketplace_eligible=report.grade in ("A", "B"),
            badge_label=badge_labels.get(report.grade, f"{report.grade} - Unknown"),
            computed_at=utcnow(),
            component_summary={
                "connectivity": report.connectivity,
                "diversity": report.diversity,
                "freshness": report.freshness,
                "consolidation_ratio": report.consolidation_ratio,
                "orphan_rate": report.orphan_rate,
                "activation_efficiency": report.activation_efficiency,
                "recall_confidence": report.recall_confidence,
            },
        )

    # ── Empty brain helper ───────────────────────────────────────

    @staticmethod
    def _empty_brain_report(
        neuron_count: int,
        synapse_count: int,
        fiber_count: int,
    ) -> BrainHealthReport:
        """Return a minimal report for an empty brain."""
        return BrainHealthReport(
            purity_score=0.0,
            grade="F",
            connectivity=0.0,
            diversity=0.0,
            freshness=0.0,
            consolidation_ratio=0.0,
            orphan_rate=0.0,
            activation_efficiency=0.0,
            recall_confidence=0.0,
            neuron_count=neuron_count,
            synapse_count=synapse_count,
            fiber_count=fiber_count,
            warnings=(
                DiagnosticWarning(
                    severity=WarningSeverity.CRITICAL,
                    code="EMPTY_BRAIN",
                    message="Brain has no memories stored.",
                ),
            ),
            recommendations=("Start storing memories with smem_remember.",),
        )

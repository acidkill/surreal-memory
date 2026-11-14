"""Post-consolidation quality delta report.

Wraps DiagnosticsEngine + ConsolidationEngine to compute a before/after
health snapshot. Keeps the two engines decoupled — composition happens
here at the call site.

Usage:
    delta = await run_with_delta(storage, brain_id, strategies=strategies)
    print(delta.summary())
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.engine.consolidation import (
        ConsolidationConfig,
        ConsolidationReport,
        ConsolidationStrategy,
    )
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import TierConfig


@dataclass(frozen=True)
class HealthSnapshot:
    """Frozen snapshot of key health metrics at a point in time."""

    purity_score: float
    grade: str
    connectivity: float
    diversity: float
    freshness: float
    consolidation_ratio: float
    orphan_rate: float
    activation_efficiency: float
    recall_confidence: float
    neuron_count: int
    synapse_count: int
    fiber_count: int


@dataclass(frozen=True)
class ConsolidationDelta:
    """Immutable before/after health comparison from a consolidation run.

    Attributes:
        before: Health snapshot before consolidation.
        after: Health snapshot after consolidation.
        report: The raw consolidation report.
        computed_at: When this delta was computed.
    """

    before: HealthSnapshot | None
    after: HealthSnapshot | None
    report: ConsolidationReport
    computed_at: datetime

    @property
    def purity_delta(self) -> float | None:
        """Change in purity score (positive = improvement)."""
        if self.before is None or self.after is None:
            return None
        return round(self.after.purity_score - self.before.purity_score, 1)

    @property
    def connectivity_delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return round(self.after.connectivity - self.before.connectivity, 4)

    @property
    def orphan_rate_delta(self) -> float | None:
        """Change in orphan rate (negative = improvement)."""
        if self.before is None or self.after is None:
            return None
        return round(self.after.orphan_rate - self.before.orphan_rate, 4)

    def summary(self) -> str:
        """Human-readable delta summary."""
        if self.before is None or self.after is None:
            return (
                self.report.summary() + "\n  Health delta unavailable: diagnostic snapshot failed"
            )
        status = self.report.extra.get("consolidation_status")
        if status == "blocked" or (
            status in {"failed", "lease_lost", "timed_out"} and self.before == self.after
        ):
            return self.report.summary()
        lines = [self.report.summary(), "", "Health Delta:"]

        metrics = [
            ("Purity", self.before.purity_score, self.after.purity_score, self.purity_delta),
            (
                "Connectivity",
                self.before.connectivity,
                self.after.connectivity,
                self.connectivity_delta,
            ),
            (
                "Orphan rate",
                self.before.orphan_rate,
                self.after.orphan_rate,
                self.orphan_rate_delta,
            ),
            (
                "Freshness",
                self.before.freshness,
                self.after.freshness,
                round(self.after.freshness - self.before.freshness, 4),
            ),
            (
                # Named for what it measures, not for the pass it appears under. It is
                # the share of fibers that reached the SEMANTIC maturation stage — NOT a
                # compression or merge ratio, which is how the old label read directly
                # beneath the merge counters. The wire field keeps its old name so the
                # API and dashboard contract is unchanged.
                "Semantic maturation ratio",
                self.before.consolidation_ratio,
                self.after.consolidation_ratio,
                round(self.after.consolidation_ratio - self.before.consolidation_ratio, 4),
            ),
        ]

        for label, before_val, after_val, delta in metrics:
            assert delta is not None  # Both snapshots were checked above.
            sign = "+" if delta > 0 else ""
            lines.append(f"  {label}: {before_val} -> {after_val} ({sign}{delta})")

        grade_changed = self.before.grade != self.after.grade
        if grade_changed:
            lines.append(f"  Grade: {self.before.grade} -> {self.after.grade}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize for MCP/CLI JSON output."""
        fields = (
            "purity_score",
            "grade",
            "connectivity",
            "diversity",
            "orphan_rate",
            "neuron_count",
            "synapse_count",
            "fiber_count",
        )

        def snapshot(value: HealthSnapshot | None) -> dict[str, object] | None:
            return (
                {key: item for key, item in asdict(value).items() if key in fields}
                if value
                else None
            )

        return {
            "before": snapshot(self.before),
            "after": snapshot(self.after),
            "delta": {
                "purity": self.purity_delta,
                "connectivity": self.connectivity_delta,
                "orphan_rate": self.orphan_rate_delta,
            },
            "grade_changed": (
                self.before.grade != self.after.grade
                if self.before is not None and self.after is not None
                else None
            ),
            "diagnostic_errors": self.report.extra.get("diagnostic_errors", {}),
        }


def _snapshot_from_report(report: object) -> HealthSnapshot:
    """Extract a HealthSnapshot from a BrainHealthReport."""
    from surreal_memory.engine.diagnostics import BrainHealthReport

    assert isinstance(report, BrainHealthReport)
    return HealthSnapshot(
        purity_score=report.purity_score,
        grade=report.grade,
        connectivity=report.connectivity,
        diversity=report.diversity,
        freshness=report.freshness,
        consolidation_ratio=report.consolidation_ratio,
        orphan_rate=report.orphan_rate,
        activation_efficiency=report.activation_efficiency,
        recall_confidence=report.recall_confidence,
        neuron_count=report.neuron_count,
        synapse_count=report.synapse_count,
        fiber_count=report.fiber_count,
    )


async def run_with_delta(
    storage: NeuralStorage,
    brain_id: str,
    *,
    strategies: list[ConsolidationStrategy] | None = None,
    dry_run: bool = False,
    config: ConsolidationConfig | None = None,
    reference_time: datetime | None = None,
    tier_config: TierConfig | None = None,
    on_progress: Callable[[str], None] | None = None,
    resume_run_id: str | None = None,
    supersede_failed_semantic_link: bool = False,
) -> ConsolidationDelta:
    """Run consolidation and compute a before/after health delta.

    1. DiagnosticsEngine.analyze() -> before snapshot
    2. ConsolidationEngine.run() -> report
    3. DiagnosticsEngine.analyze() -> after snapshot
    4. Return ConsolidationDelta with all three

    Args:
        storage: Neural storage backend.
        brain_id: Brain to consolidate and diagnose.
        strategies: Consolidation strategies (default: ALL).
        dry_run: If True, preview only.
        config: Optional consolidation config overrides.
        reference_time: Reference time for consolidation.
        resume_run_id: Require this unfinished run under the consolidation lease.
        supersede_failed_semantic_link: Let a run of another strategy replace the record of a
            failed (or too old to resume) lone semantic_link run instead of being refused.
            Ignored together with ``resume_run_id``.

    Returns:
        ConsolidationDelta with before/after snapshots and report.
    """
    from surreal_memory.engine.consolidation import ConsolidationEngine
    from surreal_memory.engine.diagnostics import DiagnosticsEngine

    diagnostics = DiagnosticsEngine(storage)
    engine = ConsolidationEngine(
        storage,
        config,
        tier_config=tier_config,
        supersede_failed_semantic_link=supersede_failed_semantic_link,
    )

    diagnostic_errors: dict[str, str] = {}

    async def snapshot(stage: str) -> HealthSnapshot | None:
        try:
            return _snapshot_from_report(await diagnostics.analyze(brain_id))
        except Exception as exc:
            diagnostic_errors[stage] = type(exc).__name__
            return None

    # Diagnostics are advisory: failure must not block a saved run's finalization.
    before = await snapshot("before")

    # Step 2: Run consolidation
    report = await engine.run(
        strategies=strategies,
        dry_run=dry_run,
        reference_time=reference_time,
        on_progress=on_progress,
        resume_run_id=resume_run_id,
    )

    # Step 3: After snapshot
    after = await snapshot("after")
    if diagnostic_errors:
        report.extra["diagnostic_errors"] = diagnostic_errors

    return ConsolidationDelta(
        before=before,
        after=after,
        report=report,
        computed_at=utcnow(),
    )

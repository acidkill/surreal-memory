"""Cognitive layer — pure functions for hypothesis/prediction reasoning.

All functions are stateless and testable. No storage, no LLM dependency.
"""

from __future__ import annotations

from typing import Literal

# --------------- Confidence Update (Bayesian-inspired) ---------------


def update_confidence(
    current: float,
    evidence_type: Literal["for", "against"],
    weight: float = 0.5,
    for_count: int = 0,
    against_count: int = 0,
) -> float:
    """Update hypothesis confidence based on new evidence.

    Uses a sigmoid-dampened shift that:
    - Moves faster when evidence is surprising (against prior)
    - Moves slower when evidence confirms (diminishing returns)
    - Never reaches exactly 0.0 or 1.0 (always revisable)

    Args:
        current: Current confidence [0.01, 0.99].
        evidence_type: "for" or "against".
        weight: Evidence strength [0.1, 1.0]. Default 0.5.
        for_count: Total evidence-for count (BEFORE this update).
        against_count: Total evidence-against count (BEFORE this update).

    Returns:
        Updated confidence clamped to [0.01, 0.99].
    """
    if evidence_type not in ("for", "against"):
        raise ValueError(f"evidence_type must be 'for' or 'against', got {evidence_type!r}")

    weight = max(0.1, min(1.0, weight))
    current = max(0.01, min(0.99, current))

    direction = 1.0 if evidence_type == "for" else -1.0

    # Surprise factor: evidence against strong belief moves more
    surprise = (1.0 - current) if direction > 0 else current

    # Dampening: more total evidence = smaller individual impact
    total_evidence = for_count + against_count
    dampening = 1.0 / (1.0 + 0.1 * total_evidence)

    shift = direction * weight * surprise * dampening * 0.3
    new_confidence = current + shift

    return max(0.01, min(0.99, new_confidence))


# --------------- Auto-Resolution ---------------


def detect_auto_resolution(
    confidence: float,
    for_count: int,
    against_count: int,
) -> str | None:
    """Check if a hypothesis should be auto-resolved.

    Args:
        confidence: Current confidence.
        for_count: Evidence supporting the hypothesis.
        against_count: Evidence against the hypothesis.

    Returns:
        "confirmed" if high confidence with enough evidence,
        "refuted" if low confidence with enough evidence,
        None if still active.
    """
    if confidence >= 0.9 and for_count >= 3:
        return "confirmed"
    if confidence <= 0.1 and against_count >= 3:
        return "refuted"
    return None


# --------------- Calibration Score ---------------


def compute_calibration(correct_count: int, total_resolved: int) -> float:
    """Compute prediction accuracy (calibration score).

    Args:
        correct_count: Number of correct predictions.
        total_resolved: Total resolved predictions (correct + wrong).

    Returns:
        Calibration score [0.0, 1.0]. Returns 0.5 if no data.
    """
    if total_resolved == 0:
        return 0.5
    return correct_count / total_resolved


# --------------- Hot Index Scoring ---------------


def score_hypothesis(
    confidence: float,
    evidence_count: int,
    age_days: float,
) -> float:
    """Score a hypothesis for hot index ranking.

    Active hypotheses with more evidence and moderate confidence rank higher.
    Extreme confidence (near 0 or 1) ranks lower (already resolved).

    Args:
        confidence: Current confidence.
        evidence_count: Total evidence items.
        age_days: Days since creation.

    Returns:
        Score [0.0, ~10.0]. Higher = more relevant for hot index.
    """
    # Middle confidence is more interesting than extremes
    confidence_interest = 1.0 - abs(confidence - 0.5) * 2.0

    # More evidence = more developed hypothesis
    evidence_factor = min(evidence_count / 5.0, 1.0)

    # Recent hypotheses rank higher
    recency = 1.0 / (1.0 + age_days / 30.0)

    return confidence_interest * 3.0 + evidence_factor * 4.0 + recency * 3.0


def score_prediction(days_until_deadline: float) -> float:
    """Score a prediction for hot index ranking.

    Predictions closer to deadline rank higher.

    Args:
        days_until_deadline: Days remaining. Negative = overdue.

    Returns:
        Score [0.0, ~10.0]. Higher = more urgent.
    """
    if days_until_deadline < 0:
        return 10.0  # Overdue predictions are most urgent
    return 10.0 / (1.0 + days_until_deadline / 3.0)


# --------------- Knowledge Gap Priority ---------------


_SOURCE_BASE_PRIORITY: dict[str, float] = {
    "contradicting_evidence": 0.8,
    "low_confidence_hypothesis": 0.7,
    "user_flagged": 0.6,
    "recall_miss": 0.5,
    "stale_schema": 0.4,
}


def gap_priority(detection_source: str) -> float:
    """Get default priority for a knowledge gap based on detection source.

    Args:
        detection_source: How the gap was detected.

    Returns:
        Priority [0.0, 1.0].
    """
    return _SOURCE_BASE_PRIORITY.get(detection_source, 0.5)


# --------------- Summary Truncation ---------------


def truncate_summary(content: str, max_length: int = 120) -> str:
    """Truncate content to a one-line summary for hot index.

    Args:
        content: Full content text.
        max_length: Maximum length.

    Returns:
        Truncated string with ellipsis if needed.
    """
    line = content.replace("\n", " ").strip()
    if len(line) <= max_length:
        return line
    return line[: max_length - 1] + "\u2026"

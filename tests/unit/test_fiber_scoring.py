"""Tests for activation-aware fiber scoring (B5)."""

from __future__ import annotations

import math
from datetime import timedelta

from neural_memory.core.fiber import Fiber
from neural_memory.engine.activation import ActivationResult
from neural_memory.utils.timeutils import utcnow


def _make_activation(neuron_id: str, level: float) -> ActivationResult:
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=1,
        path=[neuron_id],
        source_anchor="anchor-0",
    )


def _make_fiber(
    fiber_id: str,
    neuron_ids: set[str],
    salience: float = 0.5,
    conductivity: float = 1.0,
    stage: str = "episodic",
    hours_ago: float = 24.0,
) -> Fiber:
    from dataclasses import replace

    now = utcnow()
    fiber = Fiber.create(
        neuron_ids=neuron_ids,
        synapse_ids=set(),
        anchor_neuron_id=next(iter(neuron_ids)),
        fiber_id=fiber_id,
    )
    fiber = replace(
        fiber,
        salience=salience,
        conductivity=conductivity,
        last_conducted=now - timedelta(hours=hours_ago),
    )
    # stage is a DB column not in Fiber dataclass — set via attribute for testing
    object.__setattr__(fiber, "stage", stage)
    return fiber


class TestActivationAwareFiberScoring:
    """Fiber scoring uses neuron activations for relevance ranking."""

    def test_high_coverage_fiber_ranks_higher(self) -> None:
        """Fiber with more activated neurons scores higher than one with fewer."""
        activations = {
            "n1": _make_activation("n1", 0.8),
            "n2": _make_activation("n2", 0.7),
            "n3": _make_activation("n3", 0.6),
        }

        # Fiber A: 3/3 neurons activated (100% coverage)
        fiber_a = _make_fiber("f-a", {"n1", "n2", "n3"})
        # Fiber B: 1/3 neurons activated (33% coverage)
        fiber_b = _make_fiber("f-b", {"n1", "n4", "n5"})

        score_a = _compute_fiber_score(fiber_a, activations)
        score_b = _compute_fiber_score(fiber_b, activations)

        assert score_a > score_b, "Higher coverage fiber should score higher"

    def test_strong_activation_ranks_higher(self) -> None:
        """Fiber with strongly activated neurons scores higher."""
        activations_strong = {
            "n1": _make_activation("n1", 0.9),
        }
        activations_weak = {
            "n1": _make_activation("n1", 0.2),
        }

        fiber = _make_fiber("f1", {"n1", "n2"})

        score_strong = _compute_fiber_score(fiber, activations_strong)
        score_weak = _compute_fiber_score(fiber, activations_weak)

        assert score_strong > score_weak

    def test_semantic_stage_bonus(self) -> None:
        """Semantic fibers get 1.1x stage multiplier."""
        activations = {"n1": _make_activation("n1", 0.7)}

        fiber_semantic = _make_fiber("f-sem", {"n1"}, stage="semantic")
        fiber_episodic = _make_fiber("f-epi", {"n1"}, stage="episodic")

        score_sem = _compute_fiber_score(fiber_semantic, activations)
        score_epi = _compute_fiber_score(fiber_episodic, activations)

        assert score_sem > score_epi
        # Should be exactly 1.1x
        assert abs(score_sem / score_epi - 1.1) < 0.01

    def test_no_activated_neurons_minimal_score(self) -> None:
        """Fiber with zero activated neurons gets minimal activation signal."""
        activations = {"n1": _make_activation("n1", 0.8)}

        # Fiber has only n2, n3 — none are activated
        fiber = _make_fiber("f1", {"n2", "n3"}, salience=0.8)

        score = _compute_fiber_score(fiber, activations)
        # Should be very low (0.05 activation signal)
        assert score < 0.1

    def test_recency_affects_score(self) -> None:
        """More recently accessed fibers score higher."""
        activations = {"n1": _make_activation("n1", 0.7)}

        fiber_recent = _make_fiber("f-recent", {"n1"}, hours_ago=1)
        fiber_old = _make_fiber("f-old", {"n1"}, hours_ago=200)

        score_recent = _compute_fiber_score(fiber_recent, activations)
        score_old = _compute_fiber_score(fiber_old, activations)

        assert score_recent > score_old

    def test_salience_affects_score(self) -> None:
        """Higher salience fibers score higher."""
        activations = {"n1": _make_activation("n1", 0.7)}

        fiber_high = _make_fiber("f-high", {"n1"}, salience=0.9)
        fiber_low = _make_fiber("f-low", {"n1"}, salience=0.2)

        score_high = _compute_fiber_score(fiber_high, activations)
        score_low = _compute_fiber_score(fiber_low, activations)

        assert score_high > score_low

    def test_conductivity_multiplier(self) -> None:
        """Conductivity acts as a multiplier on the final score."""
        activations = {"n1": _make_activation("n1", 0.7)}

        fiber_conductive = _make_fiber("f-c", {"n1"}, conductivity=1.0)
        fiber_weak = _make_fiber("f-w", {"n1"}, conductivity=0.3)

        score_c = _compute_fiber_score(fiber_conductive, activations)
        score_w = _compute_fiber_score(fiber_weak, activations)

        assert score_c > score_w
        # Should be proportional to conductivity ratio
        ratio = score_c / score_w
        assert abs(ratio - 1.0 / 0.3) < 0.5

    def test_coverage_formula(self) -> None:
        """Coverage = activated_count / total_neuron_count."""
        activations = {
            "n1": _make_activation("n1", 0.5),
            "n2": _make_activation("n2", 0.5),
        }

        # 2 out of 4 neurons activated = 50% coverage
        fiber = _make_fiber("f1", {"n1", "n2", "n3", "n4"})
        # 2 out of 2 neurons activated = 100% coverage
        fiber_full = _make_fiber("f2", {"n1", "n2"})

        score_partial = _compute_fiber_score(fiber, activations)
        score_full = _compute_fiber_score(fiber_full, activations)

        # Full coverage should score higher (same activations)
        assert score_full > score_partial


def _compute_fiber_score(
    fiber: Fiber,
    activations: dict[str, ActivationResult],
    freshness_weight: float = 0.0,
) -> float:
    """Replicate the _fiber_score logic from retrieval.py for testing."""
    # Base quality
    recency = 0.5
    if fiber.last_conducted:
        hours_ago = (utcnow() - fiber.last_conducted).total_seconds() / 3600
        recency = max(0.1, 1.0 / (1.0 + math.exp((hours_ago - 72) / 36)))

    base_score = fiber.salience * recency * fiber.conductivity

    if freshness_weight > 0.0 and fiber.created_at:
        from neural_memory.safety.freshness import evaluate_freshness

        age_result = evaluate_freshness(fiber.created_at)
        base_score *= (1.0 - freshness_weight) + freshness_weight * age_result.score

    # Activation relevance
    activated = [nid for nid in fiber.neuron_ids if nid in activations]
    if activated:
        coverage = len(activated) / max(len(fiber.neuron_ids), 1)
        max_act = max(activations[nid].activation_level for nid in activated)
        mean_act = sum(activations[nid].activation_level for nid in activated) / len(activated)
        activation_signal = max_act * 0.5 + coverage * 0.3 + mean_act * 0.2
        activation_signal = max(0.05, activation_signal)
    else:
        activation_signal = 0.05

    # Stage bonus
    stage = getattr(fiber, "stage", None)
    stage_multiplier = 1.1 if stage == "semantic" else 1.0

    return base_score * activation_signal * stage_multiplier

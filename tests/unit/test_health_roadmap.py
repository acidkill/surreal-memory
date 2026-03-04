"""Tests for enhanced health roadmap with dynamic actions and timeframes."""

from __future__ import annotations

from typing import Any

from neural_memory.engine.diagnostics import _build_dynamic_action, _rank_penalty_factors


class TestBuildDynamicAction:
    """Test dynamic action string generation with actual metrics."""

    def test_connectivity_action_includes_ratio(self) -> None:
        metrics: dict[str, Any] = {
            "neuron_count": 100,
            "synapse_count": 50,
            "fiber_count": 30,
        }
        action = _build_dynamic_action("connectivity", "fallback", metrics)
        assert "0.5 synapses/neuron" in action
        assert "target: 3.0" in action

    def test_connectivity_action_includes_gap(self) -> None:
        metrics: dict[str, Any] = {
            "neuron_count": 100,
            "synapse_count": 50,
            "fiber_count": 30,
        }
        action = _build_dynamic_action("connectivity", "fallback", metrics)
        # gap = 3*100 - 50 = 250
        assert "250" in action

    def test_diversity_action_includes_types_used(self) -> None:
        metrics: dict[str, Any] = {"types_used": 3}
        action = _build_dynamic_action("diversity", "fallback", metrics)
        assert "3 of 8" in action

    def test_freshness_action_includes_count(self) -> None:
        metrics: dict[str, Any] = {
            "freshness": 0.2,
            "fiber_count": 50,
        }
        action = _build_dynamic_action("freshness", "fallback", metrics)
        assert "10 active in last 7 days" in action

    def test_consolidation_action_includes_pct(self) -> None:
        metrics: dict[str, Any] = {"consolidation_ratio": 0.3}
        action = _build_dynamic_action("consolidation_ratio", "fallback", metrics)
        assert "70%" in action
        assert "episodic" in action

    def test_orphan_rate_action_includes_count(self) -> None:
        metrics: dict[str, Any] = {
            "neuron_count": 200,
            "orphan_rate": 0.25,
        }
        action = _build_dynamic_action("orphan_rate", "fallback", metrics)
        assert "50 neurons have no synapses or fiber links" in action

    def test_activation_efficiency_action(self) -> None:
        metrics: dict[str, Any] = {"activation_efficiency": 0.6}
        action = _build_dynamic_action("activation_efficiency", "fallback", metrics)
        assert "40%" in action
        assert "never accessed" in action

    def test_recall_confidence_action(self) -> None:
        metrics: dict[str, Any] = {"recall_confidence": 0.35}
        action = _build_dynamic_action("recall_confidence", "fallback", metrics)
        assert "0.35" in action
        assert "target: 0.50" in action

    def test_fallback_when_no_metrics(self) -> None:
        action = _build_dynamic_action("connectivity", "static fallback", None)
        assert action == "static fallback"

    def test_unknown_component_returns_static(self) -> None:
        action = _build_dynamic_action("unknown_component", "static", {"neuron_count": 10})
        assert action == "static"


class TestRankPenaltyFactorsWithMetrics:
    """Test that penalty factors use dynamic actions when metrics provided."""

    def test_penalties_have_dynamic_actions(self) -> None:
        scores = {
            "connectivity": 0.3,
            "diversity": 0.5,
            "freshness": 0.2,
            "consolidation_ratio": 0.1,
            "orphan_rate": 0.4,
            "activation_efficiency": 0.3,
            "recall_confidence": 0.4,
        }
        metrics = {
            "neuron_count": 100,
            "synapse_count": 30,
            "fiber_count": 50,
            "freshness": 0.2,
            "orphan_rate": 0.4,
            "activation_efficiency": 0.3,
            "recall_confidence": 0.4,
            "consolidation_ratio": 0.1,
            "types_used": 3,
        }
        penalties = _rank_penalty_factors(scores, metrics=metrics)
        # All penalties should have concrete numbers or metrics in actions
        for p in penalties:
            # Dynamic actions contain numbers (not just static text)
            has_number = any(c.isdigit() for c in p.action)
            assert has_number, f"Action for {p.component} should contain numbers: {p.action}"

    def test_penalties_without_metrics_use_static(self) -> None:
        scores = {
            "connectivity": 0.3,
            "diversity": 0.5,
            "freshness": 0.8,
            "consolidation_ratio": 0.8,
            "orphan_rate": 0.1,
            "activation_efficiency": 0.8,
            "recall_confidence": 0.8,
        }
        penalties = _rank_penalty_factors(scores)
        # Top penalty should be connectivity (lowest score, highest weight)
        assert penalties[0].component == "connectivity"
        assert "causes" in penalties[0].action or "connections" in penalties[0].action

    def test_sorted_by_penalty_points(self) -> None:
        scores = {
            "connectivity": 0.1,  # Very low → high penalty
            "diversity": 0.9,  # Very high → low penalty
            "freshness": 0.5,
            "consolidation_ratio": 0.5,
            "orphan_rate": 0.2,
            "activation_efficiency": 0.5,
            "recall_confidence": 0.5,
        }
        penalties = _rank_penalty_factors(scores, top_n=3)
        assert len(penalties) == 3
        assert penalties[0].penalty_points >= penalties[1].penalty_points
        assert penalties[1].penalty_points >= penalties[2].penalty_points

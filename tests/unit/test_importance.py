"""Tests for A4-B: Auto-Importance Scoring — heuristic priority for memories."""

from __future__ import annotations

from neural_memory.engine.importance import auto_importance_score


class TestAutoImportanceScore:
    """Test heuristic importance scoring for nmem_remember."""

    def test_default_is_5(self) -> None:
        """Plain content with no signals scores baseline 5."""
        score = auto_importance_score("Some generic text about a topic we discussed", "fact", [])
        assert score == 5

    def test_decision_type_bonus(self) -> None:
        """Decision type gets +2 bonus."""
        score = auto_importance_score("Chose the database engine for the project", "decision", [])
        assert score >= 7  # 5 base + 2 type

    def test_error_type_bonus(self) -> None:
        """Error type gets +2 bonus."""
        score = auto_importance_score("Root cause was null pointer", "error", [])
        assert score >= 7

    def test_preference_type_bonus(self) -> None:
        """Preference type gets +3 bonus."""
        score = auto_importance_score("User prefers dark mode", "preference", [])
        assert score >= 8

    def test_instruction_type_bonus(self) -> None:
        """Instruction type gets +3 bonus."""
        score = auto_importance_score("Always run tests before commit", "instruction", [])
        assert score >= 8

    def test_context_type_penalty(self) -> None:
        """Context type gets -1 penalty."""
        score = auto_importance_score("Working on project X", "context", [])
        assert score <= 5

    def test_causal_language_bonus(self) -> None:
        """Content with causal language gets +1."""
        score = auto_importance_score(
            "Chose PostgreSQL over MongoDB because ACID needed",
            "fact",
            [],
        )
        assert score >= 6  # 5 base + 1 causal

    def test_multiple_causal_patterns(self) -> None:
        """Only one causal bonus even with multiple patterns."""
        score_single = auto_importance_score("Failed because of timeout", "fact", [])
        score_double = auto_importance_score(
            "Failed because of timeout due to network issues",
            "fact",
            [],
        )
        # Both should get +1, not +2
        assert score_double == score_single

    def test_comparative_language_bonus(self) -> None:
        """Content with comparative language gets +1."""
        score = auto_importance_score(
            "Redis is 3x faster than Memcached for session reads",
            "fact",
            [],
        )
        assert score >= 6

    def test_entity_richness_bonus(self) -> None:
        """Content with 2+ capitalized entities gets +1."""
        score = auto_importance_score(
            "PostgreSQL and Redis are used in PaymentService",
            "fact",
            [],
        )
        assert score >= 6

    def test_short_content_penalty(self) -> None:
        """Very short content (<20 chars) gets -1."""
        score = auto_importance_score("Fix auth bug", "fact", [])
        assert score <= 5

    def test_capped_at_10(self) -> None:
        """Score never exceeds 10."""
        # instruction(+3) + causal(+1) + entities(+1) + comparative(+1) = 11 → capped at 10
        score = auto_importance_score(
            "Always use PostgreSQL over MongoDB because ACID is faster than eventual consistency",
            "instruction",
            [],
        )
        assert score <= 10

    def test_floored_at_1(self) -> None:
        """Score never goes below 1."""
        score = auto_importance_score("x", "context", [])
        assert score >= 1

    def test_type_bonus_stacks_with_content_signals(self) -> None:
        """Type bonus and content signals are additive."""
        base = auto_importance_score("Some plain error report", "fact", [])
        decision_causal = auto_importance_score(
            "Chose X over Y because of performance", "decision", []
        )
        assert decision_causal > base

    def test_unknown_type_no_bonus(self) -> None:
        """Unknown memory type gets no type bonus."""
        score = auto_importance_score(
            "Some text about an unknown memory type here", "unknown_type", []
        )
        assert score == 5  # Just base, no bonus

    def test_tags_dont_affect_score(self) -> None:
        """Tags are accepted but don't change score (future use)."""
        score_no_tags = auto_importance_score("Some text", "fact", [])
        score_with_tags = auto_importance_score("Some text", "fact", ["project-x", "auth"])
        assert score_no_tags == score_with_tags

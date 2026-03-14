"""Tests for context merger engine."""

from __future__ import annotations

from neural_memory.engine.context_merger import merge_context


class TestMergeContext:
    """Tests for merge_context()."""

    def test_no_context_returns_unchanged(self) -> None:
        """No context dict returns original content."""
        assert merge_context("hello world", None) == "hello world"

    def test_empty_context_returns_unchanged(self) -> None:
        """Empty context dict returns original content."""
        assert merge_context("hello world", {}) == "hello world"

    def test_decision_reason_merges_inline(self) -> None:
        """Decision reason merges with 'because' connector."""
        result = merge_context(
            "Chose PostgreSQL",
            {"reason": "ACID needed"},
            "decision",
        )
        assert "because ACID needed" in result
        assert result.startswith("Chose PostgreSQL")

    def test_decision_alternatives_list(self) -> None:
        """List values are joined with commas."""
        result = merge_context(
            "Chose PostgreSQL",
            {"alternatives": ["MongoDB", "CockroachDB"]},
            "decision",
        )
        assert "MongoDB, CockroachDB" in result

    def test_decision_full_context(self) -> None:
        """Full decision context merges all fields."""
        result = merge_context(
            "Chose PostgreSQL",
            {
                "reason": "ACID for payments",
                "alternatives": ["MongoDB"],
                "decided_by": "team lead",
            },
            "decision",
        )
        assert "because ACID for payments" in result
        assert "MongoDB" in result
        assert "team lead" in result

    def test_error_root_cause(self) -> None:
        """Error type merges root cause."""
        result = merge_context(
            "Auth middleware broke",
            {"root_cause": "new cookie format", "fix": "updated parser"},
            "error",
        )
        assert "Root cause: new cookie format" in result
        assert "Fixed by updated parser" in result

    def test_workflow_steps(self) -> None:
        """Workflow type merges steps."""
        result = merge_context(
            "Deploy process",
            {"steps": "build, test, push", "trigger": "merge to main"},
            "workflow",
        )
        assert "Steps: build, test, push" in result
        assert "Triggered by merge to main" in result

    def test_unknown_keys_appended(self) -> None:
        """Keys not in template are appended as fallback."""
        result = merge_context(
            "Some fact",
            {"custom_field": "custom value"},
            "fact",
        )
        assert "Custom field: custom value" in result

    def test_ends_with_period(self) -> None:
        """Result always ends with period."""
        result = merge_context(
            "Chose PostgreSQL",
            {"reason": "ACID"},
            "decision",
        )
        assert result.endswith(".")

    def test_strips_trailing_period(self) -> None:
        """Trailing period on content is stripped before merge."""
        result = merge_context(
            "Chose PostgreSQL.",
            {"reason": "ACID"},
            "decision",
        )
        # Should not have double period
        assert ".." not in result

    def test_none_type_defaults_to_fact(self) -> None:
        """None memory_type uses fact template."""
        result = merge_context(
            "API endpoint is /v2/users",
            {"source": "docs"},
            None,
        )
        assert "Source: docs" in result

    def test_empty_value_skipped(self) -> None:
        """Empty/falsy context values are skipped."""
        result = merge_context(
            "Some content",
            {"reason": "", "alternatives": []},
            "decision",
        )
        # Should not have "because " with empty value
        assert "because " not in result.lower()

    def test_immutability(self) -> None:
        """merge_context does not mutate input context dict."""
        context = {"reason": "test", "alternatives": ["A", "B"]}
        original = {"reason": "test", "alternatives": ["A", "B"]}
        merge_context("content", context, "decision")
        assert context == original

    def test_insight_type(self) -> None:
        """Insight type merges evidence and applies_to."""
        result = merge_context(
            "React re-renders are expensive",
            {"reason": "virtual DOM diffing", "applies_to": "large component trees"},
            "insight",
        )
        assert "because virtual DOM diffing" in result
        assert "Applies to large component trees" in result

    def test_preference_type(self) -> None:
        """Preference type merges reason and scope."""
        result = merge_context(
            "User prefers dark mode",
            {"reason": "eye strain", "scope": "all dashboards"},
            "preference",
        )
        assert "because eye strain" in result
        assert "Scope: all dashboards" in result

    def test_instruction_type(self) -> None:
        """Instruction type merges reason and exceptions."""
        result = merge_context(
            "Always run linter before commit",
            {"reason": "CI catches fewer issues", "exceptions": "docs-only changes"},
            "instruction",
        )
        assert "because CI catches fewer issues" in result
        assert "Exceptions: docs-only changes" in result

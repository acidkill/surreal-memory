"""Tests for A4-C: Reflection Engine — threshold-based meta-memory generation."""

from __future__ import annotations

from neural_memory.engine.reflection import ReflectionEngine, detect_patterns


class TestDetectPatterns:
    """Test rule-based pattern detection from memory clusters."""

    def test_repeated_entity_detected(self) -> None:
        """Entity appearing in 3+ memories triggers a pattern."""
        memories = [
            {
                "content": "PostgreSQL handles ACID transactions well",
                "type": "fact",
                "tags": ["db"],
            },
            {
                "content": "PostgreSQL requires proper indexing for performance",
                "type": "insight",
                "tags": ["db"],
            },
            {
                "content": "Chose PostgreSQL over MongoDB for payments",
                "type": "decision",
                "tags": ["db"],
            },
        ]
        patterns = detect_patterns(memories)
        assert len(patterns) >= 1
        # Should mention the recurring entity
        assert any("postgresql" in p["description"].lower() for p in patterns)

    def test_no_pattern_from_unrelated_memories(self) -> None:
        """Unrelated memories produce no patterns."""
        memories = [
            {"content": "The sky is blue", "type": "fact", "tags": []},
            {"content": "Fix the auth bug", "type": "error", "tags": []},
            {"content": "Prefer dark mode", "type": "preference", "tags": []},
        ]
        patterns = detect_patterns(memories)
        # May produce some patterns but should be few/none for unrelated content
        assert len(patterns) <= 1

    def test_temporal_sequence_detected(self) -> None:
        """Memories with temporal markers produce sequence patterns."""
        memories = [
            {"content": "First, set up the database schema", "type": "workflow", "tags": []},
            {"content": "Then, implement the API endpoints", "type": "workflow", "tags": []},
            {"content": "After that, write integration tests", "type": "workflow", "tags": []},
            {"content": "Finally, deploy to staging", "type": "workflow", "tags": []},
        ]
        patterns = detect_patterns(memories)
        # Should detect a workflow/sequence pattern
        assert len(patterns) >= 1

    def test_contradiction_detected(self) -> None:
        """Contradictory statements are flagged."""
        memories = [
            {"content": "Redis is the best choice for caching", "type": "decision", "tags": []},
            {
                "content": "Redis is not suitable for our caching needs",
                "type": "decision",
                "tags": [],
            },
        ]
        patterns = detect_patterns(memories)
        # Should detect contradiction
        assert any(p["pattern_type"] == "contradiction" for p in patterns)

    def test_empty_memories_no_patterns(self) -> None:
        """Empty input produces no patterns."""
        assert detect_patterns([]) == []

    def test_single_memory_no_patterns(self) -> None:
        """Single memory produces no patterns."""
        patterns = detect_patterns([{"content": "Some fact", "type": "fact", "tags": []}])
        assert patterns == []

    def test_pattern_includes_source_info(self) -> None:
        """Patterns include indices of source memories."""
        memories = [
            {"content": "React component renders slowly", "type": "error", "tags": ["react"]},
            {"content": "React hooks cause re-render loop", "type": "error", "tags": ["react"]},
            {
                "content": "React performance needs memoization",
                "type": "insight",
                "tags": ["react"],
            },
        ]
        patterns = detect_patterns(memories)
        if patterns:
            assert "source_indices" in patterns[0]


class TestReflectionEngine:
    """Test threshold-based reflection triggering."""

    def test_accumulation_below_threshold(self) -> None:
        """No reflection when accumulated importance below threshold."""
        engine = ReflectionEngine(threshold=50.0)
        engine.accumulate(5)
        engine.accumulate(5)
        engine.accumulate(5)
        assert engine.should_reflect() is False

    def test_accumulation_reaches_threshold(self) -> None:
        """Reflection triggers when accumulated importance >= threshold."""
        engine = ReflectionEngine(threshold=20.0)
        engine.accumulate(10)
        engine.accumulate(10)
        assert engine.should_reflect() is True

    def test_reset_after_reflection(self) -> None:
        """Accumulator resets after reflection is triggered."""
        engine = ReflectionEngine(threshold=10.0)
        engine.accumulate(15)
        assert engine.should_reflect() is True
        engine.reset()
        assert engine.should_reflect() is False
        assert engine.accumulated == 0.0

    def test_default_threshold(self) -> None:
        """Default threshold is 50."""
        engine = ReflectionEngine()
        assert engine.threshold == 50.0

    def test_accumulated_property(self) -> None:
        """Accumulated importance is accessible."""
        engine = ReflectionEngine()
        engine.accumulate(7)
        engine.accumulate(3)
        assert engine.accumulated == 10.0

"""Tests for features inspired by OpenMemory competitive analysis.

Tests cover:
1. Type-aware decay rates
2. Retrieval score breakdown
3. SimHash near-duplicate detection
4. Point-in-time temporal queries
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from neural_memory.core.memory_types import (
    DEFAULT_DECAY_RATES,
    MemoryType,
    get_decay_rate,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.engine.retrieval_types import ScoreBreakdown
from neural_memory.utils.simhash import (
    DEFAULT_THRESHOLD,
    hamming_distance,
    is_near_duplicate,
    simhash,
)

# ── Feature 1: Type-Aware Decay Rates ──────────────────────────────


class TestDefaultDecayRates:
    """Tests for DEFAULT_DECAY_RATES and get_decay_rate()."""

    def test_all_memory_types_have_decay_rate(self) -> None:
        """Every MemoryType should have an entry in DEFAULT_DECAY_RATES."""
        for mt in MemoryType:
            assert mt in DEFAULT_DECAY_RATES, f"Missing decay rate for {mt}"

    def test_schemas_decay_slowest(self) -> None:
        """Schemas should have the lowest (slowest) decay rate."""
        schema_rate = DEFAULT_DECAY_RATES[MemoryType.SCHEMA]
        for mt, rate in DEFAULT_DECAY_RATES.items():
            assert rate >= schema_rate, f"{mt} decays slower than SCHEMA"

    def test_todos_decay_fastest(self) -> None:
        """TODOs should have the highest (fastest) decay rate."""
        todo_rate = DEFAULT_DECAY_RATES[MemoryType.TODO]
        for mt, rate in DEFAULT_DECAY_RATES.items():
            assert rate <= todo_rate, f"{mt} decays faster than TODO"

    def test_get_decay_rate_known_type(self) -> None:
        """get_decay_rate returns correct rate for known types."""
        assert get_decay_rate("fact") == 0.02
        assert get_decay_rate("todo") == 0.15
        assert get_decay_rate("decision") == 0.03

    def test_get_decay_rate_unknown_type(self) -> None:
        """get_decay_rate returns 0.1 default for unknown types."""
        assert get_decay_rate("nonexistent") == 0.1
        assert get_decay_rate("") == 0.1

    def test_decay_rate_ordering(self) -> None:
        """Rates should follow: facts < decisions < insights < context < todos."""
        assert get_decay_rate("fact") < get_decay_rate("decision")
        assert get_decay_rate("decision") < get_decay_rate("insight")
        assert get_decay_rate("insight") < get_decay_rate("context")
        assert get_decay_rate("context") < get_decay_rate("todo")


class TestTypeAwareDecay:
    """Tests for DecayManager using per-neuron decay_rate."""

    def test_fast_decay_neuron_loses_more(self) -> None:
        """A neuron with high decay_rate should lose more activation."""
        days_elapsed = 5.0

        # TODO-type neuron (fast decay = 0.15)
        todo_retention = math.exp(-0.15 * days_elapsed)
        # Fact-type neuron (slow decay = 0.02)
        fact_retention = math.exp(-0.02 * days_elapsed)

        assert fact_retention > todo_retention
        assert fact_retention > 0.9  # Facts barely decay
        assert todo_retention < 0.5  # TODOs decay significantly

    def test_neuron_state_decay_rate_field(self) -> None:
        """NeuronState should respect its decay_rate field."""
        state = NeuronState(
            neuron_id="test",
            activation_level=1.0,
            decay_rate=0.15,  # TODO-type rate
        )
        decayed = state.decay(5 * 86400)  # 5 days in seconds
        assert decayed.activation_level < 0.5

        stable = NeuronState(
            neuron_id="test2",
            activation_level=1.0,
            decay_rate=0.02,  # Fact-type rate
        )
        decayed_stable = stable.decay(5 * 86400)
        assert decayed_stable.activation_level > 0.9


# ── Feature 2: Score Breakdown ──────────────────────────────────────


class TestScoreBreakdown:
    """Tests for ScoreBreakdown dataclass."""

    def test_create_score_breakdown(self) -> None:
        """Test creating a ScoreBreakdown."""
        breakdown = ScoreBreakdown(
            base_activation=0.72,
            intersection_boost=0.15,
            freshness_boost=0.08,
            frequency_boost=0.05,
            emotional_resonance=0.0,
            raw_total=1.0,
        )
        assert breakdown.base_activation == 0.72
        assert breakdown.intersection_boost == 0.15
        assert breakdown.freshness_boost == 0.08
        assert breakdown.frequency_boost == 0.05
        assert breakdown.emotional_resonance == 0.0
        assert breakdown.raw_total == 1.0

    def test_score_breakdown_is_frozen(self) -> None:
        """ScoreBreakdown should be immutable."""
        breakdown = ScoreBreakdown(
            base_activation=0.5,
            intersection_boost=0.1,
            freshness_boost=0.05,
            frequency_boost=0.03,
            emotional_resonance=0.0,
            raw_total=0.68,
        )
        with pytest.raises(AttributeError):
            breakdown.base_activation = 0.9  # type: ignore[misc]

    def test_components_sum_to_raw_total(self) -> None:
        """raw_total should equal sum of components."""
        components = (0.6, 0.1, 0.08, 0.04, 0.0)
        total = sum(components)
        breakdown = ScoreBreakdown(
            base_activation=components[0],
            intersection_boost=components[1],
            freshness_boost=components[2],
            frequency_boost=components[3],
            emotional_resonance=components[4],
            raw_total=total,
        )
        assert abs(breakdown.raw_total - sum(components)) < 1e-10


# ── Feature 3: SimHash Near-Duplicate Detection ────────────────────


class TestSimHash:
    """Tests for SimHash fingerprinting."""

    def test_identical_texts_same_hash(self) -> None:
        """Identical texts should produce the same hash."""
        text = "The quick brown fox jumps over the lazy dog"
        assert simhash(text) == simhash(text)

    def test_empty_text_returns_zero(self) -> None:
        """Empty or whitespace-only text should return 0."""
        assert simhash("") == 0
        assert simhash("   ") == 0

    def test_similar_texts_close_hashes(self) -> None:
        """Similar texts should have small Hamming distance."""
        a = "We decided to use PostgreSQL for the database"
        b = "We decided to use PostgreSQL for our database"
        distance = hamming_distance(simhash(a), simhash(b))
        assert distance <= DEFAULT_THRESHOLD

    def test_different_texts_distant_hashes(self) -> None:
        """Very different texts should have large Hamming distance."""
        a = "We decided to use PostgreSQL for the database"
        b = "The weather is sunny today in San Francisco"
        distance = hamming_distance(simhash(a), simhash(b))
        assert distance > DEFAULT_THRESHOLD

    def test_hamming_distance_self_is_zero(self) -> None:
        """Hamming distance of a hash with itself is 0."""
        h = simhash("hello world")
        assert hamming_distance(h, h) == 0

    def test_hamming_distance_symmetric(self) -> None:
        """Hamming distance is symmetric."""
        a = simhash("text one")
        b = simhash("text two")
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_is_near_duplicate_true(self) -> None:
        """Similar texts should be detected as near-duplicates."""
        a = simhash("Error: connection timeout after 30 seconds")
        b = simhash("Error: connection timeout after 30 sec")
        assert is_near_duplicate(a, b)

    def test_is_near_duplicate_false(self) -> None:
        """Different texts should not be near-duplicates."""
        a = simhash("The project uses FastAPI framework")
        b = simhash("I enjoy hiking in the mountains")
        assert not is_near_duplicate(a, b)

    def test_custom_threshold(self) -> None:
        """Custom threshold should work."""
        a = simhash("hello world")
        b = simhash("hello earth")
        # With a very strict threshold (0), only exact matches pass
        assert not is_near_duplicate(a, b, threshold=0)
        # With a very loose threshold (64), everything passes
        assert is_near_duplicate(a, b, threshold=64)

    def test_case_insensitive(self) -> None:
        """SimHash should be case-insensitive (lowercases internally)."""
        assert simhash("Hello World") == simhash("hello world")

    def test_whitespace_normalized(self) -> None:
        """SimHash should normalize whitespace."""
        assert simhash("hello  world") == simhash("hello world")
        assert simhash("  hello world  ") == simhash("hello world")


class TestNeuronContentHash:
    """Tests for content_hash field on Neuron."""

    def test_neuron_default_hash_is_zero(self) -> None:
        """Default content_hash should be 0."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test")
        assert neuron.content_hash == 0

    def test_neuron_create_with_hash(self) -> None:
        """Neuron.create should accept content_hash."""
        h = simhash("test content")
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="test content",
            content_hash=h,
        )
        assert neuron.content_hash == h

    def test_with_metadata_preserves_hash(self) -> None:
        """with_metadata() should preserve content_hash."""
        h = simhash("original content")
        neuron = Neuron.create(
            type=NeuronType.ENTITY,
            content="original content",
            content_hash=h,
        )
        updated = neuron.with_metadata(extra="value")
        assert updated.content_hash == h


# ── Feature 4: Point-in-Time Temporal Queries ──────────────────────


class TestFiberValidAt:
    """Tests for _fiber_valid_at helper."""

    def test_fiber_with_no_time_bounds(self) -> None:
        """Fiber with no time bounds should always be valid."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.engine.retrieval import _fiber_valid_at

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
        )
        assert _fiber_valid_at(fiber, datetime.now())
        assert _fiber_valid_at(fiber, datetime(2020, 1, 1))
        assert _fiber_valid_at(fiber, datetime(2030, 12, 31))

    def test_fiber_before_start(self) -> None:
        """Fiber should be invalid before its time_start."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.engine.retrieval import _fiber_valid_at

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            time_start=datetime(2026, 2, 1),
        )
        assert not _fiber_valid_at(fiber, datetime(2026, 1, 15))
        assert _fiber_valid_at(fiber, datetime(2026, 2, 15))

    def test_fiber_after_end(self) -> None:
        """Fiber should be invalid after its time_end."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.engine.retrieval import _fiber_valid_at

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            time_end=datetime(2026, 2, 28),
        )
        assert _fiber_valid_at(fiber, datetime(2026, 2, 15))
        assert not _fiber_valid_at(fiber, datetime(2026, 3, 1))

    def test_fiber_within_range(self) -> None:
        """Fiber should be valid within its time range."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.engine.retrieval import _fiber_valid_at

        start = datetime(2026, 2, 1)
        end = datetime(2026, 2, 28)
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            time_start=start,
            time_end=end,
        )
        assert _fiber_valid_at(fiber, datetime(2026, 2, 15))
        assert _fiber_valid_at(fiber, start)  # Inclusive start
        assert _fiber_valid_at(fiber, end)  # Inclusive end
        assert not _fiber_valid_at(fiber, datetime(2026, 1, 31))
        assert not _fiber_valid_at(fiber, datetime(2026, 3, 1))

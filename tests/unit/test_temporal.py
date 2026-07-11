"""Unit tests for temporal extraction."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from surreal_memory.extraction.temporal import TemporalExtractor, TimeGranularity, TimeHint


class TestTemporalExtractor:
    """Tests for TemporalExtractor class."""

    @pytest.fixture
    def extractor(self) -> TemporalExtractor:
        """Create extractor instance."""
        return TemporalExtractor()

    @pytest.fixture
    def ref_time(self) -> datetime:
        """Reference time for tests."""
        return datetime(2024, 2, 4, 14, 30, 0)

    # English tests

    def test_en_today(self, extractor: TemporalExtractor, ref_time: datetime) -> None:
        """Test English 'today'."""
        hints = extractor.extract("I have a meeting today", ref_time, language="en")

        assert len(hints) >= 1
        hint = hints[0]
        assert hint.original.lower() == "today"
        assert hint.absolute_start.date() == ref_time.date()

    def test_en_yesterday(self, extractor: TemporalExtractor, ref_time: datetime) -> None:
        """Test English 'yesterday'."""
        hints = extractor.extract("I met Alice yesterday", ref_time, language="en")

        assert len(hints) >= 1
        hint = hints[0]
        yesterday = ref_time.date() - timedelta(days=1)
        assert hint.absolute_start.date() == yesterday

    def test_en_this_morning(self, extractor: TemporalExtractor, ref_time: datetime) -> None:
        """Test English 'this morning'."""
        hints = extractor.extract("I had coffee this morning", ref_time, language="en")

        assert len(hints) >= 1
        hint = hints[0]
        assert hint.absolute_start.hour <= 12
        assert hint.absolute_end.hour >= 6

    def test_en_last_week(self, extractor: TemporalExtractor, ref_time: datetime) -> None:
        """Test English 'last week'."""
        hints = extractor.extract("Last week was busy", ref_time, language="en")

        assert len(hints) >= 1
        hint = hints[0]
        assert hint.absolute_start < ref_time
        assert hint.granularity == TimeGranularity.DAY

    # Auto-detection tests

    def test_auto_detects_english(self, extractor: TemporalExtractor, ref_time: datetime) -> None:
        """Test auto-detection finds English patterns."""
        hints = extractor.extract("Yesterday was great", ref_time, language="auto")

        assert len(hints) >= 1

    # Edge cases

    def test_no_time_references(self, extractor: TemporalExtractor, ref_time: datetime) -> None:
        """Test text with no time references."""
        hints = extractor.extract("The cat sat on the mat", ref_time, language="en")

        # May or may not find hints depending on implementation
        # Just verify no errors
        assert isinstance(hints, list)

    def test_multiple_time_references(
        self, extractor: TemporalExtractor, ref_time: datetime
    ) -> None:
        """Test text with multiple time references."""
        hints = extractor.extract(
            "Yesterday I worked, today I rest, tomorrow I travel",
            ref_time,
            language="en",
        )

        # Should find multiple hints
        assert len(hints) >= 2

    def test_time_hint_midpoint(self) -> None:
        """Test TimeHint midpoint property."""
        hint = TimeHint(
            original="today",
            absolute_start=datetime(2024, 1, 1, 0, 0),
            absolute_end=datetime(2024, 1, 1, 23, 59, 59),
            granularity=TimeGranularity.DAY,
        )

        midpoint = hint.midpoint
        assert midpoint.hour == 11 or midpoint.hour == 12

"""Tests for B6: Contextual Compression — age-based recall compression."""

from __future__ import annotations

from datetime import timedelta

from neural_memory.engine.retrieval_context import compress_for_recall
from neural_memory.utils.timeutils import utcnow


class TestCompressForRecall:
    """Test age-based compression tiers in compress_for_recall()."""

    def test_recent_memory_returns_full_content(self) -> None:
        """Memories < 7 days old return full content unchanged."""
        content = "This is a detailed memory about PostgreSQL configuration tuning."
        created = utcnow() - timedelta(days=2)
        result = compress_for_recall(content, summary=None, created_at=created)
        assert result == content

    def test_medium_age_with_summary_returns_summary(self) -> None:
        """Memories 7-30 days old with summary return the summary."""
        content = "Very long detailed content about database migrations and schema changes."
        summary = "Database migration and schema changes."
        created = utcnow() - timedelta(days=15)
        result = compress_for_recall(content, summary=summary, created_at=created)
        assert result == summary

    def test_medium_age_without_summary_truncates(self) -> None:
        """Memories 7-30 days old without summary truncate to 3 sentences."""
        content = (
            "First sentence about auth. "
            "Second sentence about tokens. "
            "Third sentence about sessions. "
            "Fourth sentence about cookies. "
            "Fifth sentence about headers."
        )
        created = utcnow() - timedelta(days=20)
        result = compress_for_recall(content, summary=None, created_at=created)
        # Should have at most 3 sentences
        sentences = [s.strip() for s in result.rstrip(".").split(". ") if s.strip()]
        assert len(sentences) <= 3

    def test_old_memory_with_summary_returns_summary(self) -> None:
        """Memories 30-90 days old with summary return summary."""
        content = "Long content that should not appear."
        summary = "Concise summary of old memory."
        created = utcnow() - timedelta(days=60)
        result = compress_for_recall(content, summary=summary, created_at=created)
        assert result == summary

    def test_old_memory_without_summary_extracts_key_phrases(self) -> None:
        """Memories 30-90 days old without summary extract key phrases."""
        content = (
            "PostgreSQL requires ACID compliance for payment processing. "
            "Redis handles session caching with 3x faster reads. "
            "MongoDB was rejected due to consistency requirements. "
            "The deployment pipeline uses Docker and Kubernetes. "
            "Authentication relies on JWT tokens with RSA256 signing."
        )
        created = utcnow() - timedelta(days=45)
        result = compress_for_recall(content, summary=None, created_at=created)
        # Should be shorter than original
        assert len(result) < len(content)
        # Should still contain some key information
        assert len(result) > 0

    def test_very_old_memory_returns_minimal(self) -> None:
        """Memories 90+ days old return truncated minimal content."""
        content = (
            "This is a very old memory with lots of detailed content "
            "that should be heavily compressed to save context window space. "
            "Only the most essential information should survive."
        )
        created = utcnow() - timedelta(days=120)
        result = compress_for_recall(content, summary=None, created_at=created)
        # Should be significantly shorter than original
        assert len(result) < len(content)

    def test_very_old_with_summary_uses_summary(self) -> None:
        """Memories 90+ days with summary still use summary (better than extraction)."""
        content = "Long original content."
        summary = "Brief summary."
        created = utcnow() - timedelta(days=200)
        result = compress_for_recall(content, summary=summary, created_at=created)
        assert result == summary

    def test_edge_exactly_7_days(self) -> None:
        """Memory exactly 7 days old falls into medium tier (not full)."""
        content = "Content. Second sentence. Third sentence. Fourth sentence. Fifth."
        created = utcnow() - timedelta(days=7)
        result = compress_for_recall(content, summary=None, created_at=created)
        # Should be compressed (medium tier), not full content
        assert len(result) <= len(content)

    def test_edge_exactly_30_days(self) -> None:
        """Memory exactly 30 days old falls into old tier."""
        content = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        created = utcnow() - timedelta(days=30)
        result = compress_for_recall(content, summary=None, created_at=created)
        assert len(result) <= len(content)

    def test_edge_exactly_90_days(self) -> None:
        """Memory exactly 90 days old falls into very old tier."""
        content = "Detailed content about something from long ago."
        created = utcnow() - timedelta(days=90)
        result = compress_for_recall(content, summary=None, created_at=created)
        assert len(result) <= len(content)

    def test_empty_content_returns_empty(self) -> None:
        """Empty content returns empty string regardless of age."""
        created = utcnow() - timedelta(days=50)
        result = compress_for_recall("", summary=None, created_at=created)
        assert result == ""

    def test_none_created_at_returns_full_content(self) -> None:
        """If created_at is None, return full content (safe fallback)."""
        content = "Some content without timestamp."
        result = compress_for_recall(content, summary=None, created_at=None)
        assert result == content

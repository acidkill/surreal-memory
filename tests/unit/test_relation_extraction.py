"""Tests for relation extraction from text."""

from surreal_memory.core.synapse import SynapseType
from surreal_memory.extraction.relations import (
    RelationExtractor,
    RelationType,
)


class TestCausalExtraction:
    """Test causal relation pattern extraction."""

    def setup_method(self) -> None:
        self.extractor = RelationExtractor()

    def test_extract_causal_because(self) -> None:
        """'X because Y' should produce CAUSED_BY relation."""
        text = "The deployment failed because the database was down."
        relations = self.extractor.extract(text)

        assert len(relations) >= 1
        causal = [r for r in relations if r.synapse_type == SynapseType.CAUSED_BY]
        assert len(causal) >= 1
        assert causal[0].relation_type == RelationType.CAUSAL
        assert causal[0].confidence > 0

    def test_extract_causal_due_to(self) -> None:
        """'X due to Y' should produce CAUSED_BY relation."""
        text = "The outage occurred due to a network partition."
        relations = self.extractor.extract(text)

        caused = [r for r in relations if r.synapse_type == SynapseType.CAUSED_BY]
        assert len(caused) >= 1
        assert caused[0].relation_type == RelationType.CAUSAL

    def test_extract_causal_caused_by(self) -> None:
        """'X caused by Y' should produce CAUSED_BY relation."""
        text = "Memory corruption was caused by buffer overflow in the parser."
        relations = self.extractor.extract(text)

        caused = [r for r in relations if r.synapse_type == SynapseType.CAUSED_BY]
        assert len(caused) >= 1

    def test_extract_causal_therefore(self) -> None:
        """'X therefore Y' should produce LEADS_TO relation."""
        text = "The cache was stale therefore the response was incorrect."
        relations = self.extractor.extract(text)

        leads = [r for r in relations if r.synapse_type == SynapseType.LEADS_TO]
        assert len(leads) >= 1
        assert leads[0].relation_type == RelationType.CAUSAL

    def test_extract_causal_leads_to(self) -> None:
        """'X leads to Y' should produce LEADS_TO relation."""
        text = "High memory usage leads to increased latency in the system."
        relations = self.extractor.extract(text)

        leads = [r for r in relations if r.synapse_type == SynapseType.LEADS_TO]
        assert len(leads) >= 1

    def test_extract_causal_results_in(self) -> None:
        """'X results in Y' should produce LEADS_TO relation."""
        text = "Improper error handling results in silent data corruption."
        relations = self.extractor.extract(text)

        leads = [r for r in relations if r.synapse_type == SynapseType.LEADS_TO]
        assert len(leads) >= 1


class TestComparativeExtraction:
    """Test comparative relation pattern extraction."""

    def setup_method(self) -> None:
        self.extractor = RelationExtractor()

    def test_extract_comparative_similar(self) -> None:
        """'X similar to Y' should produce SIMILAR_TO relation."""
        text = "Redis caching is similar to Memcached in its approach."
        relations = self.extractor.extract(text)

        similar = [r for r in relations if r.synapse_type == SynapseType.SIMILAR_TO]
        assert len(similar) >= 1
        assert similar[0].relation_type == RelationType.COMPARATIVE

    def test_extract_comparative_better_than(self) -> None:
        """'X better than Y' should produce SIMILAR_TO relation."""
        text = "PostgreSQL performs better than MySQL for complex queries."
        relations = self.extractor.extract(text)

        similar = [r for r in relations if r.synapse_type == SynapseType.SIMILAR_TO]
        assert len(similar) >= 1

    def test_extract_comparative_unlike(self) -> None:
        """'X unlike Y' should produce CONTRADICTS relation."""
        text = "This new approach is unlike the previous implementation."
        relations = self.extractor.extract(text)

        contradicts = [r for r in relations if r.synapse_type == SynapseType.CONTRADICTS]
        assert len(contradicts) >= 1
        assert contradicts[0].relation_type == RelationType.COMPARATIVE

    def test_extract_comparative_different_from(self) -> None:
        """'X different from Y' should produce CONTRADICTS relation."""
        text = "The REST API is different from the GraphQL approach."
        relations = self.extractor.extract(text)

        contradicts = [r for r in relations if r.synapse_type == SynapseType.CONTRADICTS]
        assert len(contradicts) >= 1


class TestSequentialExtraction:
    """Test sequential relation pattern extraction."""

    def setup_method(self) -> None:
        self.extractor = RelationExtractor()

    def test_extract_sequential_then(self) -> None:
        """'X then Y' should produce BEFORE relation."""
        text = "We deployed the service then ran the integration tests."
        relations = self.extractor.extract(text)

        before = [r for r in relations if r.synapse_type == SynapseType.BEFORE]
        assert len(before) >= 1
        assert before[0].relation_type == RelationType.SEQUENTIAL

    def test_extract_sequential_after(self) -> None:
        """'after X, Y' should produce BEFORE relation (X happened first)."""
        text = "After the migration completed, we verified the data integrity."
        relations = self.extractor.extract(text)

        before = [r for r in relations if r.synapse_type == SynapseType.BEFORE]
        assert len(before) >= 1

    def test_extract_sequential_first_then(self) -> None:
        """'first X then Y' should produce BEFORE relation with high confidence."""
        text = "First backup the database, then apply the schema changes."
        relations = self.extractor.extract(text)

        before = [r for r in relations if r.synapse_type == SynapseType.BEFORE]
        assert len(before) >= 1
        # At least one BEFORE relation should have high confidence from first...then
        max_conf = max(r.confidence for r in before)
        assert max_conf >= 0.7

    def test_extract_sequential_followed_by(self) -> None:
        """'X followed by Y' should produce BEFORE relation."""
        text = "The build step was followed by automated testing."
        relations = self.extractor.extract(text)

        before = [r for r in relations if r.synapse_type == SynapseType.BEFORE]
        assert len(before) >= 1


class TestEdgeCases:
    """Test edge cases and deduplication."""

    def setup_method(self) -> None:
        self.extractor = RelationExtractor()

    def test_no_relations_plain_text(self) -> None:
        """Plain text without relation markers should yield empty result."""
        text = "Hello world. This is a simple memory."
        relations = self.extractor.extract(text)
        assert len(relations) == 0

    def test_empty_text(self) -> None:
        """Empty text should yield empty result."""
        assert self.extractor.extract("") == []
        assert self.extractor.extract("short") == []

    def test_multiple_relations_compound(self) -> None:
        """Compound text with multiple relation types."""
        text = (
            "The server crashed because of high load. "
            "First we restarted the service, then we scaled up the cluster."
        )
        relations = self.extractor.extract(text)
        types = {r.relation_type for r in relations}
        assert RelationType.CAUSAL in types
        assert RelationType.SEQUENTIAL in types

    def test_dedup_overlapping(self) -> None:
        """Same relation shouldn't be extracted twice."""
        text = "The build failed because the tests were broken."
        relations = self.extractor.extract(text)

        # Count unique (source, target, type) combinations
        keys = {(r.source_span.lower(), r.target_span.lower(), r.synapse_type) for r in relations}
        assert len(keys) == len(relations)

    def test_confidence_in_range(self) -> None:
        """All confidence values should be between 0.0 and 1.0."""
        text = (
            "The cache expired because of the TTL setting. "
            "Redis is similar to Memcached. "
            "First we build, then we deploy."
        )
        relations = self.extractor.extract(text)
        for r in relations:
            assert 0.0 <= r.confidence <= 1.0

    def test_short_spans_rejected(self) -> None:
        """Spans shorter than 3 characters should be filtered out."""
        # This should not match because "X" is too short
        text = "X because Y."
        relations = self.extractor.extract(text)
        # Spans "X" and "Y" are under 3 chars, should be filtered
        for r in relations:
            assert len(r.source_span) >= 3
            assert len(r.target_span) >= 3

    def test_relation_candidate_is_frozen(self) -> None:
        """RelationCandidate should be immutable."""
        text = "The server crashed because the memory was full."
        relations = self.extractor.extract(text)
        if relations:
            r = relations[0]
            try:
                r.confidence = 0.5  # type: ignore[misc]
                raise AssertionError("Should not allow mutation")
            except AttributeError:
                pass  # Expected — frozen dataclass

    def test_position_tracking(self) -> None:
        """Source and target positions should be valid character offsets."""
        text = "The deployment failed because the database connection timed out."
        relations = self.extractor.extract(text)
        for r in relations:
            assert r.source_start >= 0
            assert r.source_end > r.source_start
            assert r.target_start >= 0
            assert r.target_end > r.target_start
            # Positions should be within text bounds
            assert r.source_end <= len(text)
            assert r.target_end <= len(text)

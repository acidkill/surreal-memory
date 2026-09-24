"""Unit tests for pattern extraction — episodic → semantic concept formation."""

from __future__ import annotations

from datetime import datetime

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import NeuronType
from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.engine.pattern_extraction import extract_patterns


def _make_fiber(
    fiber_id: str,
    neuron_ids: set[str],
    tags: set[str] | None = None,
) -> Fiber:
    """Create a test fiber."""
    return Fiber(
        id=fiber_id,
        neuron_ids=neuron_ids,
        synapse_ids=set(),
        anchor_neuron_id=next(iter(neuron_ids)) if neuron_ids else "",
        agent_tags=tags or set(),
        created_at=datetime(2026, 1, 1),
    )


def _make_maturation(
    fiber_id: str,
    stage: MemoryStage = MemoryStage.EPISODIC,
    rehearsal_count: int = 3,
) -> MaturationRecord:
    """Create a test maturation record."""
    return MaturationRecord(
        fiber_id=fiber_id,
        brain_id="test-brain",
        stage=stage,
        rehearsal_count=rehearsal_count,
    )


class TestPatternExtraction:
    """Tests for semantic pattern extraction from episodic fibers."""

    def test_empty_fibers(self) -> None:
        """No fibers should produce no patterns."""
        patterns, report = extract_patterns([], {})
        assert patterns == []
        assert report.fibers_analyzed == 0

    def test_insufficient_fibers(self) -> None:
        """Fewer than min_cluster_size fibers should produce no patterns."""
        fibers = [_make_fiber("f1", {"n1"}, {"tag-a"})]
        maturations = {"f1": _make_maturation("f1")}
        patterns, report = extract_patterns(fibers, maturations, min_cluster_size=3)
        assert patterns == []

    def test_non_episodic_excluded(self) -> None:
        """Only EPISODIC fibers should be eligible."""
        fibers = [_make_fiber(f"f{i}", {"n1", "n2"}, {"tag-a", "tag-b"}) for i in range(5)]
        maturations = {
            f"f{i}": _make_maturation(f"f{i}", stage=MemoryStage.SHORT_TERM) for i in range(5)
        }
        patterns, report = extract_patterns(fibers, maturations)
        assert patterns == []
        assert report.fibers_analyzed == 0

    def test_low_rehearsal_excluded(self) -> None:
        """Fibers with insufficient rehearsal should be excluded."""
        fibers = [_make_fiber(f"f{i}", {"n1", "n2"}, {"tag-a", "tag-b"}) for i in range(5)]
        maturations = {f"f{i}": _make_maturation(f"f{i}", rehearsal_count=1) for i in range(5)}
        patterns, report = extract_patterns(fibers, maturations, min_rehearsal_count=3)
        assert patterns == []

    def test_cluster_with_common_tags(self) -> None:
        """Fibers with similar tags should cluster and produce a pattern."""
        common_neurons = {"n1", "n2", "n3"}
        fibers = [
            _make_fiber("f1", common_neurons, {"code-review", "alice", "pr"}),
            _make_fiber("f2", common_neurons, {"code-review", "alice", "feedback"}),
            _make_fiber("f3", common_neurons, {"code-review", "alice", "quality"}),
        ]
        maturations = {f.id: _make_maturation(f.id) for f in fibers}

        patterns, report = extract_patterns(
            fibers,
            maturations,
            min_cluster_size=3,
            tag_overlap_threshold=0.3,
        )
        assert len(patterns) >= 1
        assert report.patterns_extracted >= 1

        pattern = patterns[0]
        assert pattern.concept_neuron.type == NeuronType.CONCEPT
        assert len(pattern.common_tags) >= 1
        assert "code-review" in pattern.common_tags
        assert "alice" in pattern.common_tags

    def test_pattern_creates_synapses(self) -> None:
        """Extracted pattern should have IS_A synapses to common entities."""
        common_neurons = {"n1", "n2"}
        fibers = [_make_fiber(f"f{i}", common_neurons, {"tag-a", "tag-b"}) for i in range(4)]
        maturations = {f.id: _make_maturation(f.id) for f in fibers}

        patterns, _ = extract_patterns(
            fibers,
            maturations,
            min_cluster_size=3,
            tag_overlap_threshold=0.3,
        )
        assert len(patterns) >= 1
        assert len(patterns[0].synapses) > 0

    def test_prefiltered_candidates_need_no_maturation_map(self) -> None:
        """The caller may pass the same eligible candidates without a join map."""
        common_neurons = {"n1", "n2"}
        fibers = [_make_fiber(f"f{i}", common_neurons, {"tag-a", "tag-b"}) for i in range(4)]
        maturations = {fiber.id: _make_maturation(fiber.id) for fiber in fibers}

        mapped_patterns, mapped_report = extract_patterns(
            fibers, maturations, min_cluster_size=3, tag_overlap_threshold=0.3
        )
        filtered_patterns, filtered_report = extract_patterns(
            fibers, None, min_cluster_size=3, tag_overlap_threshold=0.3
        )

        assert len(filtered_patterns) == len(mapped_patterns)
        assert filtered_report.fibers_analyzed == mapped_report.fibers_analyzed == len(fibers)

    def test_disjoint_tags_no_cluster(self) -> None:
        """Fibers with disjoint tags should not cluster."""
        fibers = [
            _make_fiber("f1", {"n1"}, {"alpha"}),
            _make_fiber("f2", {"n2"}, {"beta"}),
            _make_fiber("f3", {"n3"}, {"gamma"}),
        ]
        maturations = {f.id: _make_maturation(f.id) for f in fibers}

        patterns, report = extract_patterns(
            fibers,
            maturations,
            min_cluster_size=3,
            tag_overlap_threshold=0.5,
        )
        assert patterns == []

    def test_report_metrics(self) -> None:
        """Report should accurately reflect extraction metrics."""
        common_neurons = {"n1", "n2"}
        fibers = [
            _make_fiber(f"f{i}", common_neurons, {"shared-tag", f"unique-{i}"}) for i in range(5)
        ]
        maturations = {f.id: _make_maturation(f.id) for f in fibers}

        _, report = extract_patterns(
            fibers,
            maturations,
            min_cluster_size=3,
            tag_overlap_threshold=0.3,
        )
        assert report.fibers_analyzed == 5
        assert report.clusters_found >= 1

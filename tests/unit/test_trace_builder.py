"""U4: build_retrieval_trace maps a recall result -> compact RetrievalTrace."""

from __future__ import annotations

from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.engine.trace_builder import build_retrieval_trace


def _result(fiber_ids: list[str], anchors: list[str]) -> RetrievalResult:
    return RetrievalResult(
        answer="a",
        confidence=0.83,
        depth_used=DepthLevel.CONTEXT,
        neurons_activated=3,
        fibers_matched=fiber_ids,
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=anchors),
        context="ctx",
        latency_ms=12.5,
        tokens_used=10,
        synthesis_method="single",
    )


class TestBuildRetrievalTrace:
    def test_maps_core_fields(self) -> None:
        res = _result(["f1", "f2"], ["a1"])
        trace = build_retrieval_trace(
            res,
            query="where does emma live",
            brain_id="brain-x",
            mode="associative",
            args={"tags": ["geo"], "min_trust": 0.5, "valid_at": "2026-02-01T00:00:00"},
            config_snapshot={"trust_weight": 0.0, "recency_weight": 1.0},
            session_id="sess-1",
        )
        assert trace.brain_id == "brain-x"
        assert trace.session_id == "sess-1"
        assert trace.query == "where does emma live"
        assert trace.mode == "associative"
        assert trace.depth_used == int(DepthLevel.CONTEXT)
        assert trace.confidence == 0.83
        assert trace.latency_ms == 12.5
        assert trace.fiber_ids == ("f1", "f2")
        assert trace.anchor_ids == ("a1",)
        assert trace.retrievers == ("single",)
        assert trace.config_snapshot == {"trust_weight": 0.0, "recency_weight": 1.0}
        # Filters carry mode + the applied recall filters (small, JSON-scalar-ish).
        assert trace.filters["mode"] == "associative"
        assert trace.filters["tags"] == ["geo"]
        assert trace.filters["min_trust"] == 0.5
        assert trace.filters["valid_at"] == "2026-02-01T00:00:00"

    def test_bounds_fiber_ids_to_ten(self) -> None:
        many = [f"f{i}" for i in range(25)]
        trace = build_retrieval_trace(_result(many, []), query="q", brain_id="b", mode="exact")
        assert len(trace.fiber_ids) == 10

    def test_empty_filters_default_to_mode_only(self) -> None:
        trace = build_retrieval_trace(
            _result(["f1"], []), query="q", brain_id="b", mode="exact", args={}
        )
        assert trace.filters == {"mode": "exact"}

    def test_never_raises_on_sparse_result(self) -> None:
        class _Bare:
            pass

        # A near-empty object must still yield a valid, bounded trace.
        trace = build_retrieval_trace(_Bare(), query="q", brain_id="b", mode="fast")
        assert trace.brain_id == "b"
        assert trace.fiber_ids == ()
        assert trace.anchor_ids == ()
        assert trace.depth_used == 0

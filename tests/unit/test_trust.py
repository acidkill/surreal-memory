"""Unit tests for engine/trust.py resolve_effective_trust (U2)."""

from __future__ import annotations

from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.source import Source, SourceType
from surreal_memory.engine.trust import (
    DEFAULT_LABEL_TRUST,
    DEFAULT_SOURCE_TYPE_TRUST,
    resolve_effective_trust,
)

DEFAULT = 0.7


def _tm(trust_score: float | None = None, source: str = "user_input") -> TypedMemory:
    return TypedMemory.create(
        fiber_id="f1", memory_type=MemoryType.FACT, source=source, trust_score=trust_score
    )


def _src(trust: float | None = None, source_type: SourceType = SourceType.DOCUMENT) -> Source:
    return Source.create(brain_id="b", name="doc", source_type=source_type, trust=trust)


class TestPriorityCascade:
    def test_typed_memory_trust_score_wins(self) -> None:
        tm = _tm(trust_score=0.42, source="user_input")
        src = _src(trust=0.9, source_type=SourceType.LAW)
        assert resolve_effective_trust(tm, src, DEFAULT) == 0.42

    def test_source_trust_beats_type_and_label(self) -> None:
        tm = _tm(trust_score=None, source="user_input")
        src = _src(trust=0.33, source_type=SourceType.LAW)
        assert resolve_effective_trust(tm, src, DEFAULT) == 0.33

    def test_source_type_default_when_no_explicit_trust(self) -> None:
        tm = _tm(trust_score=None, source="user_input")
        src = _src(trust=None, source_type=SourceType.LAW)
        assert (
            resolve_effective_trust(tm, src, DEFAULT) == DEFAULT_SOURCE_TYPE_TRUST[SourceType.LAW]
        )

    def test_label_default_when_no_source(self) -> None:
        tm = _tm(trust_score=None, source="verified")
        assert resolve_effective_trust(tm, None, DEFAULT) == DEFAULT_LABEL_TRUST["verified"]

    def test_mcp_label_normalised(self) -> None:
        tm = _tm(trust_score=None, source="mcp:claude_code")
        assert resolve_effective_trust(tm, None, DEFAULT) == DEFAULT_LABEL_TRUST["mcp_tool"]

    def test_falls_back_to_config_default(self) -> None:
        tm = _tm(trust_score=None, source="unknown_label_xyz")
        assert resolve_effective_trust(tm, None, DEFAULT) == DEFAULT

    def test_none_tm_and_source_returns_default(self) -> None:
        assert resolve_effective_trust(None, None, DEFAULT) == DEFAULT


class TestOrdering:
    def test_law_more_trusted_than_website(self) -> None:
        assert (
            DEFAULT_SOURCE_TYPE_TRUST[SourceType.LAW]
            > DEFAULT_SOURCE_TYPE_TRUST[SourceType.WEBSITE]
        )

    def test_verified_more_trusted_than_auto_capture(self) -> None:
        assert DEFAULT_LABEL_TRUST["verified"] > DEFAULT_LABEL_TRUST["auto_capture"]

    def test_all_defaults_in_unit_range(self) -> None:
        for v in list(DEFAULT_SOURCE_TYPE_TRUST.values()) + list(DEFAULT_LABEL_TRUST.values()):
            assert 0.0 <= v <= 1.0

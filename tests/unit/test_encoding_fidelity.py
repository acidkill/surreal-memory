"""Tests for Phase D — Encoding Fidelity.

Covers:
- Token normalizer (Vietnamese compound, diacritics stripping)
- FTS5 phrase query builder
- Phrase match heuristic
- Raw keyword storage in anchor neuron metadata
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.token_normalizer import (
    build_fts_phrase_query,
    normalize_for_search,
    normalize_vietnamese_compound,
    should_use_phrase_match,
)

# ── Vietnamese compound normalization ────────────────────────────


class TestNormalizeVietnameseCompound:
    def test_space_to_underscore(self) -> None:
        result = normalize_vietnamese_compound("doanh thu")
        assert "doanh thu" in result
        assert "doanh_thu" in result

    def test_underscore_to_space(self) -> None:
        result = normalize_vietnamese_compound("doanh_thu")
        assert "doanh_thu" in result
        assert "doanh thu" in result

    def test_single_word(self) -> None:
        result = normalize_vietnamese_compound("hello")
        assert result == ["hello"]

    def test_empty(self) -> None:
        result = normalize_vietnamese_compound("")
        assert result == []

    def test_no_separator(self) -> None:
        result = normalize_vietnamese_compound("shorttrend")
        assert result == ["shorttrend"]


# ── Full search normalization ────────────────────────────────────


class TestNormalizeForSearch:
    def test_vietnamese_with_diacritics(self) -> None:
        result = normalize_for_search("tự thân")
        assert "tự thân" in result
        assert "tự_thân" in result
        # Diacritics-stripped variant
        assert "tu than" in result

    def test_vietnamese_compound_variants(self) -> None:
        result = normalize_for_search("doanh thu")
        assert "doanh thu" in result
        assert "doanh_thu" in result

    def test_english_no_extra_variants(self) -> None:
        result = normalize_for_search("authentication")
        assert result == ["authentication"]

    def test_empty(self) -> None:
        result = normalize_for_search("")
        assert result == []

    def test_lowercased(self) -> None:
        result = normalize_for_search("API")
        assert all(v == v.lower() for v in result)

    def test_diacritics_stripped_compound(self) -> None:
        result = normalize_for_search("lợi nhuận")
        assert "lợi nhuận" in result
        assert "lợi_nhuận" in result
        assert "loi nhuan" in result
        assert "loi_nhuan" in result


# ── FTS5 phrase query builder ────────────────────────────────────


class TestBuildFtsPhraseQuery:
    def test_simple_phrase(self) -> None:
        result = build_fts_phrase_query("tự thân")
        assert result == '"tự thân"'

    def test_empty(self) -> None:
        result = build_fts_phrase_query("")
        assert result == '""'

    def test_quotes_escaped(self) -> None:
        result = build_fts_phrase_query('say "hello"')
        assert result == '"say ""hello"""'

    def test_single_word(self) -> None:
        result = build_fts_phrase_query("test")
        assert result == '"test"'


# ── Phrase match heuristic ───────────────────────────────────────


class TestShouldUsePhraseMatch:
    def test_vietnamese_compound(self) -> None:
        assert should_use_phrase_match("tự thân") is True

    def test_vietnamese_three_words(self) -> None:
        assert should_use_phrase_match("lợi nhuận ròng") is True

    def test_single_word(self) -> None:
        assert should_use_phrase_match("test") is False

    def test_long_english_phrase(self) -> None:
        assert should_use_phrase_match("this is a long english sentence") is False

    def test_short_english_compound(self) -> None:
        # All words ≤ 5 chars, 2 words → True
        assert should_use_phrase_match("api key") is True

    def test_long_english_words(self) -> None:
        assert should_use_phrase_match("authentication authorization") is False


# ── Raw keyword storage in anchor ────────────────────────────────


class TestRawKeywordStorage:
    @pytest.mark.asyncio
    async def test_anchor_has_raw_keywords(self) -> None:
        """CreateAnchorStep should store _raw_keywords in anchor metadata."""
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import CreateAnchorStep

        ctx = PipelineContext(
            content="shorttrend tự thân analysis report",
            timestamp=datetime(2026, 1, 1),
            metadata={},
            tags=set(),
            language="auto",
        )

        storage = AsyncMock()
        storage.add_neuron = AsyncMock()
        config = MagicMock()

        step = CreateAnchorStep()
        await step.execute(ctx, storage, config)

        # Anchor neuron should have _raw_keywords in metadata
        neuron_call = storage.add_neuron.call_args_list[0]
        neuron = neuron_call[0][0]
        raw_kws = neuron.metadata.get("_raw_keywords")
        assert raw_kws is not None
        assert isinstance(raw_kws, list)
        assert len(raw_kws) > 0
        # "shorttrend" should be in raw keywords (it's a significant keyword)
        assert any("shorttrend" in kw.lower() for kw in raw_kws)

    @pytest.mark.asyncio
    async def test_anchor_raw_keywords_capped(self) -> None:
        """Raw keywords should be capped at 10."""
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import CreateAnchorStep

        # Long content with many potential keywords
        words = " ".join(f"keyword{i}" for i in range(50))
        ctx = PipelineContext(
            content=words,
            timestamp=datetime(2026, 1, 1),
            metadata={},
            tags=set(),
            language="auto",
        )

        storage = AsyncMock()
        storage.add_neuron = AsyncMock()
        config = MagicMock()

        step = CreateAnchorStep()
        await step.execute(ctx, storage, config)

        neuron = storage.add_neuron.call_args_list[0][0][0]
        raw_kws = neuron.metadata.get("_raw_keywords", [])
        assert len(raw_kws) <= 10

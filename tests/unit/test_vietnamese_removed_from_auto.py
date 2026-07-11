"""Regression tests: removing the Vietnamese stop-word set must not filter
common English/Polish words in "auto"-mode keyword extraction.

The removed STOP_WORDS_VI set contained ASCII-only entries (ai, anh, bao, cho,
em, khi, ra, sao, trong) that collided with ordinary English/Polish vocabulary
and were silently dropped from every "auto"-mode extraction. Vietnamese support
has since been removed entirely, so these words must always survive extraction.
"""

from __future__ import annotations

import pytest

from surreal_memory.extraction.keywords import extract_weighted_keywords


def _unigrams(text: str, language: str = "auto") -> set[str]:
    return {r.text for r in extract_weighted_keywords(text, language=language) if " " not in r.text}


def _bigrams(text: str, language: str = "auto") -> set[str]:
    return {r.text for r in extract_weighted_keywords(text, language=language) if " " in r.text}


class TestConfirmedCollisions:
    """Real, live-reproduced collisions: EN/PL content must keep these words."""

    def test_ai_survives_as_unigram_in_auto_mode(self) -> None:
        text = "Zbudowalismy system agentow AI ktory uczy sie sam"
        assert "ai" in _unigrams(text)

    def test_ai_forms_bigrams_in_auto_mode(self) -> None:
        text = "Zbudowalismy system agentow AI ktory uczy sie sam"
        bigrams = _bigrams(text)
        assert "agentow ai" in bigrams
        assert "ai uczy" in bigrams

    def test_em_survives_as_unigram_in_auto_mode(self) -> None:
        text = "This is an em dash in a sentence"
        assert "em" in _unigrams(text)

    def test_em_dash_bigram_forms_in_auto_mode(self) -> None:
        text = "This is an em dash in a sentence"
        assert "em dash" in _bigrams(text)


class TestFormerViEntriesSurvive:
    """Every former ASCII-only Vietnamese stop word must extract normally now."""

    @pytest.mark.parametrize(
        "word,sentence",
        [
            ("anh", "The report was reviewed by anh from the team"),
            ("bao", "We ordered a pork bao for lunch today"),
            ("cho", "The engineer named cho signed off on the release"),
            ("khi", "The variable khi tracks the rotation angle"),
            ("ra", "The RA team approved the compliance report"),
            ("sao", "The satellite sao completed its orbit"),
            ("trong", "The word trong appears in this sentence"),
        ],
    )
    def test_word_survives_as_unigram_in_auto_mode(self, word: str, sentence: str) -> None:
        assert word in _unigrams(sentence), f"{word!r} was filtered in auto mode: {sentence!r}"

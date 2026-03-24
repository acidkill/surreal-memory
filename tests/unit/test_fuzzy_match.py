"""Tests for Phase B — Trigram Fuzzy Search.

Covers:
- Levenshtein distance (ASCII + Unicode)
- Fuzzy match filtering by distance and length
- Prefix variant generation
- Edge cases
"""

from __future__ import annotations

from neural_memory.engine.fuzzy_match import (
    find_fuzzy_matches,
    generate_prefix_variants,
    levenshtein_distance,
)

# ── Levenshtein distance ─────────────────────────────────────────


class TestLevenshteinDistance:
    def test_identical(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_single_insert(self) -> None:
        assert levenshtein_distance("cat", "cats") == 1

    def test_single_delete(self) -> None:
        assert levenshtein_distance("cats", "cat") == 1

    def test_single_replace(self) -> None:
        assert levenshtein_distance("cat", "car") == 1

    def test_typo_authentication(self) -> None:
        assert levenshtein_distance("authentication", "authentcation") == 1

    def test_typo_config(self) -> None:
        assert levenshtein_distance("config", "conifg") == 2

    def test_empty_strings(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "xyz") == 3

    def test_completely_different(self) -> None:
        assert levenshtein_distance("abc", "xyz") == 3

    def test_vietnamese_diacritics(self) -> None:
        """Vietnamese diacritics count as single characters."""
        # "lỗi" vs "loi": ỗ→o is 1 replacement = distance 1
        assert levenshtein_distance("lỗi", "loi") == 1

    def test_unicode_characters(self) -> None:
        assert levenshtein_distance("café", "cafe") == 1


# ── Fuzzy match finding ──────────────────────────────────────────


class TestFindFuzzyMatches:
    def test_exact_match(self) -> None:
        result = find_fuzzy_matches("hello", ["hello", "world"])
        assert len(result) >= 1
        assert result[0] == ("hello", 0)

    def test_typo_match(self) -> None:
        candidates = ["authentication", "authorization", "database"]
        result = find_fuzzy_matches("authentcation", candidates, max_distance=2)
        assert any(c == "authentication" for c, _ in result)

    def test_filters_by_distance(self) -> None:
        candidates = ["cat", "car", "xyz"]
        result = find_fuzzy_matches("cat", candidates, max_distance=1)
        matched = [c for c, _ in result]
        assert "cat" in matched
        assert "car" in matched
        assert "xyz" not in matched

    def test_skips_length_mismatch(self) -> None:
        candidates = ["a", "ab", "abcdefghijklmnop"]
        result = find_fuzzy_matches("abc", candidates, max_distance=1)
        # "a" (len diff = 2) should be skipped, "abcdefghijklmnop" (len diff > 1) should be skipped
        assert all(abs(len(c) - 3) <= 1 for c, _ in result)

    def test_sorted_by_distance(self) -> None:
        candidates = ["cat", "car", "cab"]
        result = find_fuzzy_matches("cat", candidates, max_distance=2)
        distances = [d for _, d in result]
        assert distances == sorted(distances)

    def test_empty_candidates(self) -> None:
        result = find_fuzzy_matches("test", [])
        assert result == []

    def test_empty_query(self) -> None:
        result = find_fuzzy_matches("", ["hello"])
        # Empty query with max_distance=2 should match short strings
        assert all(d <= 2 for _, d in result)


# ── Prefix variant generation ────────────────────────────────────


class TestGeneratePrefixVariants:
    def test_normal_word(self) -> None:
        result = generate_prefix_variants("authentication")
        assert "aut" in result
        assert len(result) == 2  # "aut" + half-word prefix

    def test_short_word(self) -> None:
        result = generate_prefix_variants("api")
        assert result == ["api"]

    def test_very_short_word(self) -> None:
        result = generate_prefix_variants("ab")
        assert result == ["ab"]

    def test_empty(self) -> None:
        result = generate_prefix_variants("")
        assert result == []

    def test_min_prefix_respected(self) -> None:
        result = generate_prefix_variants("database", min_prefix=4)
        assert result[0] == "data"

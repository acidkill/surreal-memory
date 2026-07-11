"""Tests for Phase A — Smart Query Expansion.

Covers:
- Synonym expansion (English)
- Abbreviation expansion (forward + reverse)
- Compound awareness (space ↔ underscore)
- Custom synonym maps
- Max expansion cap
- Input immutability
"""

from __future__ import annotations

from surreal_memory.engine.query_expander import (
    ABBREVIATION_MAP,
    SYNONYM_MAP,
    expand_terms,
)

# ── Synonym expansion ────────────────────────────────────────────


class TestSynonymExpansion:
    def test_english_synonym_cost(self) -> None:
        result = expand_terms(["cost"], enable_abbreviations=False)
        assert "expense" in result
        assert "spending" in result

    def test_english_synonym_error(self) -> None:
        result = expand_terms(["error"], enable_abbreviations=False)
        assert "bug" in result
        assert "issue" in result

    def test_synonym_bidirectional(self) -> None:
        """Looking up any member returns others in the group."""
        result_a = expand_terms(["expense"], enable_abbreviations=False)
        result_b = expand_terms(["cost"], enable_abbreviations=False)
        # Both should contain the other
        assert "cost" in result_a
        assert "expense" in result_b

    def test_unknown_keyword_passes_through(self) -> None:
        result = expand_terms(["xyzzy123"], enable_abbreviations=False)
        assert result == ["xyzzy123"]

    def test_synonyms_disabled(self) -> None:
        result = expand_terms(
            ["cost"],
            enable_synonyms=False,
            enable_abbreviations=False,
        )
        # Only original + compound variant (none for "cost")
        assert result == ["cost"]


# ── Abbreviation expansion ───────────────────────────────────────


class TestAbbreviationExpansion:
    def test_abbreviation_forward(self) -> None:
        result = expand_terms(["api"], enable_synonyms=False)
        assert "application programming interface" in result

    def test_abbreviation_roe(self) -> None:
        result = expand_terms(["roe"], enable_synonyms=False)
        assert "return on equity" in result

    def test_abbreviation_reverse(self) -> None:
        """Full form should expand to abbreviation."""
        result = expand_terms(["return on equity"], enable_synonyms=False)
        assert "roe" in result

    def test_abbreviation_case_insensitive(self) -> None:
        result = expand_terms(["API"], enable_synonyms=False)
        assert "application programming interface" in result

    def test_abbreviation_disabled(self) -> None:
        result = expand_terms(
            ["api"],
            enable_synonyms=False,
            enable_abbreviations=False,
        )
        assert "application programming interface" not in result


# ── Compound awareness (space ↔ underscore) ──────────────────────


class TestCompoundExpansion:
    def test_space_to_underscore(self) -> None:
        result = expand_terms(
            ["machine learning"],
            enable_synonyms=False,
            enable_abbreviations=False,
        )
        assert "machine_learning" in result

    def test_underscore_to_space(self) -> None:
        result = expand_terms(
            ["machine_learning"],
            enable_synonyms=False,
            enable_abbreviations=False,
        )
        assert "machine learning" in result


# ── Custom synonyms ──────────────────────────────────────────────


class TestCustomSynonyms:
    def test_custom_synonym_map(self) -> None:
        custom = {"shorttrend": ["trade", "short term trade"]}
        result = expand_terms(
            ["shorttrend"],
            enable_abbreviations=False,
            custom_synonyms=custom,
        )
        assert "trade" in result
        assert "short term trade" in result

    def test_custom_synonym_bidirectional(self) -> None:
        custom = {"shorttrend": ["trade"]}
        result = expand_terms(
            ["trade"],
            enable_abbreviations=False,
            custom_synonyms=custom,
        )
        assert "shorttrend" in result


# ── Caps and edge cases ─────────────────────────────────────────


class TestCapsAndEdgeCases:
    def test_max_expansion_cap(self) -> None:
        result = expand_terms(
            ["error"],
            enable_synonyms=True,
            enable_abbreviations=True,
            max_per_term=2,
        )
        # Original "error" + at most 2 expansions = max 3
        non_original = [r for r in result if r != "error"]
        assert len(non_original) <= 2

    def test_empty_input(self) -> None:
        result = expand_terms([])
        assert result == []

    def test_empty_string_keyword(self) -> None:
        result = expand_terms(["", "  "])
        assert result == []

    def test_no_mutation_of_input(self) -> None:
        original = ["cost", "error"]
        original_copy = list(original)
        expand_terms(original)
        assert original == original_copy

    def test_deduplication(self) -> None:
        result = expand_terms(["cost", "cost"])
        # "cost" should appear only once
        assert result.count("cost") == 1

    def test_all_lowercase(self) -> None:
        result = expand_terms(["API", "ROE"])
        for term in result:
            assert term == term.lower(), f"'{term}' is not lowercase"


# ── Map integrity ────────────────────────────────────────────────


class TestMapIntegrity:
    def test_synonym_map_populated(self) -> None:
        assert len(SYNONYM_MAP) > 20

    def test_abbreviation_map_populated(self) -> None:
        assert len(ABBREVIATION_MAP) > 15

    def test_synonym_map_all_lowercase(self) -> None:
        for key in SYNONYM_MAP:
            assert key == key.lower(), f"SYNONYM_MAP key '{key}' not lowercase"

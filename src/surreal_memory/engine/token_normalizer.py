"""Token normalizer — consistent tokenization for encode AND recall paths.

Provides a pass-through search normalizer and an FTS5 phrase query builder
used at both encode and recall time.
"""

from __future__ import annotations


def normalize_for_search(text: str) -> list[str]:
    """Produce search variants for a keyword.

    English-only pass-through: returns the lowercased text as the single
    variant (no diacritics or compound expansion).

    Args:
        text: A keyword to normalize.

    Returns:
        A single-element list with the lowercased text, or an empty list
        if the input is blank.
    """
    normalized = text.strip().lower()
    if not normalized:
        return []
    return [normalized]


def build_fts_phrase_query(phrase: str) -> str:
    """Build an FTS5 phrase query (exact phrase match, not AND).

    For multi-word queries, this produces a phrase query `"a b"` instead
    of `"a" "b"`.

    Args:
        phrase: Multi-word phrase to search as exact sequence.

    Returns:
        FTS5 MATCH expression for phrase matching.
    """
    phrase = phrase.strip()
    if not phrase:
        return '""'
    # Escape double quotes inside the phrase
    escaped = phrase.replace('"', '""')
    return f'"{escaped}"'


def should_use_phrase_match(text: str) -> bool:
    """Heuristic: should this query use FTS5 phrase matching?

    Returns True for short multi-word terms (≤3 words, all short) that
    likely form a compound term.

    Args:
        text: Query text to check.

    Returns:
        True if phrase matching is recommended.
    """
    words = text.strip().split()
    if len(words) < 2:
        return False

    # Short multi-word term: all words ≤ 5 chars (likely a compound)
    if len(words) <= 3 and all(len(w) <= 5 for w in words):
        return True

    return False

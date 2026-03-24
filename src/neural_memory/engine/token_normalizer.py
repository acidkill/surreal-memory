"""Token normalizer — consistent tokenization for encode AND recall paths.

Ensures Vietnamese compounds are normalized the same way at both encode
and recall time, producing all search variants (space, underscore,
diacritics-stripped) for comprehensive FTS5 matching.
"""

from __future__ import annotations

import re
import unicodedata

# Vietnamese diacritics detection — comprehensive character class
# Covers all Vietnamese-specific chars including composed forms (ợ, ậ, etc.)
_VI_CHARS_RE = re.compile(
    r"[ăắằẳẵặâấầẩẫậđêếềểễệôốồổỗộơớờởỡợưứừửữự"
    r"àáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]",
    re.IGNORECASE,
)


def _strip_diacritics(text: str) -> str:
    """Remove diacritics from text, matching FTS5 remove_diacritics=2 behavior.

    Decomposes Unicode characters and removes combining marks, then
    handles Vietnamese-specific chars (đ→d, Đ→D).
    """
    # Handle đ/Đ before NFD decomposition (NFD doesn't decompose đ)
    text = text.replace("đ", "d").replace("Đ", "D")
    # NFD decompose → remove combining marks
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _is_vietnamese(text: str) -> bool:
    """Quick heuristic: does text contain Vietnamese diacritics?"""
    return bool(_VI_CHARS_RE.search(text))


def normalize_vietnamese_compound(text: str) -> list[str]:
    """Normalize a Vietnamese compound word bidirectionally.

    Produces both space-separated and underscore-joined variants.

    Args:
        text: A keyword that may be a compound word.

    Returns:
        List of variants (always includes the original).
    """
    text = text.strip()
    if not text:
        return []

    variants: list[str] = [text]

    if " " in text:
        # Space form → add underscore form
        underscore = text.replace(" ", "_")
        if underscore != text:
            variants.append(underscore)
    elif "_" in text:
        # Underscore form → add space form
        space = text.replace("_", " ")
        if space != text:
            variants.append(space)

    return variants


def normalize_for_search(text: str) -> list[str]:
    """Produce all search variants for a keyword.

    Generates compound variants (space ↔ underscore) and diacritics-stripped
    variants. Used at both encode and recall time for consistency.

    Args:
        text: A keyword to normalize.

    Returns:
        Deduplicated list of all search variants.
    """
    text = text.strip().lower()
    if not text:
        return []

    seen: set[str] = set()
    result: list[str] = []

    def _add(variant: str) -> None:
        v = variant.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)

    # Original form
    _add(text)

    # Compound variants (space ↔ underscore)
    compounds = normalize_vietnamese_compound(text)
    for compound in compounds:
        _add(compound)

    # Diacritics-stripped variants (if text has diacritics)
    if _is_vietnamese(text):
        stripped = _strip_diacritics(text)
        _add(stripped)
        # Also strip compound variants (reuse already-computed list)
        for compound in compounds:
            _add(_strip_diacritics(compound))

    return result


def build_fts_phrase_query(phrase: str) -> str:
    """Build an FTS5 phrase query (exact phrase match, not AND).

    For multi-word queries that look like compounds (e.g. "tự thân"),
    this produces a phrase query `"tự thân"` instead of `"tự" "thân"`.

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

    Returns True for multi-word text that looks like a Vietnamese
    compound or a short multi-word term (≤3 words, all short).

    Args:
        text: Query text to check.

    Returns:
        True if phrase matching is recommended.
    """
    words = text.strip().split()
    if len(words) < 2:
        return False

    # Vietnamese compound: 2-3 short words with diacritics
    if len(words) <= 3 and _is_vietnamese(text):
        return True

    # Short multi-word term: all words ≤ 5 chars (likely a compound)
    if len(words) <= 3 and all(len(w) <= 5 for w in words):
        return True

    return False

"""Fuzzy matching for typo-tolerant keyword recall.

Uses Levenshtein distance (pure Python, no external deps) to find
approximate matches when exact FTS5 keyword search yields sparse results.
"""

from __future__ import annotations


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    Pure Python DP implementation.  Handles Unicode correctly
    (Vietnamese diacritics count as single characters).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Minimum number of single-character edits (insert, delete, replace).
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Use two-row DP for O(min(m,n)) space
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    prev_row = list(range(len(s1) + 1))
    for j, c2 in enumerate(s2):
        curr_row = [j + 1]
        for i, c1 in enumerate(s1):
            if c1 == c2:
                curr_row.append(prev_row[i])
            else:
                curr_row.append(1 + min(prev_row[i], prev_row[i + 1], curr_row[i]))
        prev_row = curr_row

    return prev_row[-1]


def find_fuzzy_matches(
    query_term: str,
    candidates: list[str],
    max_distance: int = 2,
) -> list[tuple[str, int]]:
    """Find candidates within edit distance of query_term.

    Optimizations:
    - Skips candidates whose length differs by more than max_distance
    - Returns results sorted by distance ascending

    Args:
        query_term: The (possibly misspelled) query keyword.
        candidates: List of candidate strings to match against.
        max_distance: Maximum allowed edit distance.

    Returns:
        List of (candidate, distance) tuples, sorted by distance.
    """
    query_lower = query_term.lower()
    query_len = len(query_lower)
    matches: list[tuple[str, int]] = []

    for candidate in candidates:
        cand_lower = candidate.lower()
        # Quick length check: skip if length diff > max_distance
        if abs(len(cand_lower) - query_len) > max_distance:
            continue

        dist = levenshtein_distance(query_lower, cand_lower)
        if dist <= max_distance:
            matches.append((candidate, dist))

    matches.sort(key=lambda x: x[1])
    return matches


def generate_prefix_variants(term: str, min_prefix: int = 3) -> list[str]:
    """Generate prefix variants for FTS5 prefix search.

    Produces short prefixes that can be used with FTS5's ``prefix*``
    syntax to fetch candidate neurons for fuzzy matching.

    Args:
        term: The keyword to generate prefixes for.
        min_prefix: Minimum prefix length.

    Returns:
        List of prefix strings (at most 2 to limit query count).
    """
    term = term.strip().lower()
    if len(term) < min_prefix:
        return [term] if term else []

    prefixes: list[str] = []
    # First prefix: min_prefix length
    prefixes.append(term[:min_prefix])

    # Second prefix: half the word (gives broader coverage)
    half = len(term) // 2
    if half > min_prefix:
        prefixes.append(term[:half])

    return prefixes

"""IDF-weighted anchor selection for keyword retrieval.

Replaces the fixed ``limit=2`` per keyword with a dynamic limit based
on each keyword's IDF score.  Rare terms get more anchor slots (up to 5),
common terms get fewer (down to 1).
"""

from __future__ import annotations

import math


def compute_idf_scores(
    keyword_df: dict[str, int],
    total_docs: int,
) -> dict[str, float]:
    """Compute normalized IDF scores for keywords.

    Formula: ``log((N + 1) / (1 + df)) / log(N + 1)`` → [0, 1].

    Args:
        keyword_df: Mapping of keyword → document frequency.
        total_docs: Total number of fibers (documents) in the brain.

    Returns:
        Dict mapping keyword → IDF score in [0, 1].
    """
    if total_docs <= 0:
        # Cold start: all keywords are equally "rare"
        return dict.fromkeys(keyword_df, 1.0)

    log_denom = math.log(total_docs + 1)
    if log_denom == 0:
        return dict.fromkeys(keyword_df, 1.0)

    scores: dict[str, float] = {}
    for kw, df in keyword_df.items():
        raw = math.log((total_docs + 1) / (1 + df))
        scores[kw] = raw / log_denom

    return scores


def compute_anchor_limit(
    idf_score: float,
    min_limit: int = 1,
    max_limit: int = 5,
) -> int:
    """Convert an IDF score to a dynamic anchor limit.

    Args:
        idf_score: Normalized IDF score in [0, 1].
        min_limit: Minimum anchors per keyword.
        max_limit: Maximum anchors per keyword.

    Returns:
        Integer anchor limit.
    """
    span = max_limit - min_limit + 1
    return max(min_limit, min(max_limit, math.ceil(idf_score * span)))


def compute_keyword_limits(
    keywords: list[str],
    keyword_df: dict[str, int],
    total_docs: int,
    min_limit: int = 1,
    max_limit: int = 5,
) -> dict[str, int]:
    """Compute per-keyword anchor limits based on IDF.

    Keywords not found in ``keyword_df`` (new/rare) get ``max_limit``.

    Args:
        keywords: List of keywords to compute limits for.
        keyword_df: Mapping of keyword → document frequency.
        total_docs: Total fiber count.
        min_limit: Minimum anchors per keyword.
        max_limit: Maximum anchors per keyword.

    Returns:
        Dict mapping keyword → anchor limit.
    """
    if not keywords:
        return {}

    # Build DF for requested keywords (missing = 0)
    requested_df = {kw: keyword_df.get(kw.lower(), 0) for kw in keywords}
    idf_scores = compute_idf_scores(requested_df, total_docs)

    limits: dict[str, int] = {}
    for kw in keywords:
        score = idf_scores.get(kw, 1.0)
        if score >= 1.0 or requested_df.get(kw, 0) == 0:
            # Unknown/rare keyword → max limit
            limits[kw] = max_limit
        else:
            limits[kw] = compute_anchor_limit(score, min_limit, max_limit)

    return limits

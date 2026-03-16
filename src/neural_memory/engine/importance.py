"""Auto-importance scoring for memories (Phase A4-B).

Heuristic-based priority scoring when user doesn't set explicit priority.
Uses content signals (causal language, entities, comparisons) and memory
type to assign a meaningful priority instead of a flat default 5.
"""

from __future__ import annotations

import re

# Type-based priority bonuses (relative to base score of 5)
_TYPE_BONUS: dict[str, int] = {
    "error": 2,
    "decision": 2,
    "preference": 3,
    "instruction": 3,
    "insight": 1,
    "workflow": 1,
    "fact": 0,
    "context": -1,
    "todo": -1,
    "reference": 0,
    "tool": 0,
    "hypothesis": 1,
    "prediction": 1,
}

# Causal language patterns (case-insensitive)
_CAUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbecause\b",
        r"\bcaused by\b",
        r"\bchose\s+\w+\s+over\b",
        r"\bdue to\b",
        r"\broot cause\b",
        r"\brejected\b.*\bdue\b",
        r"\bafter\s+\w+ing\b.*\bthen\b",
        r"\bleads? to\b",
    ]
]

# Comparative language patterns
_COMPARATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bfaster than\b",
        r"\bslower than\b",
        r"\bbetter than\b",
        r"\bworse than\b",
        r"\breplaced\b.*\bwith\b",
        r"\d+x\s+(faster|slower|better|more|less)",
        r"\binstead of\b",
    ]
]

# Entity detection: capitalized multi-word or known tech patterns
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[A-Z][a-zA-Z]*(?:SQL|DB|JS|API)\b")


def auto_importance_score(
    content: str,
    memory_type: str,
    tags: list[str],
) -> int:
    """Score memory importance heuristically based on content signals.

    Args:
        content: Memory content text.
        memory_type: Memory type string (e.g. "decision", "fact").
        tags: Memory tags (reserved for future use).

    Returns:
        Priority score 1-10.
    """
    score = 5  # Base

    # Type bonus
    score += _TYPE_BONUS.get(memory_type, 0)

    # Causal language bonus (+1 max)
    if any(p.search(content) for p in _CAUSAL_PATTERNS):
        score += 1

    # Comparative language bonus (+1 max)
    if any(p.search(content) for p in _COMPARATIVE_PATTERNS):
        score += 1

    # Entity richness bonus (+1 if 2+ entities detected)
    entities = _ENTITY_RE.findall(content)
    if len(entities) >= 2:
        score += 1

    # Short content penalty (-1 if <20 chars)
    if len(content) < 20:
        score -= 1

    return max(1, min(10, score))

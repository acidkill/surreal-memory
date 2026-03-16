"""Reflection engine for meta-memory generation (Phase A4-C).

Accumulates importance from saved memories. When threshold is reached,
analyzes recent high-priority memories for patterns (repeated entities,
temporal sequences, contradictions) and generates higher-order insights.

All processing is rule-based — zero LLM calls.
"""

from __future__ import annotations

import re
from typing import Any

# Entity extraction (capitalized words, tech terms)
_ENTITY_RE = re.compile(
    r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"  # CamelCase
    r"|\b[A-Z][a-zA-Z]*(?:SQL|DB|JS|API)\b"  # TechTerms
    r"|\b[A-Z][a-z]{2,}\b"  # Capitalized words (3+ chars)
)

# Temporal sequence markers
_TEMPORAL_MARKERS = re.compile(
    r"\b(first|then|after that|next|finally|before|later|afterward)\b",
    re.IGNORECASE,
)

# Negation patterns for contradiction detection
_NEGATION_PAIRS = [
    (re.compile(r"\bis\s+(?:the\s+)?best\b", re.I), re.compile(r"\bis\s+not\s+suitable\b", re.I)),
    (re.compile(r"\bshould\s+use\b", re.I), re.compile(r"\bshould\s+not\s+use\b", re.I)),
    (re.compile(r"\bchose\b", re.I), re.compile(r"\brejected\b", re.I)),
    (re.compile(r"\bworks?\s+well\b", re.I), re.compile(r"\bdoes(?:n't| not)\s+work\b", re.I)),
]

# Minimum memories needed for pattern detection
_MIN_MEMORIES = 2


def detect_patterns(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect patterns across a cluster of recent memories.

    Patterns detected:
    - Recurring entity: same entity appears in 3+ memories
    - Temporal sequence: 3+ memories with temporal markers
    - Contradiction: two memories with opposing statements about same subject

    Args:
        memories: List of dicts with "content", "type", "tags" keys.

    Returns:
        List of pattern dicts with "pattern_type", "description", "source_indices".
    """
    if len(memories) < _MIN_MEMORIES:
        return []

    patterns: list[dict[str, Any]] = []

    # 1. Recurring entities
    entity_to_indices: dict[str, list[int]] = {}
    for i, mem in enumerate(memories):
        entities = _ENTITY_RE.findall(mem.get("content", ""))
        for entity in entities:
            entity_lower = entity.lower()
            entity_to_indices.setdefault(entity_lower, []).append(i)

    for entity, indices in entity_to_indices.items():
        if len(indices) >= 3:
            patterns.append(
                {
                    "pattern_type": "recurring_entity",
                    "description": f"'{entity}' is a recurring theme across {len(indices)} memories",
                    "source_indices": indices,
                    "entity": entity,
                }
            )

    # 2. Temporal sequences
    temporal_indices = [
        i for i, mem in enumerate(memories) if _TEMPORAL_MARKERS.search(mem.get("content", ""))
    ]
    if len(temporal_indices) >= 3:
        patterns.append(
            {
                "pattern_type": "temporal_sequence",
                "description": f"Detected a workflow/sequence pattern across {len(temporal_indices)} memories",
                "source_indices": temporal_indices,
            }
        )

    # 3. Contradictions (pairwise check)
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            content_i = memories[i].get("content", "")
            content_j = memories[j].get("content", "")
            if _is_contradiction(content_i, content_j):
                patterns.append(
                    {
                        "pattern_type": "contradiction",
                        "description": f"Potential contradiction between memories {i} and {j}",
                        "source_indices": [i, j],
                    }
                )

    return patterns


def _is_contradiction(text_a: str, text_b: str) -> bool:
    """Check if two texts contain contradictory statements."""
    for pos_pattern, neg_pattern in _NEGATION_PAIRS:
        if (pos_pattern.search(text_a) and neg_pattern.search(text_b)) or (
            neg_pattern.search(text_a) and pos_pattern.search(text_b)
        ):
            return True
    return False


class ReflectionEngine:
    """Accumulates importance and triggers reflection when threshold is reached.

    Usage:
        engine = ReflectionEngine(threshold=50.0)
        engine.accumulate(fiber.priority)
        if engine.should_reflect():
            patterns = detect_patterns(recent_memories)
            engine.reset()
    """

    def __init__(self, threshold: float = 50.0) -> None:
        self.threshold = threshold
        self._accumulated: float = 0.0

    @property
    def accumulated(self) -> float:
        """Current accumulated importance."""
        return self._accumulated

    def accumulate(self, importance: float) -> None:
        """Add importance from a saved memory."""
        self._accumulated += importance

    def should_reflect(self) -> bool:
        """Check if accumulated importance has reached the reflection threshold."""
        return self._accumulated >= self.threshold

    def reset(self) -> None:
        """Reset accumulator after reflection is performed."""
        self._accumulated = 0.0

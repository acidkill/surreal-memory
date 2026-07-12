"""Trust resolution for recall scoring (U2 — pure, zero-LLM).

`resolve_effective_trust` picks the most authoritative trust signal available for a
memory, in priority order, falling back to a config default. Thresholds are module
constants so they can be tuned without a schema migration. This module has NO
storage or time dependency — it is a pure function over already-loaded objects.
"""

from __future__ import annotations

from surreal_memory.core.memory_types import TypedMemory
from surreal_memory.core.source import Source, SourceType

# Trust by a registered Source's source_type (authoritative document class).
DEFAULT_SOURCE_TYPE_TRUST: dict[SourceType, float] = {
    SourceType.LAW: 0.95,
    SourceType.CONTRACT: 0.95,
    SourceType.LEDGER: 0.90,
    SourceType.RESEARCH: 0.90,
    SourceType.DOCUMENT: 0.90,
    SourceType.BOOK: 0.85,
    SourceType.API: 0.80,
    SourceType.MANUAL: 0.75,
    SourceType.WEBSITE: 0.60,
}

# Trust by the free-form origin label stored on TypedMemory.source.
DEFAULT_LABEL_TRUST: dict[str, float] = {
    "verified": 0.95,
    "user_input": 0.80,
    "direct": 0.80,
    "observation": 0.70,
    "import": 0.60,
    "mcp_tool": 0.60,
    "ai_inference": 0.50,
    "auto_capture": 0.40,
}


def _normalize_label(label: str) -> str:
    """Normalise a source label the same way _cap_trust_score does.

    Extract the base before ':' (e.g. "mcp:claude_code" -> "mcp") and fold the
    "mcp" family to "mcp_tool" so both agree on the key used for defaults.
    """
    base = label.split(":")[0] if ":" in label else label
    return "mcp_tool" if base == "mcp" else base


def resolve_effective_trust(
    tm: TypedMemory | None,
    source: Source | None,
    default: float,
) -> float:
    """Resolve the effective trust in [0, 1] for a memory, most authoritative first.

    Priority:
      1. ``tm.trust_score`` (an explicit, already source-capped score)
      2. ``source.trust`` (a manual per-source override)
      3. per-``source_type`` default (law/contract > document > website ...)
      4. per-label default (verified > user_input > ... > auto_capture)
      5. the caller's ``default`` (BrainConfig.trust_default)
    """
    if tm is not None and tm.trust_score is not None:
        return tm.trust_score
    if source is not None and source.trust is not None:
        return source.trust
    if source is not None:
        by_type = DEFAULT_SOURCE_TYPE_TRUST.get(source.source_type)
        if by_type is not None:
            return by_type
    if tm is not None and tm.source:
        by_label = DEFAULT_LABEL_TRUST.get(_normalize_label(tm.source))
        if by_label is not None:
            return by_label
    return default

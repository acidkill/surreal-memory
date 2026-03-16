"""Context formatting for retrieval."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.activation import ActivationResult
from neural_memory.utils.timeutils import utcnow

# Average tokens per whitespace-separated word (accounts for subword tokenization)
_TOKEN_RATIO = 1.3

# Compression tier thresholds (days)
_FULL_CONTENT_DAYS = 7
_SUMMARY_DAYS = 30
_MINIMAL_DAYS = 90

# Sentence splitting regex — handles ". ", "! ", "? " followed by uppercase or end
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def compress_for_recall(
    content: str,
    summary: str | None,
    created_at: datetime | None,
    max_sentences_medium: int = 3,
    max_sentences_old: int = 2,
) -> str:
    """Compress memory content based on age for context-efficient recall.

    Tiers:
        < 7 days: full content
        7-30 days: summary (if available) or first N sentences
        30-90 days: summary (if available) or first 2 sentences
        90+ days: summary (if available) or first sentence only

    Args:
        content: Raw memory content.
        summary: Fiber summary (from consolidation), may be None.
        created_at: When the memory was created.
        max_sentences_medium: Max sentences for 7-30 day tier.
        max_sentences_old: Max sentences for 30-90 day tier.

    Returns:
        Compressed content string.
    """
    if not content:
        return ""

    # Safe fallback: no timestamp = treat as recent
    if created_at is None:
        return content

    now = utcnow()
    age_days = (now - created_at).total_seconds() / 86400

    # Tier 1: Recent — full content
    if age_days < _FULL_CONTENT_DAYS:
        return content

    # Tier 2: Medium age — summary or truncated sentences
    if age_days < _SUMMARY_DAYS:
        if summary:
            return summary
        return _truncate_to_sentences(content, max_sentences_medium)

    # Tier 3: Old — summary or key sentences
    if age_days < _MINIMAL_DAYS:
        if summary:
            return summary
        return _truncate_to_sentences(content, max_sentences_old)

    # Tier 4: Very old — summary or first sentence only
    if summary:
        return summary
    return _truncate_to_sentences(content, 1)


def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    """Extract first N sentences from text."""
    sentences = _SENTENCE_RE.split(text)
    if len(sentences) <= max_sentences:
        return text
    return ". ".join(s.rstrip(".") for s in sentences[:max_sentences]) + "."


def _estimate_tokens(text: str) -> int:
    """Estimate LLM token count from text using word-based heuristic."""
    return int(len(text.split()) * _TOKEN_RATIO)


if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.safety.encryption import MemoryEncryptor
    from neural_memory.storage.base import NeuralStorage


async def format_context(
    storage: NeuralStorage,
    activations: dict[str, ActivationResult],
    fibers: list[Fiber],
    max_tokens: int,
    encryptor: MemoryEncryptor | None = None,
    brain_id: str = "",
) -> tuple[str, int]:
    """Format activated memories into context for agent injection.

    Returns:
        Tuple of (formatted_context, token_estimate).
    """

    def _maybe_decrypt(text: str, fiber_meta: dict[str, Any]) -> str:
        """Decrypt content if fiber is encrypted and encryptor is available."""
        if encryptor and brain_id and fiber_meta.get("encrypted"):
            return encryptor.decrypt(text, brain_id)
        return text

    lines: list[str] = []
    token_estimate = 0

    # Add fiber summaries first (batch fetch anchors)
    if fibers:
        lines.append("## Relevant Memories\n")

        anchor_ids = list({f.anchor_neuron_id for f in fibers[:5] if not f.summary})
        anchor_map = await storage.get_neurons_batch(anchor_ids) if anchor_ids else {}

        for fiber in fibers[:5]:
            if fiber.summary:
                content = fiber.summary
            else:
                anchor = anchor_map.get(fiber.anchor_neuron_id)
                if anchor:
                    content = _maybe_decrypt(anchor.content, fiber.metadata)
                else:
                    continue

            # Age-based compression: older memories get compressed
            content = compress_for_recall(
                content,
                summary=fiber.summary,
                created_at=fiber.created_at,
            )

            # Format structured content if metadata has _structure
            content = _format_if_structured(content, fiber.metadata)

            # Truncate long content to fit within token budget
            remaining_budget = max_tokens - token_estimate
            if remaining_budget <= 0:
                break

            content_tokens = _estimate_tokens(content)
            if content_tokens > remaining_budget:
                # Truncate to fit: estimate words from remaining budget
                max_words = int(remaining_budget / _TOKEN_RATIO)
                if max_words < 10:
                    break
                words = content.split()
                content = " ".join(words[:max_words]) + "..."

            line = f"- {content}"
            token_estimate += _estimate_tokens(line)
            lines.append(line)

    # Add individual activated neurons (batch fetch)
    if token_estimate < max_tokens:
        lines.append("\n## Related Information\n")

        sorted_activations = sorted(
            activations.values(),
            key=lambda a: a.activation_level,
            reverse=True,
        )

        top_ids = [r.neuron_id for r in sorted_activations[:20]]
        neuron_map = await storage.get_neurons_batch(top_ids)

        for result in sorted_activations[:20]:
            neuron = neuron_map.get(result.neuron_id)
            if neuron is None:
                continue

            # Skip time neurons in context (they're implicit)
            if neuron.type == NeuronType.TIME:
                continue

            line = f"- [{neuron.type.value}] {neuron.content}"
            token_estimate += _estimate_tokens(line)

            if token_estimate > max_tokens:
                break

            lines.append(line)

    return "\n".join(lines), token_estimate


def _format_if_structured(content: str, metadata: dict[str, Any]) -> str:
    """Format content using structure metadata if available.

    If the neuron/fiber has _structure metadata (set by StructureDetectionStep),
    re-format the content for readable output. Otherwise return as-is.
    """
    structure = metadata.get("_structure")
    if not structure or not isinstance(structure, dict):
        return content

    fmt = structure.get("format", "plain")
    if fmt == "plain":
        return content

    fields = structure.get("fields", [])
    if not fields:
        return content

    # Rebuild StructuredContent from stored metadata for formatting
    from neural_memory.extraction.structure_detector import (
        ContentFormat,
        StructuredContent,
        StructuredField,
        format_structured_output,
    )

    sc = StructuredContent(
        format=ContentFormat(fmt),
        fields=tuple(
            StructuredField(
                name=f.get("name", ""),
                value=f.get("value", ""),
                field_type=f.get("type", "text"),
            )
            for f in fields
        ),
        raw=content,
    )
    return format_structured_output(sc)

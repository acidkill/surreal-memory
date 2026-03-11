"""Context formatting for retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import NeuronType
from neural_memory.engine.activation import ActivationResult

# Average tokens per whitespace-separated word (accounts for subword tokenization)
_TOKEN_RATIO = 1.3


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

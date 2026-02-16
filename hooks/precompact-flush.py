#!/usr/bin/env python3
"""PreCompact hook — emergency flush memories before context compaction.

Fires before Claude Code compresses the conversation. Reads the transcript,
detects memorable patterns, saves them to the neural graph, and injects
a session summary as additionalContext so critical info survives compaction.

No external dependencies beyond neural-memory itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("neural_memory.hooks.precompact")

# Maximum characters to analyze from the transcript tail
_MAX_TRANSCRIPT_CHARS = 40_000
# Maximum messages to scan from the end of the transcript
_MAX_MESSAGES = 80


def _read_hook_input() -> dict:
    """Read JSON hook input from stdin."""
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_transcript_text(transcript_path: str) -> str:
    """Extract recent text content from transcript JSONL file."""
    path = Path(transcript_path)
    if not path.is_file():
        return ""

    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            # Take last N lines (most recent messages)
            recent = all_lines[-_MAX_MESSAGES:]

        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract text from various transcript formats
            content = ""
            if isinstance(msg.get("content"), str):
                content = msg["content"]
            elif isinstance(msg.get("content"), list):
                # Content blocks format: [{"type": "text", "text": "..."}]
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "") + "\n"
            elif isinstance(msg.get("message"), dict):
                inner = msg["message"]
                if isinstance(inner.get("content"), str):
                    content = inner["content"]
                elif isinstance(inner.get("content"), list):
                    for block in inner["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "") + "\n"

            if content.strip():
                role = msg.get("role", msg.get("type", ""))
                lines.append(f"[{role}] {content.strip()}")

    except OSError:
        return ""

    text = "\n\n".join(lines)
    # Trim to max chars from the end (most recent content)
    if len(text) > _MAX_TRANSCRIPT_CHARS:
        text = text[-_MAX_TRANSCRIPT_CHARS:]
    return text


async def _flush_memories(text: str) -> tuple[int, list[str]]:
    """Analyze text and flush detected memories to storage.

    Returns (saved_count, summary_lines).
    """
    if not text:
        return 0, []

    from neural_memory.mcp.auto_capture import analyze_text_for_memories
    from neural_memory.safety.sensitive import auto_redact_content

    detected = analyze_text_for_memories(
        text,
        capture_decisions=True,
        capture_errors=True,
        capture_todos=True,
        capture_facts=True,
        capture_insights=True,
        capture_preferences=True,
    )

    if not detected:
        return 0, []

    # Emergency threshold: lower than normal to capture more before compaction
    emergency_threshold = 0.5
    eligible = [item for item in detected if item["confidence"] >= emergency_threshold]

    if not eligible:
        return 0, []

    # Boost priority for emergency-captured memories
    boosted = [{**item, "priority": min(item.get("priority", 5) + 2, 10)} for item in eligible]

    # Import storage and encoding machinery
    from neural_memory.core.typed_memory import MemoryType, Priority, TypedMemory
    from neural_memory.engine.encoder import MemoryEncoder
    from neural_memory.unified_config import get_shared_storage
    from neural_memory.utils.timeutils import utcnow

    storage = await get_shared_storage()
    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        return 0, []

    encoder = MemoryEncoder(storage, brain.config)
    saved = 0
    summary_lines: list[str] = []

    for item in boosted:
        content = item["content"]
        # Auto-redact sensitive content
        redacted, _, _ = auto_redact_content(content, min_severity=3)
        mem_type_str = item.get("type", "context")

        try:
            mem_type = MemoryType(mem_type_str)
        except ValueError:
            mem_type = MemoryType.CONTEXT

        priority_val = item.get("priority", 5)
        try:
            mem_priority = Priority(min(max(priority_val, 0), 10))
        except ValueError:
            mem_priority = Priority(5)

        try:
            storage.disable_auto_save()
            result = await encoder.encode(
                content=redacted,
                timestamp=utcnow(),
                tags={"emergency_flush", "precompact"},
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=mem_priority,
                source="precompact_hook",
                tags={"emergency_flush", "precompact"},
            )
            await storage.add_typed_memory(typed_mem)
            await storage.batch_save()
            saved += 1
            summary_lines.append(f"- [{mem_type_str}] {redacted[:80]}")
        except Exception:
            logger.debug("Failed to save memory during precompact flush", exc_info=True)
            continue

    await storage.close()
    return saved, summary_lines


def _build_additional_context(saved: int, summary_lines: list[str]) -> str:
    """Build additionalContext string for Claude to preserve across compaction."""
    parts = ["NeuralMemory PreCompact: Session context preserved."]

    if saved > 0:
        parts.append(f"Auto-saved {saved} memories before compaction:")
        parts.extend(summary_lines[:10])  # Limit to 10 lines

    parts.append(
        "Use nmem_recall or nmem_recap to retrieve session context after compaction."
    )
    return "\n".join(parts)


def main() -> None:
    hook_input = _read_hook_input()
    transcript_path = hook_input.get("transcript_path", "")

    # Extract text from transcript
    text = _extract_transcript_text(transcript_path)

    # Run flush
    try:
        saved, summary_lines = asyncio.run(_flush_memories(text))
    except Exception:
        logger.debug("PreCompact flush failed", exc_info=True)
        saved, summary_lines = 0, []

    # Build output with additionalContext
    additional = _build_additional_context(saved, summary_lines)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreCompact",
            "additionalContext": additional,
        },
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()

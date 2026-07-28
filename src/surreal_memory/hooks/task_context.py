"""Task-context hook: persist a rich, structured note after a completed task.

Unlike the Stop hook (which pattern-detects atomic memories from a transcript),
this hook stores a single, agent-authored summary of a finished task verbatim
as one ``context`` memory, scoped to the current project. It is intended to be
driven by an agent-type Claude Code hook that synthesises the note (what was
requested, what was done and how, commands used, problems hit and how they were
solved, the value delivered, and any user preferences) and pipes it in.

Usage as a command (driven by an agent hook):
    printf '%s' "<structured note>" | smem-hook-task-context
    smem-hook-task-context --text "<structured note>" --project myrepo

The note is saved against the current project (git repo basename, or ``cwd``
basename) so recall can be filtered per project on the shared brain.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Below this length the note is treated as too trivial to persist.
MIN_NOTE_CHARS = 40
# Cap stored note size (matches other hooks' flush ceiling).
MAX_NOTE_CHARS = 100_000
# Task summaries are important context — store at high priority.
TASK_CONTEXT_PRIORITY = 7


async def save_task_context(note: str, project_name: str | None) -> dict[str, Any]:
    """Persist a single structured task note as a project-scoped context memory."""
    from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
    from surreal_memory.engine.dedup.factory import build_dedup_pipeline
    from surreal_memory.engine.encoder import MemoryEncoder
    from surreal_memory.safety.input_firewall import check_content
    from surreal_memory.safety.sensitive import auto_redact_content
    from surreal_memory.unified_config import get_config, get_shared_storage
    from surreal_memory.utils.timeutils import utcnow

    # Gate 1: Input firewall — block garbage/adversarial content.
    fw = check_content(note)
    if fw.blocked:
        return {"saved": 0, "message": f"Input blocked: {fw.reason}"}
    if fw.sanitized:
        note = fw.sanitized

    config = get_config()
    storage = await get_shared_storage(config.current_brain)
    try:
        brain = await storage.get_brain(config.current_brain)
        if not brain:
            return {"saved": 0, "error": "No brain configured"}

        tags = {"task_context", "claude_code"}
        if project_name:
            tags.add(f"project:{project_name}")

        redacted, matches, _ = auto_redact_content(
            note, min_severity=config.safety.auto_redact_min_severity
        )
        if matches:
            logger.debug("Auto-redacted %d matches in task-context note", len(matches))

        encoder = MemoryEncoder(storage, brain.config, dedup_pipeline=build_dedup_pipeline(storage))
        storage.disable_auto_save()

        result = await encoder.encode(
            content=redacted,
            timestamp=utcnow(),
            tags=set(tags),
        )
        # The encoder decomposes content into concept neurons; persist the
        # verbatim note as the fiber summary so it is recallable as-is.
        await storage.update_fiber(result.fiber.with_summary(redacted))

        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=MemoryType.CONTEXT,
            priority=Priority.from_int(TASK_CONTEXT_PRIORITY),
            source="task_context_hook",
            # The repo name doubles as the project id (opaque scope label).
            project_id=project_name,
            tags=set(tags),
        )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()

        return {
            "saved": 1,
            "project": project_name,
            "message": f"Task context saved for project '{project_name}'",
        }
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("storage.close() failed (non-fatal)", exc_info=True)


def _read_note(args: Any) -> str:
    """Resolve the note text from --text or stdin."""
    if args.text:
        return str(args.text)
    # Read piped stdin; guard against blocking on an interactive terminal.
    if not sys.stdin.isatty():
        try:
            return sys.stdin.read()
        except OSError:
            return ""
    return ""


def main() -> None:
    """Entry point for the task-context hook / standalone CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Surreal-Memory task-context hook — persist a structured task note"
    )
    parser.add_argument("--text", help="Note text (alternative to stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read note from stdin (default)")
    parser.add_argument(
        "--project",
        "-P",
        help="Project scope (defaults to git repo / cwd basename)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, stream=sys.stderr)

    note = _read_note(args).strip()
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS]

    if len(note) < MIN_NOTE_CHARS:
        print("[Surreal-Memory] task-context: note too short, skipping", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    from surreal_memory.hooks.project_context import derive_project_name

    project_name = args.project or derive_project_name(None)

    try:
        result = asyncio.run(save_task_context(note, project_name))
    except Exception as exc:  # Never block Claude Code.
        print(f"[Surreal-Memory] task-context error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    if result.get("saved"):
        print(  # noqa: T201
            f"[Surreal-Memory] Task context saved for project '{project_name}'",
            file=sys.stderr,
        )
    else:
        print(  # noqa: T201
            f"[Surreal-Memory] task-context: {result.get('message', 'not saved')}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

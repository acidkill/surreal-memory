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
import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from surreal_memory.hooks.receipts import emit_receipt, receipt

logger = logging.getLogger(__name__)

# Below this length the note is treated as too trivial to persist.
MIN_NOTE_CHARS = 40
# Cap stored note size (matches other hooks' flush ceiling).
MAX_NOTE_CHARS = 100_000
# Task summaries are important context — store at high priority.
TASK_CONTEXT_PRIORITY = 7


class CheckpointConflictError(ValueError):
    """Same checkpoint scope and revision were already stored with other data."""


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
            "memory_ids": [result.fiber.id],
            "message": f"Task context saved for project '{project_name}'",
        }
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("storage.close() failed (non-fatal)")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def validate_checkpoint(envelope: Any) -> dict[str, Any]:
    """Validate the complete checkpoint envelope before opening storage."""
    if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
        raise ValueError("checkpoint schema_version must be 1")
    scope = envelope.get("scope")
    state = envelope.get("state")
    if not isinstance(scope, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint scope and state must be objects")
    required_scope = {"brain", "project", "session_id", "agent_id"}
    if set(scope) != required_scope or any(
        not isinstance(scope[k], str) or not scope[k] for k in required_scope
    ):
        raise ValueError("checkpoint scope is invalid")
    project_path = Path(scope["project"])
    if not project_path.is_absolute() or str(project_path.resolve()) != scope["project"]:
        raise ValueError("checkpoint project must be an absolute canonical path")
    state_fields = {
        "goal",
        "constraints",
        "approvals",
        "decisions",
        "completed",
        "next_actions",
        "blockers",
        "processes",
    }
    if (
        set(state) != state_fields
        or not isinstance(state["goal"], str)
        or not state["goal"].strip()
    ):
        raise ValueError("checkpoint state fields are invalid")
    if any(
        not isinstance(state[field], list)
        or any(not isinstance(item, str) for item in state[field])
        for field in state_fields - {"goal"}
    ):
        raise ValueError("checkpoint state arrays are invalid")
    revision = envelope.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("checkpoint revision must be a positive integer")
    canonical_scope = _canonical_json(scope)
    canonical_state = _canonical_json(state)
    scope_id = hashlib.sha256(canonical_scope.encode("utf-8")).hexdigest()
    content_hash = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()
    if envelope.get("scope_id") != scope_id:
        raise ValueError("checkpoint scope_id does not match scope")
    if envelope.get("content_hash") != content_hash:
        raise ValueError("checkpoint content_hash does not match state")
    return {
        "schema_version": 1,
        "scope": scope,
        "scope_id": scope_id,
        "revision": revision,
        "content_hash": content_hash,
        "state": state,
        "content": canonical_state,
    }


async def save_checkpoint(
    checkpoint: dict[str, Any], project_name: str | None = None
) -> dict[str, Any]:
    """Idempotently persist and read back one scoped Codex task checkpoint."""
    from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
    from surreal_memory.engine.dedup.factory import build_dedup_pipeline
    from surreal_memory.engine.encoder import MemoryEncoder
    from surreal_memory.safety.input_firewall import check_content
    from surreal_memory.safety.sensitive import auto_redact_content
    from surreal_memory.unified_config import get_config, get_shared_storage
    from surreal_memory.utils.timeutils import utcnow

    config = get_config()
    if config.current_brain != checkpoint["scope"]["brain"]:
        raise ValueError("checkpoint brain does not match configured brain")
    project_name = project_name if project_name is not None else checkpoint["scope"]["project"]
    project_path = Path(project_name)
    if not project_path.is_absolute() or str(project_path.resolve()) != project_name:
        raise ValueError("checkpoint project must be an absolute canonical path")
    content = checkpoint["content"]
    firewall = check_content(content)
    redacted, matches, _ = auto_redact_content(
        content, min_severity=config.safety.auto_redact_min_severity
    )
    if firewall.blocked or firewall.sanitized != content or matches or redacted != content:
        return {"saved": 0, "status": "failed", "error": "checkpoint_sensitive"}
    from surreal_memory.hooks.capture_state import local_scope_lock

    storage = await get_shared_storage(config.current_brain)
    scope_id = checkpoint["scope_id"]
    revision = checkpoint["revision"]
    content_hash = checkpoint["content_hash"]
    tags = {
        "codex_checkpoint",
        f"codex_scope:{scope_id}",
        f"codex_revision:{revision}",
        f"codex_hash:{content_hash}",
    }
    metadata = {
        "schema_version": 1,
        "scope_id": scope_id,
        "revision": revision,
        "content_hash": content_hash,
    }
    try:
        with local_scope_lock(scope_id):
            fibers = await storage.find_fibers(
                tags={"codex_checkpoint", f"codex_scope:{scope_id}", f"codex_revision:{revision}"},
                limit=100,
            )
            matching = [
                f
                for f in fibers
                if f.metadata.get("scope_id") == scope_id and f.metadata.get("revision") == revision
            ]
            hash_tag = f"codex_hash:{content_hash}"
            if any(
                f.metadata.get("content_hash") not in (None, content_hash) for f in matching
            ) or any(
                tag.startswith("codex_hash:") and tag != hash_tag for f in fibers for tag in f.tags
            ):
                raise CheckpointConflictError("checkpoint revision conflict")
            fiber = next(
                (
                    f
                    for f in fibers
                    if hash_tag in f.tags and f.metadata.get("scope_id") in (None, scope_id)
                ),
                None,
            )
            if fiber is None:
                brain = await storage.get_brain(config.current_brain)
                if not brain:
                    return {
                        "saved": 0,
                        "status": "failed",
                        "error": "checkpoint_brain_unavailable",
                    }
                encoder = MemoryEncoder(
                    storage, brain.config, dedup_pipeline=build_dedup_pipeline(storage)
                )
                storage.disable_auto_save()
                result = await encoder.encode(
                    content=checkpoint["content"], timestamp=utcnow(), tags=tags
                )
                fiber = result.fiber

            if any(fiber.metadata.get(key) not in (None, value) for key, value in metadata.items()):
                raise CheckpointConflictError("checkpoint fiber conflict")
            fiber = replace(
                fiber,
                summary=checkpoint["content"],
                metadata={**fiber.metadata, "codex_checkpoint": True, **metadata},
            )
            await storage.update_fiber(fiber)

            typed_memory = await storage.get_typed_memory(fiber.id)
            if typed_memory is None:
                typed_memory = TypedMemory.create(
                    fiber_id=fiber.id,
                    memory_type=MemoryType.CONTEXT,
                    priority=Priority.from_int(TASK_CONTEXT_PRIORITY),
                    source="codex_checkpoint_hook",
                    project_id=project_name
                    if project_name is not None
                    else checkpoint["scope"]["project"],
                    tags=tags,
                    metadata={"codex_checkpoint": True, **metadata},
                )
                await storage.add_typed_memory(typed_memory)
            elif (
                any(typed_memory.metadata.get(key) != value for key, value in metadata.items())
                or typed_memory.source != "codex_checkpoint_hook"
                or typed_memory.project_id != project_name
                or not tags.issubset(typed_memory.tags)
            ):
                raise CheckpointConflictError("typed checkpoint conflict")

            await storage.batch_save()
            readback = await storage.find_fibers(tags=tags, limit=100)
            verified_fiber = next(
                (
                    f
                    for f in readback
                    if f.id == fiber.id
                    and all(
                        f.metadata.get(k) == v
                        for k, v in {"codex_checkpoint": True, **metadata}.items()
                    )
                ),
                None,
            )
            verified_typed = await storage.get_typed_memory(fiber.id)
            if (
                verified_fiber is None
                or verified_fiber.summary != checkpoint["content"]
                or verified_fiber.metadata.get("content_hash") != content_hash
                or verified_typed is None
                or not tags.issubset(verified_typed.tags)
                or verified_typed.source != "codex_checkpoint_hook"
                or verified_typed.project_id != project_name
            ):
                raise RuntimeError("checkpoint readback did not confirm fiber and typed memory")
            if any(
                verified_typed.metadata.get(k) != v
                for k, v in {"codex_checkpoint": True, **metadata}.items()
            ):
                raise RuntimeError("checkpoint typed memory readback metadata mismatch")
            return {
                "saved": 1,
                "status": "saved",
                "memory_ids": [fiber.id],
                "project": project_name
                if project_name is not None
                else checkpoint["scope"]["project"],
                "message": "Task checkpoint saved and verified",
            }
    except ValueError:
        raise
    except Exception:
        logger.warning("Checkpoint persistence is pending retry")
        return {"saved": 0, "status": "pending", "error": "checkpoint_pending"}
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("storage.close() failed (non-fatal)")


def _read_note(args: Any) -> str:
    """Resolve the note text from --text or stdin."""
    if getattr(args, "checkpoint", False):
        try:
            return sys.stdin.read()
        except OSError:
            return ""
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
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Read a versioned checkpoint JSON envelope from stdin",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit a versioned JSON receipt on stdout"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, stream=sys.stderr)

    raw_note = _read_note(args).strip()
    if args.checkpoint:
        try:
            checkpoint = validate_checkpoint(json.loads(raw_note))
        except (json.JSONDecodeError, ValueError, TypeError, OSError):
            if args.json:
                emit_receipt(receipt("failed", error="checkpoint_invalid"))
            print(  # noqa: T201
                "[Surreal-Memory] task-context checkpoint error: checkpoint_invalid",
                file=sys.stderr,
            )
            if args.json:
                sys.exit(1)
            sys.exit(0)
        project_name = args.project or checkpoint["scope"]["project"]
        try:
            result = asyncio.run(save_checkpoint(checkpoint, project_name))
        except CheckpointConflictError:
            result = {"saved": 0, "status": "failed", "error": "checkpoint_conflict"}
        except ValueError:
            result = {"saved": 0, "status": "failed", "error": "checkpoint_invalid"}
        except Exception:
            result = {"saved": 0, "status": "pending", "error": "checkpoint_pending"}
        if args.json:
            status = result.get("status") or (
                "failed" if result.get("error") else ("saved" if result.get("saved") else "skipped")
            )
            emit_receipt(
                receipt(
                    status,
                    result.get("saved", 0),
                    checkpoint["scope"]["session_id"],
                    result.get("memory_ids"),
                    result.get("error"),
                    checkpoint["scope_id"],
                    checkpoint["revision"],
                    checkpoint["content_hash"],
                )
            )
        print(  # noqa: T201
            f"[Surreal-Memory] task-context: {result.get('error', result.get('message', 'not saved'))}",
            file=sys.stderr,
        )
        if args.json and result.get("status") != "saved":
            sys.exit(1)
        return

    note = raw_note
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS]

    if len(note) < MIN_NOTE_CHARS:
        if args.json:
            emit_receipt(receipt("skipped"))
        print("[Surreal-Memory] task-context: note too short, skipping", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    from surreal_memory.hooks.project_context import derive_project_name

    project_name = args.project or derive_project_name(None)

    try:
        result = asyncio.run(save_task_context(note, project_name))
    except Exception:  # Never block Claude Code.
        if args.json:
            emit_receipt(receipt("failed", error="task_context_failed"))
        print("[Surreal-Memory] task-context error: task_context_failed", file=sys.stderr)  # noqa: T201
        if args.json:
            sys.exit(1)
        sys.exit(0)

    if args.json:
        status = (
            "failed" if result.get("error") else ("saved" if result.get("saved") else "skipped")
        )
        emit_receipt(
            receipt(
                status,
                result.get("saved", 0),
                memory_ids=result.get("memory_ids"),
                error="task_context_failed" if result.get("error") else None,
            )
        )
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
        if args.json and result.get("error"):
            sys.exit(1)


if __name__ == "__main__":
    main()

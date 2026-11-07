"""Shared, verified task checkpoints for coding agents and MCP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from surreal_memory.hooks.agent_journal import acknowledge, enqueue, event_key, scope_lock
from surreal_memory.hooks.task_context import (
    CheckpointConflictError,
    save_checkpoint,
    validate_checkpoint,
)
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.unified_config import UnifiedConfig


def envelope(scope: Any, state: Any, revision: Any) -> dict[str, Any]:
    """Build and validate the existing v1 contract; hashes are server-computed."""

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest()

    result = {
        "schema_version": 1,
        "scope": scope,
        "state": state,
        "revision": revision,
        "scope_id": digest(scope),
        "content_hash": digest(state),
    }
    if len(json.dumps(result).encode()) > 100_000:
        raise ValueError("checkpoint exceeds 100000 bytes")
    validated = validate_checkpoint(result)
    from surreal_memory.safety.input_firewall import check_content

    firewall = check_content(validated["content"])
    if firewall.blocked or firewall.sanitized != validated["content"]:
        raise ValueError("checkpoint rejected by input firewall")
    return validated


async def load_checkpoint(storage: NeuralStorage, scope: dict[str, str]) -> dict[str, Any] | None:
    """Load an exact scope, verifying both the pointer and referenced typed record."""
    probe = envelope(
        scope,
        {
            "goal": "scope validation",
            **{
                k: []
                for k in (
                    "constraints",
                    "approvals",
                    "decisions",
                    "completed",
                    "next_actions",
                    "blockers",
                    "processes",
                )
            },
        },
        1,
    )
    sid = probe["scope_id"]
    head = await storage.get_fiber("coding_head_" + sid)
    if head:
        fiber = await storage.get_fiber(head.metadata["checkpoint_fiber_id"])
        fibers = [fiber] if fiber else []
    else:
        fibers = await storage.find_fibers(
            tags={"codex_checkpoint", "codex_scope:" + sid}, limit=1000
        )
        if len(fibers) >= 1000:
            raise ValueError(
                "legacy checkpoint history exceeds scan limit; explicit revision required"
            )
    confirmed: list[dict[str, Any]] = []
    for fiber in fibers:
        if not fiber.summary:
            continue
        state = json.loads(fiber.summary)
        revision = fiber.metadata.get("revision")
        item = envelope(scope, state, revision)
        typed = await storage.get_typed_memory(fiber.id)
        if (
            typed is None
            or typed.project_id != scope["project"]
            or typed.metadata.get("scope_id") != sid
            or fiber.metadata.get("scope_id") != sid
            or fiber.metadata.get("content_hash") != item["content_hash"]
            or typed.metadata.get("content_hash") != item["content_hash"]
            or typed.metadata.get("revision") != revision
        ):
            raise ValueError("checkpoint readback mismatch")
        confirmed.append({**item, "memory_ids": [fiber.id]})
    if head and not confirmed:
        raise ValueError("checkpoint pointer is incomplete")
    return max(confirmed, key=lambda value: value["revision"], default=None)


async def checkpoint(
    storage: NeuralStorage, config: UnifiedConfig, args: dict[str, Any]
) -> dict[str, Any]:
    """Save/load a task checkpoint; failures never masquerade as successful writes."""
    action = args.get("action", "load")
    try:
        if action not in ("save", "load"):
            raise ValueError("checkpoint action must be save or load")
        scope = args.get("scope")
        if not isinstance(scope, dict):
            raise ValueError("checkpoint scope must be an object")
        probe = envelope(
            scope,
            args.get("state")
            if action == "save"
            else {
                "goal": "scope validation",
                **{
                    k: []
                    for k in (
                        "constraints",
                        "approvals",
                        "decisions",
                        "completed",
                        "next_actions",
                        "blockers",
                        "processes",
                    )
                },
            },
            args.get("revision", 1),
        )
        if scope["brain"] != config.current_brain or storage.brain_id != scope["brain"]:
            raise ValueError("checkpoint brain does not match active brain")
        if action == "save":
            from surreal_memory.safety.sensitive import auto_redact_content

            _, matches, _ = auto_redact_content(probe["content"], min_severity=2)
            if matches:
                raise ValueError("sensitive checkpoint rejected")
        if config.storage_backend == "memory":
            if action == "save":
                enqueue("checkpoint", args)
            return {
                "status": "pending",
                "saved": 0,
                "error": "durable_backend_required",
                "memory_ids": [],
            }
        async with scope_lock(probe["scope_id"]):
            current = await load_checkpoint(storage, scope)
            if action == "load":
                return (
                    {"status": "saved", "saved": 0, **current}
                    if current
                    else {"status": "skipped", "saved": 0, "memory_ids": []}
                )
            if current and probe["revision"] <= current["revision"]:
                if (
                    probe["revision"] == current["revision"]
                    and probe["content_hash"] == current["content_hash"]
                ):
                    acknowledge(event_key("checkpoint", args))
                    return {"status": "saved", "saved": 0, **current}
                raise CheckpointConflictError("checkpoint revision conflict")
            # Write-ahead queue survives cancellation and partial database commits.
            key = enqueue("checkpoint", args)
            result = await save_checkpoint(
                probe, storage=storage, config=config, already_locked=True
            )
            if result.get("status") != "saved":
                return {**result, "scope_id": probe["scope_id"], "revision": probe["revision"]}
            fiber = await storage.get_fiber(result["memory_ids"][0])
            if fiber is None:
                raise OSError("checkpoint fiber unavailable")
            head = replace(
                fiber,
                id="coding_head_" + probe["scope_id"],
                auto_tags=set(),
                agent_tags={"coding_checkpoint_head"},
                metadata={"checkpoint_fiber_id": fiber.id},
            )
            previous_head = await storage.get_fiber(head.id)
            if previous_head:
                await storage.update_fiber(head)
            else:
                await storage.add_fiber(head)
            await storage.batch_save()
            verified = await load_checkpoint(storage, scope)
            if verified is None or verified["content_hash"] != probe["content_hash"]:
                raise OSError("checkpoint pointer not verified")
            acknowledge(key)
            return {**result, **probe}
    except CheckpointConflictError:
        return {"status": "failed", "saved": 0, "error": "checkpoint_conflict", "memory_ids": []}
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "failed", "saved": 0, "error": str(exc), "memory_ids": []}
    except Exception:
        if action == "save":
            try:
                enqueue("checkpoint", args)
            except (ValueError, OSError):
                return {
                    "status": "failed",
                    "saved": 0,
                    "error": "checkpoint_not_queued",
                    "memory_ids": [],
                }
        return {
            "status": "pending",
            "saved": 0,
            "error": "checkpoint_unavailable",
            "memory_ids": [],
        }

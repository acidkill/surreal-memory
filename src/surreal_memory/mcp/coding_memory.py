"""Scoped knowledge reads; legacy unscoped handlers remain unchanged."""

from __future__ import annotations

from typing import Any

from surreal_memory.core.memory_types import MemoryType
from surreal_memory.engine.retrieval import DepthLevel, ReflexPipeline
from surreal_memory.hooks.agent_scope import project_identity, scope_tag
from surreal_memory.storage.base import NeuralStorage


async def read_scoped(
    storage: NeuralStorage, args: dict[str, Any], *, recall: bool
) -> dict[str, Any]:
    """Apply hard scope filters before retrieval limits, including HOT and exact reads."""
    tag = scope_tag(args)
    assert tag is not None
    if args.get("brains"):
        raise ValueError("scoped reads cannot combine brains")
    limit = max(1, min(int(args.get("limit", 10)), 200))
    budget = max(50, min(int(args.get("max_tokens", 1500)), 10000)) * 4
    tags = {tag, *args.get("tags", [])}
    query = args.get("query", "")
    confidence = 0.0
    if recall:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required and must be a string")
        key = next(
            (
                query[len(prefix) :].strip()
                for prefix in ("recall:fiber:", "fiber:")
                if query.startswith(prefix)
            ),
            None,
        )
        if key is not None:
            fiber = await storage.get_fiber(key)
            fibers = [fiber] if fiber and tags.issubset(fiber.tags) else []
        else:
            brain = await storage.get_brain(storage.brain_id or "")
            if brain is None:
                raise ValueError("No brain configured")
            result = await ReflexPipeline(storage, brain.config).query(
                query=query,
                depth=DepthLevel(args.get("depth", 1)),
                max_tokens=budget // 4,
                tags=tags,
                reconsolidate=False,
            )
            fibers = []
            for fid in result.fibers_matched:
                fiber = await storage.get_fiber(fid)
                if fiber and tags.issubset(fiber.tags):
                    fibers.append(fiber)
            confidence = result.confidence
    else:
        memories = await storage.find_typed_memories(tags=tags, limit=limit)
        fibers = []
        for memory in memories:
            fiber = await storage.get_fiber(memory.fiber_id)
            if fiber and tags.issubset(fiber.tags):
                fibers.append(fiber)
    if args.get("include_global_preferences") and tag != "coding_global_preference":
        preferences = await storage.find_typed_memories(
            memory_type=MemoryType.PREFERENCE, tags={"coding_global_preference"}, limit=10
        )
        for preference in preferences:
            fiber = await storage.get_fiber(preference.fiber_id)
            if fiber:
                fibers.append(fiber)
    parts: list[str] = []
    ids: list[str] = []
    for fiber in fibers:
        typed_memory = await storage.get_typed_memory(fiber.id)
        if typed_memory is None or typed_memory.is_expired or fiber.id in ids:
            continue
        text = fiber.summary or fiber.essence
        if not text and fiber.anchor_neuron_id:
            neuron = await storage.get_neuron(fiber.anchor_neuron_id)
            text = neuron.content if neuron else ""
        if text:
            parts.append("- " + text)
            ids.append(fiber.id)
    context = "\n".join(parts)[:budget]
    return {
        "context": context,
        "answer": context,
        "count": len(ids),
        "fibers_matched": ids,
        "confidence": confidence,
        "scope": args["scope"],
    }


async def remember_scoped(server: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Write-ahead queue plus scoped content identity spans hook and skill writes."""
    import asyncio
    import hashlib
    import json

    from surreal_memory.hooks.agent_journal import acknowledge, enqueue, scope_lock

    try:
        args = dict(args)
        if args.get("scope") == "project":
            args.setdefault("project_id", project_identity())
        tag = scope_tag(args)
        if args.get("scope") == "global_preference" and args.get("type") != "preference":
            raise ValueError("global_preference scope requires explicit type='preference'")
        if not isinstance(args.get("content"), str) or not args["content"].strip():
            raise ValueError("content is required")
        if args.get("_auto_capture") and not server.config.write_gate.auto_capture_enabled:
            return {"capture_skipped": True, "error": "Automatic capture disabled"}
        digest = hashlib.sha256(
            json.dumps(
                [tag, args["content"].strip(), args.get("type"), args.get("context")],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        identity = "coding_content:" + digest
        memory = {**args, "tags": list({*args.get("tags", []), identity})}
        payload = {
            "brain": server.config.current_brain,
            "project_id": args.get("project_id"),
            "memory": memory,
        }
        async with scope_lock(digest):
            key = enqueue("memory", payload)
            if server.config.storage_backend == "memory":
                return {
                    "status": "pending",
                    "success": False,
                    "saved": 0,
                    "error": "durable_backend_required",
                    "memory_ids": [],
                }
            try:
                storage = await asyncio.wait_for(server.get_storage(), timeout=5)
                existing = await storage.find_typed_memories(tags={identity, tag}, limit=1)
                if existing:
                    fiber = await storage.get_fiber(existing[0].fiber_id)
                    if fiber is not None and tag in fiber.tags:
                        acknowledge(key)
                        return {
                            "success": True,
                            "status": "saved",
                            "saved": 0,
                            "fiber_id": fiber.id,
                            "memory_ids": [fiber.id],
                            "duplicate": True,
                        }
                result: dict[str, Any] = await asyncio.wait_for(
                    server._remember_impl(dict(memory)), timeout=10
                )
                if result.get("success"):
                    fiber_id = result["fiber_id"]
                    fiber = await storage.get_fiber(fiber_id)
                    typed = await storage.get_typed_memory(fiber_id)
                    if fiber is None or typed is None or not {tag, identity}.issubset(fiber.tags):
                        raise OSError("scoped memory readback failed")
                    acknowledge(key)
                    result.update(status="saved", saved=1, memory_ids=[fiber_id])
                elif result.get("capture_skipped") or "error" in result:
                    # The write gate and invalid content are terminal rejections, not outages.
                    acknowledge(key)
                return result
            except Exception:
                return {
                    "success": False,
                    "status": "pending",
                    "saved": 0,
                    "memory_ids": [],
                    "error": "memory_pending",
                }
    except (ValueError, TypeError, OSError) as exc:
        return {
            "success": False,
            "status": "failed",
            "saved": 0,
            "memory_ids": [],
            "error": str(exc),
        }

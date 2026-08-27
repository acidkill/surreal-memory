"""MCP handler mixin for memory edit, forget, consolidation, tool stats, and lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    Priority,
)
from surreal_memory.mcp.constants import MAX_CONTENT_LENGTH
from surreal_memory.mcp.tool_handler_utils import _get_brain_or_error, _require_brain_id
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.core.neuron import Neuron
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


async def _content_refreshed(storage: NeuralStorage, neuron: Neuron, new_content: str) -> Neuron:
    """Return *neuron* with its content replaced and the derived fields refreshed.

    ``content_hash`` is a pure function of the content, so it is recomputed
    unconditionally — keeping the old fingerprint would feed near-duplicate
    detection the SimHash of text that no longer exists.

    The embedding is recomputed only when the neuron already carries one
    (``metadata["_embedding"]``, surfaced by the storage read). ``update_neuron``
    writes whatever vector that key holds, so without a refresh a content edit
    actively re-saves the OLD vector against the NEW text: the memory stays
    retrievable by what it used to say, and ``reindex --missing-only`` cannot
    repair it because the field is never empty. Re-embedding goes through the
    same provider path and bounded wait the write path uses; if the provider is
    unavailable the edit still succeeds, but the stale vector is reported with a
    warning instead of being rewritten silently.
    """
    from dataclasses import replace as dc_replace

    from surreal_memory.extraction.structure_detector import detect_structure
    from surreal_memory.utils.simhash import simhash

    meta = dict(neuron.metadata)

    # _structure is derived from the content exactly like content_hash is, and
    # recall reads it back to answer with the memory's fields. Leaving it behind
    # meant an edited memory kept describing text that no longer exists.
    #
    # Removed, not merely overwritten: when the new content has no structure at
    # all, an overwrite-on-hit would preserve the previous fields — the worst
    # case, because recall would then surface fields present nowhere.
    structure = detect_structure(new_content)
    if structure.is_structured:
        meta["_structure"] = {
            "format": structure.format.value,
            "fields": [
                {"name": f.name, "value": f.value, "type": f.field_type} for f in structure.fields
            ],
            "confidence": structure.confidence,
        }
    else:
        meta.pop("_structure", None)

    updated = dc_replace(
        neuron, content=new_content, content_hash=simhash(new_content), metadata=meta
    )
    if "_embedding" not in meta:
        return updated
    try:
        import asyncio

        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine.encoder import _inline_embed_timeout
        from surreal_memory.engine.semantic_discovery import _create_provider

        brain = await storage.get_brain(storage.brain_id or "")
        provider = _create_provider(
            brain.config if brain else BrainConfig(), task_type="RETRIEVAL_DOCUMENT"
        )
        embed = provider.embed_batch([updated.embedding_text()])
        budget = _inline_embed_timeout()
        vectors = await (asyncio.wait_for(embed, timeout=budget) if budget else embed)
        meta["_embedding"] = list(vectors[0])
    except TimeoutError:
        logger.warning(
            "smem_edit changed the content of neuron %s but re-embedding exceeded "
            "its time budget — the stored vector still describes the old content; "
            "run `smem reindex --all` to repair it.",
            neuron.id,
        )
    except Exception:
        logger.warning(
            "smem_edit changed the content of neuron %s but could not recompute its "
            "embedding — the stored vector still describes the old content; "
            "run `smem reindex --all` to repair it.",
            neuron.id,
            exc_info=True,
        )
    return updated


class LifecycleHandler:
    """Mixin providing edit, forget, consolidate, tool_stats, and lifecycle handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _edit(self, args: dict[str, Any]) -> dict[str, Any]:
        """Edit an existing memory's type, content, priority, or tier."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        new_type = args.get("type")
        new_content = args.get("content")
        new_priority = args.get("priority")
        new_tier = args.get("tier")
        if new_tier is not None:
            new_tier = str(new_tier).lower().strip()

        if new_type is None and new_content is None and new_priority is None and new_tier is None:
            return {"error": "At least one of type, content, priority, or tier must be provided"}

        if new_type is not None:
            try:
                MemoryType(new_type)
            except ValueError:
                return {"error": f"Invalid memory type: {new_type}"}

        if new_tier is not None:
            try:
                MemoryTier(new_tier)
            except ValueError:
                return {"error": f"Invalid tier: {new_tier}. Must be hot, warm, or cold."}

        if new_content is not None and len(new_content) > MAX_CONTENT_LENGTH:
            return {
                "error": f"Content too long ({len(new_content)} chars). Max: {MAX_CONTENT_LENGTH}."
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for edit")
            return {"error": "No brain configured"}

        # Try as fiber_id first, then as neuron_id
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if typed_mem and fiber:
            # Edit via fiber path
            changes: list[str] = []

            # Update typed_memory (type, priority, tier)
            if new_type is not None or new_priority is not None or new_tier is not None:
                from dataclasses import replace as dc_replace

                updated_tm = typed_mem
                if new_type is not None:
                    updated_tm = dc_replace(updated_tm, memory_type=MemoryType(new_type))
                    changes.append(f"type: {typed_mem.memory_type.value} → {new_type}")
                    # Sync type into fiber.metadata to keep both stores consistent
                    updated_meta = {**fiber.metadata, "type": new_type}
                    fiber = dc_replace(fiber, metadata=updated_meta)
                    await storage.update_fiber(fiber)
                    # Enforce boundary invariant: boundaries are always HOT
                    if (
                        updated_tm.memory_type == MemoryType.BOUNDARY
                        and updated_tm.tier != MemoryTier.HOT
                    ):
                        old_tier = updated_tm.tier
                        updated_tm = updated_tm.with_tier(MemoryTier.HOT)
                        changes.append(f"tier: {old_tier} → hot (boundary auto-promote)")
                if new_priority is not None:
                    updated_tm = dc_replace(updated_tm, priority=Priority.from_int(new_priority))
                    changes.append(f"priority: {typed_mem.priority.value} → {new_priority}")
                if new_tier is not None:
                    old_tier = updated_tm.tier
                    updated_tm = updated_tm.with_tier(new_tier)
                    if updated_tm.tier != old_tier:
                        changes.append(f"tier: {old_tier} → {updated_tm.tier}")
                await storage.update_typed_memory(updated_tm)

            # Update anchor neuron content (and the fields derived from it)
            if new_content is not None:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    updated_neuron = await _content_refreshed(storage, anchor, new_content)
                    await storage.update_neuron(updated_neuron)
                    changes.append(f"content updated ({len(new_content)} chars)")

            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        # Try as direct neuron_id
        neuron = await storage.get_neuron(memory_id)
        if neuron:
            from dataclasses import replace as dc_replace

            changes = []
            if new_content is not None:
                neuron = await _content_refreshed(storage, neuron, new_content)
                changes.append(f"content updated ({len(new_content)} chars)")
            if new_type is not None:
                from surreal_memory.core.neuron import NeuronType

                try:
                    neuron = dc_replace(neuron, type=NeuronType(new_type))
                    changes.append(f"neuron type → {new_type}")
                except ValueError:
                    pass  # NeuronType doesn't map 1:1 to MemoryType
            await storage.update_neuron(neuron)
            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        return {"error": "Memory not found"}

    async def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        """Explicitly delete or close a specific memory."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        hard = args.get("hard", False)
        reason = args.get("reason", "")

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for forget")
            return {"error": "No brain configured"}

        # Look up the memory
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if not typed_mem and not fiber:
            # Try as neuron_id — find its fiber
            neuron = await storage.get_neuron(memory_id)
            if not neuron:
                return {"error": "Memory not found"}
            # For neuron-only delete in hard mode
            if hard:
                await storage.delete_neuron(memory_id)
                return {
                    "status": "hard_deleted",
                    "memory_id": memory_id,
                    "message": "Neuron permanently deleted",
                }
            return {
                "error": f"No typed memory found for neuron {memory_id}. Use hard=true for neuron deletion."
            }

        if hard:
            # Permanent deletion: fiber + typed_memory + neurons
            storage.disable_auto_save()
            try:
                # Delete typed memory
                await storage.delete_typed_memory(memory_id)

                # Delete fiber (CASCADE handles fiber_neurons junction)
                if fiber:
                    await storage.delete_fiber(memory_id)

                await storage.batch_save()
            finally:
                storage.enable_auto_save()

            logger.info("Hard-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "hard_deleted",
                "memory_id": memory_id,
                "message": "Memory permanently deleted with cascade cleanup",
            }
        else:
            # Soft delete: expire immediately
            from dataclasses import replace as dc_replace

            assert typed_mem is not None  # guaranteed by early return above
            expired_tm = dc_replace(typed_mem, expires_at=utcnow())
            await storage.update_typed_memory(expired_tm)

            logger.info("Soft-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "soft_deleted",
                "memory_id": memory_id,
                "message": "Memory marked as expired (will be cleaned up on next consolidation)",
            }

    async def _consolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run memory consolidation on the current brain."""
        from surreal_memory.engine.consolidation import (
            ConsolidationConfig,
            ConsolidationStrategy,
        )
        from surreal_memory.engine.consolidation_delta import run_with_delta

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for consolidate")
            return {"error": "No brain configured"}

        # Parse strategy
        strategy_str = args.get("strategy", "all")
        try:
            strategy = ConsolidationStrategy(strategy_str)
        except ValueError:
            valid = [s.value for s in ConsolidationStrategy]
            return {"error": f"Invalid strategy: {strategy_str}. Valid: {valid}"}

        strategies = [strategy]
        dry_run = bool(args.get("dry_run", False))

        # Build config with optional overrides (bounded to valid ranges)
        config_kwargs: dict[str, Any] = {}
        if "prune_weight_threshold" in args:
            val = args["prune_weight_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_weight_threshold"] = max(0.0, min(float(val), 1.0))
        if "merge_overlap_threshold" in args:
            val = args["merge_overlap_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["merge_overlap_threshold"] = max(0.0, min(float(val), 1.0))
        if "prune_min_inactive_days" in args:
            val = args["prune_min_inactive_days"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_min_inactive_days"] = max(0, int(val))

        config = ConsolidationConfig(**config_kwargs) if config_kwargs else None

        try:
            # Pass tier_config for auto-tier (Pro feature, runs post-consolidation)
            tier_config = self.config.tiers if self.config.is_pro() else None
            delta = await run_with_delta(
                storage,
                brain_id,
                strategies=strategies,
                dry_run=dry_run,
                config=config,
                tier_config=tier_config,
            )
        except Exception:
            logger.error("Consolidation failed", exc_info=True)
            return {"error": "Consolidation failed unexpectedly"}

        result = delta.to_dict()
        result["strategy"] = strategy_str
        result["dry_run"] = dry_run
        result["summary"] = delta.report.summary()
        # Machine-readable twin of the counters in ``summary`` whose bare value
        # is ambiguous. A client that wants to know whether "0 new links" meant
        # "already linked" or "every attempt failed" — or whether 0 drift
        # clusters meant "none exist" or "detection never ran" — should not have
        # to parse prose.
        #
        # Narrow on purpose, in both directions: the named fields are listed one
        # by one rather than taken from ``asdict(report)``, and ``extra`` is
        # filtered to the dedup diagnostics. Exporting ``extra`` wholesale would
        # silently widen this MCP contract every time an unrelated strategy grew
        # a new key.
        extra = {
            key: value
            for key, value in delta.report.extra.items()
            if key.startswith(("alias_", "dedup_", "merge_", "semantic_link_", "summaries_"))
        }
        result["report"] = {
            "duplicates_found": delta.report.duplicates_found,
            "new_alias_links": delta.report.new_alias_links,
            "alias_links_existing": delta.report.alias_links_existing,
            # Detected vs persisted, so the promise made in the comment above is
            # actually kept: a client can now tell "no clusters exist" from "writes
            # failed" without parsing prose.
            "drift_clusters_found": delta.report.drift_clusters_found,
            "drift_clusters_persisted": delta.report.drift_clusters_persisted,
            "extra": extra,
        }
        return result

    async def _tool_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get tool usage analytics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "summary")
        try:
            days = max(1, min(int(args.get("days", 30)), 365))
            limit = max(1, min(int(args.get("limit", 20)), 200))
        except (TypeError, ValueError):
            return {"error": "days and limit must be integers"}

        if action == "summary":
            result: dict[str, Any] = await storage.get_tool_stats(brain.id, days=days)
            return result
        elif action == "daily":
            daily = await storage.get_tool_stats_by_period(brain.id, days=days, limit=limit)
            return {"daily": daily, "days": days}
        else:
            return {"error": f"Unknown action: {action}"}

    async def _lifecycle(self, args: dict[str, Any]) -> dict[str, Any]:
        """Memory lifecycle management: status, recover, freeze, thaw."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "status")
        neuron_id: str | None = args.get("id") or args.get("neuron_id")

        if action == "status":
            try:
                distribution = await storage.get_lifecycle_distribution()
            except Exception:
                logger.error("smem_lifecycle status failed", exc_info=True)
                return {"error": "Failed to retrieve lifecycle distribution"}
            total = sum(distribution.values())
            return {
                "brain": brain.id,
                "distribution": distribution,
                "total_neurons": total,
            }

        if action in ("recover", "freeze", "thaw"):
            if not neuron_id:
                return {"error": f"action='{action}' requires 'id' (neuron_id)"}

            if action == "recover":
                # Find which fiber contains this neuron, then recover.
                from surreal_memory.engine.compression import CompressionEngine

                fibers = await storage.find_fibers(contains_neuron=neuron_id)
                if not fibers:
                    # Try decompress by fiber_id directly (caller may pass fiber_id as id)
                    engine = CompressionEngine(storage)
                    success = await engine.recover_fiber(neuron_id)
                    if success:
                        return {"recovered": True, "fiber_id": neuron_id}
                    return {
                        "recovered": False,
                        "reason": "No fiber found for neuron and direct recovery failed",
                    }

                fiber = fibers[0]
                engine = CompressionEngine(storage)
                success = await engine.recover_fiber(fiber.id)
                return {
                    "recovered": success,
                    "fiber_id": fiber.id,
                    "neuron_id": neuron_id,
                }

            elif action == "freeze":
                try:
                    await storage.update_neuron_frozen(neuron_id, frozen=True)
                except Exception:
                    logger.error("smem_lifecycle freeze failed for %s", neuron_id, exc_info=True)
                    return {"error": "Failed to freeze neuron"}
                return {"frozen": True, "neuron_id": neuron_id}

            elif action == "thaw":
                try:
                    await storage.update_neuron_frozen(neuron_id, frozen=False)
                except Exception:
                    logger.error("smem_lifecycle thaw failed for %s", neuron_id, exc_info=True)
                    return {"error": "Failed to thaw neuron"}
                return {"frozen": False, "neuron_id": neuron_id}

        if action == "backfill_supersession":
            limit = min(int(args.get("limit", 1000)), 5000)
            result = await self._lifecycle_backfill_supersession(storage, limit=limit)
            result["brain"] = brain.id
            return result

        return {
            "error": (
                f"Unknown action: {action}. "
                "Valid: status, recover, freeze, thaw, backfill_supersession"
            )
        }

    async def _lifecycle_backfill_supersession(
        self, storage: NeuralStorage, *, limit: int
    ) -> dict[str, Any]:
        """Retroactively stamp A-side supersession lineage for existing conflicts.

        Pre-U3 data has old anchors marked ``_superseded`` (C-side) with a CONTRADICTS
        edge from the newer anchor, but no A-side validity. Walk CONTRADICTS synapses,
        and for each genuinely-superseded old fact whose fiber is unambiguous, stamp
        valid_until/superseded_by + a SUPERSEDES synapse (idempotent, via
        ``engine.supersession``). Reports counts; ambiguous fibers are skipped.
        """
        from surreal_memory.core.synapse import SynapseType
        from surreal_memory.engine.supersession import (
            resolve_fibers_for_neurons,
            supersede_typed_memory,
        )

        synapses = await storage.get_synapses(type=SynapseType.CONTRADICTS)
        truncated = len(synapses) > limit
        synapses = synapses[:limit]

        scanned = 0
        backfilled = 0
        already_linked = 0
        skipped_ambiguous = 0
        for syn in synapses:
            new_anchor = syn.source_id
            old_anchor = syn.target_id
            old_neuron = await storage.get_neuron(old_anchor)
            if old_neuron is None or not old_neuron.metadata.get("_superseded"):
                continue
            scanned += 1
            fiber_map = await resolve_fibers_for_neurons(storage, [old_anchor, new_anchor])
            old_fiber_id = fiber_map.get(old_anchor)
            new_fiber_id = fiber_map.get(new_anchor)
            if not old_fiber_id or not new_fiber_id or old_fiber_id == new_fiber_id:
                skipped_ambiguous += 1
                continue
            outcome = await supersede_typed_memory(
                storage,
                old_fiber_id=old_fiber_id,
                new_fiber_id=new_fiber_id,
                new_anchor_id=new_anchor,
                old_anchor_id=old_anchor,
                reason="backfill:supersession",
            )
            if outcome.superseded:
                backfilled += 1
            else:
                already_linked += 1

        return {
            "action": "backfill_supersession",
            "scanned": scanned,
            "backfilled": backfilled,
            "already_linked": already_linked,
            "skipped_ambiguous": skipped_ambiguous,
            "truncated": truncated,
        }

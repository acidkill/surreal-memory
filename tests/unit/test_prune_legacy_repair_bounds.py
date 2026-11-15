"""Legacy repair must bound work by fibers, rather than only missing edge IDs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.prune import SurrealDBPruneMixin

_OWNER = "synthetic"


class RepairDB(SurrealDBPruneMixin):
    def __init__(self, count: int = 17) -> None:
        self.fibers = {f"fiber-{i}": {"edge-a", "edge-b", "keep"} for i in range(count)}
        self.present: set[str] = set()
        self.mutation_counts: list[int] = []
        self.responses: list[list[dict[str, Any]]] = []
        self.journal_ids: list[str] = []
        self.before_raw = None

    def _get_brain_id(self):
        return "synthetic"

    def _ensure_conn(self):
        return self

    async def _ensure_conn_on_current_loop(self):
        pass

    async def _record_changes_bulk(self, entity_type, operation, entities):
        assert (entity_type, operation) == ("fiber", "update")
        self.journal_ids.extend(entity.id for entity in entities)

    async def query_raw(self, sql, params):
        if self.before_raw:
            await self.before_raw()
        bounded = "-- bounded legacy prune repair" in sql
        if bounded:
            absent = {item["public_id"] for item in params["candidates"]} - self.present
            matches = [fid for fid, refs in self.fibers.items() if refs & absent]
            selected = matches[:8]
            for fid in selected:
                self.fibers[fid] -= absent
            result = [
                {
                    "updated_ids": ["fiber:" + fid.replace("-", "_") for fid in selected],
                    "has_more": len(matches) > 8,
                }
            ]
            self.mutation_counts.append(len(selected))
        else:
            # Prior implementation rewrites each matching fiber once per edge,
            # retaining every whole-fiber snapshot in the same RPC response.
            result = []
            for key, sid in params.items():
                if not key.startswith("sid_") or sid in self.present:
                    continue
                for fid, refs in self.fibers.items():
                    if sid in refs:
                        refs.discard(sid)
                        result.append(
                            {
                                "id": "fiber:" + fid.replace("-", "_"),
                                "neuron_ids": ["anchor"],
                                "synapse_ids": sorted(refs),
                                "anchor_neuron_id": "anchor",
                                "metadata": {"synthetic_large_field": "x" * 1024},
                            }
                        )
            self.mutation_counts.append(len(result))
        self.responses.append(deepcopy(result))
        return {"result": [{"status": "OK", "result": result}]}


async def test_repair_bounds_fibers_and_avoids_repeated_full_payloads():
    db = RepairDB()
    result = await db.repair_prune_synapse_refs_if_absent(
        ["edge-a", "edge-b"],
        lease_owner_token=_OWNER,
    )
    assert max(db.mutation_counts) <= 8
    assert result == 17
    assert len(db.journal_ids) == len(set(db.journal_ids)) == 17
    assert all(refs == {"keep"} for refs in db.fibers.values())
    assert all(set(row) == {"updated_ids", "has_more"} for page in db.responses for row in page)


async def test_budget_pause_between_pages_replays_remaining_refs_without_duplicates():
    from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

    db = RepairDB()
    checks = 0

    async def budget():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ConsolidationPausedError("synthetic budget pause")

    with pytest.raises(ConsolidationPausedError):
        await db.repair_prune_synapse_refs_if_absent(
            ["edge-a", "edge-b"], lease_owner_token=_OWNER, budget_check=budget
        )
    assert db.mutation_counts == [8]
    assert sum(bool(refs & {"edge-a", "edge-b"}) for refs in db.fibers.values()) == 9
    assert (
        await db.repair_prune_synapse_refs_if_absent(["edge-a", "edge-b"], lease_owner_token=_OWNER)
        == 9
    )
    assert len(db.journal_ids) == len(set(db.journal_ids)) == 17
    assert all(refs == {"keep"} for refs in db.fibers.values())


async def test_present_same_id_edge_protects_its_refs_while_other_missing_ref_is_removed():
    db = RepairDB(1)
    db.present = {"edge-a"}
    assert (
        await db.repair_prune_synapse_refs_if_absent(["edge-a", "edge-b"], lease_owner_token=_OWNER)
        == 1
    )
    assert db.fibers["fiber-0"] == {"edge-a", "keep"}


async def test_cancel_after_committed_page_rechecks_current_state_on_replay():
    import asyncio

    db = RepairDB()
    original = db.query_raw

    async def lost_reply(sql, params):
        await original(sql, params)
        db.query_raw = original
        raise asyncio.CancelledError

    db.query_raw = lost_reply
    with pytest.raises(asyncio.CancelledError):
        await db.repair_prune_synapse_refs_if_absent(["edge-a", "edge-b"], lease_owner_token=_OWNER)
    assert (
        await db.repair_prune_synapse_refs_if_absent(["edge-a", "edge-b"], lease_owner_token=_OWNER)
        == 9
    )
    assert all(refs == {"keep"} for refs in db.fibers.values())
    assert max(db.mutation_counts) <= 8


async def test_fiber_update_journal_still_has_null_payload():
    from surreal_memory.storage.surrealdb.store import _change_payload

    db = RepairDB(1)

    async def journal(entity_type, operation, entities):
        assert [entity.id for entity in entities] == ["fiber-0"]
        assert all(_change_payload(entity) is None for entity in entities)

    db._record_changes_bulk = journal
    await db.repair_prune_synapse_refs_if_absent(["edge-a"], lease_owner_token=_OWNER)

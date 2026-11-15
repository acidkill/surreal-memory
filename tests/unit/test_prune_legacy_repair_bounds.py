"""Offline raw-page adapter contracts; real SQL plans are checked in DB CI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.prune import SurrealDBPruneMixin

_OWNER = "synthetic"


@dataclass
class Record:
    table_name: str
    id: Any

    def __str__(self):
        value = f"⟨{self.id}⟩" if isinstance(self.id, str) and self.id.isdigit() else self.id
        return f"{self.table_name}:{value}"


class RepairDB(SurrealDBPruneMixin):
    def __init__(self, count: int = 17) -> None:
        self.fibers: dict[Any, set[str]] = {
            f"fiber-{i}": {"edge-a", "edge-b", "keep"} for i in range(count)
        }
        self.foreign: set[Any] = set()
        self.present: set[str] = set()
        self.mutation_counts: list[int] = []
        self.responses: list[list[dict[str, Any]]] = []
        self.journal_ids: list[str] = []
        self.cursors: list[Any] = []
        self.groups: list[list[str]] = []
        self.before_raw = None
        self.page_override = None

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
        self.cursors.append(params["cursor"])
        self.groups.append([item["public_id"] for item in params["candidates"]])
        records = [
            (fid, Record("fiber", fid.replace("-", "_") if isinstance(fid, str) else fid))
            for fid in self.fibers
        ]
        records.sort(key=lambda row: (0 if type(row[1].id) is int else 1, row[1].id))
        start = 0
        if params["has_cursor"]:
            assert "type::record('fiber', $cursor>..)" in sql
            start = next(
                i + 1
                for i, (_, record) in enumerate(records)
                if type(record.id) is type(params["cursor"]) and record.id == params["cursor"]
            )
        raw = records[start : start + 9]
        processed = raw[:8]
        absent = {item["public_id"] for item in params["candidates"]} - self.present
        selected = [
            (fid, record)
            for fid, record in processed
            if fid not in self.foreign and self.fibers[fid] & absent
        ]
        for fid, _ in selected:
            self.fibers[fid] -= absent
        page = {
            "updated_ids": [record for _, record in selected],
            "has_more": len(raw) > 8,
            "next_cursor": processed[-1][1] if processed else None,
            "scanned": len(raw),
        }
        if self.page_override:
            page = self.page_override(page)
        self.mutation_counts.append(len(selected))
        result = [page]
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
    assert all(
        set(row) == {"updated_ids", "has_more", "next_cursor", "scanned"}
        for page in db.responses
        for row in page
    )


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


async def test_zero_update_foreign_prefix_advances_without_skipping_disjoint_refs():
    db = RepairDB(2)
    for i in range(8):
        db.fibers[f"000{i}"] = {"edge-a", "edge-b", "foreign"}
        db.foreign.add(f"000{i}")
    db.fibers["fiber-0"] = {"edge-a", "keep"}
    db.fibers["fiber-1"] = {"edge-b", "keep"}
    assert (
        await db.repair_prune_synapse_refs_if_absent(["edge-a", "edge-b"], lease_owner_token=_OWNER)
        == 2
    )
    assert db.mutation_counts == [0, 2]
    assert all(db.fibers[f"000{i}"] == {"edge-a", "edge-b", "foreign"} for i in range(8))
    assert all(db.fibers[f"fiber-{i}"] == {"keep"} for i in range(2))


async def test_cursor_keeps_integer_and_numeric_string_identity_distinct():
    db = RepairDB(0)
    for i in range(8):
        db.fibers[i] = {"edge-a", "foreign"}
        db.foreign.add(i)
    db.fibers["7"] = {"edge-a", "keep"}
    assert await db.repair_prune_synapse_refs_if_absent(["edge-a"], lease_owner_token=_OWNER) == 1
    assert db.cursors == [None, 7]
    assert type(db.cursors[1]) is int
    assert db.fibers["7"] == {"keep"}
    assert db.fibers[7] == {"edge-a", "foreign"}


async def test_empty_page_terminates_and_group_cursor_resets():
    db = RepairDB(0)
    assert (
        await db.repair_prune_synapse_refs_if_absent(
            [f"edge-{i}" for i in range(33)], lease_owner_token=_OWNER
        )
        == 0
    )
    assert db.cursors == [None, None]
    assert [len(group) for group in db.groups] == [32, 1]


@pytest.mark.parametrize(
    "invalid",
    ["overread", "overupdate", "wrong_table", "missing_cursor", "more", "empty_cursor", "repeat"],
)
async def test_invalid_raw_page_receipts_fail_closed(invalid):
    db = RepairDB()

    def invalid_page(page):
        if invalid == "overread":
            page["scanned"] = 10
        elif invalid == "overupdate":
            page["updated_ids"] *= 2
        elif invalid == "wrong_table":
            page["next_cursor"] = Record("neuron", "wrong")
        elif invalid == "missing_cursor":
            page["next_cursor"] = None
        elif invalid == "more":
            page["has_more"] = False
        elif invalid == "empty_cursor":
            page.update(scanned=0, updated_ids=[], has_more=False)
        elif invalid == "repeat" and len(db.cursors) > 1:
            page["next_cursor"] = Record("fiber", db.cursors[-1])
        return page

    db.page_override = invalid_page
    with pytest.raises(RuntimeError, match="prune repair"):
        await db.repair_prune_synapse_refs_if_absent(["edge-a"], lease_owner_token=_OWNER)


@pytest.mark.parametrize("count,counts", [(0, [0]), (8, [8]), (16, [8, 8])])
async def test_empty_and_exact_multiple_terminal_pages(count, counts):
    db = RepairDB(count)
    assert (
        await db.repair_prune_synapse_refs_if_absent(["edge-a"], lease_owner_token=_OWNER) == count
    )
    assert db.mutation_counts == counts
    assert db.responses[-1][0]["has_more"] is False
    assert db.responses[-1][0]["scanned"] == (8 if count else 0)

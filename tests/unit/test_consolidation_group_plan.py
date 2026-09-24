from __future__ import annotations

import re
from typing import Any

import pytest

from surreal_memory.engine.consolidation_group_plan import (
    ConsolidationGroupPlanError,
    SurrealDBConsolidationGroupPlan,
)


class _PlanStorage:
    """Small query-contract fake for the immutable planner's SurrealQL operations."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.census: dict[str, dict[str, Any]] = {}
        self.page_limits: list[int] = []
        self.transaction_row_counts: list[int] = []
        self.records_by_kind: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.candidates_by_id: dict[tuple[str, str], dict[str, Any]] = {}
        self.members_by_root: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def _store_record(self, record_id: str, row: dict[str, Any]) -> None:
        stored = dict(row)
        self.records[record_id] = stored
        plan_id = str(stored["plan_id"])
        kind = str(stored["kind"])
        self.records_by_kind.setdefault((plan_id, kind), []).append(stored)
        if kind == "candidate":
            self.candidates_by_id[(plan_id, str(stored["candidate_id"]))] = stored
        elif kind == "member":
            self.members_by_root.setdefault((plan_id, str(stored["root_id"])), []).append(stored)

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        if sql.startswith("CREATE type::record('consolidation_fiber_census'"):
            stage_id = str(params["stage_id"])
            if stage_id in self.census:
                raise ValueError("record already exists")
            self.census[stage_id] = dict(params["row"])
            return []
        if sql.startswith("SELECT * FROM type::record('consolidation_fiber_census'"):
            row = self.census.get(str(params["stage_id"]))
            return [{**row, "id": "consolidation_fiber_census:test"}] if row else []
        if sql.startswith("SELECT * FROM consolidation_fiber_census "):
            self.page_limits.append(int(params.get("limit", 1)))
            found = sorted(
                (
                    row
                    for row in self.census.values()
                    if row.get("run_id") == params.get("run_id")
                    and row.get("strategy") == params.get("strategy")
                    and row.get("filter_fingerprint") == params.get("filter_fingerprint")
                    and int(row.get("page_index", -1)) > int(params.get("after_page", -1))
                ),
                key=lambda row: int(row["page_index"]),
            )
            return found[: int(params.get("limit", 1))]
        if sql.startswith("BEGIN TRANSACTION;"):
            creates = re.findall(
                r"CREATE type::record\('consolidation_group_plan', "
                r"\$record_id_(\d+)\) CONTENT \$row_(\d+)",
                sql,
            )
            record_ids = [str(params[f"record_id_{record_index}"]) for record_index, _ in creates]
            if len(set(record_ids)) != len(record_ids) or any(
                record_id in self.records for record_id in record_ids
            ):
                raise ValueError("record already exists")
            self.transaction_row_counts.append(len(creates))
            for record_index, row_index in creates:
                self._store_record(
                    str(params[f"record_id_{record_index}"]), dict(params[f"row_{row_index}"])
                )
            return []
        if sql.startswith("CREATE type::record('consolidation_group_plan'"):
            record_id = str(params["record_id"])
            if record_id in self.records:
                raise ValueError("record already exists")
            self._store_record(record_id, dict(params["row"]))
            return []
        if sql.startswith("SELECT * FROM type::record('consolidation_group_plan'"):
            row = self.records.get(str(params["record_id"]))
            return [{**row, "id": "consolidation_group_plan:test"}] if row else []

        self.page_limits.append(int(params.get("limit", 1)))
        if "AND kind = 'candidate'" in sql:
            expected_kind = "candidate"
        elif "AND kind = 'posting'" in sql:
            expected_kind = "posting"
        elif "AND kind = 'member'" in sql:
            expected_kind = "member"
        elif "AND kind = 'group'" in sql:
            expected_kind = "group"
        elif "AND kind = $kind" in sql:
            expected_kind = str(params["kind"])
        else:
            expected_kind = None
        plan_id = str(params.get("plan_id", ""))
        if expected_kind == "candidate" and "candidate_id" in params:
            candidate = self.candidates_by_id.get((plan_id, str(params["candidate_id"])))
            rows = [{**candidate, "id": "consolidation_group_plan:candidate"}] if candidate else []
        elif expected_kind == "member" and "root_id" in params:
            rows = [
                {**row, "id": f"consolidation_group_plan:member:{index}"}
                for index, row in enumerate(
                    self.members_by_root.get((plan_id, str(params["root_id"])), [])
                )
            ]
        else:
            source_rows = (
                self.records_by_kind.get((plan_id, expected_kind), [])
                if expected_kind is not None
                else [row for row in self.records.values() if row.get("plan_id") == plan_id]
            )
            rows = [
                {**row, "id": f"consolidation_group_plan:{index}"}
                for index, row in enumerate(source_rows)
            ]
        if "AND kind = 'candidate' AND candidate_id = $candidate_id" in sql:
            return [
                row
                for row in rows
                if row.get("kind") == "candidate"
                and row.get("candidate_id") == params["candidate_id"]
            ][:1]
        if "AND kind = $kind AND item_key = $item_key" in sql:
            return [
                row
                for row in rows
                if row.get("kind") == params["kind"] and row.get("item_key") == params["item_key"]
            ][:1]
        if "AND kind = 'candidate' AND candidate_id > $after" in sql:
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "candidate"
                    and str(row.get("candidate_id", "")) > str(params["after"])
                ),
                key=lambda row: str(row["candidate_id"]),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'posting' AND feature >= $start_feature" in sql:
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "posting"
                    and str(row["feature"]) >= str(params["start_feature"])
                ),
                key=lambda row: (str(row["feature"]), str(row["candidate_id"])),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'posting' AND feature > $after_feature ORDER BY" in sql:
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "posting"
                    and str(row["feature"]) > str(params["after_feature"])
                ),
                key=lambda row: (str(row["feature"]), str(row["candidate_id"])),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'posting' ORDER BY feature ASC" in sql:
            found = sorted(
                (row for row in rows if row.get("kind") == "posting"),
                key=lambda row: (str(row["feature"]), str(row["candidate_id"])),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'posting' AND (feature > $after_feature" in sql:
            cursor = (str(params["after_feature"]), str(params["after_candidate"]))
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "posting"
                    and (str(row["feature"]), str(row["candidate_id"])) > cursor
                ),
                key=lambda row: (str(row["feature"]), str(row["candidate_id"])),
            )
            return found[: int(params["limit"])]
        if "AND kind = $kind AND candidate_id = $candidate_id" in sql:
            found = [
                row
                for row in rows
                if row.get("kind") == params["kind"]
                and row.get("candidate_id") == params["candidate_id"]
                and int(row.get("sequence", -1)) < int(params["before_sequence"])
            ]
            found.sort(key=lambda row: int(row["sequence"]), reverse=True)
            return found[:1]
        if "AND kind = 'group' AND root_id > $after" in sql:
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "group" and str(row["root_id"]) > str(params["after"])
                ),
                key=lambda row: str(row["root_id"]),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'member' AND root_id > $after_root" in sql:
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "member"
                    and str(row["root_id"]) > str(params["after_root"])
                ),
                key=lambda row: (str(row["root_id"]), str(row["candidate_id"])),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'member' AND (root_id > $after_root" in sql:
            cursor = (str(params["after_root"]), str(params["after_candidate"]))
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "member"
                    and (str(row["root_id"]), str(row["candidate_id"])) > cursor
                ),
                key=lambda row: (str(row["root_id"]), str(row["candidate_id"])),
            )
            return found[: int(params["limit"])]
        if "AND kind = 'member' AND root_id = $root_id" in sql:
            found = sorted(
                (
                    row
                    for row in rows
                    if row.get("kind") == "member"
                    and row.get("root_id") == params["root_id"]
                    and str(row.get("candidate_id", "")) > str(params["after"])
                ),
                key=lambda row: str(row["candidate_id"]),
            )
            return found[: int(params["limit"])]
        raise AssertionError(f"unexpected planner query: {sql}")


def _plan(storage: _PlanStorage) -> SurrealDBConsolidationGroupPlan:
    return SurrealDBConsolidationGroupPlan(
        storage,
        brain_id="brain",
        run_id="run",
        strategy="merge",
        fingerprint="snapshot-v1",
    )


@pytest.mark.asyncio
async def test_plan_paginates_ten_thousand_candidates_and_skips_oversized_posting() -> None:
    storage = _PlanStorage()
    plan = _plan(storage)
    count = 10_005
    for index in range(count):
        candidate_id = f"fiber-{index:05d}"
        await plan.put_candidate(candidate_id, {"signature": candidate_id}, {"common"})

    candidates = 0
    async for candidate_id, payload in plan.iter_candidates():
        assert payload["signature"] == candidate_id
        candidates += 1
    assert candidates == count

    postings = [posting async for posting in plan.iter_postings(posting_limit=100)]
    assert postings == [("common", [])]  # Oversized postings are skipped after 101 rows.
    assert max(storage.page_limits) <= 500


@pytest.mark.asyncio
async def test_posting_keysets_resume_after_completed_or_partial_feature() -> None:
    storage = _PlanStorage()
    plan = _plan(storage)
    await plan.put_candidates(
        [
            (candidate_id, {}, {"alpha"} if candidate_id < "d" else {"beta"})
            for candidate_id in ("a", "b", "c", "d", "e")
        ]
    )

    first_scan = [posting async for posting in plan.iter_postings(posting_limit=2)]
    assert first_scan == [("alpha", []), ("beta", ["d", "e"])]
    assert [
        posting async for posting in plan.iter_postings(posting_limit=2, after_feature="alpha")
    ] == [("beta", ["d", "e"])]
    assert [
        posting async for posting in plan.iter_postings(posting_limit=2, start_feature="beta")
    ] == [("beta", ["d", "e"])]


@pytest.mark.asyncio
async def test_union_events_replay_exactly_after_restart_and_stream_members() -> None:
    storage = _PlanStorage()
    first = _plan(storage)
    for candidate_id in ("a", "b", "c", "d"):
        await first.put_candidate(candidate_id, {"signature": candidate_id}, set())

    assert await first.union("a", "b", 0)
    assert await first.union("c", "d", 1)
    assert await first.union("a", "c", 2)

    resumed = _plan(storage)
    # Replaying a committed union sequence is an exact immutable retry.
    assert await resumed.union("a", "b", 0)
    assert await resumed.union("c", "d", 1)
    assert await resumed.union("a", "c", 2)
    assert await resumed.find("d", 3) == await resumed.find("a", 3)

    async for candidate_id, _ in resumed.iter_candidates():
        await resumed.add_member(await resumed.find(candidate_id, 3), candidate_id)
    roots = [root async for root in resumed.iter_groups()]
    assert len(roots) == 1
    members = [candidate_id async for candidate_id, _ in resumed.iter_members(roots[0])]
    assert members == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_group_roots_keyset_pages_more_than_five_hundred_without_skips() -> None:
    storage = _PlanStorage()
    plan = _plan(storage)
    roots = [f"root-{index:04d}" for index in range(750)]
    await plan.add_members([(root, f"candidate-{root}", None) for root in roots])

    assert [root async for root in plan.iter_groups()] == roots


@pytest.mark.asyncio
async def test_candidate_postings_use_bounded_write_batches_for_wide_feature_pages() -> None:
    storage = _PlanStorage()
    plan = _plan(storage)
    features = frozenset(f"tag-{index:04d}" for index in range(1_000))
    await plan.put_candidates(
        [(f"candidate-{index}", {"signature": str(index)}, features) for index in range(5)]
    )

    assert max(storage.transaction_row_counts) <= 200
    assert len(storage.records) == 5 * (1 + len(features))


@pytest.mark.asyncio
async def test_immutable_plan_rejects_changed_candidate_payload() -> None:
    storage = _PlanStorage()
    first = _plan(storage)
    await first.put_candidate("fiber-a", {"signature": "original"}, {"tag-a"})

    resumed = _plan(storage)
    with pytest.raises(ConsolidationGroupPlanError, match="conflicts"):
        await resumed.put_candidate("fiber-a", {"signature": "changed"}, {"tag-a"})

"""Immutable, paged graph plans for resumable consolidation grouping.

The planner stores candidate projections, inverted-index postings, union-find
events, and component memberships as immutable SurrealDB rows.  Checkpoints can
therefore keep scalar cursors instead of serializing a brain-sized candidate or
parent array.  Rows are content-verified on duplicate creates so expired lease
owners can only leave harmless, identical plan data behind.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from surreal_memory.storage.errors import is_duplicate_key_error

_PAGE_SIZE = 500
_WRITE_BATCH_SIZE = 200
_MAX_UNION_DEPTH = 128


class ConsolidationGroupPlanError(RuntimeError):
    """A durable group-plan row is missing, malformed, or conflicts on replay."""


class SurrealDBConsolidationGroupPlan:
    """Run-scoped immutable candidate graph backed by ``consolidation_group_plan``."""

    def __init__(
        self,
        storage: Any,
        *,
        brain_id: str,
        run_id: str,
        strategy: str,
        fingerprint: str,
    ) -> None:
        if not all((brain_id, run_id, strategy, fingerprint)):
            raise ValueError("group plan identity fields must be non-empty")
        if not callable(getattr(storage, "_query", None)):
            raise TypeError("durable group plans require storage._query")
        self._storage = storage
        identity = "\0".join((brain_id, run_id, strategy, fingerprint))
        self.plan_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self._brain_id = brain_id
        self._run_id = run_id
        self._strategy = strategy
        self._fingerprint = fingerprint

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _record_id(plan_id: str, kind: str, item_key: str) -> str:
        return hashlib.sha256(f"{plan_id}\0{kind}\0{item_key}".encode()).hexdigest()

    @staticmethod
    def _compound_key(*parts: str | int) -> str:
        return hashlib.sha256(
            json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _row(
        self, kind: str, item_key: str, fields: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        row = {
            "plan_id": self.plan_id,
            "brain_id": self._brain_id,
            "run_id": self._run_id,
            "strategy": self._strategy,
            "fingerprint": self._fingerprint,
            "kind": kind,
            "item_key": item_key,
            **dict(fields),
        }
        record_id = self._record_id(self.plan_id, kind, item_key)
        return record_id, row

    async def _create_many(self, rows_to_create: list[tuple[str, dict[str, Any]]]) -> None:
        for start in range(0, len(rows_to_create), _WRITE_BATCH_SIZE):
            batch = rows_to_create[start : start + _WRITE_BATCH_SIZE]
            remaining = batch
            for _ in range(3):
                if not remaining:
                    break
                params: dict[str, Any] = {}
                statements = ["BEGIN TRANSACTION"]
                for index, (record_id, row) in enumerate(remaining):
                    params[f"record_id_{index}"] = record_id
                    params[f"row_{index}"] = row
                    statements.append(
                        "CREATE type::record('consolidation_group_plan', "
                        f"$record_id_{index}) CONTENT $row_{index}"
                    )
                statements.append("COMMIT TRANSACTION")
                try:
                    await self._storage._query("; ".join(statements), **params)
                    remaining = []
                except Exception as exc:
                    if not is_duplicate_key_error(exc):
                        raise
                    missing: list[tuple[str, dict[str, Any]]] = []
                    for record_id, row in remaining:
                        found = await self._storage._query(
                            "SELECT * FROM type::record('consolidation_group_plan', $record_id)",
                            record_id=record_id,
                        )
                        existing = dict(found[0]) if len(found) == 1 else None
                        if existing is not None:
                            existing.pop("id", None)
                            if self._json(existing) != self._json(row):
                                raise ConsolidationGroupPlanError(
                                    f"immutable {row['kind']} row conflicts with this consolidation plan"
                                ) from exc
                        else:
                            missing.append((record_id, row))
                    remaining = missing
            if remaining:
                raise ConsolidationGroupPlanError(
                    "immutable plan rows remained absent after duplicate-key retries"
                )

    async def _create_immutable(self, kind: str, item_key: str, fields: Mapping[str, Any]) -> None:
        await self._create_many([self._row(kind, item_key, fields)])

    async def put_item(self, kind: str, item_key: str, fields: Mapping[str, Any]) -> None:
        """Persist an immutable strategy-specific plan marker."""
        if (
            not kind
            or not item_key
            or kind in {"candidate", "posting", "parent", "rank", "member", "group"}
        ):
            raise ValueError("plan marker kind and key must be valid and strategy-specific")
        await self._create_immutable(kind, item_key, fields)

    async def put_items(self, items: list[tuple[str, str, Mapping[str, Any]]]) -> None:
        """Persist a bounded page of immutable strategy-specific markers."""
        rows: list[tuple[str, dict[str, Any]]] = []
        for kind, item_key, fields in items:
            if (
                not kind
                or not item_key
                or kind in {"candidate", "posting", "parent", "rank", "member", "group"}
            ):
                raise ValueError("plan marker kind and key must be valid and strategy-specific")
            rows.append(self._row(kind, item_key, fields))
            if len(rows) >= _WRITE_BATCH_SIZE:
                await self._create_many(rows)
                rows = []
        if rows:
            await self._create_many(rows)

    async def has_item(self, kind: str, item_key: str) -> bool:
        """Check for one immutable plan marker through its indexed identity."""
        rows = await self._storage._query(
            "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
            "AND kind = $kind AND item_key = $item_key LIMIT 1",
            plan_id=self.plan_id,
            kind=kind,
            item_key=item_key,
        )
        return bool(rows)

    async def get_item(self, kind: str, item_key: str) -> dict[str, Any]:
        """Load one immutable strategy-specific marker by its identity."""
        rows = await self._storage._query(
            "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
            "AND kind = $kind AND item_key = $item_key LIMIT 1",
            plan_id=self.plan_id,
            kind=kind,
            item_key=item_key,
        )
        if len(rows) != 1:
            raise ConsolidationGroupPlanError(f"plan item {kind}:{item_key!r} is missing")
        row = dict(rows[0])
        row.pop("id", None)
        return row

    async def iter_items(
        self, kind: str, *, after: str = ""
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Page strategy-specific markers by their immutable item key."""
        if not kind or kind in {"candidate", "posting", "parent", "rank", "member", "group"}:
            raise ValueError("kind must be a strategy-specific plan marker")
        cursor = after
        while True:
            rows = await self._storage._query(
                "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                "AND kind = $kind AND item_key > $after "
                "ORDER BY item_key ASC LIMIT $limit",
                plan_id=self.plan_id,
                kind=kind,
                after=cursor,
                limit=_PAGE_SIZE,
            )
            if not rows:
                return
            for raw in rows:
                row = dict(raw)
                item_key = str(row.get("item_key", ""))
                if not item_key or item_key <= cursor:
                    raise ConsolidationGroupPlanError("marker keyset page is malformed")
                row.pop("id", None)
                cursor = item_key
                yield item_key, row
            if len(rows) < _PAGE_SIZE:
                return

    async def put_candidate(
        self, candidate_id: str, payload: Mapping[str, Any], features: set[str] | frozenset[str]
    ) -> None:
        """Persist one candidate projection and its feature postings idempotently."""
        await self.put_candidates([(candidate_id, payload, features)])

    async def put_candidates(
        self,
        candidates: list[tuple[str, Mapping[str, Any], set[str] | frozenset[str]]],
    ) -> None:
        """Batch-create one source page while keeping every query payload bounded."""
        rows: list[tuple[str, dict[str, Any]]] = []
        for candidate_id, payload, features in candidates:
            if not candidate_id:
                raise ValueError("candidate_id must be non-empty")
            rows.append(
                self._row(
                    "candidate",
                    candidate_id,
                    {"candidate_id": candidate_id, "payload": dict(payload)},
                )
            )
            if len(rows) >= _WRITE_BATCH_SIZE:
                await self._create_many(rows)
                rows = []
            for feature in features:
                posting_key = hashlib.sha256(f"{feature}\0{candidate_id}".encode()).hexdigest()
                rows.append(
                    self._row(
                        "posting",
                        posting_key,
                        {"feature": feature, "candidate_id": candidate_id},
                    )
                )
                if len(rows) >= _WRITE_BATCH_SIZE:
                    await self._create_many(rows)
                    rows = []
        if rows:
            await self._create_many(rows)

    async def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        rows = await self._storage._query(
            "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
            "AND kind = 'candidate' AND candidate_id = $candidate_id LIMIT 1",
            plan_id=self.plan_id,
            candidate_id=candidate_id,
        )
        if len(rows) != 1 or not isinstance(rows[0].get("payload"), dict):
            raise ConsolidationGroupPlanError(f"candidate {candidate_id!r} is missing")
        return dict(rows[0]["payload"])

    async def iter_candidates(
        self, *, after: str = ""
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        cursor = after
        while True:
            rows = await self._storage._query(
                "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                "AND kind = 'candidate' AND candidate_id > $after "
                "ORDER BY candidate_id ASC LIMIT $limit",
                plan_id=self.plan_id,
                after=cursor,
                limit=_PAGE_SIZE,
            )
            if not rows:
                return
            for row in rows:
                candidate_id = str(row.get("candidate_id", ""))
                payload = row.get("payload")
                if not candidate_id or not isinstance(payload, dict) or candidate_id <= cursor:
                    raise ConsolidationGroupPlanError("candidate keyset page is malformed")
                cursor = candidate_id
                yield candidate_id, dict(payload)
            if len(rows) < _PAGE_SIZE:
                return

    async def iter_postings(
        self,
        *,
        posting_limit: int = 100,
        after_feature: str = "",
        start_feature: str | None = None,
    ) -> AsyncIterator[tuple[str, list[str]]]:
        """Yield bounded postings, using feature keysets to resume a partial scan."""
        if posting_limit < 1:
            raise ValueError("posting_limit must be positive")
        if start_feature is not None and after_feature:
            raise ValueError("start_feature and after_feature are mutually exclusive")
        cursor_feature = start_feature if start_feature is not None else after_feature
        cursor_candidate = ""
        active_feature: str | None = None
        members: list[str] = []
        first_page = True
        while True:
            if first_page:
                if start_feature is not None:
                    sql = (
                        "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                        "AND kind = 'posting' AND feature >= $start_feature "
                        "ORDER BY feature ASC, candidate_id ASC LIMIT $limit"
                    )
                    params: dict[str, Any] = {"start_feature": start_feature}
                elif after_feature:
                    sql = (
                        "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                        "AND kind = 'posting' AND feature > $after_feature "
                        "ORDER BY feature ASC, candidate_id ASC LIMIT $limit"
                    )
                    params = {"after_feature": after_feature}
                else:
                    sql = (
                        "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                        "AND kind = 'posting' ORDER BY feature ASC, candidate_id ASC LIMIT $limit"
                    )
                    params = {}
                first_page = False
            else:
                sql = (
                    "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                    "AND kind = 'posting' AND (feature > $after_feature OR "
                    "(feature = $after_feature AND candidate_id > $after_candidate)) "
                    "ORDER BY feature ASC, candidate_id ASC LIMIT $limit"
                )
                params = {
                    "after_feature": cursor_feature,
                    "after_candidate": cursor_candidate,
                }
            rows = await self._storage._query(sql, plan_id=self.plan_id, limit=_PAGE_SIZE, **params)
            if not rows:
                break
            skip_rest_of_feature: str | None = None
            for row in rows:
                feature = str(row.get("feature", ""))
                candidate_id = str(row.get("candidate_id", ""))
                if (
                    not feature
                    or not candidate_id
                    or (feature, candidate_id)
                    <= (
                        cursor_feature,
                        cursor_candidate,
                    )
                ):
                    raise ConsolidationGroupPlanError("posting keyset page is malformed")
                if active_feature is not None and feature != active_feature:
                    if members:
                        yield active_feature, members
                    members = []
                if feature != active_feature:
                    active_feature = feature
                members.append(candidate_id)
                cursor_feature, cursor_candidate = feature, candidate_id
                if len(members) > posting_limit:
                    yield feature, []
                    active_feature = None
                    members = []
                    cursor_feature = feature
                    cursor_candidate = ""
                    skip_rest_of_feature = feature
                    break
            if skip_rest_of_feature is not None:
                first_page = True
                after_feature = skip_rest_of_feature
                start_feature = None
                continue
            if len(rows) < _PAGE_SIZE:
                break
        if active_feature is not None and members:
            yield active_feature, members

    async def iter_posting_candidates(
        self, feature: str, *, after_candidate: str = ""
    ) -> AsyncIterator[str]:
        """Page candidate IDs for a single feature without retaining its posting set."""
        if not feature:
            raise ValueError("feature must be non-empty")
        cursor = after_candidate
        while True:
            rows = await self._storage._query(
                "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                "AND kind = 'posting' AND feature = $feature AND candidate_id > $after "
                "ORDER BY candidate_id ASC LIMIT $limit",
                plan_id=self.plan_id,
                feature=feature,
                after=cursor,
                limit=_PAGE_SIZE,
            )
            if not rows:
                return
            for row in rows:
                candidate_id = str(row.get("candidate_id", ""))
                if not candidate_id or candidate_id <= cursor:
                    raise ConsolidationGroupPlanError("feature posting page is malformed")
                cursor = candidate_id
                yield candidate_id
            if len(rows) < _PAGE_SIZE:
                return

    async def _latest_event(
        self, kind: str, candidate_id: str, before_sequence: int
    ) -> dict[str, Any] | None:
        rows = await self._storage._query(
            "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
            "AND kind = $kind AND candidate_id = $candidate_id "
            "AND sequence < $before_sequence ORDER BY sequence DESC LIMIT 1",
            plan_id=self.plan_id,
            kind=kind,
            candidate_id=candidate_id,
            before_sequence=before_sequence,
        )
        return dict(rows[0]) if rows else None

    async def find(self, candidate_id: str, before_sequence: int) -> str:
        """Resolve one root using only persisted parent events (no O(N) array)."""
        current = candidate_id
        visited: set[str] = set()
        for _ in range(_MAX_UNION_DEPTH):
            if current in visited:
                raise ConsolidationGroupPlanError("union-find parent events contain a cycle")
            visited.add(current)
            event = await self._latest_event("parent", current, before_sequence)
            if event is None:
                return current
            parent_id = event.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id:
                raise ConsolidationGroupPlanError("union-find parent event is malformed")
            current = parent_id
        raise ConsolidationGroupPlanError("union-find tree exceeds its bounded depth")

    async def _rank(self, candidate_id: str, before_sequence: int) -> int:
        event = await self._latest_event("rank", candidate_id, before_sequence)
        if event is None:
            return 0
        rank = event.get("rank")
        if type(rank) is not int or rank < 0:
            raise ConsolidationGroupPlanError("union-find rank event is malformed")
        return rank

    async def union(self, left: str, right: str, sequence: int) -> bool:
        """Append deterministic union-by-rank events at one replayable sequence."""
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        root_left = await self.find(left, sequence)
        root_right = await self.find(right, sequence)
        if root_left == root_right:
            return False
        rank_left = await self._rank(root_left, sequence)
        rank_right = await self._rank(root_right, sequence)
        if rank_left < rank_right or (rank_left == rank_right and root_left > root_right):
            child, root = root_left, root_right
            root_rank = rank_right
        else:
            child, root = root_right, root_left
            root_rank = rank_left
        event_rows = [
            self._row(
                "parent",
                self._compound_key(child, sequence),
                {"candidate_id": child, "sequence": sequence, "parent_id": root},
            )
        ]
        if rank_left == rank_right:
            event_rows.append(
                self._row(
                    "rank",
                    self._compound_key(root, sequence),
                    {"candidate_id": root, "sequence": sequence, "rank": root_rank + 1},
                )
            )
        await self._create_many(event_rows)
        return True

    async def add_member(
        self, root_id: str, candidate_id: str, signature: str | None = None
    ) -> None:
        await self.add_members([(root_id, candidate_id, signature)])

    async def add_members(self, members: list[tuple[str, str, str | None]]) -> None:
        rows: list[tuple[str, dict[str, Any]]] = []
        batch_roots: set[str] = set()
        for root_id, candidate_id, signature in members:
            rows.append(
                self._row(
                    "member",
                    self._compound_key(root_id, candidate_id),
                    {
                        "root_id": root_id,
                        "candidate_id": candidate_id,
                        "signature": signature,
                    },
                )
            )
            if root_id not in batch_roots:
                rows.append(self._row("group", root_id, {"root_id": root_id}))
                batch_roots.add(root_id)
            if len(rows) >= _WRITE_BATCH_SIZE:
                await self._create_many(rows)
                rows = []
                batch_roots = set()
        if rows:
            await self._create_many(rows)

    async def iter_groups(self, *, after: str = "") -> AsyncIterator[str]:
        """Page group roots by their stored, un-hashed key."""
        cursor = after
        while True:
            rows = await self._storage._query(
                "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                "AND kind = 'group' AND root_id > $after ORDER BY root_id ASC LIMIT $limit",
                plan_id=self.plan_id,
                after=cursor,
                limit=_PAGE_SIZE,
            )
            if not rows:
                return
            for row in rows:
                root_id = str(row.get("root_id", ""))
                item_key = str(row.get("item_key", ""))
                if not root_id or item_key != root_id or root_id <= cursor:
                    raise ConsolidationGroupPlanError("group keyset page is malformed")
                cursor = item_key
                yield root_id
            if len(rows) < _PAGE_SIZE:
                return

    async def iter_member_signatures(
        self, root_id: str, *, after: str = ""
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Page component membership without reloading candidate projections."""
        cursor = after
        while True:
            rows = await self._storage._query(
                "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                "AND kind = 'member' AND root_id = $root_id AND candidate_id > $after "
                "ORDER BY candidate_id ASC LIMIT $limit",
                plan_id=self.plan_id,
                root_id=root_id,
                after=cursor,
                limit=_PAGE_SIZE,
            )
            if not rows:
                return
            for row in rows:
                candidate_id = str(row.get("candidate_id", ""))
                if not candidate_id or candidate_id <= cursor:
                    raise ConsolidationGroupPlanError("membership keyset page is malformed")
                cursor = candidate_id
                signature = row.get("signature")
                if signature is not None and not isinstance(signature, str):
                    raise ConsolidationGroupPlanError("component source signature is malformed")
                yield candidate_id, signature
            if len(rows) < _PAGE_SIZE:
                return

    async def add_merge_manifest_members(
        self,
        root_id: str,
        members: list[tuple[str, str, str | None, str | None]],
    ) -> None:
        """Persist a bounded batch of immutable source signatures for one merge group."""
        rows: list[tuple[str, dict[str, Any]]] = []
        for candidate_id, fiber_signature, typed_signature, maturation_signature in members:
            if not root_id or not candidate_id or not fiber_signature:
                raise ValueError("merge manifest identity and fiber signature must be non-empty")
            rows.append(
                self._row(
                    "merge_manifest",
                    self._compound_key(root_id, candidate_id),
                    {
                        "root_id": root_id,
                        "candidate_id": candidate_id,
                        "fiber_signature": fiber_signature,
                        "typed_signature": typed_signature,
                        "maturation_signature": maturation_signature,
                    },
                )
            )
            if len(rows) >= _WRITE_BATCH_SIZE:
                await self._create_many(rows)
                rows = []
        if rows:
            await self._create_many(rows)

    async def iter_merge_manifest_members(
        self, root_id: str, *, after: str = ""
    ) -> AsyncIterator[dict[str, str | None]]:
        """Page a merge unit's frozen fiber/typed/maturation source signatures."""
        cursor = after
        while True:
            rows = await self._storage._query(
                "SELECT * FROM consolidation_group_plan WHERE plan_id = $plan_id "
                "AND kind = 'merge_manifest' AND root_id = $root_id "
                "AND candidate_id > $after ORDER BY candidate_id ASC LIMIT $limit",
                plan_id=self.plan_id,
                root_id=root_id,
                after=cursor,
                limit=_PAGE_SIZE,
            )
            if not rows:
                return
            for row in rows:
                candidate_id = str(row.get("candidate_id", ""))
                fiber_signature = row.get("fiber_signature")
                typed_signature = row.get("typed_signature")
                maturation_signature = row.get("maturation_signature")
                if (
                    str(row.get("root_id", "")) != root_id
                    or not candidate_id
                    or candidate_id <= cursor
                    or not isinstance(fiber_signature, str)
                    or not fiber_signature
                    or (typed_signature is not None and not isinstance(typed_signature, str))
                    or (
                        maturation_signature is not None
                        and not isinstance(maturation_signature, str)
                    )
                ):
                    raise ConsolidationGroupPlanError("merge member manifest page is malformed")
                cursor = candidate_id
                yield {
                    "candidate_id": candidate_id,
                    "fiber_signature": fiber_signature,
                    "typed_signature": typed_signature,
                    "maturation_signature": maturation_signature,
                }
            if len(rows) < _PAGE_SIZE:
                return

    async def iter_members(
        self, root_id: str, *, after: str = ""
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Page component members and load one candidate projection at a time."""
        async for candidate_id, _signature in self.iter_member_signatures(root_id, after=after):
            yield candidate_id, await self.get_candidate(candidate_id)

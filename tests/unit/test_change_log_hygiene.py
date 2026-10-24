"""Change-log growth and read-cost guards.

Both halves of this file pin defects measured on a brain whose
``change_log`` had grown unbounded:

* ``get_change_log_stats`` issued four separate full aggregates, each with a
  *parameterised* ``brain_id``. SurrealDB's planner only uses the ``brain_id``
  index for an inline literal, so every one of them degraded to a full scan:
  roughly 25x slower parameterised than inlined, measured in both orders, twice.
  Four of those is why the dashboard's sync card took ~117 s to answer.
* nothing ever pruned the table. ``prune_synced_changes`` had no call site
  anywhere in ``src/``, only deleted ``synced = true`` rows (of which that brain
  had none, because sync had never completed), and returned a hardcoded ``0``
  regardless of what it deleted.

Such logs come to be dominated by ``update`` entries for edges re-weighted on
every consolidation pass. For
state replication only the newest entry per entity carries information, so
collapsing superseded pending updates is lossless, which is what the equivalence
test below asserts.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import textwrap

import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "surreal_memory"


@pytest_asyncio.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="change-log-hygiene", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    await store.close()


# ── Read cost ────────────────────────────────────────────────────────────────


class TestChangeLogStatsQueryShape:
    """The stats read must stay index-eligible and must not fan out.

    A behavioural test cannot catch this: a parameterised query returns the
    correct numbers, just 24x slower. The defect lives in the SQL text, so the
    guard reads the SQL text.
    """

    @staticmethod
    def _source() -> str:
        return inspect.getsource(SurrealDBStorage.get_change_log_stats)

    def test_brain_id_is_never_parameterised(self) -> None:
        offenders = re.findall(r"brain_id\s*=\s*\$\w+", self._source())
        assert not offenders, (
            f"parameterised brain_id defeats the brain_id index: {offenders}. "
            "Inline it with _brain_literal() -- roughly 25x on a large change_log."
        )

    def test_synced_is_derived_not_counted(self) -> None:
        """Three aggregates cost a third scan AND can contradict each other.

        ``synced`` is ``total - pending``. Counting it separately is both waste
        and a correctness hazard: the three scans are not atomic, which is how
        this endpoint once reported more pending rows than total rows.
        """
        counts = len(re.findall(r"count\(\)", self._source()))
        assert counts <= 2, (
            f"{counts} count() aggregates -- total and pending are counted, "
            "synced is derived by subtraction"
        )

    def test_the_synced_filter_is_backed_by_an_index(self) -> None:
        """Filtering an unindexed field re-reads every row (well over an order of magnitude)."""
        schema = (SRC_ROOT / "storage" / "surrealdb" / "schema.py").read_text(encoding="utf-8")
        assert "ON change_log FIELDS brain_id, synced" in schema, (
            "get_change_log_stats filters on `synced`, so change_log needs a "
            "(brain_id, synced) index -- without it the count is a full read"
        )


# ── Growth control ───────────────────────────────────────────────────────────


class TestPruneSyncedChangesReportsWhatItDid:
    def test_surreal_backend_does_not_hardcode_its_return(self) -> None:
        """The SurrealDB implementation returned a literal 0 for its whole life.

        The in-memory backend always counted correctly, so backend parity tests
        passed while the production backend reported nothing. Only the SurrealDB
        source can show this -- the number it returns is not observable without
        a live database.
        """
        source = textwrap.dedent(inspect.getsource(SurrealDBStorage.prune_synced_changes))
        returns = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Return) and node.value is not None
        ]
        assert returns, "prune_synced_changes returns nothing at all"
        computed = [r for r in returns if not isinstance(r.value, ast.Constant)]
        assert computed, (
            "every return in prune_synced_changes is a literal -- the count is "
            "hardcoded rather than measured. An early `return 0` for the "
            "nothing-matched case is fine; a literal on EVERY path is the bug."
        )

    async def test_returns_the_number_deleted(self, storage: InMemoryStorage) -> None:
        """A hardcoded `return 0` is the counter-lies defect, again."""
        for i in range(3):
            await storage.record_change("neuron", f"n-{i}", "insert")
        await storage.mark_synced(await _last_sequence(storage))

        deleted = await storage.prune_synced_changes(older_than_days=0)

        assert deleted == 3, f"deleted 3 rows but reported {deleted}"

    async def test_returns_zero_when_nothing_matched(self, storage: InMemoryStorage) -> None:
        await storage.record_change("neuron", "n-1", "insert")  # left unsynced

        assert await storage.prune_synced_changes(older_than_days=0) == 0


class TestCollapsePendingUpdates:
    """Superseded pending updates are redundant; everything else is not."""

    async def test_bounded_cursor_reaches_later_duplicates_and_is_brain_scoped(
        self, storage: InMemoryStorage
    ) -> None:
        main_brain = storage.current_brain_id
        assert main_brain is not None
        await storage.record_change("fiber", "synced", "update")
        await storage.mark_synced(await _last_sequence(storage))
        for entity_id in ("unique-1", "unique-2", "unique-3"):
            await storage.record_change("synapse", entity_id, "update")
        first = await storage.record_change("synapse", "tail", "update", payload={"v": 1})
        await storage.record_change("synapse", "tail", "update", payload={"v": 2})
        await storage.record_change("neuron", "protected", "insert")
        await storage.record_change("neuron", "protected", "insert")
        await storage.record_change("fiber", "synced", "update")

        foreign = Brain.create(name="foreign-change-log")
        await storage.save_brain(foreign)
        storage.set_brain(foreign.id)
        await storage.record_change("synapse", "tail", "update", payload={"v": 99})
        storage.set_brain(main_brain)

        cursor = None
        deleted = 0
        pages = 0
        while True:
            removed, cursor, exhausted = await storage.collapse_pending_updates(
                max_rows=2, cursor=cursor
            )
            deleted += removed
            pages += 1
            if exhausted:
                break

        assert pages >= 3, "unique prefix pages must not starve later duplicate keys"
        assert deleted == 1
        main = storage._change_log[main_brain]
        assert any(entry.id == first for entry in main) is False
        assert {
            (entry.entity_type, entry.entity_id, entry.payload.get("v"))
            for entry in main
            if entry.operation == "update"
        } == {
            ("fiber", "synced", None),
            ("synapse", "unique-1", None),
            ("synapse", "unique-2", None),
            ("synapse", "unique-3", None),
            ("synapse", "tail", 2),
            ("fiber", "synced", None),
        }
        assert sum(entry.synced for entry in main) == 1
        assert sum(entry.operation == "insert" for entry in main) == 2

        storage.set_brain(foreign.id)
        foreign_entries = await storage.get_unsynced_changes(limit=10)
        assert len(foreign_entries) == 1
        assert foreign_entries[0].payload == {"v": 99}

    async def test_keeps_only_the_newest_update_per_entity(self, storage: InMemoryStorage) -> None:
        for _ in range(5):
            await storage.record_change("synapse", "s-1", "update")
        await storage.record_change("synapse", "s-2", "update")

        removed = await _collapse_all(storage)

        assert removed == 4, f"5 updates of s-1 collapse to 1, so 4 go; got {removed}"
        pending = await storage.get_unsynced_changes()
        by_entity = [c.entity_id for c in pending]
        assert sorted(by_entity) == ["s-1", "s-2"]

    async def test_the_survivor_is_the_newest_not_the_oldest(
        self, storage: InMemoryStorage
    ) -> None:
        """Keeping the oldest would replicate a stale payload."""
        first = await storage.record_change("synapse", "s-1", "update", payload={"weight": 0.1})
        last = await storage.record_change("synapse", "s-1", "update", payload={"weight": 0.9})

        await _collapse_all(storage)

        pending = await storage.get_unsynced_changes()
        assert len(pending) == 1
        survivor = pending[0]
        assert survivor.id == last, f"kept sequence {survivor.id}, expected the newest ({last})"
        assert survivor.id != first
        assert survivor.payload.get("weight") == 0.9, "the surviving payload must be the newest"

    async def test_inserts_and_deletes_are_never_collapsed(self, storage: InMemoryStorage) -> None:
        """Dropping an insert would leave a peer updating an entity it never saw.

        Unlike updates, insert/delete are not idempotent-by-latest: their
        ordering relative to each other carries meaning.
        """
        await storage.record_change("neuron", "n-1", "insert")
        await storage.record_change("neuron", "n-1", "insert")
        await storage.record_change("neuron", "n-1", "delete")
        await storage.record_change("neuron", "n-1", "delete")

        removed = await _collapse_all(storage)

        assert removed == 0
        assert len(await storage.get_unsynced_changes()) == 4

    async def test_already_synced_rows_are_left_alone(self, storage: InMemoryStorage) -> None:
        """Synced history belongs to prune_synced_changes, not to the collapse."""
        await storage.record_change("synapse", "s-1", "update")
        await storage.record_change("synapse", "s-1", "update")
        await storage.mark_synced(await _last_sequence(storage))
        await storage.record_change("synapse", "s-1", "update")

        removed = await _collapse_all(storage)

        assert removed == 0, "no two PENDING updates share an entity here"
        assert await _total(storage) == 3, "synced rows must survive the collapse"

    async def test_collapse_is_lossless_for_replication(self, storage: InMemoryStorage) -> None:
        """A peer replaying the collapsed log reaches the same end state.

        This is the property that makes the collapse safe at all: replication
        applies the newest payload per entity, so superseded updates cannot
        change the outcome.
        """
        for weight in (0.1, 0.4, 0.7, 0.9):
            await storage.record_change("synapse", "s-1", "update", payload={"weight": weight})
        await storage.record_change("neuron", "n-1", "insert", payload={"content": "hi"})

        def end_state(changes: list) -> dict[tuple[str, str], dict]:
            state: dict[tuple[str, str], dict] = {}
            for change in sorted(changes, key=lambda c: c.id):
                state[(change.entity_type, change.entity_id)] = change.payload
            return state

        before = end_state(await storage.get_unsynced_changes())
        await _collapse_all(storage)
        after = end_state(await storage.get_unsynced_changes())

        assert after == before, "collapsing changed what a peer would converge to"

    async def test_nothing_to_collapse_is_not_an_error(self, storage: InMemoryStorage) -> None:
        assert await _collapse_all(storage) == 0


class TestCollapseIsWiredIn:
    """prune_synced_changes shipped with zero call sites for its whole life.

    A growth control nobody calls is indistinguishable from no growth control,
    and the endpoint that would have shown the table was itself too slow to load.
    """

    def test_some_production_path_calls_the_collapse(self) -> None:
        call_sites = [
            path.relative_to(SRC_ROOT).as_posix()
            for path in SRC_ROOT.rglob("*.py")
            if "collapse_pending_updates" in path.read_text(encoding="utf-8")
            and path.name not in {"base.py", "store.py", "memory_sync_ops.py"}
        ]
        assert call_sites, (
            "collapse_pending_updates is defined but never called from a "
            "production path -- exactly how such a log grows unbounded"
        )


async def _last_sequence(storage: InMemoryStorage) -> int:
    stats = await storage.get_change_log_stats()
    return int(stats["last_sequence"])


async def _collapse_all(storage: InMemoryStorage, *, max_rows: int = 200_000) -> int:
    cursor = None
    removed = 0
    while True:
        page_removed, cursor, exhausted = await storage.collapse_pending_updates(
            max_rows=max_rows, cursor=cursor
        )
        removed += page_removed
        if exhausted:
            return removed


async def _total(storage: InMemoryStorage) -> int:
    stats = await storage.get_change_log_stats()
    return int(stats["total"])

"""`cleanup_orphaned_maturations` must also reach rows whose brain is gone.

The scan was scoped ``WHERE brain_id = $brain_id``, so it could only ever see
rows belonging to the brain being cleaned. Rows left behind by a *deleted* brain
were therefore structurally unreachable: no consolidation could ever remove
them, however many times it ran.
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb.maturation import SurrealDBMaturationMixin

BRAIN = "default"


class _FakeConn:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, record_id: str) -> None:
        self.deleted.append(record_id)


class _Store(SurrealDBMaturationMixin):
    """Drives the mixin against in-memory rows, mimicking the SurrealQL shapes."""

    def __init__(
        self,
        maturations: list[dict[str, Any]],
        fibers: list[dict[str, Any]],
        brains: list[dict[str, Any]],
    ) -> None:
        self._maturations = maturations
        self._fibers = fibers
        self._brains = brains
        self.conn = _FakeConn()

    def _get_brain_id(self) -> str:
        return BRAIN

    def _ensure_conn(self) -> _FakeConn:
        return self.conn

    async def _query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if "FROM brain" in query:
            return list(self._brains)
        if "FROM fiber" in query:
            bid = params.get("brain_id")
            return [f for f in self._fibers if f["brain_id"] == bid]
        if "FROM maturation" in query:
            if "brain_id = $brain_id" in query:
                bid = params.get("brain_id")
                return [m for m in self._maturations if m["brain_id"] == bid]
            return list(self._maturations)
        return []


def _mat(rid: str, brain_id: str, fiber_id: str) -> dict[str, Any]:
    return {"id": f"maturation:{rid}", "brain_id": brain_id, "fiber_id": fiber_id}


class TestOrphanedByMissingFiber:
    """The pre-existing behaviour must keep working."""

    async def test_removes_a_row_whose_fiber_is_gone(self) -> None:
        store = _Store(
            maturations=[_mat("m1", BRAIN, "f1"), _mat("m2", BRAIN, "f2")],
            fibers=[{"id": "fiber:f1", "brain_id": BRAIN}],
            brains=[{"id": "brain:uuid-1", "name": BRAIN}],
        )

        removed = await store.cleanup_orphaned_maturations()

        assert removed == 1
        assert store.conn.deleted == ["maturation:m2"]

    async def test_keeps_rows_whose_fiber_survives(self) -> None:
        store = _Store(
            maturations=[_mat("m1", BRAIN, "f1")],
            fibers=[{"id": "fiber:f1", "brain_id": BRAIN}],
            brains=[{"id": "brain:uuid-1", "name": BRAIN}],
        )

        assert await store.cleanup_orphaned_maturations() == 0
        assert store.conn.deleted == []


class TestOrphanedByDeletedBrain:
    """The case a brain-scoped scan could never reach."""

    async def test_removes_rows_belonging_to_a_deleted_brain(self) -> None:
        store = _Store(
            maturations=[
                _mat("m1", BRAIN, "f1"),
                _mat("dead1", "dead-brain-1", "gone-1"),
                _mat("dead2", "dead-brain-2", "gone-2"),
            ],
            fibers=[{"id": "fiber:f1", "brain_id": BRAIN}],
            brains=[{"id": "brain:uuid-1", "name": BRAIN}],
        )

        removed = await store.cleanup_orphaned_maturations()

        assert removed == 2
        assert sorted(store.conn.deleted) == ["maturation:dead1", "maturation:dead2"]

    async def test_a_live_brain_addressed_by_its_record_uuid_is_not_an_orphan(self) -> None:
        """Legacy rows may carry the record UUID; that brain is still alive."""
        store = _Store(
            maturations=[_mat("legacy", "uuid-1", "f1")],
            fibers=[{"id": "fiber:f1", "brain_id": BRAIN}],
            brains=[{"id": "brain:uuid-1", "name": BRAIN}],
        )

        assert await store.cleanup_orphaned_maturations() == 0
        assert store.conn.deleted == []

    async def test_underscore_spelling_of_a_record_uuid_also_counts_as_alive(self) -> None:
        store = _Store(
            maturations=[_mat("legacy", "uuid-1-2", "f1")],
            fibers=[{"id": "fiber:f1", "brain_id": BRAIN}],
            brains=[{"id": "brain:uuid_1_2", "name": BRAIN}],
        )

        assert await store.cleanup_orphaned_maturations() == 0

    async def test_an_unreadable_brain_table_deletes_nothing(self) -> None:
        """A transient read failure must not be read as 'every row is an orphan'."""
        store = _Store(
            maturations=[_mat("m1", "some-brain", "f1"), _mat("m2", "other", "f2")],
            fibers=[],
            brains=[],
        )

        removed = await store.cleanup_orphaned_maturations()

        assert removed == 0
        assert store.conn.deleted == []

    async def test_both_orphan_kinds_are_counted_together(self) -> None:
        store = _Store(
            maturations=[
                _mat("keep", BRAIN, "f1"),
                _mat("nofiber", BRAIN, "vanished"),
                _mat("nobrain", "dead-brain", "whatever"),
            ],
            fibers=[{"id": "fiber:f1", "brain_id": BRAIN}],
            brains=[{"id": "brain:uuid-1", "name": BRAIN}],
        )

        removed = await store.cleanup_orphaned_maturations()

        assert removed == 2
        assert "maturation:keep" not in store.conn.deleted
        assert sorted(store.conn.deleted) == ["maturation:nobrain", "maturation:nofiber"]


@pytest.mark.parametrize("dead_count", [1, 5, 24])
async def test_scales_to_however_many_dead_brains_exist(dead_count: int) -> None:
    store = _Store(
        maturations=[_mat(f"d{i}", f"dead-{i}", f"g{i}") for i in range(dead_count)],
        fibers=[],
        brains=[{"id": "brain:uuid-1", "name": BRAIN}],
    )

    assert await store.cleanup_orphaned_maturations() == dead_count

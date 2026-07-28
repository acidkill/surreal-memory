"""Regression tests for fiber-id normalisation in the SurrealDB maturation mixin.

THE BUG: ``Fiber.create`` mints a dash-form uuid4, ``BuildFiberStep`` passed that
straight to ``save_maturation``, and the mixin stored it verbatim in the
``fiber_id`` FIELD — while the ``fiber`` table (and this table's own record id)
folds ids to the underscore form via ``_to_surreal_id``. Consequences on the live
brain, all measured:

* 1277 of 1920 maturation rows carried a dash-form ``fiber_id``.
* ``lifecycle.reinforce`` looks up ``get_maturation(fiber.id)`` with an id read
  back from the DB (underscore form), so those 1277 rows were never found:
  ALL 77 rehearsed rows and ALL 9 semantic rows were underscore-form, and zero
  dash-form rows had ever been rehearsed.
* No rehearsals => the EPISODIC->SEMANTIC spacing gate can never be met =>
  consolidation_ratio 0.0037 and "all memories still episodic" after months.
* ``extract_patterns`` keys maturations by ``fiber_id`` and tests ``f.id in
  maturations``, so the same rows were invisible to pattern extraction too.
* Repair was impossible from inside: the existence check missed, the insert
  branch then collided with the row's already-underscore record id, and
  ``save_maturation``'s blanket ``except Exception`` swallowed it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.storage.surrealdb.maturation import SurrealDBMaturationMixin

_DASH = "0019e251-94df-4a6b-b446-72c76a083635"
_UNDERSCORE = "0019e251_94df_4a6b_b446_72c76a083635"


class _FakeMaturationStore(SurrealDBMaturationMixin):
    """In-memory stand-in that mimics SurrealDB's field/record-id semantics.

    Rows are keyed by record id (which the real engine enforces as a primary key),
    so an insert onto an occupied id raises — exactly the collision that used to be
    swallowed. ``_query`` understands only the mixin's own WHERE shapes.
    """

    def __init__(self, brain_id: str = "default") -> None:
        self._brain = brain_id
        self.rows: dict[str, dict] = {}
        conn = MagicMock()
        conn.merge = AsyncMock(side_effect=self._merge)
        conn.insert = AsyncMock(side_effect=self._insert)
        self._conn = conn

    async def _merge(self, record_id: str, data: dict) -> None:
        self.rows[record_id].update(data)

    async def _insert(self, table: str, data: dict) -> None:
        rid = f"{table}:{data['id']}"
        if rid in self.rows:
            raise RuntimeError(f"Database record `{rid}` already exists")
        row = dict(data)
        row["id"] = rid
        self.rows[rid] = row

    def seed_legacy_row(self, fiber_id: str, **overrides: object) -> str:
        """Insert a row the way the pre-fix code did: raw field, folded record id."""
        rid = f"maturation:{self._brain}_{_to_surreal_id(fiber_id)}"
        self.rows[rid] = {
            "id": rid,
            "fiber_id": fiber_id,
            "brain_id": self._brain,
            "stage": "episodic",
            "stage_entered_at": "2026-06-20T00:00:00Z",
            "rehearsal_count": 0,
            "reinforcement_timestamps": [],
            **overrides,
        }
        return rid

    def _ensure_conn(self) -> MagicMock:
        return self._conn

    def _get_brain_id(self) -> str:
        return self._brain

    async def _query(self, sql: str, **params: object) -> list[dict]:
        """Model the three WHERE shapes the mixin issues, brain-scoped."""
        candidates = [r for r in self.rows.values() if r["brain_id"] == params["brain_id"]]
        if "sid" in params:  # record-id probe
            candidates = [r for r in candidates if r["id"] == f"maturation:{params['sid']}"]
        elif "fiber_id" in params:  # field probe (brain-rename fallback)
            wanted = (params.get("fiber_id"), params.get("legacy_fiber_id"))
            candidates = [r for r in candidates if r["fiber_id"] in wanted]
        else:  # find_maturations filters
            if "stage" in params:
                candidates = [r for r in candidates if r["stage"] == params["stage"]]
            if "min_rc" in params:
                candidates = [r for r in candidates if r["rehearsal_count"] >= params["min_rc"]]
        return [dict(r) for r in candidates]


@pytest.mark.asyncio
async def test_write_then_read_roundtrips_a_dash_form_id() -> None:
    """The exact BuildFiberStep path: write with a dash-form uuid4, read it back."""
    store = _FakeMaturationStore()

    await store.save_maturation(
        MaturationRecord(fiber_id=_DASH, brain_id="default", stage=MemoryStage.EPISODIC)
    )
    found = await store.get_maturation(_DASH)

    assert found is not None
    assert found.stage is MemoryStage.EPISODIC


@pytest.mark.asyncio
async def test_both_id_forms_resolve_to_the_same_row() -> None:
    """Dash and underscore spellings must hit one row, not two."""
    store = _FakeMaturationStore()

    await store.save_maturation(
        MaturationRecord(fiber_id=_DASH, brain_id="default", stage=MemoryStage.EPISODIC)
    )

    by_dash = await store.get_maturation(_DASH)
    by_underscore = await store.get_maturation(_UNDERSCORE)

    assert len(store.rows) == 1
    assert by_dash is not None
    assert by_underscore is not None
    assert by_dash == by_underscore


@pytest.mark.asyncio
async def test_stored_field_uses_the_form_the_fiber_table_uses() -> None:
    """The persisted field must match ``fiber:<id>`` or no join can ever succeed."""
    store = _FakeMaturationStore()

    await store.save_maturation(MaturationRecord(fiber_id=_DASH, brain_id="default"))

    (row,) = store.rows.values()
    assert row["fiber_id"] == _UNDERSCORE
    assert row["id"] == f"maturation:default_{_UNDERSCORE}"


@pytest.mark.asyncio
async def test_rehearsal_lands_on_a_legacy_dash_form_row() -> None:
    """THE promotion bug: reinforce() looks up by the DB (underscore) form.

    Pre-fix the legacy row was invisible, so ``rehearse`` never ran and the
    3-distinct-days gate to SEMANTIC could never be satisfied.
    """
    store = _FakeMaturationStore()
    rid = store.seed_legacy_row(_DASH)

    record = await store.get_maturation(_UNDERSCORE)
    assert record is not None, "legacy dash-form row must be reachable by the fiber's DB id"

    await store.save_maturation(record.rehearse())

    assert len(store.rows) == 1
    assert store.rows[rid]["rehearsal_count"] == 1
    # ...and the write self-heals the field to the canonical form.
    assert store.rows[rid]["fiber_id"] == _UNDERSCORE


@pytest.mark.asyncio
async def test_save_over_legacy_row_updates_instead_of_colliding() -> None:
    """Pre-fix this took the insert branch and died on the record-id collision."""
    store = _FakeMaturationStore()
    rid = store.seed_legacy_row(_DASH, rehearsal_count=4)

    await store.save_maturation(
        MaturationRecord(
            fiber_id=_UNDERSCORE,
            brain_id="default",
            stage=MemoryStage.SEMANTIC,
            rehearsal_count=5,
        )
    )

    assert len(store.rows) == 1
    assert store.rows[rid]["stage"] == "semantic"
    assert store.rows[rid]["rehearsal_count"] == 5


@pytest.mark.asyncio
async def test_find_maturations_returns_ids_that_join_to_fibers() -> None:
    """``extract_patterns`` does ``f.id in maturations`` — the keys must match."""
    store = _FakeMaturationStore()
    store.seed_legacy_row(_DASH)
    store.seed_legacy_row("aaaa1111-2222-3333-4444-555566667777")

    records = await store.find_maturations()

    # Fiber ids as _row_to_fiber hands them out (record-id suffix, underscore form).
    fiber_ids = {_UNDERSCORE, "aaaa1111_2222_3333_4444_555566667777"}
    assert {m.fiber_id for m in records} == fiber_ids


@pytest.mark.asyncio
async def test_backfill_skips_fibers_that_already_have_a_legacy_row() -> None:
    """Pre-fix the skip check missed and the resulting insert collided silently."""

    class _Fiber:
        def __init__(self, fid: str) -> None:
            self.id = fid
            self.time_start = None
            self.created_at = None

    class _WithFibers(_FakeMaturationStore):
        async def get_fibers(self, limit: int = 0) -> list[_Fiber]:
            return [_Fiber(_UNDERSCORE)]

    store = _WithFibers()
    store.seed_legacy_row(_DASH)

    counts = await store.backfill_maturations()

    assert counts["skipped"] == 1
    assert len(store.rows) == 1

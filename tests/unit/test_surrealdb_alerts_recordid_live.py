"""Live-DB proof that the alert lookups compare against a record id, not a string.

``alerts.py`` used to bind ``rid=f"alerts:{sid}"`` — a *string* — and compare it
with ``id = $rid``. On SurrealDB the left side is a record id, so the predicate
is unconditionally false and the row can never match. Three lookups sat behind
that predicate and all three were structurally dead:

* ``mark_alerts_seen``      — an alert could never leave ``active``.
* ``mark_alert_acknowledged`` — acknowledging silently reported "not found".
* ``get_alert``             — reading a single alert always returned ``None``.

``resolve_alerts_by_type`` was already correct (it reuses the ``id`` returned by
its own query) and is covered here as the *positive control*: if it also failed,
the fixture would be broken rather than the predicate.

Every status assertion re-reads the row with a ``brain_id``-only query and
matches in Python, so the check never leans on the very construct under test.
Skipped unless ``SURREALDB_URL`` points at a running SurrealDB.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from surreal_memory.core.alert import Alert, AlertStatus, AlertType
from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)

BRAIN_NAME = "alerts-recordid-live"


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name=BRAIN_NAME)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store

    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    finally:
        await store.close()


async def _seed(store, alert_id: str, alert_type: AlertType = AlertType.STALE_FIBERS):  # type: ignore[no-untyped-def]
    """Persist one active alert and assert it really landed (positive control)."""
    alert = Alert(
        id=alert_id,
        brain_id=store._get_brain_id(),
        alert_type=alert_type,
        severity="medium",
        message=f"recordid-live {alert_id}",
        recommended_action="none",
        status=AlertStatus.ACTIVE,
        created_at=utcnow(),
    )
    assert await store.record_alert(alert) == alert_id
    assert await _status_of(store, alert_id) == "active", "seeding did not persist the alert"
    return alert


async def _status_of(store, alert_id: str) -> str | None:  # type: ignore[no-untyped-def]
    """Status straight from the DB, found without an ``id = ...`` predicate.

    Deliberately brain-scoped only: using ``type::record`` here would make the
    verification share the fix's own construct, and the test would pass for the
    wrong reason if the construct were wrong.

    The comparison is on the *sanitised* id, because that is what a record id
    actually holds ('-' is not a legal record-name character). Matching on the
    caller's raw id instead makes the dashed-uuid case fail inside this helper —
    a broken probe, not a broken lookup.
    """
    rows = await store._query(
        "SELECT id, status FROM alerts WHERE brain_id = $brain_id",
        brain_id=store._get_brain_id(),
    )
    want = _to_surreal_id(alert_id)
    for row in rows:
        if str(row.get("id")).split(":", 1)[-1].strip("⟨⟩") == want:
            return str(row.get("status"))
    return None


async def test_get_alert_finds_the_alert_it_just_stored(storage) -> None:  # type: ignore[no-untyped-def]
    alert_id = uuid4().hex[:16]
    await _seed(storage, alert_id)

    found = await storage.get_alert(alert_id)

    assert found is not None, "get_alert returned None for an alert that exists"
    assert found.id == alert_id
    assert found.status == AlertStatus.ACTIVE


async def test_mark_alerts_seen_moves_the_row_out_of_active(storage) -> None:  # type: ignore[no-untyped-def]
    alert_id = uuid4().hex[:16]
    await _seed(storage, alert_id)

    updated = await storage.mark_alerts_seen([alert_id])

    assert updated == 1, "mark_alerts_seen reported no rows for an active alert"
    # The return value is a claim; the row is the fact.
    assert await _status_of(storage, alert_id) == "seen"


async def test_mark_alert_acknowledged_persists_the_status(storage) -> None:  # type: ignore[no-untyped-def]
    alert_id = uuid4().hex[:16]
    await _seed(storage, alert_id)

    assert await storage.mark_alert_acknowledged(alert_id) is True
    assert await _status_of(storage, alert_id) == "acknowledged"


async def test_lookups_work_for_a_dashed_uuid_id(storage) -> None:  # type: ignore[no-untyped-def]
    """Ids are sanitised ('-' → '_') before becoming a record id.

    The production alert id is a bare hex slug, so a fix that forgot the
    sanitisation step would still pass the tests above. A dashed uuid makes the
    stored record id differ from the caller's id and pins that half down.

    What comes back is the *stored* id, not the caller's spelling: the alerts
    table keeps no separate raw-id field, so the record id is the only identity
    there is. That is not a leak in the round trip for real callers —
    ``AlertHandler`` mints ``uuid4().hex[:16]`` (already in the record-name
    charset) and every id it later acknowledges came out of
    ``get_active_alerts``, i.e. already sanitised, and ``_to_surreal_id`` is
    idempotent on that form.
    """
    alert_id = str(uuid4())
    assert "-" in alert_id
    await _seed(storage, alert_id, AlertType.HIGH_ORPHAN_RATIO)

    found = await storage.get_alert(alert_id)
    assert found is not None, "get_alert cannot resolve a dashed uuid alert id"
    assert found.id == _to_surreal_id(alert_id)
    assert _to_surreal_id(found.id) == found.id, "the returned id must be a fixed point"

    assert await storage.mark_alert_acknowledged(alert_id) is True
    assert await _status_of(storage, alert_id) == "acknowledged"
    # And the sanitised spelling resolves too, which is the form a caller
    # actually receives from get_active_alerts.
    assert await storage.get_alert(found.id) is not None


async def test_unknown_ids_are_still_reported_as_missing(storage) -> None:  # type: ignore[no-untyped-def]
    """Negative control: the fix must not turn the predicate into a no-op.

    ``type::record`` on an id that was never stored has to keep matching
    nothing — otherwise every assertion above would pass for a fix that simply
    dropped the id filter.

    The brain must NOT be empty for that to mean anything: against an empty
    table a lookup with no id filter also returns nothing, and this test would
    pass for a build with the filter deleted. One decoy alert is what makes the
    control discriminate.
    """
    decoy = uuid4().hex[:16]
    await _seed(storage, decoy)
    missing = uuid4().hex[:16]

    assert await storage.get_alert(missing) is None
    assert await storage.mark_alert_acknowledged(missing) is False
    assert await storage.mark_alerts_seen([missing]) == 0

    # The decoy must be untouched: a lookup that ignored the id would have
    # matched it and moved it out of 'active'.
    assert await _status_of(storage, decoy) == "active"


async def test_an_all_digit_id_updates_the_row_it_matched(storage) -> None:  # type: ignore[no-untyped-def]
    """A write must target the row the SELECT matched, not a rebuilt id.

    ``record_alert`` stores the id as a *string* record id. Rebuilding
    ``f"alerts:{sid}"`` for the write sends an all-digit sid back through the
    SDK as a *numeric* record id — a different record. The row stays put while
    the call reports success, which is worse than the bug being fixed here:
    before it at least returned False. ``uuid4().hex[:16]`` is all digits
    roughly once in 1150 alerts, so this is reachable, not theoretical.
    """
    all_digits = "9" + "".join(str(int(c, 16) % 10) for c in uuid4().hex[:15])
    assert all_digits.isdigit()
    await _seed(storage, all_digits, AlertType.HIGH_SYNAPSE_COUNT)

    assert await storage.mark_alert_acknowledged(all_digits) is True
    assert await _status_of(storage, all_digits) == "acknowledged", (
        "acknowledge reported success but the row it matched was not updated"
    )

    seen_id = "8" + "".join(str(int(c, 16) % 10) for c in uuid4().hex[:15])
    await _seed(storage, seen_id, AlertType.HIGH_NEURON_COUNT)
    assert await storage.mark_alerts_seen([seen_id]) == 1
    assert await _status_of(storage, seen_id) == "seen"


async def test_record_alert_can_replace_a_clashing_all_digit_row(storage) -> None:  # type: ignore[no-untyped-def]
    """The insert-retry path has to actually clear the row it collides with.

    ``record_alert`` falls back to delete-then-insert when the first insert
    raises. Addressing that delete with a rebuilt ``f"alerts:{sid}"`` string
    hits a numeric record id for an all-digit sid: nothing is deleted, nothing
    is raised, and the retry hits the same collision.
    """
    all_digits = "5" + "".join(str(int(c, 16) % 10) for c in uuid4().hex[:15])
    assert all_digits.isdigit()
    await _seed(storage, all_digits, AlertType.HIGH_FIBER_COUNT)

    conn = storage._ensure_conn()
    await storage._query("DELETE type::record('alerts', $sid)", sid=all_digits)
    assert await _status_of(storage, all_digits) is None

    # Re-create it, then force the collision path by inserting the same id again.
    await _seed(storage, all_digits, AlertType.HIGH_FIBER_COUNT)
    rows = await storage._query(
        "SELECT id FROM alerts WHERE brain_id = $brain_id", brain_id=storage._get_brain_id()
    )
    before = len(rows)
    assert before >= 1

    # A second record_alert for the same id must not leave a duplicate behind.
    await conn.merge(rows[0]["id"], {"status": "resolved"})  # clear the dedup cooldown
    await _seed(storage, all_digits, AlertType.HIGH_FIBER_COUNT)
    rows = await storage._query(
        "SELECT id FROM alerts WHERE brain_id = $brain_id", brain_id=storage._get_brain_id()
    )
    assert len(rows) == before, "the retry path duplicated or stranded a row"
    assert await _status_of(storage, all_digits) == "active"


async def test_the_id_get_active_alerts_hands_back_can_be_acknowledged(storage) -> None:  # type: ignore[no-untyped-def]
    """The caller's actual route, end to end, for a letter-free id.

    ``AlertHandler`` never types an id: it lists alerts and acknowledges what
    the listing gave it. SurrealDB renders a record id carrying no letter in
    its quoted form (``alerts:⟨1122334455667788⟩``), so a row mapper that only
    strips the table prefix hands the guillemets back to the caller — and
    feeding those into ``_to_surreal_id`` maps them to underscores, producing
    an id that no longer addresses its own row. Fixing the SELECT is not enough
    if the id the caller receives cannot be passed back in.
    """
    all_digits = "1" + "".join(str(int(c, 16) % 10) for c in uuid4().hex[:15])
    assert all_digits.isdigit()
    await _seed(storage, all_digits, AlertType.LOW_CONNECTIVITY)

    listed = await storage.get_active_alerts()
    ids = [a.id for a in listed]
    assert all_digits in ids, f"get_active_alerts returned a mangled id: {ids!r}"

    assert await storage.mark_alert_acknowledged(ids[ids.index(all_digits)]) is True
    assert await _status_of(storage, all_digits) == "acknowledged"


async def test_resolve_alerts_by_type_still_works(storage) -> None:  # type: ignore[no-untyped-def]
    """Positive control on the sibling that was never broken.

    ``resolve_alerts_by_type`` reuses the ``id`` its own SELECT returned, so it
    worked before this fix and must keep working after it. A red result here
    would mean the fixture is broken, not the predicate.
    """
    alert_id = uuid4().hex[:16]
    await _seed(storage, alert_id, AlertType.LOW_CONNECTIVITY)

    resolved = await storage.resolve_alerts_by_type([AlertType.LOW_CONNECTIVITY.value])

    assert resolved >= 1
    assert await _status_of(storage, alert_id) == "resolved"

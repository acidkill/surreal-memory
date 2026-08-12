"""Shared helpers for the live-SurrealDB unit tests (gated on SURREALDB_URL).

Two isolation problems used to plague these tests, fixed here in one place:

1. **Stub pollution.** Several unit-test modules stub
   ``sys.modules["surrealdb"]`` with a ``MagicMock`` so ``store.py`` can be
   imported where the optional SDK isn't installed. Because ``store.py``
   imports the SDK lazily (``from surrealdb import AsyncSurreal`` at call
   time), a stub that leaks into a full-suite run turns every live test into
   an ERROR at fixture setup (``TypeError: object MagicMock can't be used in
   'await' expression``) even though the file passes solo.
   ``ensure_real_surrealdb_sdk()`` heals ``sys.modules`` before a live
   fixture connects.

2. **Brain leakage.** Every live fixture creates a throwaway brain on the
   shared DB and used to leave it behind, so ``smem brain list`` accumulated
   rows like ``all-types-roundtrip`` forever. ``cleanup_live_brains()``
   deletes the fixture's own brain by record id at teardown, plus any *stale*
   leftovers carrying a known live-test name. It never touches ``default``
   (or any name outside the explicit test-name list), and it only deletes by
   record id — never by a name-pattern DELETE.
"""

from __future__ import annotations

import importlib
import sys
import types
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

#: Exact names of the throwaway brains the live unit tests create. Cleanup is
#: strictly limited to these names; anything else on the server is left alone.
LIVE_TEST_BRAIN_NAMES = frozenset(
    {
        "all-types-roundtrip",  # test_surrealdb_typed_memory_all_types.py
        "u1-retrieval-trace-live",  # test_surrealdb_retrieval_trace_live.py
        "u3-supersession-live",  # test_surrealdb_supersession_live.py
        "ub2-fiber-id-norm-live",  # test_surrealdb_fiber_id_norm_live.py
        "u8-geo-live",  # test_surrealdb_geo_live.py
        "ub1-recordid-fix-live",  # test_surrealdb_recordid_fix_live.py
        "parity-test-surreal",  # test_get_project_memories.py
        "snapshot-roundtrip-live",  # test_surrealdb_export_import_live.py
        "snapshot-roundtrip-live-target",  # test_surrealdb_export_import_live.py
        "pinned-expiry-test-9f3a1c",  # test_surrealdb_expiry_respects_pinned_live.py
        "bug006-tm-delete-id-live",  # test_surrealdb_typed_memory_delete_id_live.py
        "kw-df-batch-live-4b8d2e",  # test_surrealdb_keyword_df_live.py
    }
)

#: Hard deny-list: rows with these names are never deleted, even if someone
#: mistakenly adds them to LIVE_TEST_BRAIN_NAMES.
PROTECTED_BRAIN_NAMES = frozenset({"default", "my-brain.v2"})

#: Leftovers younger than this are assumed to belong to a concurrently
#: running test (xdist worker / second dev machine) and are skipped; the
#: fixture that created them deletes them by id anyway.
_STALE_AFTER = timedelta(hours=1)


def ensure_real_surrealdb_sdk() -> None:
    """Make sure ``sys.modules["surrealdb"]`` is the real installed SDK.

    If another test module replaced it with a mock/stub, drop the stub (and
    any ``surrealdb.*`` submodule stubs) and import the genuine package so
    the store's lazy ``from surrealdb import AsyncSurreal`` resolves to a
    working client. Skips the test when the SDK is genuinely not installed.
    """
    mod = sys.modules.get("surrealdb")
    if isinstance(mod, types.ModuleType) and getattr(mod, "__spec__", None) is not None:
        return  # real SDK already loaded
    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "surrealdb" or name.startswith("surrealdb.")
    }
    try:
        importlib.import_module("surrealdb")
    except ImportError:
        # SDK genuinely absent: put the stubs back (stub-based tests running
        # after this one still rely on them) and skip the live test.
        sys.modules.update(removed)
        pytest.skip("surrealdb SDK is not installed")


def _record_id_raw(rid: Any) -> str:
    """Bare id part of a brain record id ('brain:⟨x⟩' / RecordID → 'x')."""
    if hasattr(rid, "id"):  # surrealdb.RecordID
        return str(rid.id)
    return str(rid).split(":", 1)[-1].strip("⟨⟩")


def _is_stale(created_at: Any) -> bool:
    """True when a row is old enough to be a leftover from a previous run."""
    value = created_at
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return True  # unparseable → treat as leftover
    if not isinstance(value, datetime):
        return True  # missing/unknown shape → treat as leftover
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return datetime.now(UTC) - value > _STALE_AFTER


async def cleanup_live_brains(storage: Any, own_brain_id: str | None = None) -> int:
    """Delete this run's test brain and stale leftovers, by record id.

    Args:
        storage: an initialized ``SurrealDBStorage``.
        own_brain_id: the brain id the calling fixture created this test —
            always deleted regardless of age.

    Returns the number of brain rows deleted.
    """
    rows = await storage._query("SELECT id, name, created_at FROM brain")
    deleted = 0
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        name = str(row.get("name") or "")
        raw = _record_id_raw(rid)
        bid = raw.replace("_", "-")
        if name in PROTECTED_BRAIN_NAMES or bid == "default":
            continue
        is_own = own_brain_id is not None and bid == own_brain_id
        if not is_own:
            if name not in LIVE_TEST_BRAIN_NAMES or not _is_stale(row.get("created_at")):
                continue
        # Data rows key on the dashed brain-id string; legacy rows may carry
        # the underscored form, so clear both spellings when they differ.
        await storage.clear(bid)
        if raw != bid:
            await storage.clear(raw)
        if hasattr(rid, "id"):
            await storage._query("DELETE $rid", rid=rid)
        else:
            await storage._query("DELETE type::thing('brain', $raw)", raw=raw)
        deleted += 1
    return deleted

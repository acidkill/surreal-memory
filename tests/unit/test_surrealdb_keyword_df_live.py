"""Regression: increment_keyword_df must actually write its UPSERT batch.

The batch builder interpolated the per-statement index into the SQL with a
plain (non-f) string literal, so every statement carried the four literal
characters ``$sid{i}`` instead of ``$sid0``/``$sid1``/... The bound parameters
were named correctly (``sid0``, ``sid1``, ...), so the query text referenced a
parameter that never existed and SurrealDB rejected the whole batch at parse
time. The only caller wraps the call in a bare ``except Exception: pass``
("non-critical: DF update failure doesn't block encoding"), so the failure was
completely silent: keyword_document_frequency simply stopped being written,
with no error surfaced anywhere.

That silence is why a mock-level assertion is not enough here — the pre-fix SQL
still contained the right number of ``UPSERT`` keywords and the right SET
clause, which is exactly what the existing mock test checked. These tests go to
a real SurrealDB and assert the rows land.

Skipped when SURREALDB_URL is unset so CI without docker still passes.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)

#: Deliberately more than two so statement indices 1 and 2 are exercised, not
#: just index 0. Plain alphanumerics keep ``_safe_id`` collision-free.
KEYWORDS = ["alpha", "bravo", "charlie", "delta"]


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="kw-df-batch-live-4b8d2e")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    try:
        await cleanup_live_brains(storage, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


async def test_increment_keyword_df_creates_rows_for_every_keyword(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """A multi-keyword batch must create a row per keyword, not raise and write none."""
    assert await surrealdb_storage.get_keyword_df_batch(KEYWORDS) == {}

    await surrealdb_storage.increment_keyword_df(KEYWORDS)

    df = await surrealdb_storage.get_keyword_df_batch(KEYWORDS)
    assert df == dict.fromkeys(KEYWORDS, 1)


async def test_increment_keyword_df_increments_existing_rows(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """The UPSERT's increment branch must run for every statement index too."""
    await surrealdb_storage.increment_keyword_df(KEYWORDS)
    await surrealdb_storage.increment_keyword_df(KEYWORDS)
    await surrealdb_storage.increment_keyword_df(KEYWORDS)

    df = await surrealdb_storage.get_keyword_df_batch(KEYWORDS)
    assert df == dict.fromkeys(KEYWORDS, 3)


async def test_increment_keyword_df_keeps_per_keyword_counts_independent(  # type: ignore[no-untyped-def]
    surrealdb_storage,
) -> None:
    """Each statement must target its own record id.

    A shared/duplicated id placeholder would funnel several keywords into one
    row; distinct counts per keyword prove the ids are really per-index.
    """
    await surrealdb_storage.increment_keyword_df(["alpha", "bravo", "charlie"])
    await surrealdb_storage.increment_keyword_df(["bravo", "charlie"])
    await surrealdb_storage.increment_keyword_df(["charlie"])

    df = await surrealdb_storage.get_keyword_df_batch(["alpha", "bravo", "charlie"])
    assert df == {"alpha": 1, "bravo": 2, "charlie": 3}


async def test_increment_keyword_df_updates_a_row_with_a_legacy_id_prefix(  # type: ignore[no-untyped-def]
    surrealdb_storage,
) -> None:
    """Regression for a residual defect the interpolation fix's own live
    verification surfaced: writing by a COMPUTED record id (rather than by
    content) collides with rows whose id still carries a prefix from a
    historical brain rename — 911 such rows exist on the live production
    brain, each with brain_id='default' but a record id like
    'my_brain.v2_<keyword>'. Matching by (brain_id, keyword) via WHERE, not by
    a synthetic id, must find and update that row regardless of what its id
    looks like — never attempt to CREATE a second row for the same pair, which
    the (brain_id, keyword) UNIQUE index would reject and which killed the
    rest of the batch non-atomically before this fix (see the pipeline's own
    `except Exception: pass` swallow point).
    """
    brain_id = surrealdb_storage._get_brain_id()
    legacy_keyword = "legacyprefixed"
    # Simulate the exact shape found live: brain_id field is current/canonical,
    # but the record id keeps an old prefix unrelated to how ids are minted
    # today.
    await surrealdb_storage._query(
        "CREATE type::record('keyword_document_frequency', $rid)"
        " SET brain_id = $bid, keyword = $kw, fiber_count = 5, last_updated = time::now();",
        rid=f"my_brain.v2_{legacy_keyword}",
        bid=brain_id,
        kw=legacy_keyword,
    )
    assert await surrealdb_storage.get_keyword_df_batch([legacy_keyword]) == {legacy_keyword: 5}

    # Increment must land on the EXISTING legacy row, not raise a UNIQUE-index
    # conflict trying to create a fresh one at a canonically-computed id.
    await surrealdb_storage.increment_keyword_df([legacy_keyword, "freshword"])

    df = await surrealdb_storage.get_keyword_df_batch([legacy_keyword, "freshword"])
    assert df == {legacy_keyword: 6, "freshword": 1}

    # Exactly one row for the legacy keyword — not a second one alongside it.
    rows = await surrealdb_storage._query(
        "SELECT count() FROM keyword_document_frequency"
        " WHERE brain_id = $bid AND keyword = $kw GROUP ALL",
        bid=brain_id,
        kw=legacy_keyword,
    )
    assert rows == [{"count": 1}]


async def test_pre_fix_placeholder_shape_is_rejected_at_statement_zero(  # type: ignore[no-untyped-def]
    surrealdb_storage,
) -> None:
    """Confirm the pre-fix breakage was total, not partial.

    It would be easy to assume the first statement still worked (its intended
    parameter name, ``$sid0``, differs from the emitted ``$sid{i}`` only in the
    suffix) and that only indices >= 1 were lost. Assert instead: a single
    statement carrying the literal-brace placeholder is rejected outright, so
    even a one-keyword batch wrote nothing. Nothing is left behind because the
    statement never executes.
    """
    with pytest.raises(Exception) as excinfo:
        await surrealdb_storage._query(
            "UPSERT type::record('keyword_document_frequency', $sid{i})"
            " SET keyword = $kw{i}, brain_id = $bid,"
            " fiber_count = (fiber_count ?? 0) + 1, last_updated = $now;",
            sid0="x",
            kw0="alpha",
            bid=surrealdb_storage._get_brain_id(),
            now="2026-01-01T00:00:00Z",
        )
    assert "{" in str(excinfo.value)

    # ... and the table is untouched: the parse failure aborted the batch.
    assert await surrealdb_storage.get_keyword_df_batch(["alpha"]) == {}

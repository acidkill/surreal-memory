"""SurrealDB schema migrations for Surreal-Memory.

The only migration today is **v7 -> v8**: the ``synapse`` table stops being a
flat table (``source_id`` / ``target_id`` string fields) and becomes a native
``RELATION`` edge (``in`` / ``out`` RecordID endpoints). Every existing synapse
id, ``fiber.synapse_ids`` reference, ``change_log`` entry and the Merkle root are
preserved because ``INSERT RELATION`` keeps the original edge ids.

Design mirrors ``sqlite_schema.MIGRATIONS`` — a ``{(from, to): callable}``
registry driven by :func:`apply_migrations`. Unlike the DDL-only SQLite
migrations, the 7->8 step moves data in resumable phases
(``copying`` -> ``converting`` -> ``verifying``) under an atomic
``schema_meta:migration_lock``. Progress is tracked in
``schema_meta:migration_state`` so a crash resumes from the last saved phase and
cursor. ``schema_meta:version`` is stamped to 8 only after verification passes,
so a partially-migrated DB never reads as "done".

Version detection (no ``schema_meta:version`` present) is structural, via
``INFO FOR DB``: a ``synapse`` table defined ``TYPE RELATION`` is already v8; a
plain (``TYPE NORMAL``) table is v7 and must be migrated; no ``synapse`` table at
all means a fresh DB that ``ensure_schema`` already created at v8.

All SurrealQL used here was verified against a live ``surrealdb/surrealdb:v3.2.0``
image (RUN-005 U1/U2): ``INSERT RELATION`` with ``RecordID`` params, atomic
``CREATE`` locks, ``INSERT IGNORE``, id-cursor paging, ``UPSERT`` and
``INFO FOR DB`` output shape.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.storage.surrealdb.connection import (
    MIN_SERVER_VERSION,
    parse_server_version,
)
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION, SYNAPSE_V8_DDL

logger = logging.getLogger(__name__)

TARGET_VERSION = SCHEMA_VERSION  # 10
SOURCE_VERSION = 7  # flat synapse table (pre 7->8 migration)
RELATION_SYNAPSE_VERSION = 8  # synapse became a RELATION table in the 7->8 migration
TYPED_VALIDITY_VERSION = 9  # TypedMemory validity fields + retrieval_trace table

BACKUP_TABLE = "synapse_migration_backup"
BATCH_SIZE = 500

# schema_meta singleton record ids
VERSION_ID = "schema_meta:version"
LOCK_ID = "schema_meta:migration_lock"
STATE_ID = "schema_meta:migration_state"

# Lock policy: poll every LOCK_POLL_INTERVAL s, give up after LOCK_WAIT_TIMEOUT s,
# steal a lock older than LOCK_STALE_DURATION (crashed holder) — steal is a DB-side
# conditional DELETE so it is atomic and needs no client-clock comparison.
LOCK_POLL_INTERVAL = 2.0
LOCK_WAIT_TIMEOUT = 120.0
LOCK_STALE_DURATION = "10m"

# migration phases (schema_meta:migration_state.phase)
PHASE_COPYING = "copying"
PHASE_CONVERTING = "converting"
PHASE_VERIFYING = "verifying"
PHASE_DONE = "done"

# Unique-ish holder tag for diagnostics (lock correctness relies on record
# existence, not this value).
_HOLDER = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"


class MigrationError(RuntimeError):
    """Raised when the synapse->RELATE migration cannot complete safely."""


class MigrationLockError(MigrationError):
    """Raised when the migration lock cannot be acquired and the DB is unmigrated."""


# --------------------------------------------------------------------------- #
# Low-level query helpers (mirror SurrealDBStorage._query result normalisation)
# --------------------------------------------------------------------------- #
async def _query(conn: Any, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run a query and normalise the result to a list of row dicts."""
    result = await conn.query(sql, params or {})
    if result and isinstance(result, list) and len(result) > 0:
        return result[0] if isinstance(result[0], list) else result
    return []


def _is_missing_table(exc: Exception) -> bool:
    """True if *exc* is SurrealDB's 'table does not exist' error (0 rows), not a
    connectivity/auth/query failure."""
    if type(exc).__name__ == "NotFoundError":
        return True
    msg = str(exc).lower()
    return "does not exist" in msg or "no such table" in msg


async def _count(conn: Any, table: str) -> int:
    try:
        rows = await _query(conn, f"SELECT count() FROM {table} GROUP ALL")
    except Exception as exc:
        # A non-existent table has 0 rows (fresh DB / doctor-status before any
        # migration). Only swallow that — real connection/auth/query errors must
        # propagate so verification doesn't misreport an infra blip as data loss.
        if _is_missing_table(exc):
            return 0
        raise
    if rows and isinstance(rows[0], dict):
        return int(rows[0].get("count", 0) or 0)
    return 0


def _already_exists(exc: Exception) -> bool:
    """True if *exc* is SurrealDB's 'record/table already exists' error."""
    if type(exc).__name__ == "AlreadyExistsError":
        return True
    return "already exists" in str(exc).lower()


def _record_part(rid: Any) -> str:
    """Extract the identifier part of a RecordID (or a ``table:id`` string)."""
    part = getattr(rid, "id", None)
    if part is not None:
        return str(part)
    text = str(rid)
    return text.split(":", 1)[1] if ":" in text else text


# --------------------------------------------------------------------------- #
# Version detection & stamping
# --------------------------------------------------------------------------- #
async def _read_stamped_version(conn: Any) -> int | None:
    try:
        rows = await _query(conn, f"SELECT version FROM {VERSION_ID}")
    except Exception:
        return None
    if rows and isinstance(rows[0], dict) and rows[0].get("version") is not None:
        return int(rows[0]["version"])
    return None


async def detect_db_version(conn: Any) -> int:
    """Return the schema version of the connected database.

    Priority order:
    1. An explicit ``schema_meta:version`` stamp wins (only written AFTER a
       migration verifies, so it is always trustworthy).
    2. An in-progress OR failed migration — a ``schema_meta:migration_state``
       record whose ``phase`` is not ``done`` — reports :data:`SOURCE_VERSION` even
       if the ``synapse`` table is already RELATION-shaped. The converting phase
       drops-and-redefines the table BEFORE verification, so a bare RELATION table
       is NOT proof the migration succeeded; it must be resumed/re-verified.
    3. Otherwise detect structurally from ``INFO FOR DB``: fresh DBs and
       already-RELATION synapse tables report :data:`TARGET_VERSION`; a flat
       (``TYPE NORMAL``) table reports :data:`SOURCE_VERSION`.
    """
    stamped = await _read_stamped_version(conn)
    if stamped is not None:
        return stamped

    state = await _get_state(conn)
    if state is not None and state.get("phase") != PHASE_DONE:
        # A migration was started and has not verified — do not trust the table
        # shape; force a resume so verification runs (and keeps failing loudly on
        # genuine data loss instead of silently reporting success).
        return SOURCE_VERSION

    info = await conn.query("INFO FOR DB")
    tables: dict[str, Any] = {}
    if isinstance(info, dict):
        tables = info.get("tables") or {}
    elif isinstance(info, list) and info and isinstance(info[0], dict):
        tables = info[0].get("tables") or {}

    synapse_def = tables.get("synapse", "") if isinstance(tables, dict) else ""
    has_trace = isinstance(tables, dict) and "retrieval_trace" in tables
    if not synapse_def:
        return TARGET_VERSION  # fresh DB — ensure_schema already built the latest schema
    if "TYPE RELATION" in str(synapse_def):
        # Relation synapse = v8 or newer. The v9-only retrieval_trace table tells a
        # v9 DB apart from a v8 DB that still needs the additive 8->9 migration.
        return TARGET_VERSION if has_trace else RELATION_SYNAPSE_VERSION
    return SOURCE_VERSION  # flat/NORMAL table — needs the 7->8 migration


async def _stamp_version(conn: Any, version: int) -> None:
    await conn.query(f"UPSERT {VERSION_ID} SET version = $v", {"v": version})


# --------------------------------------------------------------------------- #
# Migration lock (atomic CREATE; DB-side stale steal)
# --------------------------------------------------------------------------- #
async def _try_create_lock(conn: Any) -> bool:
    try:
        await conn.query(
            f"CREATE {LOCK_ID} SET holder = $h, acquired_at = time::now()",
            {"h": _HOLDER},
        )
        return True
    except Exception as exc:
        if _already_exists(exc):
            return False
        raise


async def _acquire_lock(conn: Any) -> bool:
    """Acquire the migration lock, waiting/stealing as needed.

    Returns True once held. Returns False if another live holder keeps the lock
    for longer than :data:`LOCK_WAIT_TIMEOUT`.
    """
    start = time.monotonic()
    while True:
        if await _try_create_lock(conn):
            return True
        # Held by someone else — steal it atomically if it is stale (crashed
        # holder), then try to re-create immediately.
        await conn.query(
            f"DELETE {LOCK_ID} WHERE acquired_at < time::now() - {LOCK_STALE_DURATION}"
        )
        if await _try_create_lock(conn):
            return True
        if time.monotonic() - start > LOCK_WAIT_TIMEOUT:
            return False
        await asyncio.sleep(LOCK_POLL_INTERVAL)


async def _release_lock(conn: Any) -> None:
    try:
        await conn.query(f"DELETE {LOCK_ID}")
    except Exception as exc:
        # Swallowed on purpose: this runs in apply_migrations' finally, so raising
        # would mask the real migration error. The 10-min stale-steal self-heals a
        # lock left behind here. Log the cause so a stuck lock is diagnosable.
        logger.warning("Failed to release migration lock (%s): %s", LOCK_ID, exc)


# --------------------------------------------------------------------------- #
# Migration state (crash-resume)
# --------------------------------------------------------------------------- #
async def _get_state(conn: Any) -> dict[str, Any] | None:
    try:
        rows = await _query(conn, f"SELECT * FROM {STATE_ID}")
    except Exception:
        return None
    if rows and isinstance(rows[0], dict):
        return dict(rows[0])
    return None


async def _save_state(conn: Any, state: dict[str, Any]) -> None:
    payload = {k: v for k, v in state.items() if k != "id"}
    await conn.query(f"UPSERT {STATE_ID} CONTENT $c", {"c": payload})


def _cursor_record(cursor: Any, table: str) -> Any | None:
    """Rebuild a RecordID cursor from its stored string form (or None)."""
    if not cursor:
        return None
    from surrealdb import RecordID  # lazy: SDK not installed in CI unit env

    part = str(cursor).split(":", 1)[1] if ":" in str(cursor) else str(cursor)
    return RecordID(table, part)


async def _page(conn: Any, table: str, after: Any | None, batch: int) -> list[dict[str, Any]]:
    if after is None:
        return await _query(conn, f"SELECT * FROM {table} ORDER BY id LIMIT {batch}")
    return await _query(
        conn,
        f"SELECT * FROM {table} WHERE id > $after ORDER BY id LIMIT {batch}",
        {"after": after},
    )


# --------------------------------------------------------------------------- #
# Version gate (defensive; store.initialize() also gates before ensure_schema)
# --------------------------------------------------------------------------- #
async def _check_server_version(conn: Any) -> None:
    try:
        raw = await conn.version()
    except Exception:
        logger.warning("Could not read SurrealDB version before migration; proceeding.")
        return
    parsed = parse_server_version(str(raw))
    if parsed is not None and parsed < MIN_SERVER_VERSION:
        raise MigrationError(
            "synapse->RELATE migration requires SurrealDB >= 3.2.0 but the server reports "
            f"{raw}. Upgrade the image (docker compose -f docker-compose.surrealdb.yml pull "
            "&& up -d — the surrealdb_data volume is preserved; back it up first) and retry."
        )


# --------------------------------------------------------------------------- #
# Row transforms
# --------------------------------------------------------------------------- #
def _backup_row(row: dict[str, Any]) -> dict[str, Any]:
    """Copy a v7 synapse row into a backup row keyed by the same id part."""
    from surrealdb import RecordID  # lazy

    out = dict(row)
    out["id"] = RecordID(BACKUP_TABLE, _record_part(row["id"]))
    return out


def _to_relation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build an ``INSERT RELATION`` row from a backed-up v7 synapse row."""
    from surrealdb import RecordID  # lazy

    sid = _record_part(row["id"])
    src = _to_surreal_id(str(row.get("source_id", "")))
    tgt = _to_surreal_id(str(row.get("target_id", "")))
    relation: dict[str, Any] = {
        "id": RecordID("synapse", sid),
        "in": RecordID("neuron", src),
        "out": RecordID("neuron", tgt),
        "brain_id": row.get("brain_id", "default"),
        "type": row.get("type"),
        "weight": row.get("weight", 1.0),
        "direction": row.get("direction", "forward"),
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
        "reinforced_count": row.get("reinforced_count", 0),
    }
    last_activated = row.get("last_activated")
    if last_activated is not None:
        relation["last_activated"] = last_activated
    return relation


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
async def _phase_copying(conn: Any, state: dict[str, Any]) -> None:
    """Faithfully copy every v7 synapse row into the backup table (idempotent)."""
    after = _cursor_record(state.get("cursor"), "synapse")
    while True:
        rows = await _page(conn, "synapse", after, BATCH_SIZE)
        if not rows:
            break
        payload = [_backup_row(r) for r in rows]
        await conn.query(f"INSERT IGNORE INTO {BACKUP_TABLE} $rows", {"rows": payload})
        state["copied"] = int(state.get("copied", 0)) + len(rows)
        after = rows[-1]["id"]
        state["cursor"] = str(after)
        await _save_state(conn, state)

    backup_count = await _count(conn, BACKUP_TABLE)
    if backup_count < int(state.get("old_count", 0)):
        # Leave phase=copying (do NOT advance) so a retry re-copies; raise loudly.
        raise MigrationError(
            f"backup incomplete: {backup_count} rows backed up < {state['old_count']} originals"
        )
    state["phase"] = PHASE_CONVERTING
    state["cursor"] = None
    await _save_state(conn, state)


async def _phase_converting(conn: Any, state: dict[str, Any]) -> None:
    """Drop the flat table, (re)define the RELATION table, replay backup as edges.

    Rebuilds the RELATION table from the COMPLETE backup on every entry — including
    on a resume. Two reasons this full re-scan is correct and necessary:

    * INSERT RELATION uses explicit ids, so re-inserting is idempotent (the table is
      dropped first, so there is nothing to clash with anyway).
    * Resuming from a saved cursor would permanently lose any row that a buggy
      earlier build SKIPPED *behind* that cursor (e.g. every synapse with non-empty
      ``metadata`` before the FLEXIBLE fix). Re-scanning from the start, against the
      freshly-redefined v8 schema, is what makes those rows recoverable.
    """
    await conn.query("REMOVE TABLE IF EXISTS synapse")
    for ddl in SYNAPSE_V8_DDL:
        try:
            await conn.query(ddl)
        except Exception as exc:
            # Tolerate only "already exists" (idempotent re-run). A real DDL
            # error (syntax/permissions) must abort — inserting into a
            # half-defined table would corrupt silently.
            if not _already_exists(exc):
                raise

    # Fresh full pass over the backup (the source of truth); reset counters so
    # `verifying` compares against a clean, complete conversion.
    state["converted"] = 0
    state["skipped"] = 0
    state["cursor"] = None
    await _save_state(conn, state)

    after: Any | None = None
    while True:
        rows = await _page(conn, BACKUP_TABLE, after, BATCH_SIZE)
        if not rows:
            break
        relation_rows = [_to_relation_row(r) for r in rows]
        try:
            await conn.query("INSERT RELATION INTO synapse $rows", {"rows": relation_rows})
        except Exception:
            for rr in relation_rows:
                try:
                    await conn.query("INSERT RELATION INTO synapse $row", {"row": rr})
                except Exception as exc:
                    if _already_exists(exc):
                        continue  # already inserted on a prior run — resume
                    state["skipped"] = int(state.get("skipped", 0)) + 1
                    logger.error(
                        "synapse->RELATE: SKIPPED edge %s (data loss for this row): %s",
                        rr.get("id"),
                        exc,
                    )
        state["converted"] = int(state.get("converted", 0)) + len(rows)
        after = rows[-1]["id"]
        state["cursor"] = str(after)
        await _save_state(conn, state)

    state["phase"] = PHASE_VERIFYING
    state["cursor"] = None
    await _save_state(conn, state)


async def _phase_verifying(conn: Any, state: dict[str, Any]) -> None:
    """Verify edge count, then stamp version 8. Backup is kept for rollback.

    On failure the phase stays ``verifying`` (never advances to ``done`` and never
    stamps the version), so ``detect_db_version`` keeps reporting the DB as
    unmigrated and every subsequent startup re-verifies and re-raises — a genuine
    data-loss failure surfaces loudly instead of being silently accepted.
    """
    edge_count = await _count(conn, "synapse")
    old_count = int(state.get("old_count", 0))
    skipped = int(state.get("skipped", 0))
    expected = old_count - skipped

    if skipped > 0:
        logger.error(
            "synapse->RELATE: %s of %s edges were SKIPPED (data loss). Backup table '%s' retains "
            "the originals for recovery.",
            skipped,
            old_count,
            BACKUP_TABLE,
        )

    # Total-loss guard: a systematic conversion failure could skip every row and
    # still satisfy ``edge_count >= old_count - skipped`` (0 >= 0). Reject it.
    if old_count > 0 and edge_count == 0:
        raise MigrationError(
            f"verification failed: 0 edges migrated from {old_count} originals "
            f"(skipped={skipped}). Originals preserved in '{BACKUP_TABLE}'."
        )
    if edge_count < expected:
        raise MigrationError(
            f"verification failed: {edge_count} migrated edges < expected {expected} "
            f"(old_count={old_count}, skipped={skipped}). Originals preserved in '{BACKUP_TABLE}'."
        )

    await _stamp_version(conn, RELATION_SYNAPSE_VERSION)
    state["phase"] = PHASE_DONE
    await _save_state(conn, state)
    logger.info(
        "synapse->RELATE migration complete: %s edges (skipped=%s). Backup table '%s' retained "
        "for rollback (clean up with `smem doctor --synapse-migration purge-backup`).",
        edge_count,
        skipped,
        BACKUP_TABLE,
    )


async def _migrate_7_to_8(conn: Any) -> None:
    """Convert the flat ``synapse`` table into native RELATION edges (resumable).

    State is created ONCE (first invocation, no prior state) and then only
    resumed — never reset — so ``old_count`` is captured against the ORIGINAL
    flat table and stays authoritative across crashes/failures. A resume after a
    failure re-enters the exact phase it left off in (reading originals from the
    backup table once converting has started), so it can never recompute
    ``old_count`` from the already-converted RELATION table.
    """
    await _check_server_version(conn)

    state = await _get_state(conn)
    if state is None:
        old_count = await _count(conn, "synapse")
        state = {
            "phase": PHASE_COPYING,
            "cursor": None,
            "old_count": old_count,
            "copied": 0,
            "converted": 0,
            "skipped": 0,
        }
        await _save_state(conn, state)

    if state["phase"] == PHASE_COPYING:
        await _phase_copying(conn, state)
    if state["phase"] == PHASE_CONVERTING:
        await _phase_converting(conn, state)
    if state["phase"] == PHASE_VERIFYING:
        await _phase_verifying(conn, state)


# Additive schema-v9 DDL (validity fields, source.trust, retrieval_trace table),
# as individual statements. Applied one-by-one (like schema.ensure_schema) so a
# re-run over already-defined fields is tolerated instead of aborting the batch:
# SurrealDB raises "already exists" on a bare re-DEFINE, and a single multi-statement
# query would fail the whole batch on the first such field. Mirrors the statements in
# schema.SCHEMA_SQL so a fresh ensure_schema build and a migrated DB converge.
_V9_DDL: tuple[str, ...] = (
    "DEFINE FIELD valid_from    ON typed_memory TYPE option<datetime>",
    "DEFINE FIELD valid_until   ON typed_memory TYPE option<datetime>",
    "DEFINE FIELD superseded_by ON typed_memory TYPE option<string>",
    "DEFINE INDEX idx_typed_valid   ON typed_memory FIELDS brain_id, valid_until",
    "DEFINE INDEX idx_typed_expires ON typed_memory FIELDS brain_id, expires_at",
    "DEFINE FIELD trust ON source TYPE option<float>",
    "DEFINE TABLE retrieval_trace SCHEMAFULL",
    "DEFINE FIELD id           ON retrieval_trace TYPE string",
    "DEFINE FIELD brain_id     ON retrieval_trace TYPE string",
    "DEFINE FIELD session_id   ON retrieval_trace TYPE option<string>",
    "DEFINE FIELD query        ON retrieval_trace TYPE string DEFAULT ''",
    "DEFINE FIELD depth_used   ON retrieval_trace TYPE int DEFAULT 0",
    "DEFINE FIELD mode         ON retrieval_trace TYPE string DEFAULT ''",
    "DEFINE FIELD confidence   ON retrieval_trace TYPE float DEFAULT 0.0",
    "DEFINE FIELD latency_ms   ON retrieval_trace TYPE float DEFAULT 0.0",
    "DEFINE FIELD fiber_ids    ON retrieval_trace TYPE array<string> DEFAULT []",
    "DEFINE FIELD payload      ON retrieval_trace TYPE object FLEXIBLE DEFAULT {}",
    "DEFINE FIELD created_at   ON retrieval_trace TYPE datetime DEFAULT time::now()",
    "DEFINE INDEX idx_trace_brain   ON retrieval_trace FIELDS brain_id",
    "DEFINE INDEX idx_trace_created ON retrieval_trace FIELDS brain_id, created_at",
)


async def _migrate_8_to_9(conn: Any) -> None:
    """Additive schema-v9 migration.

    Adds TypedMemory validity fields (+ indexes), Source.trust, and the
    retrieval_trace table. Pure idempotent DDL — no data movement, no resumable
    state (unlike 7->8). Statements run individually and an "already exists"
    re-DEFINE is skipped, so the migration is safe to run twice. Stamps v9 last.
    """
    for stmt in _V9_DDL:
        try:
            await conn.query(stmt + ";")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                logger.debug("v9 migration statement skipped (already exists): %s", stmt[:80])
            else:
                # Fail loudly on a genuine DDL error (stricter than the tolerant
                # ensure_schema): swallowing it would stamp v9 over an incomplete
                # schema. Only the idempotent "already exists" re-DEFINE is skipped.
                logger.error("v9 migration statement failed: %s (%s)", stmt[:80], exc)
                raise
    # Stamp 9 explicitly, not TARGET_VERSION: this step brings a database to v9
    # and nothing further. Stamping the moving target claimed every later schema
    # version without running its DDL.
    await _stamp_version(conn, TYPED_VALIDITY_VERSION)


_V10_DDL = ("DEFINE INDEX idx_changelog_synced ON change_log FIELDS brain_id, synced",)


async def _migrate_9_to_10(conn: Any) -> None:
    """Additive schema-v10 migration: index change_log on (brain_id, synced).

    Every count filtered on ``synced`` previously degraded to a full read of the
    brain's change_log, because the planner had to fetch the field from each row.
    Measured on a large log: the same count was well over an order of magnitude slower without this index, which is the difference between the dashboard's sync card
    answering and appearing to hang.

    Pure idempotent DDL, no data movement. Building the index does scan the
    existing table once, so on a large neglected log this migration is not
    instant -- it is still bounded, one-off, and far cheaper than paying the full
    read on every dashboard load. Stamps v10 last.
    """
    for stmt in _V10_DDL:
        try:
            await conn.query(stmt + ";")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                logger.debug("v10 migration statement skipped (already exists): %s", stmt[:80])
            else:
                logger.error("v10 migration statement failed: %s (%s)", stmt[:80], exc)
                raise
    await _stamp_version(conn, TARGET_VERSION)


# Registry mirrors sqlite_schema.MIGRATIONS: {(from, to): migrate_callable}.
MIGRATIONS = {
    (SOURCE_VERSION, RELATION_SYNAPSE_VERSION): _migrate_7_to_8,  # (7, 8)
    # Pinned to explicit version constants, NOT to TARGET_VERSION: this entry
    # used to read (RELATION_SYNAPSE_VERSION, TARGET_VERSION), so bumping the
    # schema silently re-registered the 8->9 migration as 8->10 and left 9->10
    # missing entirely. A v8 database would then have jumped straight to a v10
    # stamp without ever running the v10 DDL.
    (RELATION_SYNAPSE_VERSION, TYPED_VALIDITY_VERSION): _migrate_8_to_9,  # (8, 9)
    (TYPED_VALIDITY_VERSION, TARGET_VERSION): _migrate_9_to_10,  # (9, 10)
}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def apply_migrations(conn: Any) -> int:
    """Bring the connected database up to :data:`TARGET_VERSION`.

    Intended to be wired into ``store.initialize()`` right after ``ensure_schema()``
    (RUN-005 U4 does that wiring + the pre-schema version gate). Fresh and
    already-migrated DBs are a cheap no-op (version stamp only). A DB needing the
    7->8 migration runs it once under the migration lock; concurrent callers wait
    (or observe the completed migration and return).
    """
    current = await detect_db_version(conn)
    if current >= TARGET_VERSION:
        await _stamp_version(conn, TARGET_VERSION)
        return TARGET_VERSION

    acquired = await _acquire_lock(conn)
    if not acquired:
        # Another live holder kept the lock. If it finished, we are done;
        # otherwise surface the contention rather than migrate twice.
        current = await detect_db_version(conn)
        if current >= TARGET_VERSION:
            return TARGET_VERSION
        raise MigrationLockError(
            "Could not acquire the synapse migration lock and the database is not yet migrated. "
            "Another process may be migrating; retry shortly."
        )

    try:
        # Double-checked locking: another process may have finished between the
        # first detect and acquiring the lock.
        current = await detect_db_version(conn)
        if current >= TARGET_VERSION:
            await _stamp_version(conn, TARGET_VERSION)
            return TARGET_VERSION

        version = current
        while version < TARGET_VERSION:
            migrate = MIGRATIONS.get((version, version + 1))
            if migrate is None:
                raise MigrationError(f"no migration registered for {version} -> {version + 1}")
            await migrate(conn)
            version += 1
        return version
    finally:
        await _release_lock(conn)

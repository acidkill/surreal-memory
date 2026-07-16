"""Unit tests for the synapse -> RELATE migration (storage/surrealdb/migrations.py).

These are pure-logic tests: the surrealdb SDK is stubbed via sys.modules (repo
convention, see test_surrealdb_store.py) and the connection is a scripted fake
that records every query and returns programmed results. Live-DB behaviour is
covered by the integration test (RUN-005 U6, real-db-test-runner).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Stub the optional surrealdb SDK so the lazy `from surrealdb import RecordID`
# inside migrations.py resolves. RecordID becomes a MagicMock factory — fine, the
# tests assert on query text/state transitions, not RecordID internals.
# Stub ONLY when the SDK is genuinely not installed: an `if not in sys.modules`
# guard would shadow an installed SDK for the rest of the pytest session and
# break the live (SURREALDB_URL) tests that run after this module.
try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()

from surreal_memory.storage.surrealdb import migrations as M  # noqa: N812


class AlreadyExistsError(Exception):
    """Mimics surrealdb.errors.AlreadyExistsError (matched by class name)."""


class ScriptedConn:
    """Async fake connection: records queries, routes results by SQL substring.

    Routes are checked in registration order; first match wins. A route value may
    be a plain result (list/dict) or a callable ``(sql, params) -> result``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.routes: list[tuple[str, object]] = []
        self.version_str: str = "surrealdb-3.2.0"
        self.version_exc: Exception | None = None

    def route(self, substr: str, result: object) -> ScriptedConn:
        self.routes.append((substr, result))
        return self

    async def query(self, sql: str, params: dict | None = None):
        self.calls.append((sql, dict(params or {})))
        for substr, result in self.routes:
            if substr in sql:
                return result(sql, dict(params or {})) if callable(result) else result
        return []

    async def version(self):
        if self.version_exc is not None:
            raise self.version_exc
        return self.version_str

    # test helpers ------------------------------------------------------------
    def sqls(self) -> list[str]:
        return [c[0] for c in self.calls]

    def first_index(self, substr: str) -> int:
        for i, (sql, _) in enumerate(self.calls):
            if substr in sql:
                return i
        return -1


def _rid(table: str, ident: str):
    """A minimal RecordID stand-in whose str() is 'table:ident' and .id is ident."""

    class _R:
        def __init__(self, t, i):
            self.table_name = t
            self.id = i

        def __str__(self):
            return f"{self.table_name}:{self.id}"

    return _R(table, ident)


def _pager(pages: list[list[dict]]):
    """Return a callable yielding successive pages, then [] forever."""
    box = {"i": 0}

    def _fn(sql, params):
        i = box["i"]
        box["i"] += 1
        return pages[i] if i < len(pages) else []

    return _fn


# --------------------------------------------------------------------------- #
# detect_db_version
# --------------------------------------------------------------------------- #
class TestDetectVersion:
    @pytest.mark.asyncio
    async def test_stamped_version_wins(self):
        conn = ScriptedConn().route("SELECT version FROM schema_meta:version", [{"version": 8}])
        assert await M.detect_db_version(conn) == 8

    @pytest.mark.asyncio
    async def test_relation_table_without_trace_is_v8(self):
        # Relation synapse but no v9-only retrieval_trace table -> still needs the 8->9 migration.
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route(
            "INFO FOR DB",
            {"tables": {"synapse": "DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron"}},
        )
        assert await M.detect_db_version(conn) == M.RELATION_SYNAPSE_VERSION

    @pytest.mark.asyncio
    async def test_relation_table_with_trace_is_v9(self):
        # Relation synapse + retrieval_trace table => fully at TARGET_VERSION (v9).
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route(
            "INFO FOR DB",
            {
                "tables": {
                    "synapse": "DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron",
                    "retrieval_trace": "DEFINE TABLE retrieval_trace SCHEMAFULL",
                }
            },
        )
        assert await M.detect_db_version(conn) == M.TARGET_VERSION

    @pytest.mark.asyncio
    async def test_flat_table_is_v7(self):
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route("INFO FOR DB", {"tables": {"synapse": "DEFINE TABLE synapse TYPE NORMAL"}})
        assert await M.detect_db_version(conn) == M.SOURCE_VERSION

    @pytest.mark.asyncio
    async def test_fresh_db_is_v8(self):
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route("INFO FOR DB", {"tables": {}})
        assert await M.detect_db_version(conn) == M.TARGET_VERSION


# --------------------------------------------------------------------------- #
# _migrate_7_to_8 — full happy path + statement ordering
# --------------------------------------------------------------------------- #
def _happy_conn(monkeypatch, old_count: int = 2) -> ScriptedConn:
    rows = [
        {
            "id": _rid("synapse", f"e{i}"),
            "source_id": f"n{i}",
            "target_id": f"m{i}",
            "brain_id": "default",
            "type": "associative",
            "weight": 1.0,
        }
        for i in range(old_count)
    ]
    backup_rows = [
        {
            "id": _rid(M.BACKUP_TABLE, f"e{i}"),
            "source_id": f"n{i}",
            "target_id": f"m{i}",
            "brain_id": "default",
            "type": "associative",
            "weight": 1.0,
        }
        for i in range(old_count)
    ]
    conn = ScriptedConn()
    conn.route("SELECT * FROM schema_meta:migration_state", [])  # no prior state
    conn.route("count() FROM synapse_migration_backup", [{"count": old_count}])
    conn.route("count() FROM synapse GROUP", [{"count": old_count}])
    conn.route("* FROM synapse_migration_backup ORDER", _pager([backup_rows]))
    conn.route("* FROM synapse_migration_backup WHERE", [])
    conn.route("* FROM synapse ORDER", _pager([rows]))
    conn.route("* FROM synapse WHERE", [])
    return conn


class TestMigrateHappyPath:
    @pytest.mark.asyncio
    async def test_full_migration_stamps_version_8(self, monkeypatch):
        conn = _happy_conn(monkeypatch)
        await M._migrate_7_to_8(conn)
        assert any("UPSERT schema_meta:version SET version" in s for s in conn.sqls())
        # phase reached 'done'
        done_saves = [
            p
            for s, p in conn.calls
            if "UPSERT schema_meta:migration_state" in s and p.get("c", {}).get("phase") == "done"
        ]
        assert done_saves, "final state should be phase=done"

    @pytest.mark.asyncio
    async def test_backup_happens_before_remove_table(self, monkeypatch):
        """CRITICAL safety invariant: the flat table is backed up BEFORE it is dropped."""
        conn = _happy_conn(monkeypatch)
        await M._migrate_7_to_8(conn)
        backup_idx = conn.first_index("INSERT IGNORE INTO synapse_migration_backup")
        remove_idx = conn.first_index("REMOVE TABLE IF EXISTS synapse")
        assert backup_idx != -1 and remove_idx != -1
        assert backup_idx < remove_idx, "backup must precede REMOVE TABLE synapse"

    @pytest.mark.asyncio
    async def test_insert_relation_used_not_plain_insert(self, monkeypatch):
        conn = _happy_conn(monkeypatch)
        await M._migrate_7_to_8(conn)
        assert any("INSERT RELATION INTO synapse" in s for s in conn.sqls())


# --------------------------------------------------------------------------- #
# Version gate
# --------------------------------------------------------------------------- #
class TestVersionGate:
    @pytest.mark.asyncio
    async def test_old_server_raises(self):
        conn = ScriptedConn()
        conn.version_str = "surrealdb-3.1.1"
        with pytest.raises(M.MigrationError) as exc:
            await M._check_server_version(conn)
        assert "3.2.0" in str(exc.value)

    @pytest.mark.asyncio
    async def test_current_server_passes(self):
        conn = ScriptedConn()
        conn.version_str = "surrealdb-3.2.0"
        await M._check_server_version(conn)  # no raise

    @pytest.mark.asyncio
    async def test_unparsable_version_warns_and_continues(self):
        conn = ScriptedConn()
        conn.version_str = "weird-build"
        await M._check_server_version(conn)  # no raise

    @pytest.mark.asyncio
    async def test_failed_probe_warns_and_continues(self):
        conn = ScriptedConn()
        conn.version_exc = RuntimeError("no version endpoint")
        await M._check_server_version(conn)  # no raise

    def test_parse_version_strips_prefix(self):
        from surreal_memory.storage.surrealdb.connection import parse_server_version

        assert parse_server_version("surrealdb-3.2.0") == (3, 2, 0)
        assert parse_server_version("3.2.10") == (3, 2, 10)
        assert parse_server_version("nope") is None


# --------------------------------------------------------------------------- #
# Lock acquisition: wait + stale steal
# --------------------------------------------------------------------------- #
class TestLock:
    @pytest.mark.asyncio
    async def test_acquire_waits_then_succeeds(self, monkeypatch):
        slept: list[float] = []

        async def fake_sleep(sec):
            slept.append(sec)

        monkeypatch.setattr(M.asyncio, "sleep", fake_sleep)

        state = {"create_calls": 0}

        def create_lock(sql, params):
            state["create_calls"] += 1
            # first CREATE (and the immediate post-steal retry) fail; later succeed
            if state["create_calls"] <= 2:
                raise AlreadyExistsError(
                    "Database record `schema_meta:migration_lock` already exists"
                )
            return []

        conn = ScriptedConn()
        conn.route("CREATE schema_meta:migration_lock", create_lock)
        conn.route("DELETE schema_meta:migration_lock WHERE acquired_at", [])  # steal no-op

        assert await M._acquire_lock(conn) is True
        assert slept, "should have polled at least once"

    @pytest.mark.asyncio
    async def test_stale_lock_is_stolen_without_waiting(self, monkeypatch):
        async def fake_sleep(sec):  # should not be reached
            raise AssertionError("stale steal must not wait")

        monkeypatch.setattr(M.asyncio, "sleep", fake_sleep)

        state = {"create_calls": 0}

        def create_lock(sql, params):
            state["create_calls"] += 1
            if state["create_calls"] == 1:
                raise AlreadyExistsError("already exists")  # held
            return []  # after steal, second create succeeds

        conn = ScriptedConn()
        conn.route("CREATE schema_meta:migration_lock", create_lock)
        conn.route("DELETE schema_meta:migration_lock WHERE acquired_at", [])

        assert await M._acquire_lock(conn) is True
        # the stale-steal DELETE must have been issued
        assert any("DELETE schema_meta:migration_lock WHERE acquired_at" in s for s in conn.sqls())

    @pytest.mark.asyncio
    async def test_acquire_times_out_when_lock_never_frees(self, monkeypatch):
        async def fast_sleep(sec):
            return None

        monkeypatch.setattr(M.asyncio, "sleep", fast_sleep)
        # freeze the clock so the timeout branch is reached after one poll
        ticks = iter([0.0, 1000.0, 2000.0])
        monkeypatch.setattr(M.time, "monotonic", lambda: next(ticks, 3000.0))

        def always_held(sql, params):
            raise AlreadyExistsError("already exists")

        conn = ScriptedConn()
        conn.route("CREATE schema_meta:migration_lock", always_held)
        conn.route("DELETE schema_meta:migration_lock WHERE acquired_at", [])

        assert await M._acquire_lock(conn) is False


# --------------------------------------------------------------------------- #
# Resume from each phase
# --------------------------------------------------------------------------- #
class TestResume:
    @pytest.mark.asyncio
    async def test_resume_from_converting_rebuilds_without_recopy(self, monkeypatch):
        """A resume mid-converting must NOT re-copy, but MUST rebuild the RELATION
        table from the complete backup (dropping any partial/buggy conversion) so
        rows a prior attempt skipped behind its cursor are recovered."""
        backup_rows = [
            {
                "id": _rid(M.BACKUP_TABLE, "e0"),
                "source_id": "n0",
                "target_id": "m0",
                "brain_id": "default",
                "type": "associative",
                "weight": 1.0,
            }
        ]
        conn = ScriptedConn()
        conn.route(
            "SELECT * FROM schema_meta:migration_state",
            [
                {
                    "phase": "converting",
                    "cursor": "synapse_migration_backup:seen",
                    "old_count": 1,
                    "copied": 1,
                    "converted": 0,
                    "skipped": 0,
                }
            ],
        )
        conn.route("count() FROM synapse GROUP", [{"count": 1}])
        conn.route("* FROM synapse_migration_backup WHERE", _pager([backup_rows]))
        conn.route("* FROM synapse_migration_backup ORDER", _pager([backup_rows]))

        await M._migrate_7_to_8(conn)
        sqls = conn.sqls()
        assert not any("INSERT IGNORE INTO synapse_migration_backup" in s for s in sqls), (
            "must not re-copy"
        )
        assert any("REMOVE TABLE IF EXISTS synapse" in s for s in sqls), (
            "resume must rebuild the RELATION table from the complete backup"
        )
        assert any("INSERT RELATION INTO synapse" in s for s in sqls)
        assert any("UPSERT schema_meta:version SET version" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_resume_from_verifying_only_verifies_and_stamps(self, monkeypatch):
        conn = ScriptedConn()
        conn.route(
            "SELECT * FROM schema_meta:migration_state",
            [
                {
                    "phase": "verifying",
                    "cursor": None,
                    "old_count": 3,
                    "copied": 3,
                    "converted": 3,
                    "skipped": 0,
                }
            ],
        )
        conn.route("count() FROM synapse GROUP", [{"count": 3}])

        await M._migrate_7_to_8(conn)
        sqls = conn.sqls()
        assert not any("REMOVE TABLE" in s for s in sqls)
        assert not any("INSERT RELATION" in s for s in sqls)
        assert any("UPSERT schema_meta:version SET version" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_verify_failure_raises_without_stamping_version(self, monkeypatch):
        conn = ScriptedConn()
        conn.route(
            "SELECT * FROM schema_meta:migration_state",
            [
                {
                    "phase": "verifying",
                    "cursor": None,
                    "old_count": 5,
                    "copied": 5,
                    "converted": 5,
                    "skipped": 0,
                }
            ],
        )
        conn.route("count() FROM synapse GROUP", [{"count": 2}])  # lost edges → fail

        with pytest.raises(M.MigrationError):
            await M._migrate_7_to_8(conn)
        # version must NOT be stamped and phase must stay resumable (never 'done')
        assert not any("UPSERT schema_meta:version" in s for s in conn.sqls())
        done = [
            p
            for s, p in conn.calls
            if "UPSERT schema_meta:migration_state" in s and p.get("c", {}).get("phase") == "done"
        ]
        assert not done, "failed verification must never mark phase=done"


# --------------------------------------------------------------------------- #
# apply_migrations orchestration (fresh / already-migrated / lock)
# --------------------------------------------------------------------------- #
class TestApplyMigrations:
    @pytest.mark.asyncio
    async def test_fresh_db_stamps_and_skips_migration(self, monkeypatch):
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route("INFO FOR DB", {"tables": {}})  # fresh → detect 8

        result = await M.apply_migrations(conn)
        assert result == M.TARGET_VERSION
        sqls = conn.sqls()
        assert not any("REMOVE TABLE" in s for s in sqls)
        assert not any("INSERT IGNORE INTO synapse_migration_backup" in s for s in sqls)
        assert any("UPSERT schema_meta:version SET version" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_already_migrated_is_noop(self, monkeypatch):
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [{"version": M.TARGET_VERSION}])
        result = await M.apply_migrations(conn)
        assert result == M.TARGET_VERSION
        assert not any("REMOVE TABLE" in s for s in conn.sqls())

    @pytest.mark.asyncio
    async def test_lock_contention_but_peer_finished_returns_noop(self, monkeypatch):
        """Second concurrent caller: can't get lock, but peer already migrated → return 8."""
        versions = iter([[], [{"version": M.TARGET_VERSION}]])  # 1st: unmigrated; 2nd: migrated

        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", lambda s, p: next(versions, []))
        conn.route("INFO FOR DB", {"tables": {"synapse": "DEFINE TABLE synapse TYPE NORMAL"}})
        conn.route(
            "CREATE schema_meta:migration_lock",
            lambda s, p: (_ for _ in ()).throw(AlreadyExistsError("already exists")),
        )
        conn.route("DELETE schema_meta:migration_lock WHERE acquired_at", [])

        # make the lock give up fast
        monkeypatch.setattr(M, "LOCK_WAIT_TIMEOUT", -1.0)

        async def no_sleep(sec):
            return None

        monkeypatch.setattr(M.asyncio, "sleep", no_sleep)

        result = await M.apply_migrations(conn)
        assert result == M.TARGET_VERSION

    @pytest.mark.asyncio
    async def test_lock_released_after_migration(self, monkeypatch):
        conn = _happy_conn(monkeypatch)
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route("INFO FOR DB", {"tables": {"synapse": "DEFINE TABLE synapse TYPE NORMAL"}})
        conn.route("CREATE schema_meta:migration_lock", [])  # acquired first try
        conn.route("DELETE schema_meta:migration_lock", [])

        result = await M.apply_migrations(conn)
        assert result == M.TARGET_VERSION
        # a plain DELETE of the lock (release) must appear
        assert any(
            s.strip().startswith("DELETE schema_meta:migration_lock") and "WHERE" not in s
            for s in conn.sqls()
        ), "lock must be released"


class TestFailedMigrationNotSilentlyAccepted:
    """Regression tests for the U2 review CRITICAL findings: a migration that
    started but never verified must NEVER be reported as v8."""

    @pytest.mark.asyncio
    async def test_detect_reports_unmigrated_when_state_not_done(self):
        # Table is already RELATION-shaped (converting ran) but verification never
        # passed → migration_state.phase != 'done' must override the structural check.
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])  # not stamped
        conn.route(
            "SELECT * FROM schema_meta:migration_state",
            [{"phase": "verifying", "old_count": 5, "skipped": 0}],
        )
        conn.route("INFO FOR DB", {"tables": {"synapse": "DEFINE TABLE synapse TYPE RELATION"}})
        assert await M.detect_db_version(conn) == M.SOURCE_VERSION

    @pytest.mark.asyncio
    async def test_apply_reverifies_and_raises_instead_of_stamping(self, monkeypatch):
        # After a failed conversion (RELATION table, fewer edges), apply_migrations
        # must resume + re-verify + raise — never stamp 8 and return success.
        conn = ScriptedConn()
        conn.route("SELECT version FROM schema_meta:version", [])
        conn.route(
            "SELECT * FROM schema_meta:migration_state",
            [
                {
                    "phase": "verifying",
                    "cursor": None,
                    "old_count": 5,
                    "copied": 5,
                    "converted": 5,
                    "skipped": 0,
                }
            ],
        )
        conn.route("INFO FOR DB", {"tables": {"synapse": "DEFINE TABLE synapse TYPE RELATION"}})
        conn.route("count() FROM synapse GROUP", [{"count": 2}])  # still short
        conn.route("CREATE schema_meta:migration_lock", [])
        conn.route("DELETE schema_meta:migration_lock", [])

        with pytest.raises(M.MigrationError):
            await M.apply_migrations(conn)
        assert not any("UPSERT schema_meta:version" in s for s in conn.sqls())


class TestConvertingErrors:
    @pytest.mark.asyncio
    async def test_ddl_error_other_than_already_exists_aborts(self, monkeypatch):
        """A real DDL failure (not 'already exists') must abort, not silently
        insert into a half-defined table (review MEDIUM)."""

        def ddl_boom(sql, params):
            raise RuntimeError("Parse error: unexpected token near RELATION")

        conn = ScriptedConn()
        conn.route(
            "SELECT * FROM schema_meta:migration_state",
            [
                {
                    "phase": "converting",
                    "cursor": None,
                    "old_count": 1,
                    "copied": 1,
                    "converted": 0,
                    "skipped": 0,
                }
            ],
        )
        conn.route("REMOVE TABLE IF EXISTS synapse", [])
        conn.route("DEFINE TABLE synapse TYPE RELATION", ddl_boom)

        with pytest.raises(RuntimeError):
            await M._migrate_7_to_8(conn)

"""Tests for SurrealDBStorage auth fail-fast and default handling."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# surrealdb is an optional dependency not installed in the base test environment.
# Inject a stub so that the lazy `from surrealdb import AsyncSurreal` inside
# store.py succeeds and the mock can override it. Stub ONLY when the SDK is
# genuinely not installed: an `if not in sys.modules` guard would shadow an
# installed SDK for the rest of the pytest session and break the live
# (SURREALDB_URL) tests that run after this module.
try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()


class TestInitializeAuthFailFast:
    """signin on bad credentials → StorageAuthError, not raw NotAllowedError."""

    @pytest.mark.asyncio
    async def test_signin_credential_error_raises_storage_auth_error(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class FakeNotAllowedError(Exception):
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = FakeNotAllowedError(
            "There was a problem with authentication"
        )

        storage = SurrealDBStorage(url="http://localhost:8001", password="wrongpass")  # noqa: S106

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True):
            with pytest.raises(StorageAuthError) as exc_info:
                await storage.initialize()

        err = exc_info.value
        assert "wrongpass" not in str(err), "Password must not appear in error message"
        assert err.hint != "", "hint must be non-empty"
        assert "SURREALDB_PASS" in err.hint

    @pytest.mark.asyncio
    async def test_signin_credential_error_includes_user_and_url(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class NotAllowedError(Exception):  # name triggers class-name fallback
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = NotAllowedError("not allowed")

        storage = SurrealDBStorage(url="http://myhost:8001", user="myuser", password="bad")  # noqa: S106

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True):
            with pytest.raises(StorageAuthError) as exc_info:
                await storage.initialize()

        msg = str(exc_info.value)
        assert "myuser" in msg
        assert "myhost" in msg

    @pytest.mark.asyncio
    async def test_non_credential_non_connection_exception_propagates_unchanged(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        # NOT a connection-class error: those now retry with backoff (see
        # TestInitializeTransientReset below); everything else must still
        # surface on the first attempt.
        mock_conn.signin.side_effect = RuntimeError("unexpected")

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True):
            with pytest.raises(RuntimeError):
                await storage.initialize()
        assert mock_conn.signin.await_count == 1

    @pytest.mark.asyncio
    async def test_initialize_success_does_not_raise(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None

        storage = SurrealDBStorage()

        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
        ):
            await storage.initialize()  # must not raise


class TestInitializeTransientReset:
    """A transient transport reset during the connect-and-prepare window retries.

    The Integration CI job runs the live-gated tests against one container;
    each opens its own connection, and a single server hiccup mid-run aborted
    whichever test was connecting at that moment (Errno 104 — a different
    test set each run). _query already retries this class (S-01); #172
    extended it to signin, and this now covers the handshake queries too
    (`INFO FOR DB` inside apply_migrations was where the next flake moved
    to). Credential errors never retry."""

    @pytest.mark.asyncio
    async def test_transient_reset_during_signin_is_retried(self, monkeypatch):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        async def _no_sleep(_d: float) -> None:
            return None

        monkeypatch.setattr("surreal_memory.storage.surrealdb.store.asyncio.sleep", _no_sleep)

        def _conn() -> AsyncMock:
            c = AsyncMock()
            c.version.return_value = "surrealdb-3.5.0"
            return c

        conn1, conn2, conn3 = _conn(), _conn(), _conn()
        conn1.signin.side_effect = ConnectionResetError(104, "Connection reset by peer")
        conn2.signin.side_effect = ConnectionResetError(104, "Connection reset by peer")

        storage = SurrealDBStorage()

        with (
            patch("surrealdb.AsyncSurreal", side_effect=[conn1, conn2, conn3], create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await storage.initialize()  # third attempt lands

        assert storage._conn is conn3
        assert conn3.signin.await_count == 1

    @pytest.mark.asyncio
    async def test_transient_reset_during_use_is_retried(self, monkeypatch):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        async def _no_sleep(_d: float) -> None:
            return None

        monkeypatch.setattr("surreal_memory.storage.surrealdb.store.asyncio.sleep", _no_sleep)

        def _conn() -> AsyncMock:
            c = AsyncMock()
            c.version.return_value = "surrealdb-3.5.0"
            return c

        conn1, conn2 = _conn(), _conn()
        conn1.use.side_effect = ConnectionResetError(104, "Connection reset by peer")

        storage = SurrealDBStorage()

        with (
            patch("surrealdb.AsyncSurreal", side_effect=[conn1, conn2], create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await storage.initialize()

        assert storage._conn is conn2

    @pytest.mark.asyncio
    async def test_transient_reset_during_handshake_is_retried(self, monkeypatch):
        """The exact Integration failure shape: signin succeeds, then the
        schema/migration queries on the raw connection hit the reset —
        no _query retry covers them, so the whole attempt must re-run."""
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        async def _no_sleep(_d: float) -> None:
            return None

        monkeypatch.setattr("surreal_memory.storage.surrealdb.store.asyncio.sleep", _no_sleep)

        def _conn() -> AsyncMock:
            c = AsyncMock()
            c.version.return_value = "surrealdb-3.5.0"
            return c

        conn1, conn2 = _conn(), _conn()

        ensure_schema = AsyncMock(
            side_effect=[ConnectionResetError(104, "Connection reset by peer"), None]
        )

        storage = SurrealDBStorage()

        with (
            patch("surrealdb.AsyncSurreal", side_effect=[conn1, conn2], create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", ensure_schema),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await storage.initialize()

        assert storage._conn is conn2, "a fresh connection must back the retried attempt"
        assert ensure_schema.await_count == 2

    @pytest.mark.asyncio
    async def test_version_gate_rejection_is_not_retried(self, monkeypatch):
        """An old server never gets newer by reconnecting — fail fast."""
        from surreal_memory.storage.surrealdb.connection import StorageVersionError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        async def _no_sleep(_d: float) -> None:
            return None

        monkeypatch.setattr("surreal_memory.storage.surrealdb.store.asyncio.sleep", _no_sleep)

        conns = []
        for _ in range(3):  # if retried, the loop would consume these
            c = AsyncMock()
            c.version.return_value = "surrealdb-3.1.1"
            conns.append(c)

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", side_effect=conns, create=True):
            with pytest.raises(StorageVersionError):
                await storage.initialize()

        assert conns[0].signin.await_count == 1
        assert conns[1].signin.await_count == 0, "version-gate rejection must not reconnect"

    @pytest.mark.asyncio
    async def test_persistent_reset_exhausts_retries_and_raises(self, monkeypatch):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        sleeps: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleeps.append(d)

        monkeypatch.setattr("surreal_memory.storage.surrealdb.store.asyncio.sleep", _fake_sleep)

        conns = []
        for _ in range(3):
            c = AsyncMock()
            c.signin.side_effect = ConnectionResetError(104, "Connection reset by peer")
            conns.append(c)

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", side_effect=conns, create=True):
            with pytest.raises(ConnectionResetError):
                await storage.initialize()

        # Backoff happened between the three attempts.
        assert sleeps == [1.0, 3.0]

    @pytest.mark.asyncio
    async def test_credential_error_is_not_retried(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class NotAllowedError(Exception):  # name triggers class-name fallback
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = NotAllowedError("not allowed")

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True):
            with pytest.raises(StorageAuthError):
                await storage.initialize()

        assert mock_conn.signin.await_count == 1  # fail fast, no retry


class TestReconnectAuthFailFast:
    """_reconnect on bad credentials → StorageAuthError (not a loop)."""

    @pytest.mark.asyncio
    async def test_reconnect_credential_error_raises_storage_auth_error(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class NotAllowedError(Exception):  # name triggers class-name fallback
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = NotAllowedError("not allowed")

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True):
            with pytest.raises(StorageAuthError):
                await storage._reconnect()


class TestDefaultPasswordDry:
    """Default password comes from connection.py (surrealmemory), not 'root'."""

    def test_default_password_is_surrealmemory(self, monkeypatch):
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        s = SurrealDBStorage()
        assert s._password == "surrealmemory"  # noqa: S105

    def test_explicit_password_overrides_default(self, monkeypatch):
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        s = SurrealDBStorage(password="explicit")  # noqa: S106
        assert s._password == "explicit"  # noqa: S105

    def test_env_password_overrides_default(self, monkeypatch):
        monkeypatch.setenv("SURREALDB_PASS", "envpass")
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        s = SurrealDBStorage()
        assert s._password == "envpass"  # noqa: S105


class TestInitializeVersionGate:
    """store.initialize() hard-gates on SurrealDB >= 3.2.0 (RUN-005 U4)."""

    @pytest.mark.asyncio
    async def test_rejects_confirmed_old_server(self):
        from surreal_memory.storage.surrealdb.connection import StorageVersionError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None
        mock_conn.version.return_value = "surrealdb-3.1.1"

        storage = SurrealDBStorage()
        with patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True):
            with pytest.raises(StorageVersionError) as exc:
                await storage.initialize()
        assert "3.2.0" in str(exc.value)
        # gate fires BEFORE schema/migration
        mock_conn.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepts_current_server(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None
        mock_conn.version.return_value = "surrealdb-3.2.0"

        storage = SurrealDBStorage()
        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await storage.initialize()  # must not raise

    @pytest.mark.asyncio
    async def test_continues_on_unparsable_version(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None
        mock_conn.version.return_value = "weird-build-string"

        storage = SurrealDBStorage()
        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await storage.initialize()  # unparsable → warn + continue, no raise

    @pytest.mark.asyncio
    async def test_continues_when_version_probe_fails(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None
        mock_conn.version.side_effect = RuntimeError("no version endpoint")

        storage = SurrealDBStorage()
        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await storage.initialize()  # probe failure → warn + continue, no raise


class TestToSurrealIdSanitization:
    """_to_surreal_id must enforce the documented ``[A-Za-z0-9_]`` contract.

    Regression guard for the W7.3 eval/GQL injection surface: the sanitized id
    is inlined verbatim into record-id and eval::gql query strings, so any
    character that could break out of a string/record literal must be folded
    to ``_``. Legit ids (UUID4, content-hashes, prefixed record ids) must be
    preserved (modulo '-' -> '_').
    """

    def test_legit_ids_preserved(self):
        from surreal_memory.storage.surrealdb.store import _to_surreal_id

        assert _to_surreal_id("57d4c589-6a1c-490d-b0f8-6ee1a23c180b") == (
            "57d4c589_6a1c_490d_b0f8_6ee1a23c180b"
        )
        assert _to_surreal_id("neuron:abc-123") == "abc_123"  # prefix stripped, '-' -> '_'
        assert _to_surreal_id("12345") == "12345"
        assert _to_surreal_id("plain_id") == "plain_id"

    def test_output_is_always_charset_safe(self):
        from surreal_memory.storage.surrealdb.store import _to_surreal_id

        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        hostile = [
            'x"',
            'x"})',
            'x"}) RETURN s //',
            'aaa"})-[:synapse]->{1,4}(t:neuron) RETURN s //',
            'zzz"}) RETURN (MATCH (a:neuron) RETURN a) AS leaked //',
            "a' OR 1=1 --",
            "id with spaces",
            "back`tick",
            "semi;colon",
            "star*glob",
        ]
        for payload in hostile:
            out = _to_surreal_id(payload)
            assert set(out) <= allowed, f"{payload!r} -> {out!r} leaked a non-charset char"

    def test_no_quote_or_brace_survives(self):
        from surreal_memory.storage.surrealdb.store import _to_surreal_id

        for ch in '"' + "'" + "{}()[]<>;:/*\\`= \t\n":
            assert ch not in _to_surreal_id(f"a{ch}b")


class TestSafeBrainId:
    """_safe_brain_id fail-closed rejects breakout chars while allowing the
    legitimate brain-id charset ``[A-Za-z0-9_.-]`` (brain ids are inlined raw
    into ``brain:{id}`` / ``device:{brain_id}_{did}`` / raw ``UPDATE brain:{id}``
    and must NOT be folded, so this is a reject-not-fold guard)."""

    def test_valid_brain_ids_pass_unchanged(self):
        from surreal_memory.storage.surrealdb.store import _safe_brain_id

        for v in ["uruboros", "my-brain.v2", "a_b.c-d", "A1", "x" * 128]:
            assert _safe_brain_id(v) == v

    def test_hostile_brain_ids_rejected(self):
        import pytest

        from surreal_memory.storage.surrealdb.store import _safe_brain_id

        hostile = [
            'x"} REMOVE TABLE neuron; --',
            "brain:evil",
            "a b",
            "x)",
            "y{1}",
            "semi;colon",
            "back`tick",
            "star*glob",
            "",
            "x" * 129,
            "nul\x00l",
            "rtl‮override",
        ]
        for payload in hostile:
            with pytest.raises(ValueError):
                _safe_brain_id(payload)


class TestConnectionErrorDetection:
    """_is_connection_error catches dropped-transport errors so the store
    reconnects after a DB container restart (audit finding S-01)."""

    def test_stdlib_connection_errors_detected(self):
        from surreal_memory.storage.surrealdb.store import _is_connection_error

        assert _is_connection_error(ConnectionResetError("reset"))
        assert _is_connection_error(OSError("broken pipe"))

    def test_websocket_close_messages_detected(self):
        from surreal_memory.storage.surrealdb.store import _is_connection_error

        assert _is_connection_error(Exception("received 1001 (going away)"))
        assert _is_connection_error(Exception("websocket connection is closed"))
        assert _is_connection_error(Exception("Connection refused"))

    def test_classname_detected(self):
        from surreal_memory.storage.surrealdb.store import _is_connection_error

        class ConnectionClosedError(Exception):
            pass

        assert _is_connection_error(ConnectionClosedError("1011"))

    def test_query_and_auth_errors_not_flagged(self):
        # A syntax/query error must NOT trigger a needless reconnect.
        from surreal_memory.storage.surrealdb.store import _is_connection_error

        assert not _is_connection_error(Exception("Parse error: unexpected token"))
        assert not _is_connection_error(ValueError("bad field value"))

    def test_query_timeout_not_flagged(self):
        # TimeoutError (== asyncio.TimeoutError since 3.11) is an OSError subclass,
        # but a slow query hitting the HTTP transport's ClientTimeout(total=30) is a
        # query outcome, not a dropped transport — it must NOT trigger a reconnect.
        from surreal_memory.storage.surrealdb.store import _is_connection_error

        assert not _is_connection_error(TimeoutError("query exceeded 30s"))

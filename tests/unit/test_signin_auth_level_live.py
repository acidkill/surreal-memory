"""Live-DB proof that the auth level is the thing standing between the two shapes.

SurrealDB accepts a root user only when sign-in carries neither namespace nor
database, and a user defined ON DATABASE only when it carries both. These tests
create such a user and drive the real ``signin`` with each payload, so the
matrix is measured rather than asserted from documentation. Skipped unless
SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.storage.surrealdb.connection import signin_payload

SURREALDB_URL = os.getenv("SURREALDB_URL")
ROOT_USER = os.getenv("SURREALDB_USER", "root")
ROOT_PASS = os.getenv("SURREALDB_PASS", "")

pytestmark = pytest.mark.skipif(
    not (SURREALDB_URL and ROOT_PASS),
    reason="requires SURREALDB_URL and SURREALDB_PASS for a running SurrealDB",
)

_NS = "auth_level_live"
_DB = "auth_level_live"
_SCOPED_USER = "auth_level_app_user"
_SCOPED_PASS = "auth-level-live-secret"  # noqa: S105  # throwaway, created and dropped here


@pytest.fixture
async def scoped_user():  # type: ignore[no-untyped-def]
    from surrealdb import AsyncSurreal

    root = AsyncSurreal(SURREALDB_URL)
    await root.signin({"username": ROOT_USER, "password": ROOT_PASS})
    await root.query(f"DEFINE NAMESPACE IF NOT EXISTS {_NS};")
    await root.query(f"USE NS {_NS}; DEFINE DATABASE IF NOT EXISTS {_DB};")
    await root.query(
        f"USE NS {_NS} DB {_DB}; "
        f"DEFINE USER IF NOT EXISTS {_SCOPED_USER} ON DATABASE "
        f"PASSWORD '{_SCOPED_PASS}' ROLES OWNER;"
    )
    yield
    try:
        await root.query(f"USE NS {_NS} DB {_DB}; REMOVE DATABASE {_DB};")
        await root.query(f"USE NS {_NS}; REMOVE NAMESPACE {_NS};")
    except Exception:
        pass
    try:
        await root.close()
    except Exception:
        pass


async def _signin_succeeds(payload: dict) -> bool:  # type: ignore[type-arg]
    """True if the server accepted the credentials; raises on anything else.

    Deliberately narrow: mapping every exception to "rejected" would let a
    connection refusal or a typo in the URL masquerade as a correct negative
    result, and the negative half of the matrix below is exactly what makes an
    unconditional change visibly wrong. Only a credential rejection counts.
    """
    from surrealdb import AsyncSurreal

    from surreal_memory.storage.surrealdb.connection import is_credential_error

    conn = AsyncSurreal(SURREALDB_URL)
    try:
        await conn.signin(payload)
        return True
    except Exception as exc:
        if is_credential_error(exc) or "authentication" in str(exc).lower():
            return False
        raise
    finally:
        try:
            await conn.close()
        except Exception:
            pass


class TestAuthLevelMatrix:
    async def test_root_authenticates_only_at_root_level(self, scoped_user) -> None:  # type: ignore[no-untyped-def]
        """The default level is the only one a root user accepts — hence the default."""
        assert await _signin_succeeds(signin_payload(ROOT_USER, ROOT_PASS, _NS, _DB, "root"))
        assert not await _signin_succeeds(
            signin_payload(ROOT_USER, ROOT_PASS, _NS, _DB, "database")
        ), "scoping a root sign-in rejects it — which is why this cannot be unconditional"

    async def test_database_user_authenticates_only_at_database_level(self, scoped_user) -> None:  # type: ignore[no-untyped-def]
        """The case the setting exists for: this user could not connect at all before."""
        assert await _signin_succeeds(
            signin_payload(_SCOPED_USER, _SCOPED_PASS, _NS, _DB, "database")
        )
        assert not await _signin_succeeds(
            signin_payload(_SCOPED_USER, _SCOPED_PASS, _NS, _DB, "root")
        ), "credentials alone is what the code used to send unconditionally"

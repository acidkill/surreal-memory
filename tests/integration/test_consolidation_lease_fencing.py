"""Disposable live integration test for cross-client consolidation lease fencing.

Only SURREALDB_TEST_URL is honored, and it must point to loopback. The
ordinary SURREALDB_* settings are deliberately cleared before constructing
storage clients so an inherited production endpoint cannot be selected
implicitly. Without an explicitly configured local endpoint, this test skips.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

_TEST_URL = os.getenv("SURREALDB_TEST_URL")
_TEST_USER = os.getenv("SURREALDB_TEST_USER", "root")
_TEST_PASSWORD = os.getenv("SURREALDB_TEST_PASS", "root")
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

pytestmark = pytest.mark.integration


def _local_test_url() -> str:
    if not _TEST_URL:
        pytest.skip("requires an explicit loopback SURREALDB_TEST_URL")

    parsed = urlsplit(_TEST_URL)
    if (
        parsed.scheme not in {"ws", "wss", "http", "https"}
        or parsed.hostname not in _LOCAL_HOSTS
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        pytest.fail("SURREALDB_TEST_URL must be a credential-free loopback URL")

    return _TEST_URL


@pytest_asyncio.fixture
async def lease_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[SurrealDBStorage, SurrealDBStorage, str]]:
    """Connect two independent clients to one uniquely named test database."""
    url = _local_test_url()

    # SurrealDBStorage reads defaults from SURREALDB_* even when explicit
    # constructor settings are supplied. Clear inherited settings first.
    for key in tuple(os.environ):
        if key.startswith("SURREALDB_"):
            monkeypatch.delenv(key, raising=False)

    suffix = uuid.uuid4().hex
    namespace = f"it_{suffix[:12]}"
    database = f"lease_{suffix[12:24]}"
    options = {
        "url": url,
        "user": _TEST_USER,
        "password": _TEST_PASSWORD,
        "namespace": namespace,
        "database": database,
    }
    first = SurrealDBStorage(**options)
    second = SurrealDBStorage(**options)

    try:
        await first.initialize()
        await second.initialize()
        yield first, second, suffix
    finally:
        await second.close()
        await first.close()


async def test_cross_client_expiry_takeover_fences_stale_owner(
    lease_clients: tuple[SurrealDBStorage, SurrealDBStorage, str],
) -> None:
    """A second client takes over only after expiry; the former owner is fenced."""
    first, second, suffix = lease_clients
    brain_id = f"lease-fencing-{suffix}"
    owner_a = f"owner-a-{suffix}"
    owner_b = f"owner-b-{suffix}"

    assert await first.acquire_consolidation_lease(brain_id, owner_a, lease_seconds=30)
    assert not await second.acquire_consolidation_lease(brain_id, owner_b, lease_seconds=30)

    # Advance only this synthetic row's expiry instead of sleeping for the
    # minimum supported 30-second lease duration.
    expired_rows = await first._query(
        "UPDATE type::record('consolidation_lease', $record_id) "
        "SET expires_at = $expires_at WHERE brain_id = $brain_id RETURN AFTER",
        record_id=first._consolidation_record_id(brain_id),
        expires_at=utcnow() - timedelta(seconds=1),
        brain_id=brain_id,
    )
    assert expired_rows

    assert await second.acquire_consolidation_lease(brain_id, owner_b, lease_seconds=30)
    assert not await first.renew_consolidation_lease(brain_id, owner_a)
    assert not await first.release_consolidation_lease(brain_id, owner_a)
    assert await second.renew_consolidation_lease(brain_id, owner_b)
    assert await second.release_consolidation_lease(brain_id, owner_b)

"""Exact, complete brain deletion and the separate clear contract."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from surreal_memory.core.brain import Brain
from surreal_memory.server.dependencies import get_storage
from surreal_memory.server.routes import brain as brain_routes
from surreal_memory.server.routes import dashboard_api as dashboard
from surreal_memory.storage.base import BrainDeletionConflictError
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.shared_store import SharedStorage
from surreal_memory.storage.shared_store_collections import SharedStorageError
from surreal_memory.storage.surrealdb.store import _BRAIN_DELETE_TABLES, SurrealDBStorage


async def test_memory_delete_removes_all_scopes_and_versions() -> None:
    store = InMemoryStorage()
    for bid in ("delete_full_id", "other", "default"):
        await store.save_brain(Brain.create(bid, brain_id=bid))
    for name, value in vars(store).items():
        if isinstance(value, dict) and name != "_brains":
            value["delete_full_id"] = {}
            value["other"] = {"synthetic": 1}
    store._graph.add_node("target", brain_id="delete_full_id")
    store._graph.add_node("other", brain_id="other")
    store.set_brain("other")
    assert not await store.delete_brain("delete_full")
    assert await store.delete_brain("delete_full_id")
    assert not await store.delete_brain("delete_full_id")
    assert store.brain_id == "other"
    assert list(store._graph.nodes()) == ["other"]
    for value in vars(store).values():
        if isinstance(value, dict):
            assert "delete_full_id" not in value
            assert "other" in value


@pytest.mark.parametrize("bid,name", [("default", "renamed"), ("uuid", "default")])
async def test_memory_protects_default_identity_and_name(bid: str, name: str) -> None:
    store = InMemoryStorage()
    await store.save_brain(Brain.create(name, brain_id=bid))
    with pytest.raises(BrainDeletionConflictError):
        await store.delete_brain(bid)
    assert await store.get_brain(bid) is not None


class DeleteStore(SurrealDBStorage):
    """Script only the transport; keep deletion control flow and acknowledgement checks."""

    def __init__(self) -> None:
        self.exists = True
        self.rows = dict.fromkeys(_BRAIN_DELETE_TABLES, 0)
        self.rows["neuron"] = 501
        self.rows["semantic_discovery_state"] = 9
        self.calls: list[str] = []
        self.acquire_consolidation_lease = AsyncMock(return_value=True)
        self.release_consolidation_lease = AsyncMock(return_value=True)
        self.failure: str | None = None
        self.name = "target_full_id"
        self.aliases: list[dict[str, str]] = []
        self.lost_ack = False

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append(sql)
        if sql.startswith("SELECT * FROM type::record"):
            return [{"name": self.name}] if self.exists else []
        if sql.startswith("SELECT id FROM brain"):
            return self.aliases
        if sql.startswith("SELECT id FROM type::record"):
            return [{"id": "brain:target_full_id"}] if self.exists else []
        if sql.startswith("SELECT id FROM "):
            table = sql.split()[3]
            return [{"id": "synthetic"}] if self.rows[table] else []
        assert "owner_token = $owner" in sql
        assert "expires_at > time::now()" in sql
        assert "WHERE name = $name" in sql
        assert params["scopes"] == ["target_full_id"]
        if self.failure:
            raise RuntimeError(self.failure)
        if "LET $ids" in sql:
            match = re.search(
                r"SELECT VALUE id FROM (\w+) WHERE brain_id = \"target_full_id\" LIMIT (\d+)", sql
            )
            assert match
            table, limit = match.groups()
            count = min(self.rows[table], int(limit))
            self.rows[table] -= count
            return [{"deleted": count}]
        assert "DELETE type::record('brain', $brain_id)" in sql
        assert not any(self.rows.values())
        assert "brain deletion has remaining rows" in sql
        self.exists = False
        if self.lost_ack:
            raise ConnectionError("lost final response")
        return [{"deleted": True}]


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_surreal_delete_registry_last_and_lost_ack(lost_ack: bool) -> None:
    store = DeleteStore()
    store.lost_ack = lost_ack
    assert await store.delete_brain("target_full_id")
    assert not store.exists
    assert not any(store.rows.values())
    assert not await store.delete_brain("target_full_id")
    assert (
        sum(
            'FROM semantic_discovery_state WHERE brain_id = "target_full_id" LIMIT 8' in q
            for q in store.calls
        )
        == 3
    )
    store.release_consolidation_lease.assert_awaited_once()


async def test_surreal_active_lease_rejects_before_delete() -> None:
    store = DeleteStore()
    store.acquire_consolidation_lease.return_value = False
    with pytest.raises(BrainDeletionConflictError):
        await store.delete_brain("target_full_id")
    assert store.exists and store.rows["neuron"] == 501
    assert not any(q.startswith("RETURN") for q in store.calls)


@pytest.mark.parametrize("failure", ["lease lost", "batch response timed out"])
async def test_surreal_partial_failure_keeps_registry_and_retry(failure: str) -> None:
    store = DeleteStore()
    store.failure = failure
    with pytest.raises(RuntimeError, match=failure):
        await store.delete_brain("target_full_id")
    assert store.exists
    store.failure = None
    assert await store.delete_brain("target_full_id")


@pytest.mark.parametrize("name,aliases", [("default", []), ("target_full_id", [{"id": "other"}])])
async def test_surreal_protects_default_and_ambiguous_scope(name: str, aliases: list) -> None:
    store = DeleteStore()
    store.name, store.aliases = name, aliases
    with pytest.raises(BrainDeletionConflictError):
        await store.delete_brain("target_full_id")
    store.acquire_consolidation_lease.assert_not_awaited()


async def test_remote_clear_and_delete_are_separate() -> None:
    store = SharedStorage("http://127.0.0.1:8000", "target_full_id")
    store._request = AsyncMock(return_value={"status": "deleted", "brain_id": "target_full_id"})
    await store.clear("target_full_id")
    store._request.assert_awaited_with("POST", "/brain/target_full_id/clear")
    assert await store.delete_brain("target_full_id")
    store._request.assert_awaited_with("DELETE", "/brain/target_full_id")
    store._request.side_effect = SharedStorageError("gone", status_code=404)
    assert not await store.delete_brain("target_full_id")


@pytest.mark.parametrize("prefix", ["", "/api/v1"])
async def test_api_exact_delete_default_cache_and_clear(prefix: str) -> None:
    store = InMemoryStorage()
    for bid in ("default", "target_full_id", "other"):
        await store.save_brain(Brain.create(bid, brain_id=bid))
    app = FastAPI()
    app.include_router(brain_routes.router, prefix=prefix)
    app.dependency_overrides[get_storage] = lambda: store
    dashboard._BRAINS_LAST_GOOD = []
    dashboard._GRADE_LAST_GOOD["health:target_full_id"] = object()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.delete(f"{prefix}/brain/target_full")).status_code == 404
        assert (await client.delete(f"{prefix}/brain/default")).status_code == 409
        result = await client.delete(f"{prefix}/brain/target_full_id")
        assert result.status_code == 200
        assert (await client.delete(f"{prefix}/brain/target_full_id")).status_code == 404
        assert (await client.post(f"{prefix}/brain/other/clear")).status_code == 200
    assert dashboard._BRAINS_LAST_GOOD is None
    assert not dashboard._GRADE_LAST_GOOD
    assert await store.get_brain("default") is not None


async def test_cache_invalidation_cancels_inflight_refresh() -> None:
    async def refresh() -> None:
        await asyncio.sleep(10)
        dashboard._BRAINS_LAST_GOOD = []

    task = asyncio.create_task(refresh())
    dashboard._BRAINS_REFRESH_TASKS.add(task)
    try:
        await dashboard.invalidate_brain_caches()
        assert task.cancelled()
        assert dashboard._BRAINS_LAST_GOOD is None
    finally:
        dashboard._BRAINS_REFRESH_TASKS.discard(task)


async def test_api_timeout_is_incomplete_not_success() -> None:
    store = InMemoryStorage()
    store.delete_brain = AsyncMock(side_effect=TimeoutError("synthetic timeout"))
    app = FastAPI()
    app.include_router(brain_routes.router)
    app.dependency_overrides[get_storage] = lambda: store
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.delete("/brain/target_full_id")
    assert response.status_code == 503
    assert "incomplete" in response.json()["detail"]


@pytest.mark.parametrize("invalidate_during_read", [False, True])
async def test_dashboard_lists_canonical_id_and_does_not_recache_deleted_scope(
    monkeypatch: pytest.MonkeyPatch, invalidate_during_read: bool
) -> None:
    from types import SimpleNamespace

    from surreal_memory import unified_config

    store = InMemoryStorage()
    await store.save_brain(Brain.create("legacy-name", brain_id="legacy-full-id"))
    store.set_brain("legacy-name")
    store.get_brain = AsyncMock(
        return_value=Brain.create("legacy-name-longer", brain_id="wrong-brain")
    )

    async def stats(*args: Any) -> dict[str, int]:
        if invalidate_during_read:
            await dashboard.invalidate_brain_caches()
        return {"neuron_count": 1}

    store.get_stats = stats
    monkeypatch.setattr(
        unified_config, "get_config", lambda: SimpleNamespace(current_brain="default")
    )
    monkeypatch.setattr(
        unified_config, "list_available_brains", AsyncMock(return_value=["legacy-name"])
    )
    await dashboard.invalidate_brain_caches()
    try:
        results = await dashboard.list_brains_api(store)
        assert results[0].id == "legacy-full-id"
        assert results[0].name == "legacy-name"
        assert (dashboard._BRAINS_LAST_GOOD is None) == invalidate_during_read
    finally:
        await dashboard.invalidate_brain_caches()


async def test_remote_rejects_malformed_delete_acknowledgement() -> None:
    store = SharedStorage("http://127.0.0.1:8000", "target_full_id")
    store._request = AsyncMock(return_value={"status": "deleted", "brain_id": "other"})
    with pytest.raises(SharedStorageError, match="acknowledgement"):
        await store.delete_brain("target_full_id")

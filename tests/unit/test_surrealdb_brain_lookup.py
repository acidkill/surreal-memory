"""Unit tests for SurrealDB brain lookup (no live DB required).

Regression coverage for the duplicate-brain bug: get_brain() must match a
brain by its `name` field, not only by a record-id substring. When it only
matched by id, get_brain("my-brain.v2") always returned None (record ids are
random UUIDs), so the bootstrap re-created a fresh brain on every process
start, accumulating hundreds of orphan rows.
"""

from __future__ import annotations

from typing import Any

import surreal_memory.storage.surrealdb as _surrealdb_pkg
from surreal_memory import unified_config
from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _make_fake_store(*, existing: Brain | None, found_by_name: Brain | None) -> type:
    """Build a SurrealDBStorage stand-in with scripted brain lookups."""

    class _FakeStore:
        def __init__(self, **_kwargs: Any) -> None:
            self.saved: list[Brain] = []
            self.brain_context: str | None = None

        async def initialize(self) -> None:
            return None

        async def get_brain(self, _name: str) -> Brain | None:
            return existing

        async def find_brain_by_name(self, _name: str) -> Brain | None:
            return found_by_name

        async def save_brain(self, brain: Brain) -> None:
            self.saved.append(brain)

        def set_brain(self, name: str) -> None:
            self.brain_context = name

    return _FakeStore


def _fake_config() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(embedding=SimpleNamespace(enabled=False, model=None))


class _BrainLookupStore(SurrealDBStorage):
    """Instantiate without connecting; stub the query layer with fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def _ensure_conn(self) -> Any:  # type: ignore[override]
        return object()

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        return self._rows


async def test_get_brain_matches_by_name_field() -> None:
    # Record id is a random UUID; the only link to the requested brain is name.
    rows = [
        {
            "id": "brain:9f1c0e2a-0000-4000-8000-000000000000",
            "name": "my-brain.v2",
            "metadata": {},
            "created_at": "2026-05-28T00:00:00Z",
            "updated_at": "2026-05-28T00:00:00Z",
        }
    ]
    store = _BrainLookupStore(rows)
    brain = await store.get_brain("my-brain.v2")
    assert brain is not None
    assert brain.name == "my-brain.v2"


async def test_get_brain_still_matches_by_record_id() -> None:
    rows = [
        {
            "id": "brain:default",
            "name": "default",
            "metadata": {},
            "created_at": "2026-05-27T00:00:00Z",
            "updated_at": "2026-05-27T00:00:00Z",
        }
    ]
    store = _BrainLookupStore(rows)
    brain = await store.get_brain("default")
    assert brain is not None
    assert brain.name == "default"


async def test_get_brain_returns_none_when_absent() -> None:
    rows = [
        {
            "id": "brain:default",
            "name": "default",
            "metadata": {},
            "created_at": "2026-05-27T00:00:00Z",
            "updated_at": "2026-05-27T00:00:00Z",
        }
    ]
    store = _BrainLookupStore(rows)
    assert await store.get_brain("does-not-exist") is None


async def test_list_brain_names_returns_distinct_sorted() -> None:
    # Duplicate brain rows (the orphan-row leak) must collapse to one name.
    rows = [
        {"name": "my-brain.v2"},
        {"name": "default"},
        {"name": "my-brain.v2"},
    ]
    store = _BrainLookupStore(rows)
    assert await store.list_brain_names() == ["default", "my-brain.v2"]


async def test_list_brain_names_ignores_blank_names() -> None:
    rows = [{"name": "default"}, {"name": ""}, {"name": None}]
    store = _BrainLookupStore(rows)
    assert await store.list_brain_names() == ["default"]


async def test_surrealdb_bootstrap_reuses_brain_found_by_name(monkeypatch: Any) -> None:
    # get_brain misses (legacy UUID rows), but find_brain_by_name resolves it:
    # the bootstrap must reuse the existing brain and never insert a new row.
    existing = Brain.create("my-brain.v2", brain_id="my-brain.v2")
    fake = _make_fake_store(existing=None, found_by_name=existing)
    monkeypatch.setattr(_surrealdb_pkg, "SurrealDBStorage", fake)
    monkeypatch.setattr(unified_config, "_surrealdb_storage", None)

    storage = await unified_config._get_surrealdb_storage(_fake_config(), "my-brain.v2")

    assert storage.saved == []  # no duplicate brain row created
    assert storage.brain_context == "my-brain.v2"


async def test_surrealdb_bootstrap_creates_brain_with_deterministic_id(
    monkeypatch: Any,
) -> None:
    # No brain exists at all: the bootstrap must create it with a deterministic
    # brain_id == name (not a random UUID), so a re-run cannot leak duplicates.
    fake = _make_fake_store(existing=None, found_by_name=None)
    monkeypatch.setattr(_surrealdb_pkg, "SurrealDBStorage", fake)
    monkeypatch.setattr(unified_config, "_surrealdb_storage", None)

    storage = await unified_config._get_surrealdb_storage(_fake_config(), "my-brain.v2")

    assert len(storage.saved) == 1
    assert storage.saved[0].id == "my-brain.v2"  # deterministic, not a UUID
    assert storage.brain_context == "my-brain.v2"


async def test_surrealdb_bootstrap_reinitializes_after_close(monkeypatch: Any) -> None:
    # A previously close()d cached instance nulls _conn. The bootstrap must NOT
    # return that dead handle (find_fibers would raise "not initialized") — it must
    # reinitialize a fresh store. Regression for the session_start memories→reasoning
    # double-open sequence: the memories block closes shared storage, so the reasoning
    # block's get_shared_storage() must reconnect rather than reuse the closed handle.
    class _ClosedStore:
        _conn = None

        def set_brain(self, _name: str) -> None:  # pragma: no cover - must not run
            raise AssertionError("closed instance must not be reused")

    closed = _ClosedStore()
    existing = Brain.create("my-brain.v2", brain_id="my-brain.v2")
    fake = _make_fake_store(existing=None, found_by_name=existing)
    monkeypatch.setattr(_surrealdb_pkg, "SurrealDBStorage", fake)
    monkeypatch.setattr(unified_config, "_surrealdb_storage", closed)

    storage = await unified_config._get_surrealdb_storage(_fake_config(), "my-brain.v2")

    assert storage is not closed  # reinitialized, not the dead handle
    assert storage.brain_context == "my-brain.v2"


async def test_surrealdb_bootstrap_reuses_live_connection(monkeypatch: Any) -> None:
    # A cached instance with a live _conn is returned as-is (no reinitialize); only
    # set_brain runs to switch the brain context. Constructing SurrealDBStorage here
    # would mean the liveness check wrongly fell through — so make that path fail loud.
    class _LiveStore:
        _conn = object()

        def __init__(self) -> None:
            self.brain_context: str | None = None

        def set_brain(self, name: str) -> None:
            self.brain_context = name

    live = _LiveStore()

    def _boom(**_kwargs: Any) -> Any:  # pragma: no cover - must not run
        raise AssertionError("must not reinitialize a cached store with a live _conn")

    monkeypatch.setattr(_surrealdb_pkg, "SurrealDBStorage", _boom)
    monkeypatch.setattr(unified_config, "_surrealdb_storage", live)

    storage = await unified_config._get_surrealdb_storage(_fake_config(), "brain-3")

    assert storage is live  # same instance, reused
    assert live.brain_context == "brain-3"

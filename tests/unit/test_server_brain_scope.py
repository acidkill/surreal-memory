"""Regression test: the server's ``get_brain`` dependency must scope by brain *name*.

Every brain-scoped row carries ``brain_id`` as a plain string equal to the brain
*name* (``"default"``), while a brain record created by an older version has a
random uuid4 primary key. ``get_brain`` resolves before any route body runs, so
binding the storage scope to ``brain.id`` silently pointed the whole request at a
UUID scope that holds no rows.

That is also why #97's ``storage.brain_id or brain.name`` fix was a no-op on the
server: ``storage.brain_id`` had already been set to the UUID by this dependency,
so the ``or`` never reached ``brain.name``. On the live DB this split writes across
two scopes — e.g. ``reasoning_traces`` ended up with 10,548 rows under the UUID and
326 under ``"default"``, and the dashboard read the stale UUID half.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from surreal_memory.core.brain import Brain
from surreal_memory.server.dependencies import get_brain

# Mirrors the live DB: one brain record, uuid4 primary key, rows keyed by the name.
BRAIN_NAME = "default"
BRAIN_UUID = "00313cb4-61ca-4e69-9784-e51431e99ad7"


class _FakeStorage:
    """Storage whose rows only exist under the brain *name* scope."""

    def __init__(self, brain: Brain) -> None:
        self._brain = brain
        self.brain_id: str | None = None
        self.set_brain_calls: list[str] = []
        # Only the name scope holds rows; the UUID scope is empty, exactly like prod.
        self._rows: dict[str, int] = {BRAIN_NAME: 10643}

    def set_brain(self, brain_id: str) -> None:
        self.set_brain_calls.append(brain_id)
        self.brain_id = brain_id

    async def get_brain(self, brain_id: str) -> Brain | None:
        # Record lookup: only the uuid4 primary key resolves here.
        return self._brain if brain_id == self._brain.id else None

    async def find_brain_by_name(self, name: str) -> Brain | None:
        return self._brain if name == self._brain.name else None

    def neuron_count(self) -> int:
        """Rows visible in the currently bound scope."""
        return self._rows.get(self.brain_id or "", 0)


@pytest.fixture
def legacy_brain() -> Brain:
    """Brain whose record id is a uuid4 that differs from its row scope."""
    return Brain(id=BRAIN_UUID, name=BRAIN_NAME)


@pytest.fixture
def storage(legacy_brain: Brain) -> _FakeStorage:
    return _FakeStorage(legacy_brain)


async def test_scopes_by_name_when_header_carries_the_record_uuid(
    storage: _FakeStorage,
) -> None:
    brain = await get_brain(storage, brain_id=BRAIN_UUID)  # type: ignore[arg-type]

    assert brain.id == BRAIN_UUID  # resolved via the record-id lookup
    assert storage.brain_id == BRAIN_NAME, (
        f"scope bound to {storage.brain_id!r}; the UUID scope holds no rows"
    )


async def test_scopes_by_name_when_header_carries_the_name(storage: _FakeStorage) -> None:
    # Name goes through the find_brain_by_name fallback; the scope must not change.
    await get_brain(storage, brain_id=BRAIN_NAME)  # type: ignore[arg-type]

    assert storage.brain_id == BRAIN_NAME


async def test_never_binds_the_scope_to_the_record_id(storage: _FakeStorage) -> None:
    await get_brain(storage, brain_id=BRAIN_UUID)  # type: ignore[arg-type]

    assert storage.set_brain_calls == [BRAIN_NAME]
    assert BRAIN_UUID not in storage.set_brain_calls


async def test_bound_scope_actually_sees_the_rows(storage: _FakeStorage) -> None:
    await get_brain(storage, brain_id=BRAIN_UUID)  # type: ignore[arg-type]

    assert storage.neuron_count() == 10643, "the bound scope must be the one holding rows"


async def test_pr97_route_expression_resolves_to_the_name(storage: _FakeStorage) -> None:
    """Routes read ``storage.brain_id or brain.name``; that must not yield the UUID.

    This is the exact expression #97 introduced. It only ever returned the correct
    scope once this dependency stopped pre-setting ``storage.brain_id`` to the UUID.
    """
    brain = await get_brain(storage, brain_id=BRAIN_UUID)  # type: ignore[arg-type]

    assert (storage.brain_id or brain.name) == BRAIN_NAME


async def test_falls_back_to_current_brain_when_header_omitted(
    storage: _FakeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    import surreal_memory.unified_config as unified_config

    class _FakeConfig:
        current_brain = BRAIN_NAME

    monkeypatch.setattr(unified_config, "get_config", lambda: _FakeConfig())

    await get_brain(storage, brain_id=None)  # type: ignore[arg-type]

    assert storage.brain_id == BRAIN_NAME


async def test_unknown_brain_raises_404_without_touching_the_scope(
    storage: _FakeStorage,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_brain(storage, brain_id="no-such-brain")  # type: ignore[arg-type]

    assert exc_info.value.status_code == 404
    # A failed lookup must not leave a half-bound scope behind for the next caller.
    assert storage.set_brain_calls == []


async def test_scopes_by_name_for_modern_brains_where_id_equals_name() -> None:
    """Brains created since #97 have ``brain_id == name``; behaviour is unchanged."""
    brain = Brain(id="project-x", name="project-x")
    storage = _FakeStorage(brain)

    await get_brain(storage, brain_id="project-x")  # type: ignore[arg-type]

    assert storage.set_brain_calls == ["project-x"]


@pytest.mark.parametrize("header", [BRAIN_UUID, BRAIN_NAME])
async def test_scope_is_independent_of_how_the_brain_was_addressed(
    storage: _FakeStorage, header: str
) -> None:
    """Both header forms resolve the same brain, so both must bind the same scope."""
    await get_brain(storage, brain_id=header)  # type: ignore[arg-type]

    assert storage.brain_id == BRAIN_NAME

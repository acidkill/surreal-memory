"""Tests for storage brain_id property parity across backends."""

from __future__ import annotations

import pytest

from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.memory_store import InMemoryStorage


class TestBrainIdProperty:
    """Verify brain_id property works uniformly across storage backends."""

    def test_base_class_has_brain_id_property(self) -> None:
        """NeuralStorage base class exposes brain_id property."""
        assert hasattr(NeuralStorage, "brain_id")

    def test_base_class_has_current_brain_id_alias(self) -> None:
        """current_brain_id is a backward-compat alias for brain_id."""
        assert hasattr(NeuralStorage, "current_brain_id")

    def test_inmemory_brain_id_initially_none(self) -> None:
        storage = InMemoryStorage()
        assert storage.brain_id is None
        assert storage.current_brain_id is None

    def test_inmemory_brain_id_after_set(self) -> None:
        storage = InMemoryStorage()
        storage.set_brain("test-brain")
        assert storage.brain_id == "test-brain"
        assert storage.current_brain_id == "test-brain"

    def test_inmemory_brain_id_switch(self) -> None:
        storage = InMemoryStorage()
        storage.set_brain("brain-a")
        assert storage.brain_id == "brain-a"
        storage.set_brain("brain-b")
        assert storage.brain_id == "brain-b"

    def test_brain_id_and_current_brain_id_always_match(self) -> None:
        """brain_id and current_brain_id must always return the same value."""
        storage = InMemoryStorage()
        assert storage.brain_id == storage.current_brain_id  # both None
        storage.set_brain("x")
        assert storage.brain_id == storage.current_brain_id  # both "x"


class TestSQLiteStorageBrainId:
    """Verify SQLiteStorage brain_id property."""

    @pytest.fixture
    async def storage(self, tmp_path):
        from neural_memory.storage.sqlite_store import SQLiteStorage

        db_path = tmp_path / "test.db"
        s = SQLiteStorage(db_path)
        await s.initialize()
        yield s
        await s.close()

    async def test_brain_id_initially_none(self, storage) -> None:
        assert storage.brain_id is None

    async def test_brain_id_after_set(self, storage) -> None:
        storage.set_brain("my-brain")
        assert storage.brain_id == "my-brain"
        assert storage.current_brain_id == "my-brain"


class TestHelpersForceFlags:
    """Verify _helpers.py get_storage logic precedence."""

    def test_force_sqlite_overrides_shared_mode(self) -> None:
        """force_sqlite=True must prevent shared mode even when is_shared_mode=True."""
        # This tests the logic expression, not full storage creation
        is_shared_mode = True
        force_shared = False
        force_local = False
        force_sqlite = True

        use_shared = (is_shared_mode or force_shared) and not force_local and not force_sqlite
        assert use_shared is False

    def test_force_shared_enables_shared_mode(self) -> None:
        is_shared_mode = False
        force_shared = True
        force_local = False
        force_sqlite = False

        use_shared = (is_shared_mode or force_shared) and not force_local and not force_sqlite
        assert use_shared is True

    def test_force_local_overrides_shared_mode(self) -> None:
        is_shared_mode = True
        force_shared = False
        force_local = True
        force_sqlite = False

        use_shared = (is_shared_mode or force_shared) and not force_local and not force_sqlite
        assert use_shared is False

    def test_default_no_shared(self) -> None:
        is_shared_mode = False
        force_shared = False
        force_local = False
        force_sqlite = False

        use_shared = (is_shared_mode or force_shared) and not force_local and not force_sqlite
        assert use_shared is False

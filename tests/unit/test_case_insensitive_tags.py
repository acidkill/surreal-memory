"""Tests for case-insensitive tag matching (follow-up #31).

Covers all 4 normalization boundaries:
1. Fiber.create() — tags lowercased at ingestion
2. _parse_tags()  — query tags lowercased at MCP boundary
3. SurrealDBStorage.find_fibers() post-filter — query tags lowercased
4. brain_transplant._fiber_matches_tags() — both sides lowercased

The SurrealDB-backed test stubs out the DB connection so CI runs without
a live SurrealDB instance, exercising the real post-filter logic.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub the optional surrealdb dependency so store.py can be imported in CI.
# Stub ONLY when the SDK is genuinely not installed: an `if not in sys.modules`
# guard would shadow an installed SDK for the rest of the pytest session and
# break the live (SURREALDB_URL) tests that run after this module.
try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fiber_row(fiber_id: str, tags: list[str]) -> dict:
    """Build a minimal SurrealDB-style fiber row dict."""
    rid = MagicMock()
    rid.table_name = "fiber"
    rid.id = fiber_id
    return {
        "id": rid,
        "neuron_ids": ["n1"],
        "synapse_ids": [],
        "anchor_neuron_id": "n1",
        "pathway": ["n1"],
        "conductivity": 1.0,
        "coherence": 0.0,
        "salience": 0.5,
        "frequency": 0,
        "auto_tags": tags,
        "agent_tags": [],
        "metadata": {},
        "compression_tier": 0,
        "pinned": False,
    }


# ---------------------------------------------------------------------------
# Boundary 1: Fiber.create() normalizes at ingestion
# ---------------------------------------------------------------------------


class TestFiberCreateNormalizesTags:
    """Tags passed to Fiber.create() must be stored lowercased."""

    def test_uppercase_agent_tags_lowercased(self) -> None:
        from surreal_memory.core.fiber import Fiber

        f = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            tags={"KB", "Python", "BACKEND"},
        )
        assert f.agent_tags == {"kb", "python", "backend"}

    def test_uppercase_auto_tags_lowercased(self) -> None:
        from surreal_memory.core.fiber import Fiber

        f = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            auto_tags={"Frontend", "API"},
            agent_tags={"DOCS"},
        )
        assert f.auto_tags == {"frontend", "api"}
        assert f.agent_tags == {"docs"}

    def test_mixed_case_tags_property(self) -> None:
        from surreal_memory.core.fiber import Fiber

        f = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            auto_tags={"Auth"},
            agent_tags={"SESSION"},
        )
        assert f.tags == {"auth", "session"}

    def test_already_lowercase_unchanged(self) -> None:
        from surreal_memory.core.fiber import Fiber

        f = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            tags={"kb", "python"},
        )
        assert f.agent_tags == {"kb", "python"}


# ---------------------------------------------------------------------------
# Boundary 2: _parse_tags() normalizes query tags
# ---------------------------------------------------------------------------


class TestParseTagsNormalizes:
    """_parse_tags must lowercase all tag strings it returns."""

    def test_uppercase_query_tags_lowercased(self) -> None:
        from surreal_memory.mcp.tool_handler_utils import _parse_tags

        result = _parse_tags({"tags": ["KB", "Python", "BACKEND"]})
        assert result == {"kb", "python", "backend"}

    def test_mixed_case_lowercased(self) -> None:
        from surreal_memory.mcp.tool_handler_utils import _parse_tags

        result = _parse_tags({"tags": ["CamelCase", "ALLCAPS", "lower"]})
        assert result == {"camelcase", "allcaps", "lower"}

    def test_empty_list_returns_none(self) -> None:
        from surreal_memory.mcp.tool_handler_utils import _parse_tags

        assert _parse_tags({"tags": []}) is None

    def test_no_tags_key_returns_none(self) -> None:
        from surreal_memory.mcp.tool_handler_utils import _parse_tags

        assert _parse_tags({}) is None


# ---------------------------------------------------------------------------
# Boundary 3: SurrealDBStorage.find_fibers() post-filter (case-insensitive)
# ---------------------------------------------------------------------------


class TestSurrealDBFindFibersTagCaseInsensitive:
    """Querying with uppercase tags must match fibers stored with lowercase tags.

    The SurrealDB connection is stubbed — _query returns pre-built rows so that
    the post-filter logic (which lives in pure Python) is exercised directly.
    """

    @pytest.mark.asyncio
    async def test_uppercase_query_matches_lowercase_stored_tag(self) -> None:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("brain1")

        rows = [_make_fiber_row("fiber-a", ["kb", "python"])]
        storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]

        # Query with "KB" — stored as "kb" — must still match
        results = await storage.find_fibers(tags={"KB"})
        assert len(results) == 1
        assert "kb" in results[0].tags

    @pytest.mark.asyncio
    async def test_lowercase_query_matches_lowercase_stored_tag(self) -> None:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("brain1")

        rows = [_make_fiber_row("fiber-b", ["kb"])]
        storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]

        results = await storage.find_fibers(tags={"kb"})
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_mixed_case_query_matches_lowercase_stored_tag(self) -> None:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("brain1")

        rows = [_make_fiber_row("fiber-c", ["kb"])]
        storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]

        results = await storage.find_fibers(tags={"Kb"})
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_tag_mismatch_returns_empty(self) -> None:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("brain1")

        rows = [_make_fiber_row("fiber-d", ["python"])]
        storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]

        results = await storage.find_fibers(tags={"KB"})
        assert results == []

    @pytest.mark.asyncio
    async def test_all_three_case_variants_match(self) -> None:
        """Upper, lower, and mixed case queries all match the same stored fiber.

        This is the core case-insensitive assertion: querying "KB", "kb", or "Kb"
        must all match a fiber that was stored with tag "kb".
        """
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("brain1")

        rows = [_make_fiber_row("fiber-e", ["kb"])]

        for query_tag in ("KB", "kb", "Kb"):
            storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]
            results = await storage.find_fibers(tags={query_tag})
            assert len(results) == 1, f"Expected match for query tag {query_tag!r}"


# ---------------------------------------------------------------------------
# Boundary 4: brain_transplant._fiber_matches_tags()
# ---------------------------------------------------------------------------


class TestFiberMatchesTagsCaseInsensitive:
    """_fiber_matches_tags must be case-insensitive on both sides."""

    def test_uppercase_required_matches_lowercase_fiber_tag(self) -> None:
        from surreal_memory.engine.brain_transplant import _fiber_matches_tags

        fiber = {"tags": ["kb"]}
        assert _fiber_matches_tags(fiber, frozenset({"KB"}))

    def test_lowercase_required_matches_uppercase_fiber_tag(self) -> None:
        from surreal_memory.engine.brain_transplant import _fiber_matches_tags

        fiber = {"tags": ["KB"]}
        assert _fiber_matches_tags(fiber, frozenset({"kb"}))

    def test_mixed_case_both_sides(self) -> None:
        from surreal_memory.engine.brain_transplant import _fiber_matches_tags

        fiber = {"tags": ["Python"]}
        assert _fiber_matches_tags(fiber, frozenset({"PYTHON"}))

    def test_no_match_returns_false(self) -> None:
        from surreal_memory.engine.brain_transplant import _fiber_matches_tags

        fiber = {"tags": ["python"]}
        assert not _fiber_matches_tags(fiber, frozenset({"KB"}))

    def test_empty_fiber_tags(self) -> None:
        from surreal_memory.engine.brain_transplant import _fiber_matches_tags

        fiber = {"tags": []}
        assert not _fiber_matches_tags(fiber, frozenset({"kb"}))


# ---------------------------------------------------------------------------
# Boundary 0: normalize_tags_lower() utility itself
# ---------------------------------------------------------------------------


class TestNormalizeTagsLower:
    """Basic contract for the normalize_tags_lower() primitive."""

    def test_lowercases_all_tags(self) -> None:
        from surreal_memory.utils.tag_normalizer import normalize_tags_lower

        assert normalize_tags_lower({"KB", "Python", "BACKEND"}) == {"kb", "python", "backend"}

    def test_idempotent(self) -> None:
        from surreal_memory.utils.tag_normalizer import normalize_tags_lower

        tags = {"kb", "python"}
        assert normalize_tags_lower(normalize_tags_lower(tags)) == tags

    def test_empty_set(self) -> None:
        from surreal_memory.utils.tag_normalizer import normalize_tags_lower

        assert normalize_tags_lower(set()) == set()

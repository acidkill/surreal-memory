"""Regression: `smem list` and `smem cleanup --dry-run` must not surface the
`"[graph-only]"` compression sentinel as a memory's preview content.

Four call sites in `cli/commands/listing.py` used a two-step
`fiber.summary → anchor.content` fallback with no third rung. When a fiber
was compressed to the `GRAPH_ONLY` tier, `compression.py` overwrites the
anchor neuron's content with the literal string `"[graph-only]"` — that write
does NOT clear `fiber.essence`, and `_essence_backfill` in
`engine/consolidation.py` may have already generated a real essence from
`anchor.content` before compression ran. Net effect: `smem list` printed
`[graph-only]` as if it were the memory, and `smem cleanup --dry-run` used
the same broken preview right before deletion.

The fix adds a helper (`fiber.summary → anchor.content (skipping
"[graph-only]") → fiber.essence → ""`) and threads it through the four
sites. Behaviour-preserving for everything else — the only user-visible
change is that a `GRAPH_ONLY`-compressed fiber now shows its essence
instead of the tombstone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.cli.commands.listing import _fiber_preview_content, list_memories


def _fiber(
    *,
    summary: str | None = None,
    essence: str | None = None,
    anchor_neuron_id: str | None = "n0",
) -> Any:
    """Minimal fiber-like object with just the fields the helper reads."""
    return SimpleNamespace(
        summary=summary,
        essence=essence,
        anchor_neuron_id=anchor_neuron_id,
    )


def _neuron(content: str) -> Any:
    return SimpleNamespace(content=content)


class TestFiberPreviewContent:
    """The helper used by all four listing.py call sites."""

    @pytest.mark.asyncio
    async def test_returns_summary_when_present(self) -> None:
        storage = SimpleNamespace(get_neuron=AsyncMock())
        fiber = _fiber(summary="short summary", essence="deep essence")
        result = await _fiber_preview_content(fiber, storage)
        assert result == "short summary"
        # Fast-path: don't touch storage if summary already gives us text.
        storage.get_neuron.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_anchor_content_when_summary_empty(self) -> None:
        storage = SimpleNamespace(get_neuron=AsyncMock(return_value=_neuron("real anchor text")))
        fiber = _fiber(summary=None, essence="deep essence")
        result = await _fiber_preview_content(fiber, storage)
        assert result == "real anchor text"

    @pytest.mark.asyncio
    async def test_skips_graph_only_sentinel_and_falls_through_to_essence(self) -> None:
        """The regression: GRAPH_ONLY compressed anchor + backfilled essence."""
        storage = SimpleNamespace(get_neuron=AsyncMock(return_value=_neuron("[graph-only]")))
        fiber = _fiber(summary=None, essence="real human-written essence")
        result = await _fiber_preview_content(fiber, storage)
        assert result == "real human-written essence"

    @pytest.mark.asyncio
    async def test_returns_essence_when_summary_and_anchor_both_empty(self) -> None:
        storage = SimpleNamespace(get_neuron=AsyncMock(return_value=None))
        fiber = _fiber(summary=None, essence="only essence available")
        result = await _fiber_preview_content(fiber, storage)
        assert result == "only essence available"

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_nothing_available(self) -> None:
        storage = SimpleNamespace(get_neuron=AsyncMock(return_value=None))
        fiber = _fiber(summary=None, essence=None)
        result = await _fiber_preview_content(fiber, storage)
        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_lookup_when_no_anchor_neuron_id(self) -> None:
        """Guard: no anchor_neuron_id → don't hit storage, jump to essence."""
        storage = SimpleNamespace(get_neuron=AsyncMock())
        fiber = _fiber(summary=None, essence="essence only", anchor_neuron_id=None)
        result = await _fiber_preview_content(fiber, storage)
        assert result == "essence only"
        storage.get_neuron.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_still_wins_even_if_anchor_is_graph_only(self) -> None:
        """Behaviour-preserving check: summary path unchanged from before the fix."""
        storage = SimpleNamespace(get_neuron=AsyncMock())
        fiber = _fiber(summary="a summary", essence="an essence")
        result = await _fiber_preview_content(fiber, storage)
        assert result == "a summary"
        storage.get_neuron.assert_not_called()


class TestListMemoriesCallSites:
    """The helper is only half the fix — these tests go through the actual
    `list_memories` call sites. With the call sites reverted to the old
    two-step chain, both fail (the preview renders the literal tombstone);
    they exist because an earlier version of this PR's tests measured only
    the helper in isolation and passed with all four call sites unwired."""

    @staticmethod
    def _patched_storage(
        monkeypatch: pytest.MonkeyPatch, *, typed: Any | None, expired: Any | None
    ) -> Any:
        import surreal_memory.cli.commands.listing as listing_mod

        storage = SimpleNamespace()
        if typed is not None:
            storage.find_typed_memories = AsyncMock(return_value=typed)
        if expired is not None:
            storage.get_expired_memories = AsyncMock(return_value=expired)
        storage.get_fiber = AsyncMock(
            return_value=SimpleNamespace(
                summary=None,
                essence="real human essence",
                anchor_neuron_id="n1",
                created_at=datetime(2026, 9, 4, tzinfo=UTC),
                id="f1",
            )
        )
        storage.get_neuron = AsyncMock(return_value=SimpleNamespace(content="[graph-only]"))
        monkeypatch.setattr(listing_mod, "get_storage", AsyncMock(return_value=storage))
        monkeypatch.setattr(listing_mod, "get_config", lambda: SimpleNamespace())
        return storage

    def test_expired_branch_renders_essence_not_tombstone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory

        tm = TypedMemory.create(
            fiber_id="f1", memory_type=MemoryType.DECISION, priority=Priority.NORMAL, source="t"
        )
        self._patched_storage(monkeypatch, typed=None, expired=[tm])
        import surreal_memory.cli.commands.listing as listing_mod

        echoed: list[str] = []
        monkeypatch.setattr(listing_mod.typer, "echo", lambda s, **kw: echoed.append(str(s)))
        list_memories(show_expired=True, json_output=False)
        assert any("real human essence" in line for line in echoed), (
            f"expired listing must show the essence, not the tombstone; got {echoed}"
        )

    def test_typed_branch_renders_essence_not_tombstone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory

        tm = TypedMemory.create(
            fiber_id="f1", memory_type=MemoryType.DECISION, priority=Priority.NORMAL, source="t"
        )
        self._patched_storage(monkeypatch, typed=[tm], expired=None)
        import surreal_memory.cli.commands.listing as listing_mod

        echoed: list[str] = []
        monkeypatch.setattr(listing_mod.typer, "echo", lambda s, **kw: echoed.append(str(s)))
        list_memories(json_output=False)
        assert any("real human essence" in line for line in echoed), (
            f"typed-memory listing must show the essence, not the tombstone; got {echoed}"
        )

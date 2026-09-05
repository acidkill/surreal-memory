"""Regression: PUT /neurons/{id} must refresh content_hash and the embedding.

`update_neuron` in `server/routes/memory.py` used `dataclasses.replace(neuron,
**updates)`, which only touches the fields it is given. `content_hash` and
`metadata["_embedding"]` (both derived from the neuron's text) carried over
from the pre-edit neuron, so a REST update that changed `content` persisted
new text against an old fingerprint and old vector: recall by the new
content missed, recall by the old content still hit. This is the same
shape #166 fixed for `smem_edit` and #193 rolled out across the engine
and MCP handlers via `utils/content_refresh.content_refreshed`; the REST
route was the last write path left with the bug.

The fix threads `content_refreshed` through the route, guarded so it only
fires when `content` actually changed — a metadata-only PUT stays a
one-write op.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.server.routes.memory import update_neuron
from surreal_memory.utils.simhash import simhash


class _Request:
    """Minimal stand-in for NeuronUpdateRequest — only the fields the route reads."""

    def __init__(
        self,
        *,
        type: str | None = None,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.type = type
        self.content = content
        self.metadata = metadata


def _neuron(*, content: str = "old content", metadata: dict[str, Any] | None = None) -> Neuron:
    return Neuron(
        id="n0",
        type=NeuronType.CONCEPT,
        content=content,
        metadata=dict(metadata) if metadata else {},
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_content_change_refreshes_hash() -> None:
    """The regression: a content-change PUT must recompute content_hash."""
    old = _neuron(content="old content")
    old_hash = old.content_hash

    captured: dict[str, Neuron] = {}

    async def _update(neuron: Neuron) -> None:
        captured["written"] = neuron

    storage = SimpleNamespace(
        get_neuron=AsyncMock(return_value=old),
        update_neuron=_update,
        get_brain=AsyncMock(return_value=None),
    )
    brain = type("B", (), {"id": "brain-0"})()

    response = await update_neuron(
        neuron_id="n0",
        request=_Request(content="new content"),
        brain=brain,
        storage=storage,
    )

    written = captured["written"]
    assert written.content == "new content"
    assert written.content_hash == simhash("new content"), (
        f"expected content_hash to reflect the NEW content ({simhash('new content')!r}); "
        f"got {written.content_hash!r} (unchanged from old={old_hash!r})"
    )
    assert response.content == "new content"


@pytest.mark.asyncio
async def test_content_change_refreshes_embedding_when_one_exists() -> None:
    """Neuron with a stored `_embedding` — helper must refresh the vector, not
    persist the old one against the new text."""
    from unittest.mock import patch

    old = _neuron(content="old content", metadata={"_embedding": [0.1, 0.2, 0.3]})

    captured: dict[str, Neuron] = {}

    async def _update(neuron: Neuron) -> None:
        captured["written"] = neuron

    storage = SimpleNamespace(
        get_neuron=AsyncMock(return_value=old),
        update_neuron=_update,
        get_brain=AsyncMock(return_value=None),
    )
    brain = type("B", (), {"id": "brain-0"})()

    # Patch content_refreshed at its import site inside the route module,
    # so the assertion measures that the route CALLS the shared helper —
    # the exact contract this PR restores. A live provider round-trip is
    # covered by the sibling test_edit_forget suite via the same helper.
    refreshed_marker = _neuron(content="new content", metadata={"_embedding": [0.9, 0.8, 0.7]})
    from dataclasses import replace as dc_replace

    from surreal_memory.utils.simhash import simhash as _simhash

    refreshed_marker = dc_replace(refreshed_marker, content_hash=_simhash("new content"))

    async def _fake_content_refreshed(_storage: Any, _neuron: Neuron, _new: str) -> Neuron:
        return refreshed_marker

    with patch(
        "surreal_memory.server.routes.memory.content_refreshed",
        side_effect=_fake_content_refreshed,
    ) as spy:
        await update_neuron(
            neuron_id="n0",
            request=_Request(content="new content"),
            brain=brain,
            storage=storage,
        )

    spy.assert_called_once()
    written = captured["written"]
    assert written.metadata["_embedding"] == [0.9, 0.8, 0.7], (
        "embedding must be the refreshed vector, not the pre-edit one"
    )


@pytest.mark.asyncio
async def test_metadata_only_put_does_not_call_content_refresh() -> None:
    """Behaviour-preserving: metadata-only PUT stays a one-write op, no
    accidental round-trip to the embed provider."""
    from unittest.mock import patch

    old = _neuron(content="unchanged", metadata={"_embedding": [0.1]})
    captured: dict[str, Neuron] = {}

    async def _update(neuron: Neuron) -> None:
        captured["written"] = neuron

    storage = SimpleNamespace(
        get_neuron=AsyncMock(return_value=old),
        update_neuron=_update,
    )
    brain = type("B", (), {"id": "brain-0"})()

    with patch("surreal_memory.server.routes.memory.content_refreshed") as spy:
        await update_neuron(
            neuron_id="n0",
            request=_Request(metadata={"tag": "new"}),
            brain=brain,
            storage=storage,
        )

    spy.assert_not_called()
    assert captured["written"].content == "unchanged"


@pytest.mark.asyncio
async def test_content_and_metadata_put_preserves_embedding_for_refresh() -> None:
    """Regression: a PUT that changes `content` AND ships `metadata` replaces
    the whole metadata dict, so the pre-edit `_embedding` would be gone before
    `content_refreshed` ever saw it — and the old vector would stay in the
    row, describing text that no longer exists. The route must carry the
    internal key forward so the helper re-embeds."""
    from unittest.mock import patch

    old = _neuron(content="old content", metadata={"_embedding": [0.1, 0.2, 0.3]})
    old_vector = old.metadata["_embedding"]

    captured: dict[str, Neuron] = {}

    async def _update(neuron: Neuron) -> None:
        captured["written"] = neuron

    storage = SimpleNamespace(
        get_neuron=AsyncMock(return_value=old),
        update_neuron=_update,
        get_brain=AsyncMock(return_value=None),
    )
    brain = type("B", (), {"id": "brain-0"})()

    seen_by_helper: dict[str, Neuron] = {}

    async def _fake_content_refreshed(storage_: Any, neuron: Neuron, new: str) -> Neuron:
        seen_by_helper["neuron"] = neuron
        refreshed = _neuron(
            content=new, metadata={**neuron.metadata, "_embedding": [0.9, 0.8, 0.7]}
        )
        return refreshed

    with patch(
        "surreal_memory.server.routes.memory.content_refreshed",
        side_effect=_fake_content_refreshed,
    ) as spy:
        await update_neuron(
            neuron_id="n0",
            request=_Request(content="new content", metadata={"tag": "shipped-with-content"}),
            brain=brain,
            storage=storage,
        )

    spy.assert_called_once()
    handed = seen_by_helper["neuron"]
    # The helper receives the pre-content neuron plus the new content string —
    # content_refreshed itself performs the content swap. What the route must
    # guarantee is that the metadata it hands over still carries the vector.
    assert handed.content == "old content"
    assert handed.metadata["tag"] == "shipped-with-content", "caller metadata must survive"
    assert handed.metadata.get("_embedding") == old_vector, (
        "the pre-edit vector must be carried into content_refreshed so it can "
        f"be re-embedded; got {handed.metadata.get('_embedding')!r} (was {old_vector!r})"
    )
    assert captured["written"].metadata["_embedding"] == [0.9, 0.8, 0.7], (
        "the refreshed vector must be what the storage write sees"
    )


@pytest.mark.asyncio
async def test_same_content_put_skips_content_refresh() -> None:
    """Sending the same content in a PUT is a no-op on the derived fields —
    guard prevents an unnecessary embed round-trip."""
    from unittest.mock import patch

    old = _neuron(content="same text")
    captured: dict[str, Neuron] = {}

    async def _update(neuron: Neuron) -> None:
        captured["written"] = neuron

    storage = SimpleNamespace(
        get_neuron=AsyncMock(return_value=old),
        update_neuron=_update,
    )
    brain = type("B", (), {"id": "brain-0"})()

    with patch("surreal_memory.server.routes.memory.content_refreshed") as spy:
        await update_neuron(
            neuron_id="n0",
            request=_Request(content="same text"),
            brain=brain,
            storage=storage,
        )

    spy.assert_not_called()

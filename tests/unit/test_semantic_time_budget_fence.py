"""Offline regression for chunked semantic source verification in one run."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.semantic_validation import SemanticSourceValidation
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceChangedError,
)
from tests.unit.test_semantic_source_revision import (
    _DatabaseFeedSessionStorage,
    _feed_event,
    _token,
)

_OWNER = "owner-a"


class _ValidationStorage(_DatabaseFeedSessionStorage):
    def __init__(self, events: list[dict[str, Any]]) -> None:
        super().__init__(events)
        self.snapshots: dict[tuple[str, int], dict[str, Any]] = {}

    async def save_semantic_discovery_state(
        self, state_id: str, revision: int, payload: dict[str, Any]
    ) -> None:
        key = (state_id, revision)
        existing = self.snapshots.get(key)
        if existing is not None:
            assert existing == payload
        self.snapshots[key] = dict(payload)

    async def load_semantic_discovery_state(
        self, state_id: str, revision: int
    ) -> dict[str, Any] | None:
        return self.snapshots.get((state_id, revision))

    async def _load_semantic_discovery_state_page(
        self, state_id: str, revision: int, **_kwargs: Any
    ) -> dict[str, Any] | None:
        return self.snapshots.get((state_id, revision))


class _Progress:
    lease_lost = False
    brain_id = "brain-a"
    reference_time = _DatabaseFeedSessionStorage([]).now
    state = {"run_id": "run-a"}
    owner_token = _OWNER


def _feed() -> list[dict[str, Any]]:
    return [
        _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
        *(
            _feed_event(
                stamp,
                {"create": {"id": f"audit:noise-{stamp}"}},
            )
            for stamp in range(101, 107)
        ),
    ]


def _install_validation_handler(
    monkeypatch: pytest.MonkeyPatch,
    storage: _ValidationStorage,
    *,
    mutate_after_first_chunk: bool = False,
) -> tuple[ConsolidationEngine, list[int], list[str]]:
    cutoff = storage.now
    token = _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100)
    validation = SemanticSourceValidation(
        storage,
        token,
        reference_time=cutoff,
        run_id="run-a",
        owner_token=_OWNER,
    )
    engine = ConsolidationEngine(storage)
    engine._progress_session = _Progress()  # type: ignore[assignment]
    checkpoints: list[int] = []
    applied: list[str] = []

    async def checkpoint(_reference: dict[str, Any], counters: dict[str, int]) -> None:
        checkpoints.append(counters["validation_cursor"])
        if mutate_after_first_chunk and len(checkpoints) == 1:
            storage.events[3] = _feed_event(
                103,
                {"update": {"id": "neuron:original-source", "brain_id": "brain-a"}},
            )

    async def semantic_link(_report: ConsolidationReport, _dry_run: bool) -> None:
        await validation.assert_unchanged(checkpoint)
        applied.append("semantic-link-apply")

    monkeypatch.setattr(engine, "_semantic_link", semantic_link)
    return engine, checkpoints, applied


@pytest.mark.asyncio
async def test_validator_chunks_continue_inside_engine_dispatch_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    cutoff = _DatabaseFeedSessionStorage([]).now
    storage = _ValidationStorage(_feed())
    engine, checkpoints, applied = _install_validation_handler(monkeypatch, storage)

    await engine._run_strategy(
        ConsolidationStrategy.SEMANTIC_LINK,
        ConsolidationReport(),
        cutoff,
        False,
    )

    assert len(checkpoints) == 3
    assert checkpoints == [102, 104, 106]
    assert storage.queries == [(100, 1000), (102, 1000), (104, 1000), (106, 1000)]
    assert applied == ["semantic-link-apply"]


@pytest.mark.asyncio
async def test_mutation_after_committed_chunk_stops_before_semantic_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    cutoff = _DatabaseFeedSessionStorage([]).now
    storage = _ValidationStorage(_feed())
    engine, checkpoints, applied = _install_validation_handler(
        monkeypatch,
        storage,
        mutate_after_first_chunk=True,
    )

    with pytest.raises(SemanticSourceChangedError, match="changed since"):
        await engine._run_strategy(
            ConsolidationStrategy.SEMANTIC_LINK,
            ConsolidationReport(),
            cutoff,
            False,
        )

    assert len(checkpoints) == 1
    assert applied == []

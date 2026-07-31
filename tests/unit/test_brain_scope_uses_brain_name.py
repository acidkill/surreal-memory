"""Regression test: brain-scoped queries must use the brain *name*, not ``brain.id``.

On the SurrealDB backend every row is scoped by ``brain_id = <brain name>``
(``storage.brain_id``, e.g. ``"default"``), while the ``brain`` record carries a
separate UUID primary key. Callers that passed ``brain.id`` into
``get_stats``/``get_enhanced_stats``/``DiagnosticsEngine.analyze`` therefore
queried a brain scope that holds no rows, so ``smem stats`` and ``smem_health``
reported an empty brain (grade F, EMPTY_BRAIN) on a brain with >10k neurons.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

BRAIN_NAME = "default"
BRAIN_UUID = "00313cb4-61ca-4e69-9784-e51431e99ad7"


class _FakeBrain:
    """Brain record: UUID primary key, name used as the row scope."""

    def __init__(self) -> None:
        self.id = BRAIN_UUID
        self.name = BRAIN_NAME
        self.created_at = datetime(2026, 6, 22)


class _FakeStorage:
    """Storage that only holds rows under the brain *name* scope."""

    def __init__(self) -> None:
        self.brain_id = BRAIN_NAME
        self.scopes_queried: list[str] = []

    def _counts(self, brain_id: str) -> dict[str, int]:
        self.scopes_queried.append(brain_id)
        if brain_id != BRAIN_NAME:
            return {"neuron_count": 0, "synapse_count": 0, "fiber_count": 0}
        return {"neuron_count": 10643, "synapse_count": 126086, "fiber_count": 2226}

    async def get_brain(self, _brain_id: str) -> _FakeBrain:
        return _FakeBrain()

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        return self._counts(brain_id)

    async def get_enhanced_stats(
        self, brain_id: str, include_neuron_types: bool = True
    ) -> dict[str, Any]:
        return dict(self._counts(brain_id))

    async def get_fibers(self, limit: int = 100) -> list[Any]:
        return []

    async def find_typed_memories(self, **_kwargs: Any) -> list[Any]:
        return []

    async def get_expired_memories(self) -> list[Any]:
        return []


def test_cli_stats_queries_the_brain_name_scope(monkeypatch: Any) -> None:
    from surreal_memory.cli.commands import info

    storage = _FakeStorage()

    async def fake_get_storage(_config: Any) -> _FakeStorage:
        return storage

    captured: dict[str, Any] = {}

    monkeypatch.setattr(info, "get_config", lambda: object())
    monkeypatch.setattr(info, "get_storage", fake_get_storage)
    monkeypatch.setattr(info, "output_result", lambda result, _json: captured.update(result))

    info.stats(json_output=True)

    assert storage.scopes_queried == [BRAIN_NAME], (
        f"stats queried scope {storage.scopes_queried!r}; the UUID scope holds no rows"
    )
    assert captured["neuron_count"] == 10643


def test_cli_status_queries_the_brain_name_scope(monkeypatch: Any) -> None:
    from surreal_memory.cli.commands import info

    storage = _FakeStorage()

    async def fake_get_storage(_config: Any) -> _FakeStorage:
        return storage

    captured: dict[str, Any] = {}

    monkeypatch.setattr(info, "get_config", lambda: object())
    monkeypatch.setattr(info, "get_storage", fake_get_storage)
    monkeypatch.setattr(info, "output_result", lambda result, _json: captured.update(result))

    info.status(json_output=True)

    assert storage.scopes_queried == [BRAIN_NAME]


def test_mcp_health_analyzes_the_brain_name_scope(monkeypatch: Any) -> None:
    from surreal_memory.engine import diagnostics as diagnostics_mod
    from surreal_memory.mcp.stats_handler import StatsHandler

    storage = _FakeStorage()
    analyzed: list[str] = []

    class _FakeReport:
        grade = "A"
        purity_score = 1.0
        neuron_count = 10643
        synapse_count = 126086
        fiber_count = 2226
        contradiction_count = 0
        warnings: list[Any] = []
        recommendations: list[Any] = []
        top_penalties: list[Any] = []
        # Not covered by the 0.0 catch-all below: these are `dict | None` on the
        # real report, and None is the "backend can't answer" case the handler
        # checks for. Letting __getattr__ answer 0.0 would make the handler try
        # to copy a float as a mapping.
        stage_distribution: dict[str, int] | None = None
        semantic_gate_blockers: dict[str, int] | None = None

        def __getattr__(self, _name: str) -> Any:
            return 0.0

    class _FakeEngine:
        def __init__(self, _storage: Any) -> None: ...

        async def analyze(self, brain_id: str) -> _FakeReport:
            analyzed.append(brain_id)
            return _FakeReport()

    monkeypatch.setattr(diagnostics_mod, "DiagnosticsEngine", _FakeEngine)

    handler = StatsHandler.__new__(StatsHandler)

    async def fake_get_storage() -> _FakeStorage:
        return storage

    handler.get_storage = fake_get_storage  # type: ignore[method-assign]

    asyncio.run(handler._health({}))

    assert analyzed == [BRAIN_NAME], (
        f"health analyzed scope {analyzed!r}; the UUID scope reports EMPTY_BRAIN"
    )

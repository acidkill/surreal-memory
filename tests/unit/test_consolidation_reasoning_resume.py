from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

BRAIN = "reasoning-resume-brain"
MODEL = "claude-fable-5"
TRACES = [
    "I need to fix this. Let me check the error traceback. I verify the bug is gone.",
    "I need to look at this. Let me check the exception. Verify the crash is fixed.",
    "I need to resolve it. Let me check the failing traceback. I verify the error.",
]


class _Progress:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {"strategy_states": {}}
        self.pause_on_pending = True

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return self.state["strategy_states"].setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        state = self.strategy_state(strategy)
        state.update(
            phase=phase,
            cursor=cursor,
            counters=dict(counters or {}),
        )
        if pending is not None:
            state["pending"] = list(pending)
        else:
            state.pop("pending", None)
        if phase == "reasoning_pattern_pending" and pending and self.pause_on_pending:
            self.pause_on_pending = False
            raise ConsolidationPausedError("simulated interruption after pending save")


class _Namer:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire(self) -> None:
        return None

    async def release(self) -> None:
        return None

    async def rename(
        self, pattern: dict[str, Any], _traces: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls += 1
        return {**pattern, "title": "exact persisted LLM title", "naming_method": "llm"}


async def _seed(storage: InMemoryStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            {
                "trace_hash": f"{MODEL}-{index}",
                "model": MODEL,
                "session_id": "session",
                "project": "project",
                "task_context": "",
                "content": content,
                "content_chars": len(content),
                "created_at": "2026-03-01T00:00:00",
            }
            for index, content in enumerate(TRACES)
        ],
    )


def _config(tmp_path: Path) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(
            mining_enabled=True,
            min_cluster_support=2,
            min_patterns_per_category=1,
            min_confidence=0.2,
            pattern_targets={MODEL: 1},
        ),
    )


@pytest.mark.asyncio
async def test_learn_reasoning_replays_exact_pending_pattern_before_naming_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from surreal_memory.engine import reasoning_distiller

    config = _config(tmp_path)
    monkeypatch.setattr(UnifiedConfig, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(reasoning_distiller, "_get_embedder", lambda *_args, **_kwargs: None)
    namer = _Namer()
    monkeypatch.setattr(reasoning_distiller, "build_namer", lambda _config: namer)

    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage)
    progress = _Progress()
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.LEARN_REASONING
    engine._progress_session = progress  # type: ignore[assignment]

    with pytest.raises(ConsolidationPausedError, match="after pending save"):
        await engine._learn_reasoning(ConsolidationReport(), dry_run=False)

    strategy_state = progress.strategy_state("learn_reasoning")
    assert namer.calls == 1
    assert await storage.find_fibers(metadata_key="_reasoning_pattern", limit=10) == []
    assert len(strategy_state["pending"]) == 1
    saved_pattern = json.loads(strategy_state["pending"][0])
    assert saved_pattern["title"] == "exact persisted LLM title"

    report = ConsolidationReport()
    await engine._learn_reasoning(report, dry_run=False)

    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=10)
    assert namer.calls == 1
    assert len(fibers) == 1
    assert fibers[0].metadata["_reasoning_title"] == "exact persisted LLM title"
    assert report.reasoning_patterns_learned == 1
    assert "pending" not in strategy_state

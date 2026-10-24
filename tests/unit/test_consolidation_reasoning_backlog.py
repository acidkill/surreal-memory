"""Consolidation drains staged reasoning traces beyond one distillation batch."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig


def _trace_rows(model: str, contents: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "trace_hash": f"synthetic-{index}",
            "model": model,
            "session_id": "synthetic-session",
            "project": "unit-test",
            "task_context": "",
            "content": content,
            "content_chars": len(content),
            "created_at": f"2026-03-01T00:{index // 60:02d}:{index % 60:02d}",
        }
        for index, content in enumerate(contents)
    ]


class _ProgressMemoryStorage(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.progress: dict[str, Any] | None = None
        self.lease_owner: str | None = None
        self.pause_after_pending_pattern = False
        self.reject_distill_checkpoint = False
        self.reasoning_batch_fetches = 0

    async def acquire_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int
    ) -> bool:
        if self.lease_owner is not None:
            return False
        self.lease_owner = owner_token
        return True

    async def renew_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int
    ) -> bool:
        return self.lease_owner == owner_token

    async def release_consolidation_lease(self, brain_id: str, owner_token: str) -> bool:
        if self.lease_owner != owner_token:
            return False
        self.lease_owner = None
        return True

    async def get_consolidation_progress(self, brain_id: str) -> dict[str, Any] | None:
        return deepcopy(self.progress)

    async def start_consolidation_progress(self, state: dict[str, Any]) -> dict[str, Any]:
        self.progress = deepcopy(state)
        return deepcopy(state)

    async def claim_consolidation_progress(
        self, brain_id: str, owner_token: str
    ) -> dict[str, Any] | None:
        if self.progress is None:
            return None
        self.progress["owner_token"] = owner_token
        return deepcopy(self.progress)

    async def save_consolidation_progress(
        self, brain_id: str, owner_token: str, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self.lease_owner != owner_token:
            return None
        learning = (state.get("strategy_states") or {}).get("learn_reasoning", {})
        if self.reject_distill_checkpoint and learning.get("phase") == "reasoning_distill":
            return None
        self.progress = deepcopy(state)
        if (
            self.pause_after_pending_pattern
            and learning.get("phase") == "reasoning_pattern_pending"
            and learning.get("pending")
        ):
            self.pause_after_pending_pattern = False
            raise ConsolidationPausedError("simulated stop after durable pattern intent")
        return deepcopy(self.progress)

    async def get_unprocessed_reasoning_traces(
        self, brain_id: str, limit: int = 200, model: str | None = None
    ) -> list[dict[str, Any]]:
        self.reasoning_batch_fetches += 1
        return await super().get_unprocessed_reasoning_traces(brain_id, limit, model)


async def test_learn_reasoning_drains_backlog_beyond_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_id = "reasoning-backlog-test"
    model = "claude-fable-5"
    config = UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain=brain_id,
        reasoning_training=ReasoningTrainingConfig(
            mining_enabled=True,
            pattern_targets={model: 100},
        ),
    )
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: config))
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller._get_embedder",
        lambda *_args, **_kwargs: None,
    )

    storage = InMemoryStorage()
    storage.set_brain(brain_id)
    await storage.insert_reasoning_traces(
        brain_id,
        _trace_rows(model, [f"qwerty plugh foobar payload {index}" for index in range(201)]),
    )

    queued_before = await storage.get_unprocessed_reasoning_traces(brain_id, model=model, limit=500)
    assert len(queued_before) == 201

    await ConsolidationEngine(storage).run([ConsolidationStrategy.LEARN_REASONING])

    queued_after = await storage.get_unprocessed_reasoning_traces(brain_id, model=model, limit=500)
    assert queued_after == []


async def test_learn_reasoning_resumes_durable_pending_pattern_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_id = "reasoning-checkpoint-test"
    model = "claude-fable-5"
    config = UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain=brain_id,
        reasoning_training=ReasoningTrainingConfig(
            mining_enabled=True,
            min_cluster_support=3,
            pattern_targets={model: 100},
        ),
    )
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: config))
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller._get_embedder",
        lambda *_args, **_kwargs: None,
    )

    storage = _ProgressMemoryStorage()
    storage.set_brain(brain_id)
    storage.pause_after_pending_pattern = True
    debug_trace = "I need to fix the bug. Let me check the error traceback. I verify it."
    await storage.insert_reasoning_traces(
        brain_id,
        _trace_rows(
            model,
            [f"qwerty plugh foobar payload {index}" for index in range(200)] + [debug_trace] * 201,
        ),
    )

    engine = ConsolidationEngine(storage)
    first = await engine.run([ConsolidationStrategy.LEARN_REASONING])

    assert first.extra["consolidation_status"] == "paused"
    assert first.reasoning_traces_ingested == 0
    assert storage.progress is not None
    run_id = storage.progress["run_id"]
    checkpoint = storage.progress["strategy_states"]["learn_reasoning"]
    assert checkpoint["phase"] == "reasoning_pattern_pending"
    assert len(checkpoint["pending"]) == 1
    assert checkpoint["counters"] == {
        "models_done": 0,
        "models_total": 1,
        "traces_processed": 200,
        "patterns_learned": 0,
    }
    assert (
        len(await storage.get_unprocessed_reasoning_traces(brain_id, model=model, limit=500)) == 201
    )

    resumed = await engine.run([ConsolidationStrategy.LEARN_REASONING], resume_run_id=run_id)

    assert resumed.extra["consolidation_status"] == "completed"
    assert resumed.reasoning_traces_ingested == 0
    assert storage.progress["status"] == "completed"
    assert storage.progress["strategy_states"]["learn_reasoning"]["counters"] == {
        "models_done": 1,
        "models_total": 1,
        "traces_processed": 401,
        "patterns_learned": 1,
    }
    assert await storage.get_unprocessed_reasoning_traces(brain_id, model=model, limit=500) == []
    patterns = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(patterns) == 1


async def test_learn_reasoning_stops_after_rejected_batch_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_id = "reasoning-fenced-checkpoint-test"
    model = "claude-fable-5"
    config = UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain=brain_id,
        reasoning_training=ReasoningTrainingConfig(
            mining_enabled=True,
            pattern_targets={model: 100},
        ),
    )
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: config))
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller._get_embedder",
        lambda *_args, **_kwargs: None,
    )

    storage = _ProgressMemoryStorage()
    storage.set_brain(brain_id)
    storage.reject_distill_checkpoint = True
    await storage.insert_reasoning_traces(
        brain_id,
        _trace_rows(model, [f"qwerty plugh foobar payload {index}" for index in range(401)]),
    )

    report = await ConsolidationEngine(storage).run([ConsolidationStrategy.LEARN_REASONING])

    assert report.extra["consolidation_status"] == "lease_lost"
    assert storage.reasoning_batch_fetches == 1
    assert (
        len(await storage.get_unprocessed_reasoning_traces(brain_id, model=model, limit=500)) == 201
    )
    assert storage.progress is not None
    assert storage.progress["status"] == "running"
    assert "learn_reasoning" not in storage.progress["completed_strategies"]

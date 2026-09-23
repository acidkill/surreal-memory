from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.reasoning_progress import MiningProgress


class _Progress:
    def __init__(self, pause_phase: str | None = None) -> None:
        self.state: dict[str, Any] = {"strategy_states": {}}
        self.pause_phase = pause_phase
        self.writes: list[dict[str, Any]] = []

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        states = self.state["strategy_states"]
        return states.setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        current = self.strategy_state(strategy)
        current.update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )
        self.writes.append(dict(current))
        if phase == self.pause_phase:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated interruption")


def _engine(
    storage: Any,
    strategy: ConsolidationStrategy,
    progress: _Progress | None = None,
) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, ConsolidationConfig())
    engine._active_strategy = strategy
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


class _CompressionStorage:
    def __init__(self) -> None:
        self.current_brain_id = "brain-1"
        self.fibers = [
            SimpleNamespace(id="fiber-b", pinned=False, metadata={}, compression_tier=0),
            SimpleNamespace(id="fiber-a", pinned=False, metadata={}, compression_tier=0),
        ]
        self.compressed: list[str] = []

    async def get_fibers(self, *, limit: int) -> list[Any]:
        return self.fibers[:limit]

    async def get_brain(self, brain_id: str) -> object:
        assert brain_id == self.current_brain_id
        return object()


@pytest.mark.asyncio
async def test_compress_resumes_after_committed_fiber_without_recompressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    class _CompressionEngine:
        def __init__(self, storage: _CompressionStorage) -> None:
            self.storage = storage

        def determine_target_tier(
            self, fiber: Any, reference_time: datetime, *, heat_score: float
        ) -> int:
            assert reference_time == datetime(2026, 1, 1, tzinfo=UTC)
            assert heat_score == 0.0
            return 1

        async def compress_fiber(
            self,
            fiber: Any,
            target_tier: int,
            *,
            dry_run: bool,
            brain: object | None,
        ) -> Any:
            self.storage.compressed.append(fiber.id)
            if fiber.compression_tier >= target_tier:
                return SimpleNamespace(skipped=True, tokens_saved=0)
            fiber.compression_tier = target_tier
            return SimpleNamespace(skipped=False, tokens_saved=7)

    monkeypatch.setattr(compression, "CompressionEngine", _CompressionEngine)
    storage = _CompressionStorage()
    progress = _Progress(pause_phase="compress_fibers")
    reference_time = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.COMPRESS, progress)._compress(
            ConsolidationReport(), reference_time, dry_run=False
        )

    assert progress.strategy_state("compress")["cursor"] == "fiber-a"
    assert storage.compressed == ["fiber-a"]

    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.COMPRESS, progress)._compress(
        report, reference_time, dry_run=False
    )

    assert storage.compressed == ["fiber-a", "fiber-b"]
    assert report.fibers_compressed == 2
    assert report.tokens_saved == 14
    assert progress.strategy_state("compress")["cursor"] == "fiber-b"


@pytest.mark.asyncio
async def test_compress_does_not_advance_cursor_when_a_fiber_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    class _FailingCompressionEngine:
        def __init__(self, storage: _CompressionStorage) -> None:
            self.storage = storage

        def determine_target_tier(
            self, fiber: Any, _reference_time: datetime, *, heat_score: float
        ) -> int:
            return 1

        async def compress_fiber(
            self,
            fiber: Any,
            _target_tier: int,
            *,
            dry_run: bool,
            brain: object | None,
        ) -> Any:
            if fiber.id == "fiber-b":
                raise RuntimeError("transient compression failure")
            self.storage.compressed.append(fiber.id)
            return SimpleNamespace(skipped=False, tokens_saved=5)

    monkeypatch.setattr(compression, "CompressionEngine", _FailingCompressionEngine)
    storage = _CompressionStorage()
    progress = _Progress()

    with pytest.raises(RuntimeError, match="transient compression failure"):
        await _engine(storage, ConsolidationStrategy.COMPRESS, progress)._compress(
            ConsolidationReport(), datetime(2026, 1, 1, tzinfo=UTC), dry_run=False
        )

    assert storage.compressed == ["fiber-a"]
    assert progress.strategy_state("compress")["cursor"] == "fiber-a"


class _DriftStorage:
    current_brain_id = "brain-1"

    def __init__(self) -> None:
        self.saved: dict[str, int] = {}

    async def get_tag_cooccurrence(self, *, min_count: int) -> list[object]:
        assert min_count > 0
        return []

    async def get_tag_fiber_counts(self) -> dict[str, int]:
        return {}

    async def save_drift_cluster(
        self,
        *,
        cluster_id: str,
        canonical: str,
        members: list[str],
        confidence: float,
        status: str,
    ) -> None:
        assert status == "detected"
        self.saved[cluster_id] = self.saved.get(cluster_id, 0) + 1


@pytest.mark.asyncio
async def test_detect_drift_resumes_from_cluster_cursor_and_upserts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import drift_clusters

    clusters = [
        SimpleNamespace(
            cluster_id="cluster-b",
            cluster=SimpleNamespace(canonical="beta", members={"beta", "gamma"}, confidence=0.8),
        ),
        SimpleNamespace(
            cluster_id="cluster-a",
            cluster=SimpleNamespace(canonical="alpha", members={"alpha", "delta"}, confidence=0.9),
        ),
    ]
    monkeypatch.setattr(drift_clusters, "detect_clusters", lambda _cooccurrences, _counts: clusters)
    storage = _DriftStorage()
    progress = _Progress(pause_phase="drift_clusters")

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.DETECT_DRIFT, progress)._detect_drift(
            ConsolidationReport(), dry_run=False
        )

    assert progress.strategy_state("detect_drift")["cursor"] == "cluster-a"
    assert storage.saved == {"cluster-a": 1}

    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.DETECT_DRIFT, progress)._detect_drift(
        report, dry_run=False
    )

    assert storage.saved == {"cluster-a": 1, "cluster-b": 1}
    assert report.drift_clusters_found == 2
    assert report.drift_clusters_persisted == 2


class _ReasoningConfig:
    class _Training:
        mining_enabled = True

    reasoning_training = _Training()
    data_dir = Path("/tmp/surreal-memory-test")


class _ReasoningStorage:
    current_brain_id = "brain-1"


@pytest.mark.asyncio
async def test_process_reasoning_traces_resumes_at_durable_file_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import reasoning_miner
    from surreal_memory.unified_config import UnifiedConfig

    files = [
        (Path("/synthetic/transcript-a.jsonl"), "project-a"),
        (Path("/synthetic/transcript-b.jsonl"), "project-b"),
    ]
    monkeypatch.setattr(UnifiedConfig, "load", classmethod(lambda cls: _ReasoningConfig()))
    monkeypatch.setattr(
        reasoning_miner,
        "_resolve_transcript_roots",
        lambda _training, claude_dir=None: [Path("/synthetic")],
    )
    monkeypatch.setattr(reasoning_miner, "_discover_all_transcripts", lambda _roots: files)
    monkeypatch.setattr(reasoning_miner, "_STATE_SAVE_EVERY", 1)

    committed: set[str] = set()
    stop_after_first = True

    async def _ingest(
        _storage: Any,
        _brain_id: str,
        _config: Any,
        *,
        progress: Any = None,
    ) -> Any:
        nonlocal stop_after_first
        inserted = 0
        scanned = 0
        for path, _project in files:
            scanned += 1
            if path.name not in committed:
                committed.add(path.name)
                inserted += 1
            if progress is not None:
                progress(
                    MiningProgress(
                        phase="ingesting",
                        files_total=len(files),
                        files_scanned=scanned,
                        traces_found=inserted,
                        traces_ingested=inserted,
                    )
                )
            if stop_after_first:
                stop_after_first = False
                raise ConsolidationPausedError("simulated interruption")
        return SimpleNamespace(
            files_scanned=scanned,
            traces_ingested=inserted,
            traces_scanned=inserted,
        )

    monkeypatch.setattr(reasoning_miner, "ingest_reasoning_traces", _ingest)
    storage = _ReasoningStorage()
    progress = _Progress()

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(
            storage, ConsolidationStrategy.PROCESS_REASONING_TRACES, progress
        )._process_reasoning_traces(ConsolidationReport(), dry_run=False)

    first_cursor = progress.strategy_state("process_reasoning_traces")["cursor"]
    expected_a = f"file:{hashlib.sha256(str(files[0][0]).encode()).hexdigest()}"
    assert first_cursor == expected_a
    assert committed == {"transcript-a.jsonl"}

    report = ConsolidationReport()
    await _engine(
        storage, ConsolidationStrategy.PROCESS_REASONING_TRACES, progress
    )._process_reasoning_traces(report, dry_run=False)

    expected_b = f"file:{hashlib.sha256(str(files[1][0]).encode()).hexdigest()}"
    assert progress.strategy_state("process_reasoning_traces")["cursor"] == expected_b
    assert committed == {"transcript-a.jsonl", "transcript-b.jsonl"}
    assert report.reasoning_traces_ingested == 2


@pytest.mark.asyncio
async def test_learn_reasoning_resumes_from_completed_model_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import reasoning_distiller
    from surreal_memory.unified_config import UnifiedConfig

    monkeypatch.setattr(UnifiedConfig, "load", classmethod(lambda cls: _ReasoningConfig()))
    processed_models: set[str] = set()
    stop_after_first = True

    async def _distill(
        _storage: Any,
        _brain_id: str,
        _config: Any,
        *,
        progress: Any = None,
        pending_patterns: Any = (),
        pattern_checkpoint: Any = None,
    ) -> Any:
        nonlocal stop_after_first
        patterns_created = 0
        traces_processed = 0
        for model in ("model-a", "model-b"):
            if model not in processed_models:
                processed_models.add(model)
                patterns_created += 1
                traces_processed += 2
            if progress is not None:
                progress(
                    MiningProgress(
                        phase="distilling",
                        current_model=model,
                        models_done=len(processed_models),
                        models_total=2,
                        traces_processed=traces_processed,
                        patterns_learned=patterns_created,
                    )
                )
            if stop_after_first:
                stop_after_first = False
                raise ConsolidationPausedError("simulated interruption")
        return SimpleNamespace(
            patterns_learned=patterns_created,
            traces_processed=traces_processed,
            models_seen=2,
        )

    monkeypatch.setattr(reasoning_distiller, "distill_reasoning_patterns", _distill)
    storage = _ReasoningStorage()
    progress = _Progress()

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.LEARN_REASONING, progress)._learn_reasoning(
            ConsolidationReport(), dry_run=False
        )

    assert progress.strategy_state("learn_reasoning")["cursor"] == "model-a"
    assert processed_models == {"model-a"}

    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.LEARN_REASONING, progress)._learn_reasoning(
        report, dry_run=False
    )

    assert progress.strategy_state("learn_reasoning")["cursor"] == "model-b"
    assert processed_models == {"model-a", "model-b"}
    assert report.reasoning_patterns_learned == 2


@pytest.mark.asyncio
async def test_late_strategy_dry_runs_do_not_write_or_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import (
        compression,
        drift_clusters,
        reasoning_distiller,
        reasoning_miner,
    )
    from surreal_memory.unified_config import UnifiedConfig

    class _DryRunCompressionEngine:
        def __init__(self, storage: _CompressionStorage) -> None:
            self.storage = storage

        def determine_target_tier(
            self, fiber: Any, _reference_time: datetime, *, heat_score: float
        ) -> int:
            assert heat_score == 0.0
            return 1

        async def compress_fiber(
            self,
            fiber: Any,
            target_tier: int,
            *,
            dry_run: bool,
            brain: object | None,
        ) -> Any:
            assert dry_run
            assert brain is not None
            self.storage.compression_calls.append(fiber.id)
            return SimpleNamespace(skipped=False, tokens_saved=7)

    clusters = [
        SimpleNamespace(
            cluster_id="cluster-a",
            cluster=SimpleNamespace(canonical="alpha", members={"alpha"}, confidence=0.9),
        )
    ]

    async def _unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run must not invoke a reasoning writer")

    monkeypatch.setattr(compression, "CompressionEngine", _DryRunCompressionEngine)
    monkeypatch.setattr(drift_clusters, "detect_clusters", lambda _pairs, _counts: clusters)
    monkeypatch.setattr(UnifiedConfig, "load", classmethod(lambda cls: _ReasoningConfig()))
    monkeypatch.setattr(reasoning_miner, "ingest_reasoning_traces", _unexpected)
    monkeypatch.setattr(reasoning_distiller, "distill_reasoning_patterns", _unexpected)

    compression_storage = _CompressionStorage()
    compression_storage.compression_calls = []
    compression_progress = _Progress()
    compression_report = ConsolidationReport()
    await _engine(
        compression_storage, ConsolidationStrategy.COMPRESS, compression_progress
    )._compress(compression_report, datetime(2026, 1, 1, tzinfo=UTC), dry_run=True)

    drift_storage = _DriftStorage()
    drift_progress = _Progress()
    drift_report = ConsolidationReport()
    await _engine(drift_storage, ConsolidationStrategy.DETECT_DRIFT, drift_progress)._detect_drift(
        drift_report, dry_run=True
    )

    reasoning_storage = _ReasoningStorage()
    ingest_progress = _Progress()
    await _engine(
        reasoning_storage, ConsolidationStrategy.PROCESS_REASONING_TRACES, ingest_progress
    )._process_reasoning_traces(ConsolidationReport(), dry_run=True)
    distill_progress = _Progress()
    await _engine(
        reasoning_storage, ConsolidationStrategy.LEARN_REASONING, distill_progress
    )._learn_reasoning(ConsolidationReport(), dry_run=True)

    assert compression_storage.compression_calls == ["fiber-a", "fiber-b"]
    assert compression_storage.compressed == []
    assert compression_report.fibers_compressed == 2
    assert compression_report.tokens_saved == 14
    assert drift_storage.saved == {}
    assert drift_report.drift_clusters_found == 1
    assert drift_report.drift_clusters_persisted == 1
    assert not compression_progress.writes
    assert not drift_progress.writes
    assert not ingest_progress.writes
    assert not distill_progress.writes

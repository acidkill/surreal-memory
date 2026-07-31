"""Tests for engine/reasoning_distiller.py — heuristic distillation.

Runs against InMemoryStorage with the embedding provider forced OFF (the
`no_embedder` fixture), exercising the fail-soft keyword-classification +
move-set-Jaccard-clustering path (D5). Covers move segmentation, keyword
classification, clustering/pattern materialization, idempotency, mark-processed,
coverage math, and the LEARN_REASONING consolidation handler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from surreal_memory.engine.reasoning_distiller import (
    DistillResult,
    _classify_by_keywords,
    distill_reasoning_patterns,
    reasoning_coverage,
    segment_moves,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

BRAIN = "b1"


@pytest.fixture
def no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fail-soft (no embedding provider) path — keyword + Jaccard."""
    monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._get_embedder", lambda: None)


def _ucfg(tmp_path: Path, **rt_kw: object) -> UnifiedConfig:
    base: dict[str, object] = {
        "mining_enabled": True,
        "min_cluster_support": 2,
        "min_patterns_per_category": 1,
        "min_confidence": 0.2,
        # Generous per-model targets so the existing fixtures (which create at
        # most a couple of patterns) distill freely; individual tests override.
        "pattern_targets": {"claude-fable-5": 100, "claude-sonnet-5": 100},
    }
    base.update(rt_kw)
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(**base),  # type: ignore[arg-type]
    )


async def _seed(storage: InMemoryStorage, model: str, contents: list[str]) -> None:
    traces = [
        {
            "trace_hash": f"{model}-{i}",
            "model": model,
            "session_id": "s",
            "project": "p",
            "task_context": "",
            "content": c,
            "content_chars": len(c),
            "created_at": "2026-03-01T00:00:00",
        }
        for i, c in enumerate(contents)
    ]
    await storage.insert_reasoning_traces(BRAIN, traces)


# 3 debugging traces sharing the move-set {restate-goal, gather-evidence, verify}.
_DEBUG_TRACES = [
    "I need to fix this. Let me check the error traceback. I verify the bug is gone.",
    "I need to look at this. Let me check the exception. Verify the crash is fixed.",
    "I need to resolve it. Let me check the failing traceback. I verify the error.",
]

# Three 2-trace clusters with DISTINCT move-sets (→ distinct signatures), ordered
# so a small per-model batch fetches one cluster at a time.
_THREE_CLUSTERS = [
    "I need to fix the bug. Let me check the error traceback. I verify it.",
    "I need to fix the bug. Let me check the error traceback. I verify it.",
    "What if the edge case is null? Actually, wait, let me reconsider the boundary.",
    "What if the edge case is null? Actually, wait, let me reconsider the boundary.",
    "My plan: step 1 decompose the problem. Compare option A versus option B.",
    "My plan: step 1 decompose the problem. Compare option A versus option B.",
]


def test_segment_moves_detects_closed_vocab() -> None:
    moves = segment_moves("I need to fix it. Let me check the logs. Then I verify the result.")
    assert "restate-goal" in moves
    assert "gather-evidence" in moves
    assert "verify" in moves
    assert segment_moves("") == []


def test_classify_by_keywords_and_other() -> None:
    cats = ReasoningTrainingConfig().categories
    assert _classify_by_keywords("hit an error with a traceback", cats) == "debugging"
    assert _classify_by_keywords("let me refactor and clean up this module", cats) == "refactoring"
    assert _classify_by_keywords("zzz qwerty plugh foobar nothing here", cats) == "other"


async def test_distill_creates_pattern_fiber(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert isinstance(result, DistillResult)
    assert result.patterns_learned == 1
    assert result.traces_processed == 3

    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 1
    md = fibers[0].metadata
    assert md["_reasoning_pattern"] is True
    assert md["_source_model"] == "claude-fable-5"
    assert md["_reasoning_category"] == "debugging"
    assert md["_reasoning_frequency"] == 3
    assert md["_reasoning_confidence"] == 1.0
    assert md["_reasoning_signature"]
    # CONCEPT neuron for the category + EFFECTIVE_FOR synapse exist.
    cat_neuron = await storage.find_neurons(content_exact="reasoning_category:debugging", limit=1)
    assert cat_neuron


async def test_distill_honors_mining_models_filter(tmp_path: Path, no_embedder: None) -> None:
    # Pre-existing unprocessed traces for two models; a mining_models glob must
    # restrict distillation to the matching model only (HIGH-2 regression: the
    # POST /mine models= override must actually gate distillation, not just ingest).
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    await _seed(storage, "claude-sonnet-5", _DEBUG_TRACES)

    cfg = _ucfg(tmp_path, mining_models=("claude-fable-5",))
    await distill_reasoning_patterns(storage, BRAIN, cfg)

    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert {f.metadata["_source_model"] for f in fibers} == {"claude-fable-5"}
    # sonnet's traces are untouched (not consumed by the restricted run).
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, limit=100)
    assert {t["model"] for t in remaining} == {"claude-sonnet-5"}


async def test_distill_is_idempotent(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    first = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert first.patterns_learned == 1
    # Second pass: traces already processed → nothing new.
    second = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert second.patterns_learned == 0
    assert second.traces_processed == 0
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 1


async def test_distill_marks_traces_processed(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert remaining == []


async def test_below_min_cluster_support_no_pattern(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES[:1])  # only 1 trace, support=2
    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert result.patterns_learned == 0
    # The trace is still marked processed (considered, just not clustered).
    assert result.traces_processed == 1


async def test_other_category_not_distilled(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", ["qwerty plugh foobar", "zzz nothing", "blah blah"])
    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert result.patterns_learned == 0


async def test_reasoning_coverage_math(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path)
    await distill_reasoning_patterns(storage, BRAIN, cfg)

    cov = await reasoning_coverage(storage, "claude-fable-5", cfg)
    assert cov["by_category"]["debugging"] == 1
    assert cov["covered"]["debugging"] is True
    assert cov["coverage_percent"] == round(1 / len(cfg.reasoning_training.categories) * 100, 1)
    # A different model has no coverage.
    cov_other = await reasoning_coverage(storage, "claude-sonnet-5", cfg)
    assert cov_other["coverage_percent"] == 0.0


async def test_coverage_respects_min_patterns_bar(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    # Require 5 patterns/category — the single pattern is below the bar.
    strict = _ucfg(tmp_path, min_patterns_per_category=5)
    cov = await reasoning_coverage(storage, "claude-fable-5", strict)
    assert cov["by_category"]["debugging"] == 1  # still counted
    assert cov["covered"]["debugging"] is False  # but below the coverage bar
    assert cov["coverage_percent"] == 0.0


class _FakeEmbedder:
    """Deterministic one-hot-by-category embedder for the embedding code path."""

    async def embed(self, text: str) -> list[float]:
        return _fake_vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_fake_vec(t) for t in texts]


def _fake_vec(text: str) -> list[float]:
    cats = ReasoningTrainingConfig().categories
    cat = _classify_by_keywords(text, cats)
    vec = [0.0] * (len(cats) + 1)
    vec[cats.index(cat) if cat in cats else len(cats)] = 1.0
    return vec


async def test_distill_uses_embedding_path(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    # Explicit embedder -> exercises _seed_centroids / _embed_texts /
    # _classify_by_vector / cosine clustering / medoid (no live provider needed).
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path), embedder=_FakeEmbedder()
    )
    assert result.patterns_learned == 1
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert fibers[0].metadata["_reasoning_category"] == "debugging"


def test_backtrack_and_hypothesize_moves_match() -> None:
    moves = segment_moves("Actually, I was wrong. My hypothesis is that the bug is here.")
    assert "backtrack" in moves
    assert "hypothesize" in moves


class _RaisingEmbedder:
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("provider down")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider down")


async def test_fail_soft_when_embedder_raises(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    # An embedder that raises must not break distillation — it falls back to
    # keyword classification + move-set clustering (D5).
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path), embedder=_RaisingEmbedder()
    )
    assert result.patterns_learned == 1


async def test_cap_marks_only_consumed_traces(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    contents = (
        ["I need to fix the error bug. Let me check the traceback."] * 2
        + ["I need to refactor and clean up. Let me check the code."] * 2
        + ["I need to design the module architecture boundary. Let me check it."] * 2
        + ["I need to read the documentation docs. Let me check them."] * 2
    )
    await _seed(storage, "claude-fable-5", contents)
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 2})
    )
    assert result.patterns_learned == 2
    # Only the 2 fully-clustered categories are marked processed; the 2 categories
    # the budget never reached keep their traces for the next run (no silent loss).
    assert result.traces_processed == 4
    remaining = await storage.get_unprocessed_reasoning_traces(
        BRAIN, model="claude-fable-5", limit=100
    )
    assert len(remaining) == 4


async def test_distinct_clusters_same_top_moves_not_dropped(
    tmp_path: Path, no_embedder: None
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    a = "I need to fix the bug. Let me check the error. I verify it."
    b = (
        "I need to fix the bug. Let me check the error. I verify. "
        "What if the edge case? Actually, wait. Instead of option 1."
    )
    await _seed(storage, "claude-fable-5", [a, a, b, b])
    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    # Two distinct clusters (Jaccard 0.5) sharing the same top-3-moves title must
    # NOT collide on signature — both materialize.
    assert result.patterns_learned == 2
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 2
    assert len({f.metadata["_reasoning_signature"] for f in fibers}) == 2


# ── Per-model pattern targets (sliders) ──────────────────────────────────────


async def test_target_zero_skips_model_and_leaves_traces(tmp_path: Path, no_embedder: None) -> None:
    # With no target set for the model (default 0), distillation is skipped and
    # its traces stay UNPROCESSED — a preliminary Mine only detects models.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path, pattern_targets={})
    result = await distill_reasoning_patterns(storage, BRAIN, cfg, drain=True)
    assert result.patterns_learned == 0
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert len(remaining) == len(_DEBUG_TRACES)  # untouched, not marked processed


async def test_existing_patterns_count_toward_budget_and_raising_distills_remainder(
    tmp_path: Path, no_embedder: None
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _THREE_CLUSTERS)
    # First run: target 1 → only the first cluster distills (budget 1); the rest
    # stay unprocessed.
    run1 = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 1}), drain=True
    )
    assert run1.patterns_learned == 1
    # Raise the target to 3: 1 already exists → budget 2 → the 2 remaining
    # clusters distill (raising a target picks up exactly the remainder).
    run2 = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 3}), drain=True
    )
    assert run2.patterns_learned == 2
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 3


async def test_drain_clears_backlog_across_multiple_fetches(
    tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tiny per-model batch forces the drain loop to fetch more than twice.
    monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._BATCH_PER_MODEL", 2)
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _THREE_CLUSTERS)  # 3 clusters, batch=2
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 100}), drain=True
    )
    assert result.patterns_learned == 3  # all three clusters drained (3 fetches)
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert remaining == []


async def test_background_run_processes_one_batch_per_model(
    tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # drain=False (background consolidation) does ONE batch per model per run
    # even with a large budget and a bigger backlog.
    monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._BATCH_PER_MODEL", 2)
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _THREE_CLUSTERS)
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 100})
    )
    assert result.patterns_learned == 1  # only the first batch this run
    remaining = await storage.get_unprocessed_reasoning_traces(
        BRAIN, model="claude-fable-5", limit=100
    )
    assert len(remaining) == 4


async def test_drain_terminates_when_a_batch_makes_no_progress(
    tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If a batch consumes nothing (pathological no-forward-progress), the drain
    # loop must break rather than re-fetch the same traces forever.
    async def _no_progress(*_a: object, **_k: object) -> tuple[int, list[object]]:
        return 0, []

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller._process_model_batch", _no_progress
    )
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 100}), drain=True
    )
    assert result.patterns_learned == 0  # terminated without hanging


# ── Consolidation LEARN_REASONING handler ────────────────────────────────────


async def test_consolidation_learn_reasoning_runs(tmp_path: Path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    fake = _ucfg(tmp_path, mining_enabled=True)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))

    async def _fake_distill(storage, brain_id, config, **k):
        return DistillResult(patterns_learned=4, traces_processed=12)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", _fake_distill
    )
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    report = await ConsolidationEngine(storage).run([ConsolidationStrategy.LEARN_REASONING])
    assert report.reasoning_patterns_learned == 4


async def test_consolidation_learn_reasoning_guard_disabled(tmp_path: Path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    fake = _ucfg(tmp_path, mining_enabled=False)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))
    calls = {"n": 0}

    async def _spy(*a, **k):  # pragma: no cover - must NOT run
        calls["n"] += 1
        return DistillResult(patterns_learned=9)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", _spy
    )
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    report = await ConsolidationEngine(storage).run([ConsolidationStrategy.LEARN_REASONING])
    assert report.reasoning_patterns_learned == 0
    assert calls["n"] == 0


class _SpyNamer:
    """Stands in for a PatternNamer, counting renames and releases."""

    def __init__(self) -> None:
        self.renamed = 0
        self.released = 0
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1

    async def rename(self, pattern: dict, cluster_traces: list[dict]) -> dict:
        self.renamed += 1
        return {**pattern, "title": f"llm-named-{self.renamed}"}

    async def release(self) -> None:
        self.released += 1


class TestLLMNamingIsWiredIn:
    """distill_use_llm has to reach the patterns, and let go of the model after."""

    async def test_patterns_carry_the_llm_title(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _SpyNamer()
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: spy
        )
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        assert result.patterns_learned >= 1
        assert spy.renamed >= 1
        fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
        titles = [str(f.metadata.get("_reasoning_title", "")) for f in fibers]
        assert any(t.startswith("llm-named") for t in titles)

    async def test_the_model_is_released_when_the_run_finishes(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _SpyNamer()
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: spy
        )
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        assert spy.acquired == 1
        assert spy.released == 1

    async def test_the_model_is_released_even_when_the_run_blows_up(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-run must not strand a multi-gigabyte model in VRAM."""
        spy = _SpyNamer()
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: spy
        )

        async def _boom(*_a: object, **_k: object) -> tuple[int, list[object]]:
            raise RuntimeError("storage went away")

        monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._process_model_batch", _boom)
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        with pytest.raises(RuntimeError):
            await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        assert spy.acquired == 1
        assert spy.released == 1

    async def test_no_namer_means_the_heuristic_title_survives(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: None
        )
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
        assert fibers
        titles = [str(f.metadata.get("_reasoning_title", "")) for f in fibers]
        assert not any("llm-named" in t for t in titles)


# ── reprocess: rebuilding patterns from an already-processed backlog ───────────


async def test_reset_lets_distillation_revisit_a_drained_backlog(
    tmp_path: Path, no_embedder: None
) -> None:
    # The distiller marks EVERY consumed trace processed, so a brain that lost
    # its pattern fibers has no way back — re-mining finds nothing unprocessed.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    # Long retention: this test is about the reset, not the post-run prune
    # (pinned separately below).
    cfg = _ucfg(tmp_path, retention_days=100_000)
    first = await distill_reasoning_patterns(storage, BRAIN, cfg)
    assert first.patterns_learned == 1
    assert await storage.get_unprocessed_reasoning_traces(BRAIN) == []

    # Simulate the loss: patterns gone, traces still staged but all processed.
    for fiber in await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100):
        await storage.delete_fiber(str(fiber.id))
    stuck = await distill_reasoning_patterns(storage, BRAIN, cfg)
    assert (stuck.patterns_learned, stuck.traces_processed) == (0, 0)

    await storage.reset_reasoning_traces_processed(BRAIN)
    rebuilt = await distill_reasoning_patterns(storage, BRAIN, cfg)

    assert rebuilt.patterns_learned == 1
    cov = await reasoning_coverage(storage, "claude-fable-5", cfg)
    assert cov["covered"]["debugging"] is True


async def test_reset_then_redistill_does_not_duplicate_patterns(
    tmp_path: Path, no_embedder: None
) -> None:
    # Re-opening the backlog must be safe to repeat: pattern signatures make the
    # second pass a no-op rather than a duplicate-maker.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path, retention_days=100_000)
    await distill_reasoning_patterns(storage, BRAIN, cfg)
    before = len(await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100))

    await storage.reset_reasoning_traces_processed(BRAIN)
    second = await distill_reasoning_patterns(storage, BRAIN, cfg)

    after = len(await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100))
    assert after == before
    assert second.patterns_learned == 0


async def test_reset_cannot_recover_traces_the_retention_prune_removed(
    tmp_path: Path, no_embedder: None
) -> None:
    # A distill run prunes processed traces past retention, so reprocess can only
    # rebuild from what is still staged; anything older needs a backfill re-ingest
    # from the transcripts. Seeded traces are older than the 1-day retention here.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path, retention_days=1)

    await distill_reasoning_patterns(storage, BRAIN, cfg)

    assert await storage.get_reasoning_trace_models(BRAIN) == []
    assert await storage.reset_reasoning_traces_processed(BRAIN) == 0

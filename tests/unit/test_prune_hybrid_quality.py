"""Contract tests for the frozen synthetic quality benchmark."""

import json
from unittest.mock import patch

import pytest

from scripts.benchmark_prune_hybrid import FIXTURE, build, gates, score
from surreal_memory.core.brain import BrainConfig
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.engine.retrieval_eval import ReadOnlyStorage
from surreal_memory.unified_config import UnifiedConfig


def test_corpus_has_independent_judgments() -> None:
    corpus = json.loads(FIXTURE.read_text())
    cases = corpus["queries"]
    assert len(cases) >= 30
    assert sum(case["hard"] for case in cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["expected"] for case in cases)
    assert {item for case in cases for item in case["expected"]} <= {
        f"fact-{fact['id']}" for fact in corpus["facts"]
    }


def test_metrics_and_gates_fail_closed() -> None:
    cases = [
        {"id": "a", "expected": ["x"], "critical": True, "hard": True},
        {"id": "b", "expected": ["y"], "critical": False, "hard": False},
    ]
    classic = score(cases, {"a": [], "b": ["z", "y"]})
    hybrid = score(cases, {"a": ["x"], "b": ["y"]})
    assert classic["recall_at_10"] == 0.5
    assert classic["mrr_at_10"] == 0.25
    assert classic["critical_misses"] == ["a"]
    classic["cold_p95_ms"] = classic["warm_p95_ms"] = 2.0
    hybrid["cold_p95_ms"] = hybrid["warm_p95_ms"] = 4.0
    result = gates(classic, hybrid, None, 1.0)
    assert all(value is True for key, value in result.items() if key != "rss_within_limit")
    assert result["rss_within_limit"] is None
    hybrid["warm_p95_ms"] = 4.01
    assert gates(classic, hybrid, 0.5, 1.0)["warm_p95_at_most_twice_classic"] is False
    assert gates(classic, hybrid, 0.5, 1.0)["rss_within_limit"] is False


@pytest.mark.asyncio
async def test_synthetic_query_uses_read_only_real_pipeline() -> None:
    store = await build(json.loads(FIXTURE.read_text()))
    try:
        boundary = ReadOnlyStorage(store)
        with patch("surreal_memory.unified_config.get_config", return_value=UnifiedConfig()):
            pipeline = ReflexPipeline(boundary, BrainConfig(), read_only=True)
            await pipeline.query("atlas deployment", max_tokens=2000)
        assert boundary.read_counts
        assert boundary.write_attempts == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_offline_run_keeps_external_event_sentinel_untouched(tmp_path, monkeypatch) -> None:
    import surreal_memory.engine.reasoning_distiller as distiller
    import surreal_memory.plugins as plugins
    import surreal_memory.unified_config as unified_config
    from scripts import benchmark_prune_hybrid as bench
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport

    host = tmp_path / "outside" / "tool_events.jsonl"
    host.parent.mkdir()
    host.write_text('{"event_id":"sentinel"}\n')

    async def exercise(_seed, _limit):
        isolated = UnifiedConfig.load()
        assert unified_config.get_config() is isolated
        assert not isolated.reasoning_training.distill_use_llm
        assert plugins.get_compression_fn() is None
        assert distiller._get_embedder(isolated) is None
        assert isolated.data_dir != host.parent
        assert isolated.tool_memory.enabled
        assert not (isolated.data_dir / "tool_events.jsonl").exists()
        store = await bench.build(json.loads(FIXTURE.read_text()))
        try:
            report = ConsolidationReport()
            await ConsolidationEngine(store)._process_tool_events(report, False)
            assert report.extra.get("tool_events_ingested", 0) == 0
            assert host.read_text() == '{"event_id":"sentinel"}\n'
            return {"accepted": True}
        finally:
            await store.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("host discovery invoked")

    monkeypatch.setattr(UnifiedConfig, "load", forbidden)
    monkeypatch.setattr(unified_config, "get_config", forbidden)
    monkeypatch.setattr(plugins, "get_compression_fn", forbidden)
    monkeypatch.setattr(distiller, "_get_embedder", forbidden)
    monkeypatch.setattr(bench, "_run_offline", exercise)
    result = await bench.run(42, None)
    assert result["network_attempts"] == 0


@pytest.mark.asyncio
async def test_seeded_runs_have_identical_rankings() -> None:
    from scripts.benchmark_prune_hybrid import run

    first = await run(42, 512)
    second = await run(42, 512)
    assert (
        first["seed_binding"]
        == second["seed_binding"]
        == {
            "query_shuffle": 42,
            "dream_simulation": 42,
        }
    )
    for stage in ("before", "prune", "full_cycle"):
        for strategy in ("classic", "ppr", "hybrid", "auto"):
            assert (
                first["stages"][stage][strategy]["rankings"]
                == second["stages"][stage][strategy]["rankings"]
            )
    receipt = first["consolidation"]["full_cycle"]
    assert receipt["requested_handlers"] == receipt["completed_handlers"]


def test_seeded_rankings_match_across_process_hash_seeds(tmp_path) -> None:
    import os
    import subprocess
    import sys

    from scripts.benchmark_prune_hybrid import ROOT

    results = []
    for hash_seed in ("1", "2"):
        output = tmp_path / f"quality-{hash_seed}.json"
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT / "src")}
        process = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "scripts/benchmark_prune_hybrid.py"),
                "--seed",
                "42",
                "--max-rss-mib",
                "512",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert output.exists(), process.stderr
        assert process.returncode in (0, 1)
        result = json.loads(output.read_text())
        assert result["network_attempts"] == 0
        results.append(result)
    for stage in ("before", "prune", "full_cycle"):
        for strategy in ("classic", "ppr", "hybrid", "auto"):
            assert (
                results[0]["stages"][stage][strategy]["rankings"]
                == results[1]["stages"][stage][strategy]["rankings"]
            )

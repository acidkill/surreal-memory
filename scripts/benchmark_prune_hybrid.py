"""Offline synthetic retrieval benchmark; never reads a configured brain or database."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import resource
import socket
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.engine.retrieval_eval import ReadOnlyStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import UnifiedConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/prune_hybrid_golden.json"
STRATEGIES = ("classic", "ppr", "hybrid", "auto")


async def build(corpus: dict) -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(
        name="synthetic-prune-quality",
        config=BrainConfig(embedding_enabled=False, essence_generator="extractive"),
    )
    await store.save_brain(brain)
    store.set_brain(brain.id)
    hub = Neuron.create(NeuronType.CONCEPT, "unrelated shared hub", neuron_id="hub")
    await store.add_neuron(hub)
    for fact in corpus["facts"]:
        key, topic = fact["id"], fact["topic"]
        target = Neuron.create(
            NeuronType.CONCEPT,
            f"{topic} operational requirements and policy details",
            neuron_id=f"target-{key}",
        )
        anchor = Neuron.create(
            NeuronType.CONCEPT, f"{key} companion operational reference", neuron_id=f"anchor-{key}"
        )
        middle = Neuron.create(
            NeuronType.CONCEPT,
            f"intermediate association bridge {topic}",
            neuron_id=f"bridge-{key}",
        )
        for neuron in (target, anchor, middle):
            await store.add_neuron(neuron)
        for source, destination, suffix in (
            (anchor, middle, "a"),
            (middle, target, "b"),
            (middle, hub, "noise"),
        ):
            await store.add_synapse(
                Synapse.create(
                    source.id,
                    destination.id,
                    SynapseType.RELATED_TO,
                    weight=0.7 if suffix != "noise" else 0.15,
                    synapse_id=f"edge-{key}-{suffix}",
                )
            )
        await store.add_fiber(
            Fiber.create(
                neuron_ids={target.id},
                synapse_ids=set(),
                anchor_neuron_id=target.id,
                summary=fact["summary"],
                fiber_id=f"fact-{key}",
            )
        )
    weak = Neuron.create(NeuronType.CONCEPT, "obsolete unrelated noise", neuron_id="weak-noise")
    await store.add_neuron(weak)
    edge = Synapse.create(
        hub.id, weak.id, SynapseType.RELATED_TO, weight=0.01, synapse_id="old-weak-noise"
    )
    await store.add_synapse(replace(edge, created_at=datetime.fromisoformat("2020-01-01T00:00:00")))
    return store


def graph_manifest(corpus: dict) -> dict:
    return {
        "facts": corpus["facts"],
        "layout": "target topic + operational requirements and policy details; companion name + operational reference; intermediate association bridge topic; bridge->hub noise",
        "associated_edge_weight": 0.7,
        "hub_edge_weight": 0.15,
        "weak_edge": {
            "id": "old-weak-noise",
            "source": "hub",
            "target": "weak-noise",
            "weight": 0.01,
            "created_at": "2020-01-01T00:00:00",
        },
    }


def score(cases: list[dict], rankings: dict[str, list[str]]) -> dict:
    recalls, reciprocals = [], []
    critical_misses, hard_hits = [], []
    for case in cases:
        ranking = rankings[case["id"]][:10]
        expected = set(case["expected"])
        hits = expected.intersection(ranking)
        recalls.append(len(hits) / len(expected))
        reciprocals.append(
            next((1 / (i + 1) for i, item in enumerate(ranking) if item in expected), 0)
        )
        if case["critical"] and hits != expected:
            critical_misses.append(case["id"])
        if case["hard"] and hits:
            hard_hits.append(case["id"])
    return {
        "recall_at_10": sum(recalls) / len(cases),
        "mrr_at_10": sum(reciprocals) / len(cases),
        "critical_misses": critical_misses,
        "hard_hits": hard_hits,
        "hard_recall_at_10": len(hard_hits) / sum(case["hard"] for case in cases),
    }


def gates(classic: dict, hybrid: dict, max_rss_mib: float | None, rss_mib: float) -> dict:
    return {
        "critical": not hybrid["critical_misses"],
        "recall_noninferior": hybrid["recall_at_10"] >= classic["recall_at_10"],
        "mrr_noninferior": hybrid["mrr_at_10"] >= classic["mrr_at_10"],
        "additional_hard_query": hybrid["hard_recall_at_10"] > classic["hard_recall_at_10"]
        and bool(set(hybrid["hard_hits"]) - set(classic["hard_hits"])),
        "cold_p95_at_most_twice_classic": hybrid["cold_p95_ms"] <= 2 * classic["cold_p95_ms"],
        "warm_p95_at_most_twice_classic": hybrid["warm_p95_ms"] <= 2 * classic["warm_p95_ms"],
        "rss_within_limit": None if max_rss_mib is None else rss_mib <= max_rss_mib,
    }


async def measure(store: InMemoryStorage, corpus: dict, seed: int) -> dict:
    boundaries = {s: ReadOnlyStorage(store) for s in STRATEGIES}
    pipelines = {
        s: ReflexPipeline(
            boundaries[s],
            BrainConfig(activation_strategy=s, sufficiency_threshold=0.1),
            read_only=True,
        )
        for s in STRATEGIES
    }
    rankings = {s: {} for s in STRATEGIES}
    durations = {s: [] for s in STRATEGIES}
    randomizer = random.Random(seed)
    for phase in ("cold", "warm"):
        jobs = [(s, case) for s in STRATEGIES for case in corpus["queries"]]
        randomizer.shuffle(jobs)
        for strategy, case in jobs:
            pipeline = (
                ReflexPipeline(
                    boundaries[strategy],
                    BrainConfig(activation_strategy=strategy, sufficiency_threshold=0.1),
                    read_only=True,
                )
                if phase == "cold"
                else pipelines[strategy]
            )
            start = time.perf_counter()
            result = await pipeline.query(case["query"], max_tokens=2000)
            durations[strategy].append((phase, (time.perf_counter() - start) * 1000))
            if phase == "warm":
                rankings[strategy][case["id"]] = list(result.fibers_matched)
    output = {}
    for strategy in STRATEGIES:
        cold = sorted(value for phase, value in durations[strategy] if phase == "cold")
        warm = sorted(value for phase, value in durations[strategy] if phase == "warm")
        output[strategy] = {
            **score(corpus["queries"], rankings[strategy]),
            "cold_p95_ms": cold[math.ceil(len(cold) * 0.95) - 1],
            "warm_p95_ms": warm[math.ceil(len(warm) * 0.95) - 1],
            "latencies_ms": {
                phase: [v for p, v in durations[strategy] if p == phase]
                for phase in ("cold", "warm")
            },
            "read_counts": dict(boundaries[strategy].read_counts),
            "write_attempts": boundaries[strategy].write_attempts,
            "rankings": rankings[strategy],
        }
    return output


async def _run_offline(seed: int, max_rss_mib: float | None) -> dict:
    raw = FIXTURE.read_bytes()
    corpus = json.loads(raw)
    store = await build(corpus)
    stages, reports = {}, {}
    try:
        stages["before"] = await measure(store, corpus, seed)
        engine = ConsolidationEngine(store)
        report = await engine.run(strategies=[ConsolidationStrategy.PRUNE])
        reports["prune"] = {**report.extra, "synapses_pruned": report.synapses_pruned}
        stages["prune"] = await measure(store, corpus, seed)
        completed = []
        original = engine._run_strategy

        async def observe(strategy, report, reference_time, dry_run):
            await original(strategy, report, reference_time, dry_run)
            completed.append(strategy.value)

        with patch.object(engine, "_run_strategy", side_effect=observe):
            report = await engine.run()
        expected = sorted(
            strategy.value
            for strategy in ConsolidationStrategy
            if strategy != ConsolidationStrategy.ALL
        )
        reports["full_cycle"] = {
            **report.extra,
            "requested_handlers": expected,
            "completed_handlers": sorted(completed),
        }
        stages["full_cycle"] = await measure(store, corpus, seed)
    finally:
        await store.close()
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    stage_gates = {
        stage: gates(results["classic"], results["hybrid"], max_rss_mib, rss_mib)
        for stage, results in stages.items()
    }
    for name, result in stage_gates.items():
        result["no_write_attempts"] = all(not v["write_attempts"] for v in stages[name].values())
        result["consolidation_completed"] = (
            True
            if name == "before"
            else reports["prune" if name == "prune" else "full_cycle"].get("consolidation_status")
            == "completed"
        )
    stage_gates["full_cycle"]["all_handlers_returned"] = (
        reports["full_cycle"]["requested_handlers"] == reports["full_cycle"]["completed_handlers"]
    )
    stage_gates["prune"]["noise_pruned"] = reports["prune"]["synapses_pruned"] > 0
    for stage in ("prune", "full_cycle"):
        for strategy in STRATEGIES:
            before, after = stages["before"][strategy], stages[stage][strategy]
            stage_gates[stage][f"{strategy}_recall_preserved"] = (
                after["recall_at_10"] >= before["recall_at_10"]
            )
            stage_gates[stage][f"{strategy}_mrr_preserved"] = (
                after["mrr_at_10"] >= before["mrr_at_10"]
            )
            stage_gates[stage][f"{strategy}_critical_preserved"] = set(
                after["critical_misses"]
            ) <= set(before["critical_misses"])
    return {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "graph_sha256": hashlib.sha256(
            json.dumps(graph_manifest(corpus), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": seed,
        "seed_binding": {"query_shuffle": seed, "dream_simulation": seed},
        "concurrency": 1,
        "aggregate_peak_process_rss_mib": rss_mib,
        "rss_limit_mib": max_rss_mib,
        "stages": stages,
        "consolidation": reports,
        "gates": stage_gates,
        "accepted": all(
            value is True for group in stage_gates.values() for value in group.values()
        ),
    }


async def run(seed: int, max_rss_mib: float | None) -> dict:
    # Point configuration and transcript discovery at an empty temporary fixture root.
    from surreal_memory.engine import dream as dream_module

    real_dream = dream_module.dream

    async def seeded_dream(storage, config, seed=None, **kwargs):
        return await real_dream(
            storage, config, seed=seed if seed is not None else benchmark_seed, **kwargs
        )

    benchmark_seed = seed
    with tempfile.TemporaryDirectory(prefix="smem-quality-") as directory:
        root = Path(directory)
        config = UnifiedConfig(data_dir=root, current_brain="synthetic")
        config.embedding = replace(config.embedding, enabled=False)
        config.reranker = replace(config.reranker, enabled=False)
        config.reasoning_training = replace(
            config.reasoning_training,
            mining_enabled=True,
            extra_transcript_dirs=[],
            distill_use_llm=False,
        )
        with (
            patch.object(socket, "socket", side_effect=RuntimeError("network disabled")) as blocked,
            patch("surreal_memory.unified_config.get_config", return_value=config),
            patch.object(UnifiedConfig, "load", return_value=config),
            patch("surreal_memory.plugins.get_compression_fn", return_value=None),
            patch.object(dream_module, "dream", side_effect=seeded_dream),
            patch("surreal_memory.engine.reasoning_distiller._get_embedder", return_value=None),
            patch(
                "surreal_memory.engine.reasoning_miner._resolve_transcript_roots",
                return_value=[root],
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(False, "", ""),
            ),
        ):
            result = await _run_offline(seed, max_rss_mib)
        result["network_attempts"] = blocked.call_count
        result["accepted"] = result["accepted"] and blocked.call_count == 0
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rss-mib", type=float)
    args = parser.parse_args()
    result = asyncio.run(run(args.seed, args.max_rss_mib))
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()

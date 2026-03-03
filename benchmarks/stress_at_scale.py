"""
Mega stress test — benchmark encode, recall, consolidation, and diagnostics
at 1K, 5K, and 10K memories on real SQLite.

Usage:
    python benchmarks/stress_at_scale.py

Outputs:
    - Console: live progress + results
    - docs/benchmarks.md: appends SQLite-at-scale section
"""

from __future__ import annotations

import asyncio
import gc
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from neural_memory.engine.diagnostics import DiagnosticsEngine
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.storage.sqlite_store import SQLiteStorage

# ── Content generators ───────────────────────────────────────────────────────

TOPICS = [
    "Python", "JavaScript", "Rust", "Go", "TypeScript", "Java", "C++", "Ruby",
    "PostgreSQL", "Redis", "MongoDB", "MySQL", "SQLite", "Elasticsearch", "Cassandra",
    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions", "ArgoCD",
    "React", "Vue", "Angular", "Svelte", "Next.js", "FastAPI", "Django", "Flask",
    "JWT", "OAuth2", "CORS", "HTTPS", "WebSocket", "gRPC", "GraphQL", "REST",
    "AWS", "GCP", "Azure", "Cloudflare", "Vercel", "Netlify", "DigitalOcean",
    "Machine Learning", "Neural Networks", "NLP", "Computer Vision", "Transformers",
]

ACTIONS = [
    "supports", "implements", "requires", "provides", "enables", "handles",
    "manages", "processes", "validates", "optimizes", "replaces", "extends",
    "integrates with", "depends on", "supersedes", "enhances",
]

FEATURES = [
    "concurrent request handling", "type-safe data validation",
    "automatic schema generation", "efficient memory management",
    "distributed caching layers", "real-time event streaming",
    "structured error handling", "automated test discovery",
    "incremental compilation", "hot module replacement",
    "connection pooling", "query optimization", "load balancing",
    "rate limiting", "health monitoring", "circuit breaker pattern",
    "retry with exponential backoff", "blue-green deployment",
    "canary releases", "feature flag management",
    "structured logging", "distributed tracing",
    "authentication middleware", "authorization policies",
    "input sanitization", "CSRF protection",
]

MEMORY_TYPES = ["fact", "decision", "error", "insight", "todo", "workflow", "context"]

DECISION_TEMPLATES = [
    "We decided to use {topic1} instead of {topic2} because {reason}",
    "Chose {topic1} over {topic2} for {feature}",
    "After evaluating both, {topic1} was selected for {feature}",
]

ERROR_TEMPLATES = [
    "ConnectionError when {topic1} tried to connect to {topic2}: timeout after 30s",
    "ImportError in {topic1} module: missing dependency for {feature}",
    "{topic1} failed during {feature} with exit code 1",
]

INSIGHT_TEMPLATES = [
    "Pattern: {topic1} {action} {feature} more efficiently than {topic2}",
    "{topic1} and {topic2} both {action} {feature} but through different mechanisms",
    "Root cause: {topic1} {feature} breaks when {topic2} is unavailable",
]

REASONS = [
    "better performance under load", "stronger type safety",
    "more active community", "better documentation",
    "lower operational cost", "simpler deployment",
    "native async support", "better error messages",
]


def generate_diverse_memories(n: int) -> list[tuple[str, str]]:
    """Generate N unique, diverse memories with types. Returns [(content, type)]."""
    random.seed(42)
    memories: list[tuple[str, str]] = []

    for i in range(n):
        t1 = random.choice(TOPICS)
        t2 = random.choice([t for t in TOPICS if t != t1])
        action = random.choice(ACTIONS)
        feature = random.choice(FEATURES)
        reason = random.choice(REASONS)
        mtype = MEMORY_TYPES[i % len(MEMORY_TYPES)]

        if mtype == "fact":
            content = f"{t1} {action} {feature} (fact #{i})"
        elif mtype == "decision":
            tmpl = random.choice(DECISION_TEMPLATES)
            content = tmpl.format(topic1=t1, topic2=t2, feature=feature, reason=reason)
        elif mtype == "error":
            tmpl = random.choice(ERROR_TEMPLATES)
            content = tmpl.format(topic1=t1, topic2=t2, feature=feature)
        elif mtype == "insight":
            tmpl = random.choice(INSIGHT_TEMPLATES)
            content = tmpl.format(topic1=t1, topic2=t2, action=action, feature=feature)
        elif mtype == "todo":
            content = f"TODO: Implement {feature} using {t1} before next release"
        elif mtype == "workflow":
            content = f"Workflow: {t1} → {t2} → {feature} → deploy"
        else:
            content = f"Context: {t1} team meeting discussed {feature} with {t2} integration"

        memories.append((content, mtype))

    return memories


# ── Benchmark runners ────────────────────────────────────────────────────────


async def bench_encode(
    storage: SQLiteStorage,
    encoder: MemoryEncoder,
    memories: list[tuple[str, str]],
    batch_label: str,
) -> dict:
    """Encode all memories, track per-memory timing."""
    times: list[float] = []
    errors = 0

    print(f"  [{batch_label}] Encoding {len(memories)} memories...", end="", flush=True)
    t_start = time.perf_counter()

    for i, (content, mtype) in enumerate(memories):
        t0 = time.perf_counter()
        try:
            await encoder.encode(content, tags={mtype})
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n    ERROR at #{i}: {e}")
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        # Progress dots
        if (i + 1) % 500 == 0:
            print(f" {i + 1}", end="", flush=True)

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f" done ({total_ms / 1000:.1f}s)")

    return {
        "count": len(memories),
        "total_ms": round(total_ms, 1),
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 2),
        "max_ms": round(max(times), 2),
        "throughput": round(len(memories) / (total_ms / 1000), 1),
        "errors": errors,
    }


async def bench_recall(
    storage: SQLiteStorage,
    config: BrainConfig,
    queries: list[tuple[str, DepthLevel]],
    n_runs: int,
    label: str,
) -> list[dict]:
    """Run recall queries and measure latency."""
    brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
    assert brain is not None
    pipeline = ReflexPipeline(storage=storage, config=config)

    results: list[dict] = []
    for query, depth in queries:
        times: list[float] = []
        last = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = await pipeline.query(query, depth=depth)
            times.append((time.perf_counter() - t0) * 1000)
            last = result

        results.append({
            "query": query,
            "depth": depth.name,
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
            "neurons": last.neurons_activated if last else 0,
            "confidence": round(last.confidence, 2) if last else 0,
            "has_answer": bool(last and last.context),
        })

    return results


async def bench_consolidation(storage: SQLiteStorage) -> dict:
    """Run full consolidation and measure time."""
    print("  Consolidating...", end="", flush=True)
    engine = ConsolidationEngine(storage=storage)
    t0 = time.perf_counter()
    report = await engine.run(strategies=[ConsolidationStrategy.ALL])
    elapsed = (time.perf_counter() - t0) * 1000
    print(f" done ({elapsed / 1000:.1f}s)")

    return {
        "duration_ms": round(elapsed, 1),
        "synapses_pruned": report.synapses_pruned,
        "neurons_pruned": report.neurons_pruned,
        "fibers_merged": report.fibers_merged,
        "synapses_enriched": report.synapses_enriched,
    }


async def bench_diagnostics(storage: SQLiteStorage) -> dict:
    """Run diagnostics and extract health metrics."""
    brain_id = storage._current_brain_id
    assert brain_id is not None

    engine = DiagnosticsEngine(storage=storage)
    t0 = time.perf_counter()
    report = await engine.analyze(brain_id)
    elapsed = (time.perf_counter() - t0) * 1000

    stats = await storage.get_stats(brain_id)

    return {
        "diagnostics_ms": round(elapsed, 1),
        "grade": report.grade,
        "purity": round(report.purity_score, 1),
        "connectivity": round(report.connectivity, 3),
        "diversity": round(report.diversity, 3),
        "freshness": round(report.freshness, 3),
        "orphan_rate": round(report.orphan_rate, 3),
        "neuron_count": stats["neuron_count"],
        "synapse_count": stats["synapse_count"],
        "fiber_count": stats["fiber_count"],
        "warnings": len(report.warnings),
        "critical_warnings": len([w for w in report.warnings if w.severity.name == "CRITICAL"]),
    }


async def get_db_size(db_path: str) -> float:
    """Get database file size in MB."""
    path = Path(db_path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


# ── Recall queries ───────────────────────────────────────────────────────────

RECALL_QUERIES = [
    ("Python concurrency", DepthLevel.INSTANT),
    ("What database did we choose?", DepthLevel.CONTEXT),
    ("connection error Redis", DepthLevel.INSTANT),
    ("deployment workflow", DepthLevel.CONTEXT),
    ("Why did we choose PostgreSQL?", DepthLevel.DEEP),
    ("authentication JWT", DepthLevel.INSTANT),
    ("What patterns were discovered?", DepthLevel.CONTEXT),
    ("machine learning integration", DepthLevel.DEEP),
    ("rate limiting implementation", DepthLevel.INSTANT),
    ("TODO before release", DepthLevel.CONTEXT),
]


# ── Main benchmark ───────────────────────────────────────────────────────────


async def run_scale_benchmark(n_memories: int, tmpdir: str) -> dict:
    """Run full benchmark at a given scale."""
    print(f"\n{'=' * 60}")
    print(f"  SCALE: {n_memories:,} memories")
    print(f"{'=' * 60}")

    db_path = os.path.join(tmpdir, f"bench_{n_memories}.db")
    storage = SQLiteStorage(db_path=str(db_path))
    await storage.initialize()

    config = BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.15,
        max_spread_hops=4,
        max_context_tokens=1500,
    )
    brain = Brain.create(name=f"bench-{n_memories}", config=config)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage=storage, config=config)
    memories = generate_diverse_memories(n_memories)

    # Phase 1: Encode
    encode_result = await bench_encode(storage, encoder, memories, f"{n_memories:,}")

    db_size_after_encode = await get_db_size(db_path)
    print(f"  DB size after encode: {db_size_after_encode:.1f} MB")

    # Phase 2: Recall (pre-consolidation)
    print(f"  Running recall queries (pre-consolidation)...")
    recall_pre = await bench_recall(storage, config, RECALL_QUERIES, n_runs=5, label="pre")

    # Phase 3: Diagnostics (pre-consolidation)
    print(f"  Running diagnostics (pre-consolidation)...")
    diag_pre = await bench_diagnostics(storage)
    print(f"    Grade: {diag_pre['grade']} | Purity: {diag_pre['purity']} | "
          f"Neurons: {diag_pre['neuron_count']:,} | Synapses: {diag_pre['synapse_count']:,} | "
          f"Fibers: {diag_pre['fiber_count']:,}")

    # Phase 4: Consolidation
    consolidation = await bench_consolidation(storage)

    db_size_after_consolidation = await get_db_size(db_path)
    print(f"  DB size after consolidation: {db_size_after_consolidation:.1f} MB")

    # Phase 5: Recall (post-consolidation)
    print(f"  Running recall queries (post-consolidation)...")
    recall_post = await bench_recall(storage, config, RECALL_QUERIES, n_runs=5, label="post")

    # Phase 6: Diagnostics (post-consolidation)
    print(f"  Running diagnostics (post-consolidation)...")
    diag_post = await bench_diagnostics(storage)
    print(f"    Grade: {diag_post['grade']} | Purity: {diag_post['purity']} | "
          f"Neurons: {diag_post['neuron_count']:,} | Synapses: {diag_post['synapse_count']:,} | "
          f"Fibers: {diag_post['fiber_count']:,}")

    await storage.close()
    gc.collect()

    return {
        "scale": n_memories,
        "encode": encode_result,
        "db_size_mb": round(db_size_after_encode, 2),
        "db_size_post_consolidation_mb": round(db_size_after_consolidation, 2),
        "recall_pre": recall_pre,
        "recall_post": recall_post,
        "diag_pre": diag_pre,
        "diag_post": diag_post,
        "consolidation": consolidation,
    }


# ── Markdown generation ──────────────────────────────────────────────────────


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_scale_markdown(results: list[dict], timestamp: str) -> str:
    sections: list[str] = []

    sections.append("## SQLite at Scale\n")
    sections.append(f"Last updated: **{timestamp}**\n")
    sections.append("Real SQLiteStorage benchmarks with diverse memory types on Windows 11.\n")

    # ── Encode throughput ──
    sections.append("### Encode Throughput\n")
    headers = ["Memories", "Total (s)", "Mean (ms)", "Median (ms)", "P95 (ms)", "P99 (ms)", "Throughput (mem/s)", "Errors"]
    rows = []
    for r in results:
        e = r["encode"]
        rows.append([
            f"{r['scale']:,}",
            f"{e['total_ms'] / 1000:.1f}",
            str(e["mean_ms"]),
            str(e["median_ms"]),
            str(e["p95_ms"]),
            str(e["p99_ms"]),
            str(e["throughput"]),
            str(e["errors"]),
        ])
    sections.append(md_table(headers, rows))

    # ── Database size ──
    sections.append("\n### Database Size\n")
    headers = ["Memories", "After Encode (MB)", "After Consolidation (MB)", "Neurons", "Synapses", "Fibers"]
    rows = []
    for r in results:
        d = r["diag_pre"]
        rows.append([
            f"{r['scale']:,}",
            str(r["db_size_mb"]),
            str(r["db_size_post_consolidation_mb"]),
            f"{d['neuron_count']:,}",
            f"{d['synapse_count']:,}",
            f"{d['fiber_count']:,}",
        ])
    sections.append(md_table(headers, rows))

    # ── Recall latency ──
    sections.append("\n### Recall Latency (Post-Consolidation)\n")
    sections.append("10 queries, 5 runs each (median reported).\n")
    for r in results:
        sections.append(f"\n#### {r['scale']:,} memories\n")
        headers = ["Query", "Depth", "Median (ms)", "P95 (ms)", "Neurons", "Confidence", "Found"]
        rows = []
        for q in r["recall_post"]:
            rows.append([
                q["query"],
                q["depth"],
                str(q["median_ms"]),
                str(q["p95_ms"]),
                str(q["neurons"]),
                str(q["confidence"]),
                "yes" if q["has_answer"] else "no",
            ])
        # Add average row
        avg_median = round(statistics.mean(q["median_ms"] for q in r["recall_post"]), 2)
        avg_p95 = round(statistics.mean(q["p95_ms"] for q in r["recall_post"]), 2)
        avg_neurons = round(statistics.mean(q["neurons"] for q in r["recall_post"]), 1)
        rows.append([
            "**Average**", "", f"**{avg_median}**", f"**{avg_p95}**",
            f"**{avg_neurons}**", "", "",
        ])
        sections.append(md_table(headers, rows))

    # ── Consolidation ──
    sections.append("\n### Consolidation Performance\n")
    headers = ["Memories", "Duration (s)", "Synapses Pruned", "Neurons Pruned", "Fibers Merged", "Synapses Enriched"]
    rows = []
    for r in results:
        c = r["consolidation"]
        rows.append([
            f"{r['scale']:,}",
            f"{c['duration_ms'] / 1000:.1f}",
            str(c["synapses_pruned"]),
            str(c["neurons_pruned"]),
            str(c["fibers_merged"]),
            str(c["synapses_enriched"]),
        ])
    sections.append(md_table(headers, rows))

    # ── Health ──
    sections.append("\n### Health Diagnostics\n")
    headers = ["Memories", "Phase", "Grade", "Purity", "Connectivity", "Diversity", "Freshness", "Orphan Rate", "Warnings", "Diagnostics (ms)"]
    rows = []
    for r in results:
        for phase, key in [("Pre", "diag_pre"), ("Post", "diag_post")]:
            d = r[key]
            rows.append([
                f"{r['scale']:,}",
                phase,
                d["grade"],
                str(d["purity"]),
                str(d["connectivity"]),
                str(d["diversity"]),
                str(d["freshness"]),
                str(d["orphan_rate"]),
                str(d["warnings"]),
                str(d["diagnostics_ms"]),
            ])
    sections.append(md_table(headers, rows))

    # ── Methodology ──
    sections.append("\n### Methodology\n")
    sections.append("""
- **Storage**: Real SQLiteStorage (aiosqlite, WAL mode)
- **Platform**: Windows 11, single-threaded async
- **Memory types**: 7 types (fact, decision, error, insight, todo, workflow, context)
- **Content**: Diverse generated content from 50 topics × 16 actions × 26 features
- **Recall runs**: 5 per query (median reported)
- **Seed**: `random.seed(42)` for reproducibility
""".strip())

    return "\n\n".join(sections) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Default scales. Override with env: BENCH_SCALES="50000,100000"
    env_scales = os.environ.get("BENCH_SCALES", "")
    if env_scales:
        scales = [int(s.strip()) for s in env_scales.split(",")]
    else:
        scales = [1000, 5000, 10000]
    results: list[dict] = []

    with TemporaryDirectory(prefix="nmem_bench_") as tmpdir:
        for n in scales:
            result = await run_scale_benchmark(n, tmpdir)
            results.append(result)

    # Generate markdown
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    scale_md = generate_scale_markdown(results, timestamp)

    # Read existing benchmarks.md and append
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    bench_path = docs_dir / "benchmarks.md"

    if bench_path.exists():
        existing = bench_path.read_text(encoding="utf-8")
        # Remove old SQLite at Scale section if present
        marker = "## SQLite at Scale"
        if marker in existing:
            existing = existing[:existing.index(marker)].rstrip() + "\n\n"
        combined = existing + scale_md
    else:
        combined = scale_md

    bench_path.write_text(combined, encoding="utf-8")
    print(f"\nWrote results to {bench_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        e = r["encode"]
        d = r["diag_post"]
        c = r["consolidation"]
        avg_recall = round(statistics.mean(q["median_ms"] for q in r["recall_post"]), 2)
        print(f"  {r['scale']:>6,} memories: "
              f"encode={e['throughput']} mem/s, "
              f"recall_avg={avg_recall}ms, "
              f"consolidation={c['duration_ms'] / 1000:.1f}s, "
              f"grade={d['grade']}, "
              f"db={r['db_size_post_consolidation_mb']}MB")


if __name__ == "__main__":
    asyncio.run(main())

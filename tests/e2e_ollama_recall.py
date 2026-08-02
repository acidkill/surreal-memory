"""E2E test: Train motorcycle manual → Recall in English via Ollama bge-m3 embeddings."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_ollama")

for name in ["surreal_memory.engine", "surreal_memory.storage", "surreal_memory.safety"]:
    logging.getLogger(name).setLevel(logging.WARNING)

MD_PATH = os.environ.get("OLLAMA_E2E_MANUAL", "/private/tmp/husqvarna-train/husqvarna-manual.md")

# Same 6 base queries as Gemini E2E
BASE_QUERIES = [
    ("EN", "How to change oil on KTM motorcycle?"),
    ("EN", "What is the recommended tire pressure?"),
    ("EN", "engine coolant specifications"),
    ("EN", "How does the brake system work?"),
    ("EN", "What tire size is recommended?"),
    ("EN", "What type of oil is suitable for the bike?"),
]

# 100 English recall queries (same as Gemini 100-query test)
RECALL_QUERIES: list[tuple[str, str]] = [
    ("oil", "how to change engine oil on KTM motorcycle"),
    ("oil", "which engine oil is suitable for KTM motorcycle"),
    ("oil", "when should the engine oil be changed"),
    ("oil", "what is the engine oil capacity"),
    ("oil", "steps to check the engine oil level"),
    ("oil", "synthetic or mineral oil for the motorcycle"),
    ("oil", "how to replace the oil filter"),
    ("oil", "what SAE oil viscosity is recommended"),
    ("oil", "is gearbox oil different from engine oil"),
    ("oil", "how often to change the motorcycle oil"),
    ("tire", "what is the tire pressure"),
    ("tire", "how to check tire pressure"),
    ("tire", "when should the tires be replaced"),
    ("tire", "which tires are suitable for KTM motorcycle"),
    ("tire", "front and rear tire sizes"),
    ("tire", "how does the TPMS tire pressure sensor system work"),
    ("tire", "tire pressure when carrying two people"),
    ("tire", "how to change a motorcycle tire"),
    ("tire", "what causes a tire to wear on one side"),
    ("tire", "how does temperature affect tire pressure"),
    ("brake", "how does the brake system work"),
    ("brake", "how to check the brake pads"),
    ("brake", "when should the brake fluid be changed"),
    ("brake", "which brake fluid DOT rating is suitable"),
    ("brake", "how does the ABS system work"),
    ("brake", "how to bleed the brake system"),
    ("brake", "how do front and rear brakes differ"),
    ("brake", "what does the brake warning light mean"),
    ("brake", "what is the minimum brake pad thickness"),
    ("brake", "how to adjust the brake lever"),
    ("engine", "KTM motorcycle engine specifications"),
    ("engine", "normal engine operating temperature"),
    ("engine", "how to start the bike in cold weather"),
    ("engine", "what causes unusual noise from the engine"),
    ("engine", "how does the engine cooling system work"),
    ("engine", "which coolant type is suitable"),
    ("engine", "what is the maximum power of the bike"),
    ("engine", "what is the maximum torque"),
    ("engine", "what is the maximum engine rpm"),
    ("engine", "how does the EFI fuel injection system work"),
    ("elec", "how to replace the battery"),
    ("elec", "which battery type is suitable for the bike"),
    ("elec", "how to check the battery charging system"),
    ("elec", "what do the dashboard indicator lights mean"),
    ("elec", "how to replace the headlight bulb"),
    ("elec", "how does the ignition system work"),
    ("elec", "where are the fuses located"),
    ("elec", "how to check the spark plug"),
    ("elec", "what is the spark plug gap"),
    ("elec", "how to charge the battery"),
    ("chain", "how to tension the chain"),
    ("chain", "when should the chain be replaced"),
    ("chain", "which chain lubricant is best"),
    ("chain", "what is the standard chain slack"),
    ("chain", "how to clean the chain"),
    ("chain", "should the sprockets and chain be replaced together"),
    ("chain", "number of teeth on front and rear sprockets"),
    ("chain", "how to check chain wear"),
    ("chain", "is a DID or RK chain better"),
    ("chain", "how often to lubricate the chain"),
    ("susp", "how to adjust the front suspension"),
    ("susp", "what suspension stiffness is appropriate"),
    ("susp", "can the rear suspension be adjusted"),
    ("susp", "what nitrogen pressure for the suspension"),
    ("susp", "when should the suspension oil be changed"),
    ("susp", "what is the seat height"),
    ("susp", "what is the maximum load capacity"),
    ("susp", "what is the front suspension travel in mm"),
    ("susp", "how to check for a damaged suspension"),
    ("susp", "is the stock suspension any good"),
    ("maint", "KTM periodic maintenance schedule"),
    ("maint", "pre-ride inspection checklist"),
    ("maint", "how to replace the air filter"),
    ("maint", "when to clean the fuel injectors"),
    ("maint", "what is the fuel tank capacity in liters"),
    ("maint", "what fuel RON rating is suitable"),
    ("maint", "how to wash the bike properly"),
    ("maint", "how to store the bike for long periods"),
    ("maint", "what is the wheel bolt torque"),
    ("maint", "how to remove and install the wheel"),
    ("safety", "essential safety gear for riding"),
    ("safety", "how to use the different ride modes"),
    ("safety", "how does the TCS traction control system work"),
    ("safety", "how to ride safely in the rain"),
    ("safety", "what is the top speed of the bike"),
    ("safety", "how to operate the bike correctly for beginners"),
    ("safety", "how do sport and street ride modes differ"),
    ("safety", "how does the anti-slip system work"),
    ("safety", "how to park safely using the side stand"),
    ("safety", "the correct engine shutdown procedure"),
    ("clutch", "how to adjust the clutch lever"),
    ("clutch", "what is a slipper clutch"),
    ("clutch", "when should the clutch plates be replaced"),
    ("clutch", "what is the clutch lever free play"),
    ("clutch", "how does the quickshifter gearbox work"),
    ("clutch", "how to shift gears most smoothly"),
    ("clutch", "how many gears does the bike have"),
    ("clutch", "can clutch fluid and brake fluid be shared"),
    ("clutch", "is noise when releasing the clutch normal"),
    ("clutch", "how to check the clutch cable"),
]


async def main() -> None:
    import tempfile

    md_path = Path(MD_PATH)
    if not md_path.exists():
        print(f"ERROR: Manual not found at {md_path}")
        sys.exit(1)

    # Verify Ollama is running
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/version")
            version = resp.json().get("version", "unknown")
            logger.info("Ollama server v%s detected", version)
    except Exception as e:
        print(f"ERROR: Ollama server not reachable at localhost:11434: {e}")
        sys.exit(1)

    # --- Step 1: Create fresh brain with Ollama bge-m3 embeddings ---
    tmp_dir = tempfile.mkdtemp(prefix="smem_e2e_ollama_")
    logger.info("Step 1: Creating fresh brain with Ollama bge-m3 embeddings")
    logger.info("  Temp dir: %s", tmp_dir)

    from surreal_memory.core.brain import Brain, BrainConfig
    from surreal_memory.storage.memory_store import InMemoryStorage

    storage = InMemoryStorage()

    brain_config = BrainConfig(
        embedding_enabled=True,
        embedding_provider="ollama",
        embedding_model="bge-m3",
        embedding_similarity_threshold=0.5,
        max_context_tokens=3000,
    )
    brain = Brain.create(name="huskyAI", config=brain_config, brain_id="huskyAI")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    loaded_brain = await storage.get_brain("huskyAI")
    assert loaded_brain is not None, "Brain not found after save!"
    logger.info("  embedding_enabled=%s", loaded_brain.config.embedding_enabled)
    logger.info("  embedding_provider=%s", loaded_brain.config.embedding_provider)
    logger.info("  embedding_model=%s", loaded_brain.config.embedding_model)

    # --- Step 2: Train from markdown ---
    logger.info(
        "Step 2: Training from markdown file (%d lines)", len(md_path.read_text().splitlines())
    )

    from surreal_memory.engine.doc_trainer import DocTrainer

    trainer = DocTrainer(storage, brain_config)
    train_start = time.time()
    result = await trainer.train_file(md_path)
    train_elapsed = time.time() - train_start
    logger.info(
        "  Trained: %d chunks, %d neurons, %d synapses (%.1fs)",
        result.chunks_encoded,
        result.neurons_created,
        result.synapses_created,
        train_elapsed,
    )

    all_neurons = await storage.find_neurons(limit=1000)
    emb_count = sum(1 for n in all_neurons if n.metadata.get("_embedding"))
    logger.info("  Neurons with embeddings: %d / %d", emb_count, len(all_neurons))

    # --- Step 3: Base recall (6 EN) ---
    logger.info("Step 3: Testing base recall (6 queries)")

    from surreal_memory.engine.retrieval import ReflexPipeline

    pipeline = ReflexPipeline(storage, brain_config)
    logger.info("  Has embedding provider: %s", pipeline._embedding_provider is not None)

    base_results = []
    for lang, q in BASE_QUERIES:
        try:
            r = await pipeline.query(q, max_tokens=3000)
            n_fibers = len(r.fibers_matched)
            conf = r.confidence
            answer_preview = (r.answer or "(no answer)")[:120]
            base_results.append((lang, q, n_fibers, conf, answer_preview))
        except Exception as e:
            base_results.append((lang, q, -1, 0.0, f"ERROR: {e}"))

    print("\n" + "=" * 80)
    print("E2E OLLAMA bge-m3 RECALL — BASE QUERIES (6 EN)")
    print("=" * 80)
    print(f"Neurons: {len(all_neurons)} total, {emb_count} with embeddings")
    print(f"Provider: ollama / bge-m3 | Threshold: {brain_config.embedding_similarity_threshold}")
    print(f"Training time: {train_elapsed:.1f}s")
    print("-" * 80)

    base_ok = 0
    for lang, q, n_fibers, conf, answer in base_results:
        status = "OK" if n_fibers > 0 else "FAIL"
        if n_fibers > 0:
            base_ok += 1
        print(f"  [{lang}] {status} | {n_fibers:2d} fibers | conf={conf:.2f} | {q}")
        if n_fibers > 0:
            print(f"       -> {answer}")

    print(f"\nBase result: {base_ok}/{len(BASE_QUERIES)} queries returned results")

    # --- Step 4: 100 English queries ---
    logger.info("Step 4: Testing 100 queries")

    results_data: list[dict] = []
    recall_start = time.time()

    for i, (category, query) in enumerate(RECALL_QUERIES):
        try:
            r = await pipeline.query(query, max_tokens=3000)
            n_fibers = len(r.fibers_matched)
            conf = r.confidence
            answer_preview = (r.answer or "(no answer)")[:150]
            status = "OK" if n_fibers > 0 else "FAIL"
        except Exception as e:
            n_fibers = 0
            conf = 0.0
            answer_preview = f"ERROR: {e}"
            status = "ERROR"

        results_data.append(
            {
                "index": i + 1,
                "category": category,
                "query": query,
                "status": status,
                "fibers": n_fibers,
                "confidence": round(conf, 2),
                "answer_preview": answer_preview,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/100 queries done", i + 1)

    recall_elapsed = time.time() - recall_start

    # --- Stats ---
    total = len(results_data)
    ok_count = sum(1 for r in results_data if r["status"] == "OK")
    fail_count = sum(1 for r in results_data if r["status"] == "FAIL")
    error_count = sum(1 for r in results_data if r["status"] == "ERROR")
    avg_conf = sum(r["confidence"] for r in results_data if r["status"] == "OK") / max(ok_count, 1)
    avg_fibers = sum(r["fibers"] for r in results_data if r["status"] == "OK") / max(ok_count, 1)

    categories = sorted({r["category"] for r in results_data})
    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in results_data if r["category"] == cat]
        cat_ok = sum(1 for r in cat_results if r["status"] == "OK")
        cat_total = len(cat_results)
        cat_avg_conf = sum(r["confidence"] for r in cat_results if r["status"] == "OK") / max(
            cat_ok, 1
        )
        cat_stats[cat] = {"ok": cat_ok, "total": cat_total, "avg_conf": round(cat_avg_conf, 2)}

    # --- Print 100vi results ---
    print("\n" + "=" * 90)
    print("E2E OLLAMA bge-m3 RECALL — 100 QUERIES")
    print("=" * 90)
    print(f"Provider: ollama / bge-m3 | Threshold: {brain_config.embedding_similarity_threshold}")
    print(f"Recall time: {recall_elapsed:.1f}s ({recall_elapsed / total:.2f}s/query)")
    print("-" * 90)
    print(f"RESULTS: {ok_count}/{total} OK | {fail_count} FAIL | {error_count} ERROR")
    print(f"Avg confidence (OK): {avg_conf:.2f}")
    print(f"Avg fibers (OK): {avg_fibers:.1f}")
    print("-" * 90)

    print("\nPer-Category Breakdown:")
    for cat in categories:
        s = cat_stats[cat]
        pct = s["ok"] / s["total"] * 100
        print(f"  {cat:8s}: {s['ok']}/{s['total']} ({pct:5.1f}%) avg_conf={s['avg_conf']:.2f}")

    print("\nDetailed Results:")
    print("-" * 90)
    for r in results_data:
        marker = "OK" if r["status"] == "OK" else "FAIL"
        print(
            f"  {r['index']:3d}. {marker:4s} [{r['category']:6s}] fibers={r['fibers']:2d} conf={r['confidence']:.2f} | {r['query']}"
        )
        if r["status"] == "OK" and r["fibers"] > 0:
            print(f"       -> {r['answer_preview'][:100]}")

    failed = [r for r in results_data if r["status"] != "OK"]
    if failed:
        print(f"\nFailed Queries ({len(failed)}):")
        for r in failed:
            print(f"  {r['index']}. {r['query']}")

    print("-" * 90)

    # Save JSON results
    json_path = Path(__file__).resolve().parent / "results_ollama_bge_m3.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": {
                    "provider": "ollama",
                    "model": "bge-m3",
                    "total": total,
                    "ok": ok_count,
                    "fail": fail_count,
                    "error": error_count,
                    "success_rate": round(ok_count / total * 100, 1),
                    "avg_confidence": round(avg_conf, 2),
                    "avg_fibers": round(avg_fibers, 1),
                    "training_time_s": round(train_elapsed, 1),
                    "recall_time_s": round(recall_elapsed, 1),
                    "recall_per_query_s": round(recall_elapsed / total, 2),
                    "neurons_total": len(all_neurons),
                    "neurons_with_embeddings": emb_count,
                    "base_queries_ok": base_ok,
                    "base_queries_total": len(BASE_QUERIES),
                },
                "category_stats": cat_stats,
                "results": results_data,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON results saved: {json_path}")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())

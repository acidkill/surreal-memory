"""E2E test: Train motorcycle manual PDF -> Recall in English via Gemini embeddings.

Requires GEMINI_API_KEY and SURREALDB_URL. The SQLite fallback this script used
to have was removed along with the SQLite backend in v3.0.0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_gemini")

# Suppress noisy loggers
for name in ["surreal_memory.engine", "surreal_memory.storage", "surreal_memory.safety"]:
    logging.getLogger(name).setLevel(logging.WARNING)

PDF_PATH = os.environ.get(
    "SURREAL_MEMORY_TEST_PDF_PATH",
    "test_document.pdf",  # Default: expects a PDF in current directory
)

QUERIES = [
    ("EN", "How to change oil on KTM motorcycle?"),
    ("EN", "What is the recommended tire pressure?"),
    ("EN", "engine coolant specifications"),
    ("EN", "How does the brake system work?"),
    ("EN", "What tire size is recommended?"),
    ("EN", "What type of oil is suitable for the bike?"),
]


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY env var")
        sys.exit(1)

    surrealdb_url = os.environ.get("SURREALDB_URL")
    if not surrealdb_url:
        print("ERROR: Set SURREALDB_URL env var (this script's SQLite fallback")
        print("was removed along with the SQLite backend in v3.0.0)")
        sys.exit(1)

    # --- Step 1: Extract PDF to markdown ---
    logger.info("Step 1: Extracting PDF → markdown")
    try:
        import pymupdf4llm
    except ImportError:
        print("ERROR: pip install pymupdf4llm")
        sys.exit(1)

    md_text = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=False)
    logger.info("  Extracted %d chars of markdown", len(md_text))

    # Write to temp .md file
    tmp_dir = tempfile.mkdtemp(prefix="smem_e2e_")
    md_path = Path(tmp_dir) / "husqvarna_manual.md"
    md_path.write_text(md_text, encoding="utf-8")

    # --- Step 2: Create fresh brain with Gemini embeddings ---
    logger.info("Step 2: Creating fresh brain with Gemini embeddings")

    from surreal_memory.core.brain import Brain, BrainConfig
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    brain_id = f"e2e-gemini-{uuid.uuid4().hex[:8]}"
    storage = SurrealDBStorage(url=surrealdb_url)
    await storage.initialize()

    brain_config = BrainConfig(
        embedding_enabled=True,
        embedding_provider="gemini",
        embedding_model="gemini-embedding-001",
        embedding_similarity_threshold=0.5,
        max_context_tokens=3000,
    )
    brain = Brain.create(name=brain_id, config=brain_config, brain_id=brain_id)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    # Verify brain config round-trip
    loaded_brain = await storage.get_brain(brain_id)
    assert loaded_brain is not None, "Brain not found after save!"
    logger.info("  embedding_enabled=%s (stored)", loaded_brain.config.embedding_enabled)
    logger.info("  embedding_provider=%s (stored)", loaded_brain.config.embedding_provider)

    # --- Step 3: Train from markdown ---
    logger.info("Step 3: Training from markdown file")

    from surreal_memory.engine.doc_trainer import DocTrainer

    trainer = DocTrainer(storage, brain_config)
    result = await trainer.train_file(md_path)
    logger.info(
        "  Trained: %d chunks, %d neurons, %d synapses",
        result.chunks_encoded,
        result.neurons_created,
        result.synapses_created,
    )

    # Check how many neurons have embeddings
    all_neurons = await storage.find_neurons(limit=1000)
    emb_count = sum(1 for n in all_neurons if n.metadata.get("_embedding"))
    logger.info("  Neurons with embeddings: %d / %d", emb_count, len(all_neurons))

    # --- Step 4: Recall queries ---
    logger.info("Step 4: Testing recall")

    from surreal_memory.engine.retrieval import ReflexPipeline

    pipeline = ReflexPipeline(storage, brain_config)
    logger.info("  Has embedding provider: %s", pipeline._embedding_provider is not None)

    results_table = []
    for lang, q in QUERIES:
        try:
            result = await pipeline.query(q, max_tokens=3000)
            n_fibers = len(result.fibers_matched)
            conf = result.confidence
            answer_preview = (result.answer or "(no answer)")[:120]
            results_table.append((lang, q, n_fibers, conf, answer_preview))
        except Exception as e:
            results_table.append((lang, q, -1, 0.0, f"ERROR: {e}"))

    # --- Print results ---
    print("\n" + "=" * 80)
    print("E2E GEMINI RECALL RESULTS")
    print("=" * 80)
    print(f"Brain: {brain_id}")
    print(f"Total neurons: {len(all_neurons)}, with embeddings: {emb_count}")
    print(f"Embedding provider: {brain_config.embedding_provider}")
    print(f"Similarity threshold: {brain_config.embedding_similarity_threshold}")
    print("-" * 80)

    any_success = False
    for lang, q, n_fibers, conf, answer in results_table:
        status = "OK" if n_fibers > 0 else "FAIL"
        if n_fibers > 0:
            any_success = True
        print(f"  [{lang}] {status} | {n_fibers:2d} fibers | conf={conf:.2f} | {q}")
        if n_fibers > 0:
            print(f"       -> {answer}")

    print("-" * 80)
    if any_success:
        print("PASS: At least some queries returned results")
    else:
        print("FAIL: All queries returned 0 results")

    # Cleanup discipline (project hard rule): delete only this run's own
    # brain, by its exact id, never `default`. Re-fetch the real RecordID via
    # SELECT rather than reconstructing one from the string id -- a hand-built
    # `type::record('brain', $bid)` can look like it succeeded (no error)
    # while silently matching zero rows.
    await storage.clear(brain_id)
    rows = await storage._query(
        "SELECT id FROM brain WHERE id = type::record('brain', $bid)", bid=brain_id
    )
    for row in rows:
        await storage._query("DELETE $rid", rid=row["id"])
    await storage.close()
    print(f"\nTemp dir preserved at: {tmp_dir}")


if __name__ == "__main__":
    asyncio.run(main())

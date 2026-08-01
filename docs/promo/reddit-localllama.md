# Reddit r/LocalLLaMA Post

**Title:** Open-source graph memory for AI agents — spreading activation instead of RAG, $0/query, no API keys needed

**Body:**

I've been working on an alternative approach to persistent memory for AI agents. Instead of the usual RAG pipeline (embed → vector search → return chunks), it stores memories as a neural graph and retrieves them through **spreading activation** — closer to how biological memory works than cosine similarity.

## The problem with RAG for agent memory

RAG treats memory as a search problem: "find text similar to this query." It works for documents, but it loses causal structure. When you ask "why did the outage happen?", RAG returns "JWT caused the outage" — but not *why* JWT was chosen, who suggested it, or what decision it replaced.

## Spreading activation approach

[Surreal-Memory](https://github.com/acidkill/surreal-memory) stores everything as typed neurons connected by 41 explicit synapse types (`CAUSED_BY`, `LEADS_TO`, `SUGGESTED_BY`, `RESOLVED_BY`, `CONTRADICTS`, ...). Recall works by:

1. Activate seed neurons matching your query
2. Activation spreads through synapses (weighted, with decay)
3. Most-activated neurons surface as context

This gives you multi-hop causal reasoning. "Why did the outage happen?" traces: `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice ← DECIDED_AT ← Tuesday meeting`.

**No embedding API calls required for core recall.** It's graph traversal. Embeddings are optional and local-first (Ollama, sentence-transformers, or nothing at all).

## Technical details

- **Storage**: SurrealDB multi-model backend (document + graph + vector HNSW in one DB — no separate vector store)
- **Graph**: 15 memory types (fact, decision, error, insight, preference, workflow, todo, ...), 41 synapse types, fiber bundles for episodic grouping
- **Retrieval**: Spreading activation with configurable decay, threshold, and max hops
- **Lifecycle**: Ebbinghaus decay, Hebbian reinforcement, sleep consolidation (ENRICH/PRUNE/MERGE/DREAM), 5-tier compression
- **MCP server**: 58 tools (3-tool core: smem_remember, smem_recall, smem_health), stdio + HTTP, works with Claude Code, Cursor, Windsurf, Cline, Zed, Gemini CLI
- **Pro features**: cone/HNSW vector search, smart merge, directional compression — all FREE via the bundled community plugin, no license keys
- **Tests**: 7200+ unit tests, 67%+ CI coverage, mypy + ruff + pytest

## Why local/offline users care

- **$0.00/query**: core encode and recall make zero LLM or embedding API calls — works fully offline
- **No API keys needed** for core operations
- **Ollama support**: opt-in local embeddings if you want semantic similarity on top of graph traversal
- **Single portable brain**: export/import JSON, brain versioning and transplant, multi-device sync via your own Cloudflare account (Merkle delta, encrypted)
- **MIT licensed**: no paywalls, no telemetry, no vendor lock-in

## Relationship to neural-memory

Surreal-Memory is a fork of [nhadaututtheky/neural-memory](https://github.com/nhadaututtheky/neural-memory). It inherits the neuron/synapse/fiber architecture, spreading activation model, and the MCP tool surface from upstream (credit where it's due). The main differences: SurrealDB multi-model backend instead of SQLite; all Pro-tier features (vector search, smart merge, compression) unlocked free via the community plugin — upstream gates these behind a paid plan; and ongoing port of storage-agnostic upstream improvements.

## Install

```bash
# Core install
pip install "surreal-memory[surrealdb]"

# Claude Code plugin
# /plugin marketplace add acidkill/surreal-memory

# Docker (recommended for persistent brain)
docker compose -f docker-compose.surrealdb.yml up -d
```

GitHub: https://github.com/acidkill/surreal-memory
Docs: https://acidkill.github.io/surreal-memory/

MIT licensed. Early project — contributions welcome.

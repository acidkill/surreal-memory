# Reddit r/ClaudeAI Post

**Title:** I built a persistent memory system for Claude Code that works like a brain — Surreal-Memory (open source, 58 MCP tools, $0/query)

**Body:**

Claude Code forgets everything between sessions. I got tired of re-explaining project context every time, so I built Surreal-Memory — an MCP server that gives Claude a persistent, associative memory backed by SurrealDB.

## How it's different from other memory tools

Most memory MCP servers use RAG (embed text → vector search → return chunks). Surreal-Memory doesn't. It stores memories as a **neural graph** and retrieves them through **spreading activation** — the same mechanism the human brain uses for recall.

When you remember "Alice", it doesn't just find text containing "Alice". It activates the Alice neuron, which spreads to connected concepts: the meeting where you discussed rate limiting → the outage it caused → the JWT decision that led to it. You get the full causal chain, not just keyword matches.

**No LLM or embedding API required for core encode and recall.** It's pure algorithmic graph traversal. No API keys needed for core operations. Embeddings (Ollama, Gemini, OpenAI, sentence-transformers) are optional and used for semantic (vector) search only.

## What it does

- **58 MCP tools**: 3-tool core (`smem_remember`, `smem_recall`, `smem_health`) — the other 53 fire automatically
- **Spreading activation retrieval**: memories surface through association, not search
- **41 synapse types** (`CAUSED_BY`, `LEADS_TO`, `CONTRADICTS`, `RESOLVED_BY`) — causal reasoning, not just similarity
- **15 memory types**: fact, decision, error, insight, preference, workflow, todo, and more
- **Memory lifecycle**: Ebbinghaus decay, Hebbian reinforcement, sleep consolidation (ENRICH / PRUNE / MERGE / DREAM), 5-tier compression
- **ALL Pro features free**: cone/HNSW vector search, smart merge, directional compression — no license keys, no paywalls
- **Multi-device sync** via your own Cloudflare account (Merkle delta, encrypted)
- **Single portable brain**: export / import JSON, versioning, brain transplant

## Backend

SurrealDB multi-model engine — document + graph + vector HNSW in **one** database. No separate vector store, no SQLite, no extra infrastructure. Spin it up with:

```bash
docker compose -f docker-compose.surrealdb.yml up -d
```

## Quick start

```bash
pip install "surreal-memory[surrealdb]"
```

Add to Claude Code:

```bash
/plugin marketplace add acidkill/surreal-memory
```

Works with Claude Code, Cursor, Windsurf, VS Code, Cline, Zed, and Gemini CLI.

## Numbers

- **58 MCP tools** · **7200+ unit tests** · **67%+ CI coverage**
- **$0.00/query** — no API calls for core encode + recall
- 15 memory types · 41 synapse types
- Python 3.11+ · async · MIT license

## Relationship to neural-memory

Surreal-Memory is a fork of [nhadaututtheky/neural-memory](https://github.com/nhadaututtheky/neural-memory). It inherits the neuron/synapse/fiber architecture, spreading activation, and the MCP tool surface — credit where it's due. The key differences: (1) SurrealDB multi-model backend replaces SQLite; (2) ALL Pro-tier features (vector search, smart merge, compression) are unlocked free via the community plugin — upstream gates these behind a paid license; (3) ongoing port of storage-agnostic upstream improvements.

---

GitHub: https://github.com/acidkill/surreal-memory

Happy to answer questions about the architecture or how spreading activation works in practice.

# Dev.to Article

**Title:** Surreal-Memory: How Spreading Activation Gives AI Agents a Real Memory System

**Tags:** claudecode, mcp, python, ai, opensource

**Cover image:** (use dashboard-overview.png or dashboard-graph.png)

---

## The Problem

Every AI coding session starts from zero. You explain your project architecture, your conventions, your past decisions — and the AI forgets all of it when the session ends.

Most solutions reach for RAG: embed text into vectors, search by similarity, return chunks. It works for document retrieval, but it's a poor model for *memory*. When you remember something, you don't search a database — you *associate*. One thought triggers another, which triggers another, until the relevant memory surfaces.

## A Different Approach: Neural Graphs

[Surreal-Memory](https://github.com/acidkill/surreal-memory) stores memories as a graph of typed neurons connected by typed synapses:

```
outage ← CAUSED_BY ← JWT_decision ← SUGGESTED_BY ← Alice ← DECIDED_AT ← Tuesday_meeting
```

When you ask "why did the outage happen?", it doesn't just find text containing "outage." It activates the outage neuron, and activation spreads through the graph following synapse weights. You get the full causal chain — not just the closest text match.

### RAG vs Spreading Activation

| Aspect | RAG / Vector Search | Surreal-Memory |
|--------|---------------------|----------------|
| Model | Search engine | Human brain |
| LLM/Embedding | Required | Optional — core recall is pure graph traversal |
| Query | "Find similar text" | "Recall through association" |
| Relationships | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS` |
| Multi-hop | Multiple queries | Natural graph traversal |
| API Cost | ~$0.02/1K queries | $0.00 — no API keys needed for core operations |

## How It Works

### 1. Encoding

When you tell the AI to remember something, Surreal-Memory:
- Extracts entities, keywords, and temporal markers
- Creates typed neurons (ENTITY, CONCEPT, ACTION, TEMPORAL, and more)
- Creates typed synapses between them (41 explicit relationship types)
- Groups related neurons into a Fiber (episodic memory bundle)
- Persists everything in SurrealDB — document + graph + vector in one engine

### 2. Retrieval (Spreading Activation)

When you recall:
1. **Seed activation**: neurons matching your query get initial activation
2. **Spreading**: activation propagates through synapses, weighted by strength
3. **Decay**: activation decreases with each hop (configurable)
4. **Threshold**: only neurons above threshold are included in results
5. **Context assembly**: top-activated neurons are assembled into a coherent response

This naturally handles multi-hop queries. "Who suggested the thing that caused the outage?" follows the chain without explicit graph queries.

### 3. Consolidation (Sleep Cycle)

Memories have a full lifecycle modelled on human cognition:
- **Decay**: unused synapses weaken over time (Ebbinghaus forgetting curve)
- **Reinforcement**: recalled memories get stronger (Hebbian learning)
- **Pruning**: orphan neurons with no connections get cleaned up
- **Merging**: duplicate information is consolidated
- **Dreaming**: a sleep consolidation phase (ENRICH / PRUNE / MERGE / DREAM) compresses and reorganises the graph through 5-tier compression

## 56 MCP Tools

Surreal-Memory exposes **56 tools** via the [Model Context Protocol](https://modelcontextprotocol.io/). The 3-tool core drives daily use; the rest fire automatically or on demand:

| Tool | What it does |
|------|-------------|
| `smem_remember` | Store a memory with automatic extraction |
| `smem_recall` | Retrieve memories through spreading activation |
| `smem_health` | Health diagnostics with actionable recommendations |
| `smem_context` | Load recent memories at session start |
| `smem_explain` | Show WHY two concepts are connected (BFS path) |
| `smem_habits` | Detect recurring patterns in your workflow |
| `smem_consolidate` | Run the memory lifecycle (decay, prune, merge, dream) |
| `smem_session` | Save/restore session state |

Plus 48 more for brain management, import/export, training, sync, visualisation, and diagnostics. All 56 are included in the open-source package — no license keys, no paywalls.

### Pro Features, Free

The bundled community plugin unlocks what upstream forks typically gate behind a paid tier: cone/HNSW vector search, smart merge, and directional compression are all included at no cost.

## Quick Start

### pip (Python 3.11+)

```bash
pip install "surreal-memory[surrealdb]"
```

### Claude Code Plugin

```bash
/plugin marketplace add acidkill/surreal-memory
```

### Docker (full stack with SurrealDB)

```bash
docker compose -f docker-compose.surrealdb.yml up -d
```

### Manual MCP Config

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["surreal-memory[surrealdb]"]
    }
  }
}
```

Works with Claude Code, Cursor, Windsurf, VS Code (Cline/Continue), Zed, and Gemini CLI.

### Optional: Semantic (Vector) Search

Core recall works without any embedding API. Enable optional semantic (vector) search for similarity-based recall:

```toml
# ~/.surrealmemory/config.toml
[embedding]
enabled = true
provider = "gemini"                      # or ollama, sentence_transformer, openai
model = "gemini-embedding-001"           # 3072-dim; needs GEMINI_API_KEY
```

The `auto` provider cascades at runtime: ollama → sentence-transformers → gemini → openai → openrouter, so it works offline-first.

## The Backend: SurrealDB

Surreal-Memory runs entirely on [SurrealDB](https://surrealdb.com/) — a multi-model engine that handles document storage, graph traversal, and HNSW vector search in a single database. There is no separate vector store, no SQLite file, and no external search service. The neuron/synapse graph is a native SurrealDB graph; vector embeddings live alongside it in the same engine.

The brain is stored as a single portable export file (JSON) that can be versioned, snapshotted, and transplanted between installations. Multi-device sync uses Merkle-delta diffing over your own Cloudflare account — encrypted, no third-party cloud required.

## Built on neural-memory

Surreal-Memory is a fork of [nhadaututtheky/neural-memory](https://github.com/nhadaututtheky/neural-memory). It inherits the core neuron/synapse/fiber architecture, the spreading-activation retrieval model, and the MCP tool surface — credit goes upstream for that foundation.

Three things differ in this fork:

1. **SurrealDB backend** instead of SQLite. The multi-model engine enables native graph traversal and HNSW vector search without any additional infrastructure.
2. **All Pro-tier features unlocked for free** via the community plugin. Upstream gates cone/HNSW vector search, smart merge, and compression behind a paid Pro plan; this fork ships all of it open-source under MIT.
3. **Ongoing port** of storage-agnostic upstream improvements as they land, keeping the fork current.

If you need a lightweight SQLite-backed version, the upstream project is the right choice. If you want the full feature set — graph + vector in one engine, all Pro features free — Surreal-Memory is for you.

## Numbers

- **56 MCP tools** — 3-tool core, 53 fire automatically or on demand
- **5,500+ unit tests**, 67%+ CI coverage
- **15 memory types**: fact, decision, error, insight, preference, workflow, todo, and more
- **41 synapse types**: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS`, and 37 more
- **$0.00 per query** — no API keys needed for core encode + recall
- **Python 3.11+** · **MIT license** · SurrealDB backend
- **Dashboard**: FastAPI + React web UI for graph visualisation

## Links

- **GitHub**: https://github.com/acidkill/surreal-memory
- **Docs**: https://acidkill.github.io/surreal-memory/
- **PyPI**: https://pypi.org/project/surreal-memory/

---

*Surreal-Memory is open source and contributions are welcome. The spreading activation approach will feel familiar if you've worked with cognitive architectures like ACT-R or Soar — it's the same theoretical foundation applied to AI agent memory, now running on a multi-model graph database instead of flat files.*

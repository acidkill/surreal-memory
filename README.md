# Surreal-Memory

[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![CI](https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg)](https://github.com/nhadaututtheky/neural-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![SurrealDB](https://img.shields.io/badge/Powered_by-SurrealDB-ff00e5)](https://surrealdb.com/)

**Persistent graph memory for AI agents, powered by SurrealDB.**

All features are free and open source. No license keys. No paywalls. No embedding API required for basic usage.

```bash
pip install neural-memory[surrealdb]
nmem init --full
```

Restart your AI tool. Your agent now remembers.

---

## Why Surreal-Memory?

Most AI memory tools are vector databases with a search API bolted on. Surreal-Memory is a **graph that thinks** — memories are stored as interconnected neurons and recalled through spreading activation, backed by SurrealDB's multi-model engine (document + graph + vector in one database).

```
Query: "Why did Tuesday's outage happen?"

Surreal-Memory traces the chain:
outage ← CAUSED_BY ← JWT expiry ← SUGGESTED_BY ← Alice's review
```

**Relationships are explicit** — `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS` — so your agent doesn't just find memories, it *reasons* through them.

| | RAG / Vector Search | Surreal-Memory |
|--|---------------------|----------------|
| Backend | Pinecone / Chroma | **SurrealDB** (doc + graph + vector) |
| Retrieval | Similarity score | Graph traversal + vector search |
| Relationships | None | 24 explicit synapse types |
| LLM required | Yes (embeddings) | No — works fully offline |
| Multi-hop reasoning | Multiple queries | One traversal |
| Memory lifecycle | Static | Decay, reinforcement, consolidation |
| Cost per 1K queries | ~$0.02 | **$0.00** |

---

## What's Different From Neural Memory?

Surreal-Memory builds on the neural graph memory architecture but replaces the SQLite + paid-Pro model with **SurrealDB + free community plugin**:

| | Neural Memory | Surreal-Memory |
|--|---------------|----------------|
| Storage engine | SQLite (limited) | **SurrealDB** (all features free) |
| Vector search | Paid Pro feature | **Built-in** via SurrealDB HNSW |
| Semantic recall | Paid Pro feature | **Free** via community plugin |
| Smart consolidation | Paid Pro feature | **Free** via community plugin |
| Compression | Paid Pro feature | **Free** via community plugin |
| License required | Yes for Pro features | **No** — everything is free |
| Multi-model | No | **Yes** — document + graph + vector |

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env    # edit with your keys
docker compose -f docker-compose.surrealdb.yml up -d
```

Dashboard at http://localhost:8000/ui, SurrealDB at localhost:8001.

### Manual

```bash
pip install neural-memory[surrealdb]
nmem init --full
```

### First Memory

```bash
nmem remember "Fixed auth bug with null check in login.py:42"
nmem recall "auth bug"
# → "Fixed auth bug with null check in login.py:42"
```

---

## 3 Tools. That's It.

53 MCP tools are available, but you only need three:

| Tool | What it does |
|------|-------------|
| `nmem_remember` | Store a memory — auto-detects type, tags, and connections |
| `nmem_recall` | Recall through spreading activation + vector search |
| `nmem_health` | Brain health score (A–F) with actionable fix suggestions |

Everything else — sessions, context loading, habit tracking, maintenance — works transparently in the background.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │       MCP Server (53 tools)   │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     Engine (encoding +        │
                    │     retrieval pipeline)       │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │        SurrealDB Backend         │
              │  ┌─────────┬─────────┬────────┐ │
              │  │ Document │  Graph  │ Vector │ │
              │  │  Store   │ Queries │  HNSW  │ │
              │  └─────────┴─────────┴────────┘ │
              └─────────────────────────────────┘
```

### Core Data Model

- **Brain** — top-level container with configuration
- **Neuron** — atomic knowledge node (entity, concept, time, action, intent, state)
- **Synapse** — typed, directed edge between neurons (24 types: `CAUSED_BY`, `LEADS_TO`, etc.)
- **Fiber** — a memory record: typed content with metadata, priority, tags, lifecycle stage

### Engine

- **Encoding Pipeline** — composable async steps: extract entities → create neurons → link synapses → bundle into fibers
- **Reflex Retrieval** — spreading activation through the neuron graph, combined with SurrealDB vector search when available
- **Consolidation** — merges similar neurons, reinforces strong paths, prunes weak ones
- **Compression** — 5-tier lifecycle: full → summary → essence → ghost → metadata

### Community Plugin

The built-in `CommunityPlugin` provides all Pro-tier features at no cost:

- **Cone Queries** — HNSW vector search via SurrealDB for semantic recall
- **Smart Merge** — embedding-based neuron consolidation
- **Directional Compression** — multi-axis semantic preservation
- **SurrealDB Storage Backend** — registered automatically when `[surrealdb]` extra is installed

---

## Cloud Sync

Sync your brain across every machine through your own Cloudflare Worker:

```
Laptop ←→ Your Cloudflare Worker ←→ Desktop
                  ↕
              Your Phone
```

You deploy the sync hub to **your own Cloudflare account**. Your D1 database, your encryption key, your data.

```bash
nmem sync --full    # bi-directional sync
nmem sync --auto    # auto-sync after every remember/recall
```

Sync uses **Merkle delta** — only diffs travel, not the full brain.

---

## Features

#### Memory & Recall
- **14 memory types** — fact, decision, error, insight, preference, workflow, instruction, and more
- **Spreading activation** — memories surface by association, not keyword match
- **Vector search** — SurrealDB HNSW for semantic similarity (when embeddings are configured)
- **Cognitive reasoning** — hypothesize, submit evidence, make predictions, verify with Bayesian confidence

#### Knowledge Ingestion
- **Train from documents** — PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV ingested into permanent brain knowledge
- **Train from database schemas** — extract table structures and FK relationships
- **Import adapters** — migrate from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex

#### Lifecycle & Storage
- **Memory consolidation** — episodic memories mature into semantic knowledge
- **Compression tiers** — full → summary → essence → ghost → metadata
- **Brain versioning** — snapshot, rollback, diff, transplant memories between brains

#### Ecosystem
- **Web dashboard** — 7-page React UI with graph visualization, health radar, timeline
- **VS Code extension** — memory tree, graph explorer, CodeLens, WebSocket sync
- **Safety** — Fernet encryption, sensitive content auto-detection, input firewall
- **Plugin system** — extend with custom retrieval strategies, compression, and storage backends

---

## Setup by Tool

<details>
<summary><b>Claude Code (Plugin)</b></summary>

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

</details>

<details>
<summary><b>Cursor / Windsurf / Other MCP Clients</b></summary>

```bash
pip install neural-memory[surrealdb]
```

Add to your editor's MCP config:

```json
{
  "mcpServers": {
    "neural-memory": { "command": "nmem-mcp" }
  }
}
```

</details>

<details>
<summary><b>OpenClaw (Plugin)</b></summary>

```bash
pip install neural-memory[surrealdb] && npm install -g neuralmemory
```

Set memory slot in `~/.openclaw/openclaw.json`:
```json
{ "plugins": { "slots": { "memory": "neuralmemory" } } }
```

</details>

<details>
<summary><b>Docker (self-hosted)</b></summary>

```bash
cp .env.example .env          # configure SurrealDB + embeddings
docker compose -f docker-compose.surrealdb.yml up -d
```

Dashboard: http://localhost:8000/ui

</details>

---

## Python API

```python
import asyncio
from neural_memory import Brain
from neural_memory.storage import create_storage
from neural_memory.core.brain_mode import BrainModeConfig, BrainMode
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline

async def main():
    config = BrainModeConfig(mode=BrainMode.LOCAL)
    storage = await create_storage(config, brain_id="my_brain")

    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Met Alice to discuss API design")
    await encoder.encode("Decided to use FastAPI for backend")

    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What did we decide about backend?")
    print(result.context)  # "Decided to use FastAPI for backend"

asyncio.run(main())
```

---

## Development

```bash
git clone https://github.com/nhadaututtheky/neural-memory
cd neural-memory && pip install -e ".[dev]"
pytest tests/ -v          # 5500+ tests
ruff check src/ tests/    # Lint
make verify               # Full CI gate
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Surreal-Memory is built on top of [**NeuralMemory**](https://github.com/nhadaututtheky/neural-memory) by [nhadaututtheky](https://github.com/nhadaututtheky) — an exceptional graph-based memory system for AI agents. The core architecture (neurons, synapses, fibers, spreading activation, consolidation, compression, and the 53-tool MCP interface) is entirely their work.

Surreal-Memory extends it with a SurrealDB storage backend and a community plugin that makes all advanced features available for free.

> If you find Surreal-Memory useful, please also star the [original NeuralMemory project](https://github.com/nhadaututtheky/neural-memory).

## License

MIT — see [LICENSE](LICENSE).

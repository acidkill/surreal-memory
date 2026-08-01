# Surreal-Memory

[![PyPI](https://img.shields.io/pypi/v/surreal-memory.svg)](https://pypi.org/project/surreal-memory/)
[![CI](https://github.com/acidkill/surreal-memory/workflows/CI/badge.svg)](https://github.com/acidkill/surreal-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![SurrealDB](https://img.shields.io/badge/Powered_by-SurrealDB-ff00e5)](https://surrealdb.com/)

**Persistent graph memory for AI agents, powered by SurrealDB.**

All features are free and open source. No license keys. No paywalls. No embedding API required for basic usage.

```bash
pip install surreal-memory[surrealdb]
smem init --full
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
| Relationships | None | 41 explicit synapse types |
| LLM required | Yes (embeddings) | No — works fully offline |
| Multi-hop reasoning | Multiple queries | One traversal |
| Memory lifecycle | Static | Decay, reinforcement, consolidation |
| Cost per 1K queries | ~$0.02 | **$0.00** |

---

## What's Different From NeuralMemory?

Surreal-Memory builds on [NeuralMemory](https://github.com/nhadaututtheky/neural-memory)'s graph memory architecture but replaces the SQLite + paid-Pro model with **SurrealDB + free community plugin**:

| | NeuralMemory (upstream) | Surreal-Memory |
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

### Automated Setup (Claude Code)

Give this to Claude Code on any machine and it handles everything — prerequisites, Docker, MCP registration, and verification:

```
Please read INSTALL_PROMPT.md and follow the instructions to set up Surreal-Memory on this machine.
```

Or clone and point Claude Code at the file:

```bash
git clone https://github.com/acidkill/surreal-memory.git
# then in Claude Code:
# "Read INSTALL_PROMPT.md and follow the setup instructions"
```

### Docker (manual)

```bash
cp .env.example .env    # edit with your keys
docker compose -f docker-compose.surrealdb.yml up -d
```

Dashboard at http://localhost:8000/ui, SurrealDB at localhost:8001.

> **Requires SurrealDB ≥ 3.2.0** (the compose file uses `surrealdb/surrealdb:v3.2.0`).
> Upgrading from an older SurrealDB? Back up the `surrealdb_data` volume first — the synapse
> graph auto-migrates to native RELATE edges on the first connect after the upgrade.

### Manual

```bash
pip install surreal-memory[surrealdb]
smem init --full
```

### First Memory

```bash
smem remember "Fixed auth bug with null check in login.py:42"
smem recall "auth bug"
# → "Fixed auth bug with null check in login.py:42"
```

---

## 3 Tools. That's It.

58 MCP tools are available, but you only need three:

| Tool | What it does |
|------|-------------|
| `smem_remember` | Store a memory — auto-detects type, tags, and connections |
| `smem_recall` | Recall through spreading activation + vector search |
| `smem_health` | Brain health score (A–F) with actionable fix suggestions |

Everything else — sessions, context loading, habit tracking, maintenance — works transparently in the background.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │       MCP Server (58 tools)   │
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
- **Synapse** — typed, directed edge between neurons (41 types: `CAUSED_BY`, `LEADS_TO`, etc.)
- **Fiber** — a memory record: typed content with metadata, priority, tags, lifecycle stage

### Engine

- **Encoding Pipeline** — composable async steps: extract entities → create neurons → link synapses → bundle into fibers
- **Reflex Retrieval** — spreading activation through the neuron graph, combined with SurrealDB vector search when available
- **Reranking** — optional cross-encoder pass over-fetches SA candidates and blends relevance score with activation level for higher recall precision
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
smem sync --full    # bi-directional sync
smem sync --auto    # auto-sync after every remember/recall
```

Sync uses **Merkle delta** — only diffs travel, not the full brain.

---

## Features

#### Memory & Recall
- **15 memory types** — fact, decision, error, insight, preference, workflow, instruction, and more
- **Spreading activation** — memories surface by association, not keyword match
- **Vector search** — SurrealDB HNSW for semantic similarity (when embeddings are configured)
- **Cross-encoder reranking** — optional config-driven precision pass, HTTP (shared inference server) or in-process, blended with the activation score
- **Cognitive reasoning** — hypothesize, submit evidence, make predictions, verify with Bayesian confidence
- **Per-fact supersession** — when a new memory replaces an old one, recall automatically resolves the conflict (the old fact is superseded, not deleted); recall the state of the world as of a past moment with `valid_at`
- **Trust & recency scoring** — weight recall by how much you trust a source and how fresh a memory is, instead of treating everything as equally reliable forever
- **Uncertainty surfacing** — ask `smem_uncertainty` how much to trust an answer: contradictions, drift, soon-expiring memories, low-evidence facts
- **Queryable retrieval traces** — opt-in record of *why* a recall returned what it did, for debugging and auditing
- **Geospatial recall** — attach a location to a memory and filter recall to a radius around a point (`near`)

#### Knowledge Ingestion
- **Train from documents** — PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV ingested into permanent brain knowledge
- **Train from database schemas** — extract table structures and FK relationships
- **Fast bulk training** — `smem train` batches DB writes (one round-trip per N synapses via `add_synapses_batch`) and the `find_neurons` brain_id index is now used, so large docs stay cheap per chunk even on big brains (previously 7–15 s/chunk on a 68k-neuron brain; ~10× fewer DB ops/chunk). Shows live `tqdm` progress.
- **Import adapters** — migrate from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex
- **Reasoning training** (opt-in) — mine a model's own `thinking` from `~/.claude` transcripts, distill it into reusable reasoning-pattern fibers, and inject the learned strategies into other models' sessions (dashboard + `smem reasoning` CLI + `smem_reasoning` MCP tool). Off by default; traces are redacted before storage. Per-model **category coverage** — how many of the 8 categories hold enough patterns — is the number to watch; `pattern_targets` caps how many patterns each model may accumulate, though a model's own trace backlog usually runs out first.
- **Rebuilding lost patterns** — distillation marks every trace it consumes as processed, including the ones it discards, so patterns cannot normally be re-derived from a backlog that has already been mined. `reprocess` re-opens it: a checkbox beside *Backfill* in the dashboard, `smem reasoning mine --reprocess`, or `reprocess: true` on `smem_reasoning`. Repeating it is safe — pattern signatures make a second pass a no-op rather than a duplicate-maker. It respects `mining_models` (globs are resolved against the models actually present, and a filter matching nothing re-opens nothing), and `dry_run` still wins, since re-opening the backlog is a write. Traces already dropped by `retention_days` need `--backfill` to be re-ingested from the transcripts first.
- **LLM pattern naming** (opt-in, local-only) — by default a distilled pattern is named after its own mechanics (`debugging: restate-goal, gather-evidence, verify`) and described by a raw slice of the medoid trace. Point `distill_use_llm` at a **loopback** OpenAI-compatible endpoint and a local model rewrites the title, description and strategy into something readable, leaving identity and statistics untouched. A non-loopback endpoint is refused outright — reasoning traces never leave the machine — and any failure falls back to the mechanical naming. `distill_llm_load_cmd` loads the model once before the first request, so you control *how* it loads (context size, GPU layers, no vision projector) instead of accepting whatever the endpoint does implicitly; `distill_llm_unload_cmd` releases it when the run ends, so it does not sit in VRAM between runs. Both are argv lists run without a shell, `{model}` is substituted, and either one being absent or failing degrades silently to the old behavior:

  ```toml
  [reasoning_training]
  distill_use_llm = true
  distill_llm_model = "<a chat model your server serves>"
  distill_llm_endpoint = "http://127.0.0.1:PORT/v1"   # loopback only
  distill_llm_load_cmd = ["<your-launcher>", "load", "{model}", "--n-gpu-layers", "99"]
  distill_llm_unload_cmd = ["<your-launcher>", "stop", "{model}"]
  ```

- **Clustering threshold travels with the embedder** — two traces become one pattern when their embeddings exceed `cluster_cosine` (default `0.75`). Embedding models do not share a similarity scale, so a value borrowed from another model can sit above the *99th percentile* of your corpus and cluster nothing at all, silently leaving the move-set fallback to outperform the embedding path it is meant to back up. If mining reports patterns learned but categories stay empty, compare the threshold against your own pairwise distribution before raising `pattern_targets`:

  ```toml
  [reasoning_training]
  cluster_cosine = 0.75   # lower = more merging; too low collapses everything into one cluster
  ```

#### Lifecycle & Storage
- **Memory consolidation** — episodic memories mature into semantic knowledge through **spaced recall**, not through a command. A fiber reaches `semantic` after 7 days in `episodic` *plus* reinforcement spread across 3+ distinct days (or 15+ rehearsals across 5+ time windows) — recalling a memory is what advances it; `smem consolidate` cannot move it on its own. `smem health` shows where every memory sits (`stage_distribution`) and what each one is still waiting on (`semantic_gate_blockers`: dwell time, recall spacing, or already eligible), so a flat `consolidation_ratio` is diagnosable instead of mysterious. Recall rehearses the memories it actually surfaced — raise `brain.reinforcement_neuron_limit` (default 15) to widen that reach at the cost of some recall latency.
- **Compression tiers** — full → summary → essence → ghost → metadata
- **Brain versioning** — snapshot, rollback, diff, transplant memories between brains

#### Ecosystem
- **Web dashboard** — multi-page React UI at `/ui` (overview, health radar, graph, timeline, evolution, storage, tool-stats, visualize, settings, uncertainty) — every page free, no Pro gate
- **VS Code extension** — memory tree, graph explorer, CodeLens, WebSocket sync
- **LangChain adapter** — optional extra (`pip install surreal-memory[langchain]`) exposing a `BaseRetriever` and chat message history backed by a brain — see [Python API](#python-api)
- **Safety** — Fernet encryption, sensitive content auto-detection, input firewall
- **Plugin system** — extend with custom retrieval strategies, compression, and storage backends

---

## Embeddings

The keyword + graph core works with **no embedding API at all**. Embeddings are
optional — they add semantic recall (vector search via SurrealDB HNSW) so memories
surface even when the wording differs. Configure interactively with `smem setup embeddings`
or via `SURREAL_MEMORY_EMBEDDING_*` env vars.

**Recommended — Google Gemini** (`gemini-embedding-001`, 3072-dim, multilingual, free tier):

```bash
pip install "surreal-memory[surrealdb,embeddings-gemini]"
export GEMINI_API_KEY=...        # free key: https://aistudio.google.com/apikey
```

**No API key? Run it locally** — the same on-device model class ChromaDB/MemPalace use,
via `sentence-transformers` (offline, no key):

```bash
pip install "surreal-memory[surrealdb,embeddings]"
smem setup embeddings            # choose "Sentence Transformers"
```

| Provider | Default model | Key | Notes |
|----------|---------------|-----|-------|
| **Gemini** (recommended) | `gemini-embedding-001` | `GEMINI_API_KEY` | 3072-dim, multilingual, free tier |
| Local (sentence-transformers) | `all-MiniLM-L6-v2` · `paraphrase-multilingual-MiniLM-L12-v2` | — | offline, no key, ~440MB download |
| Ollama | `nomic-embed-text` · `bge-m3` | — | local server (`ollama serve`) |
| **Local OpenAI-compatible** (e.g. llamastash) | `bge-m3` | — | any server exposing `/v1/embeddings`; pairs with `bge-reranker-v2-m3` on the same host |
| OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` | paid |
| OpenRouter | `openai/text-embedding-3-small` | `OPENROUTER_API_KEY` | OpenAI-compatible |

**Running bge-m3 on your own OpenAI-compatible server** (llama.cpp, llamastash, vLLM …) — this is
the local pairing for the `bge-reranker-v2-m3` cross-encoder described under Reranking, so both
halves of retrieval stay on one host and nothing leaves the machine:

```toml
[embedding]
enabled = true
provider = "openai"          # the wire format, not the vendor
model = "bge-m3-FP16"        # whatever name your server serves it under
dimension = 1024             # bge-m3 is 1024-dim; drives the HNSW index
endpoint = "http://127.0.0.1:11435/v1"
```

`endpoint` may also come from `SURREAL_MEMORY_EMBEDDING_ENDPOINT`; the config key wins, matching
`[reranker] endpoint`. Only a loopback host is accepted for reasoning-trace work — a remote base is
refused with a warning rather than silently downgrading, because those traces never leave the machine.
The endpoint that passes that check is the one handed to the client as its base URL, so "the check
passed" means the requests really went there — a provider's own hosted default cannot override it.

Set the provider to `auto` to pick the best available option at runtime
(order: Ollama → local sentence-transformers → Gemini → OpenAI → OpenRouter).

**The combination is validated.** Every provider assumes a dimension for a model it does
not recognise, so aiming one at another provider's model name yields vectors of the wrong
width — which the vector index then rejects on every write. `smem doctor`, `smem_health`
and MCP startup report two impossible cases instead of passing them: a model outside a
hosted provider's catalogue (the report lists the models it does serve), and a known model
whose dimension contradicts the configured one. The check stays quiet where it cannot
know — a local OpenAI-compatible server serves whatever files it was pointed at, so an
unfamiliar model name there is normal, and only an exact catalogue match counts as knowing
a dimension.

> Embeddings use **one** model per brain — switching models changes vector
> dimensions and invalidates existing vectors. Pick a provider before ingesting at scale.

---

## Reranking

Spreading activation over-fetches candidates; an optional cross-encoder reranker then
scores each `(query, memory)` pair for relevance and blends that score with the
activation level (`blend_weight`, default `0.7`) for a final precision pass. Off by
default — recall works the same without it.

```toml
[reranker]
enabled = true
endpoint = "http://127.0.0.1:11435/v1"   # OpenAI-compatible /rerank (e.g. llamastash)
model_name = "BAAI/bge-reranker-v2-m3"
blend_weight = 0.7
```

- **HTTP mode** (`endpoint` set) — runs on a shared inference server (e.g. llama.cpp /
  llamastash on GPU), no `torch` dependency needed locally. Falls back to the
  `SURREAL_MEMORY_RERANKER_ENDPOINT` env var when `endpoint` is unset. For endpoints
  that require auth, set `SURREAL_MEMORY_RERANKER_API_KEY` and requests carry a
  `Bearer` header (an empty key sends none, so llamastash needs no extra config).
- **In-process mode** (no endpoint) — loads a local `sentence-transformers` `CrossEncoder`.
  Install with `pip install "surreal-memory[reranker]"`.
- Reranking never breaks recall, but it never fails *quietly* either. A failed rerank
  is retried once; if it still cannot run, recall returns the spreading-activation
  ordering **and says so** — `rerank_degraded` in the MCP response,
  `rerank_degraded_warning` in the CLI. Un-reranked results look exactly like
  reranked ones, so a silent fallback would hide a real drop in precision.

---

## Setup by Tool

<details>
<summary><b>Claude Code (Plugin)</b></summary>

```bash
/plugin marketplace add acidkill/surreal-memory
/plugin install surreal-memory@surreal-memory-marketplace
```

</details>

<details>
<summary><b>Cursor / Windsurf / Other MCP Clients</b></summary>

```bash
pip install surreal-memory[surrealdb]
```

Add to your editor's MCP config:

```json
{
  "mcpServers": {
    "surreal-memory": { "command": "smem-mcp" }
  }
}
```

</details>

<details>
<summary><b>OpenClaw (Plugin)</b></summary>

```bash
pip install surreal-memory[surrealdb] && npm install -g surrealmemory
```

Set memory slot in `~/.openclaw/openclaw.json`:
```json
{ "plugins": { "slots": { "memory": "surrealmemory" } } }
```

</details>

<details>
<summary><b>TypeScript / JavaScript (REST SDK)</b></summary>

```bash
npm install @acidkill/surreal-memory-client
```

```ts
import { SurrealMemoryClient } from "@acidkill/surreal-memory-client"

const client = new SurrealMemoryClient({
  baseUrl: "http://localhost:8000",
  brain: "myproject",
})

await client.remember({ content: "Fixed auth bug", type: "fix", priority: 7 })
const { results } = await client.recall({ query: "auth bug" })
```

Full reference: [`integrations/surreal-memory-client/README.md`](integrations/surreal-memory-client/README.md).

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
from surreal_memory import Brain
from surreal_memory.storage import create_storage
from surreal_memory.core.brain_mode import BrainModeConfig, BrainMode
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline

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

### LangChain integration

Install the extra (`pip install surreal-memory[langchain]`) and wrap a brain as a
LangChain retriever + chat-message history. Both are in-process (no REST server needed);
everything is async underneath with a sync bridge for LangChain's sync API.

```python
from surreal_memory.adapters.langchain import (
    SurrealMemoryChatMessageHistory,
    SurrealMemoryRetriever,
)

# Retriever — matched fibers become LangChain Documents (page_content = memory text,
# metadata carries fiber_id, tags, salience, confidence, source="surreal-memory").
retriever = SurrealMemoryRetriever(brain_name="my_brain", k=5)
docs = await retriever.ainvoke("what did we decide about the backend?")

# Per-session chat history — turns are stored as memories tagged lc-session:<id>.
history = SurrealMemoryChatMessageHistory("session-42", brain_name="my_brain")
history.add_user_message("Which database are we using?")
history.add_ai_message("SurrealDB.")
print(history.messages)  # replayed verbatim, in order

# Already hold a storage handle (tests, custom wiring)? Inject it:
retriever = SurrealMemoryRetriever.from_storage(storage, k=5)
```

See [`examples/langchain_rag.py`](examples/langchain_rag.py) for a full LCEL RAG chain
with `RunnableWithMessageHistory`.

---

## Development

```bash
git clone https://github.com/acidkill/surreal-memory
cd surreal-memory && pip install -e ".[dev]"
smem doctor --dev        # Verify contributor setup
pytest tests/ -v          # 7,267 tests
ruff check src/ tests/    # Lint
make verify               # Full CI gate
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Surreal-Memory is built on top of [**NeuralMemory**](https://github.com/nhadaututtheky/neural-memory) by [nhadaututtheky](https://github.com/nhadaututtheky) — an exceptional graph-based memory system for AI agents. The core architecture (neurons, synapses, fibers, spreading activation, consolidation, compression, and the 53-tool MCP interface) is entirely their work.

Surreal-Memory extends it with a SurrealDB storage backend and a community plugin that makes all advanced features available for free.

Equally important: this project would not exist without [**SurrealDB**](https://surrealdb.com/). The combined document + graph + vector model in a single engine is what made it possible to retire the SQLite + paid-Pro split. The shift specifically depends on the changes shipped in **SurrealDB 3.x** — without them, the storage backend in this fork would still be vaporware. As of v2.6.0 the synapse graph uses native RELATE edges and internal ISO GQL, so **SurrealDB ≥ 3.2.0 is required**.

> If you find Surreal-Memory useful, please also star both the [original NeuralMemory project](https://github.com/nhadaututtheky/neural-memory) and [SurrealDB](https://github.com/surrealdb/surrealdb).

## Roadmap

Items here are explicitly **not** in the current release. Community PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

### Deferred (external action required)

- **ClawHub listing** — the OpenClaw plugin still needs its ClawHub registry entry aligned to the `surreal-memory` slug. The publish workflow already targets `--slug surreal-memory`; waiting on the registry side.
- **VS Code Marketplace publisher** — `vscode-extension/package.json` now uses publisher `ai-flow-nowak`. The publisher account needs to be created / verified before the next release can push to Marketplace.
- **PyPI canonical name** — `surreal-memory` is the only published package. If any earlier-named package remains on PyPI under an account we control, flag it `Development Status :: 7 - Inactive` with a README pointing at `surreal-memory`.

### Nice-to-haves (community contributions welcome)

- **Recall-quality regression gate** — `benchmarks/ground_truth.py` and `metrics.py` can already score recall against a labelled set, but nothing runs them. Wire them into CI with a floor on precision/recall so a retrieval change cannot quietly make the product worse. For a memory system this is the single most valuable test that does not yet exist.
- **One-command trial** — `docker compose up` that brings up SurrealDB, the API, the dashboard and a pre-seeded demo brain. Evaluating the project currently means provisioning a database and choosing an embedding provider first; a newcomer should be able to see a populated graph in one command and judge it from there.
- **Reasoning-pattern injection on by default** — mining learns per-model strategies, but `injection_enabled` defaults to `false`, so the patterns sit unused unless you find the flag. Turn it on behind a token budget and a quality floor, and surface in the dashboard which pattern fired on which turn.
- **Per-project brain routing** — resolve the active brain from the project root or git remote instead of a single global `current_brain`, so changing repository changes memory without `smem brain use`. Removes the most common source of "why is this memory here".
- **PostCompact context restore hook** — analog to `session_start.py` but fired right after a Claude Code compaction. Pipes `smem recap` + `smem context --limit 20` to stdout as a `## Context restored after compaction` block, so the agent doesn't lose the thread when the 80% buffer kicks in.
- **JetBrains IDE plugin** — Kotlin / Java plugin using the same REST API as the dashboard. Parity feature for IntelliJ-family IDEs.
- **Cloudflare Pages for docs** — alternative to the removed GitHub Pages workflow. Static `mkdocs build` deployed to CF Pages, no GitHub dependency.
- **LlamaIndex retriever adapter** — same shape as the shipped LangChain adapter (`from surreal_memory.adapters.langchain import SurrealMemoryRetriever`; see the Python API section), for the LlamaIndex side of the RAG ecosystem.
- **More embedding providers** — Voyage AI, Cohere, Mistral. Current set: Gemini, OpenAI, OpenRouter, Ollama, BGE-M3, sentence-transformers.
- **Upstream sync bot** — scheduled workflow that scans `nhadaututtheky/neural-memory` for new commits, classifies them GREEN / YELLOW / RED against our fork, and opens a draft PR for the green batch.
- **Two-way Telegram bot** — the Telegram integration is one-way (release notes, backups). Extend with `smem remember` via bot commands.
- **Public benchmarks dashboard** — `benchmarks/` already has comparison scripts against mem0 and cognee; publish the results as a static dashboard so the claims are checkable.
- **Brain templates / starter packs** — pre-seeded brains for common workflows (Python dev, K8s admin, research notes).

## License

MIT — see [LICENSE](LICENSE).

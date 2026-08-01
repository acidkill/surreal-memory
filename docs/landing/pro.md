# All Features, Always Free

> Every capability, no paywall. MIT licensed, community driven.

Surreal-Memory is a complete neural memory system for AI agents -- semantic search, graph traversal, lifecycle compression, cloud sync, and more. No tiers. No keys. Install and go.

```bash
pip install surreal-memory[surrealdb]
```

---

## Feature Categories

### Storage Backends

| Backend | Type | Best For |
|---------|------|----------|
| **SurrealDB** (recommended) | Document + Graph + Vector | Full-featured: semantic search, graph queries, vector indexing in one engine |
| SQLite | Relational + FTS5 | Lightweight local-only setups, no external database |

SurrealDB is the recommended backend. It provides native vector search (HNSW), graph traversal, and document storage in a single process -- no extra services to run.

---

### Retrieval

| Method | Description |
|--------|-------------|
| Spreading activation | Graph-based recall that spreads signal through connected neurons. Depth 0-3 controls traversal range. |
| HNSW vector search | Approximate nearest neighbor via SurrealDB's built-in vector index. ~5ms at 1M neurons. |
| Consolidation | The `merge` and `dedup` passes fold near-duplicate memories together using embedding similarity rather than pairwise text comparison. |
| Keyword search | Exact and fuzzy word matching, blended with the graph and vector signals. |

**Blending the signals:** recall runs spreading activation and vector search
together, then optionally reranks. A memory can be semantically relevant (high
similarity) even if rarely accessed, and a frequently accessed one (high
activation) gets a boost at moderate similarity — both signals matter.

The weighting is configurable under `[reranker]` in `config.toml`:
`blend_weight` (default `0.7`) is the reranker's share against spreading
activation, and `min_score` (default `0.15`) drops weak candidates. The vector
floor is `SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD`.

---

### Lifecycle Management

#### 5-Tier Vector Compression

Memories age automatically through storage tiers:

```
ACTIVE   float32   1,536 bytes/neuron   Recently accessed, high priority
   |
WARM     float16     768 bytes          7-30 days old           (-50%)
   |
COOL     int8        384 bytes          30-90 days old          (-75%)
   |
FROZEN   binary       48 bytes          >90 days old            (-97%)
   |
CRYSTAL  metadata      0 bytes          Archived, vector gone   (-100%)
```

Smart rules prevent over-compression:

- Priority 8+ memories always stay ACTIVE
- Recent access auto-promotes back to a higher tier
- Ephemeral memories go straight to CRYSTAL

**Storage impact at scale:**

| Brain size | Uncompressed | With tier compression | Savings |
|-----------|-------------|----------------------|---------|
| 10K neurons | 15 MB | 5 MB | 67% |
| 100K neurons | 150 MB | 25 MB | 83% |
| 1M neurons | 1.5 GB | 120 MB | 92% |

#### Directional Compression

Multi-axis semantic preservation during text compression. Each sentence is scored against the memory's own embedding plus up to 3 related neuron embeddings, keeping sentences that preserve all relevant semantic directions.

```
Score per sentence = primary_similarity * 0.6 + max(reference_similarities) * 0.4
```

#### Consolidation

20 consolidation strategies plus smart merge:

| Strategy | Complexity | Best For |
|----------|-----------|----------|
| Brute-force pairwise | O(N^2) | Small brains (<1K neurons) |
| Embedding-similarity `merge` / `dedup` | O(N x k) | Large brains (1K-1M+ neurons) |

#### Decay and Reinforcement

- Time-based decay lowers activation of unused memories
- Reinforcement strengthens memories on repeated access
- Configurable half-life and reinforcement factor

---

### Sync

**Merkle delta sync** via self-hosted Cloudflare Worker:

- Merkle tree identifies changed neurons efficiently
- Only deltas are transferred -- minimal bandwidth
- End-to-end encryption -- data is unreadable in transit and at rest
- Runs on Cloudflare Workers free tier (100K requests/day)

Deploy with the provided Worker template. No managed service dependency.

---

### Ecosystem

| Interface | Description |
|-----------|-------------|
| **MCP Server** | 58 tools for Claude, GPT, and other agents. Recall, store, consolidate, query. |
| **Web Dashboard** | Browser-based brain inspector. Visualize neurons, fibers, and graph topology. |
| **VS Code Extension** | Inline memory panel. Recall context without leaving the editor. |
| **CLI** | Full control from the terminal. `smem recall`, `smem remember`, `smem consolidate`, etc. |

All interfaces share the same backend. Use one or use them all.

---

### Safety and Security

| Feature | Description |
|---------|-------------|
| Encryption | AES-256 encryption at rest for sensitive neuron data |
| Input firewall | Validates and sanitizes all incoming data before storage |
| Sensitive detection | Automatic flagging of potential secrets, credentials, and PII |
| Local-first | All data stored locally by default. Cloud sync is opt-in and encrypted. |

---

## Feature Comparison

Surreal-Memory vs. alternatives:

| | Surreal-Memory | RAG (naive) | Mem0 | LangChain Memory |
|--|----------------|-------------|------|------------------|
| **Storage model** | Graph + Vector + Document | Vector only | Vector + metadata | Key-value / vector |
| **Semantic search** | HNSW + spreading activation | Cosine similarity | Embedding search | Embedding search |
| **Graph traversal** | Native adjacency BFS | None | Limited | None |
| **Lifecycle management** | 5-tier auto compression | Manual | Manual | Manual |
| **Consolidation** | `merge` / `dedup` (O(N x k)) | N/A | N/A | N/A |
| **Decay / reinforcement** | Built-in | N/A | Partial | N/A |
| **Cloud sync** | Self-hosted (free tier) | N/A | Managed (paid) | N/A |
| **Encryption at rest** | Yes | Varies | No | No |
| **Input firewall** | Yes | No | No | No |
| **License** | MIT | Varies | proprietary | MIT |
| **Cost** | $0 | $0 | Paid tiers | $0 |

---

## MCP Tools

58 tools registered automatically. Key tools by category:

**Recall:**
- `smem_recall` -- query memories with configurable depth and confidence
- `smem_suggest` -- autocomplete from brain neurons

**Storage:**
- `smem_remember` -- store facts, decisions, insights, todos
- `smem_auto` -- auto-capture memories from text
- `smem_forget` -- remove specific memories

**Lifecycle:**
- `smem_tier` -- view and manage storage tier distribution
- `smem_consolidate` -- trigger consolidation strategies

**Sync:**
- `smem_transplant` -- merge brains with conflict resolution
- `smem_version` -- snapshot, rollback, diff brain state
- `smem_train` / `smem_train_db` -- ingest docs or database schema

**Monitoring:**
- `smem_health` -- brain health diagnostics
- `smem_stats` -- memory counts and freshness
- `smem_evolution` -- brain maturation progress
- `smem_session` -- track working session state

---

## Getting Started

```bash
pip install surreal-memory[surrealdb]
```

No configuration required for local use. SurrealDB starts embedded. All features available immediately.

To enable cloud sync, deploy the Cloudflare Worker template and set the endpoint in your config.

---

## Technical Specifications

| Spec | Value |
|------|-------|
| Python | >= 3.11 |
| Vector dimensions | 384 (default, configurable) |
| HNSW params | M=16, ef_construction=200 |
| Max BFS traversal | 1,000 nodes |
| Max vector candidates | 500 |
| Batch insert | Vectorized, atomic rollback |
| Crash recovery | SurrealDB WAL / SQLite WAL |
| Encryption | AES-256 |

---

*Surreal-Memory is MIT licensed. No paid tiers, no feature gates, no license keys. The code is the product.*

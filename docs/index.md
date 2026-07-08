# Surreal-Memory

<p align="center">
  <strong>Graph-based brain memory powered by SurrealDB</strong><br>
  <em>Document + graph + vector in one database. Zero LLM calls. Fully offline.</em>
</p>

<p align="center">
  <a href="https://github.com/acidkill/surreal-memory/actions"><img src="https://github.com/acidkill/surreal-memory/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/surreal-memory/"><img src="https://img.shields.io/pypi/v/surreal-memory.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

---

## What is Surreal-Memory?

Surreal-Memory stores experiences as interconnected neurons and recalls them through **spreading activation** -- mimicking how the human brain works. Instead of searching a database, memories are retrieved through associative recall.

Powered by **SurrealDB**, every memory lives in a single multi-model database that provides document storage, graph traversal, and vector search out of the box. No separate vector store, no embedding API, no LLM calls required.

```bash
# Store a memory
smem remember "Fixed auth bug with null check in login.py:42"

# Recall through association
smem recall "auth bug fix"
# -> "Fixed auth bug with null check in login.py:42"
```

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | Surreal-Memory |
|--------|---------------------|----------------|
| **Database** | Separate vector store + document DB | **SurrealDB** -- document, graph, and vector in one |
| **LLM/Embedding** | Required (embedding API calls) | **None** -- pure algorithmic graph traversal |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO` |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |
| **API Cost** | ~$0.02/1K queries | **$0.00** -- fully offline |
| **Infrastructure** | Pinecone/Weaviate + Postgres | **Single SurrealDB instance** |

!!! example "Example: Causal Query"
    **Query:** "Why did Tuesday's outage happen?"

    - **RAG**: Returns "JWT caused outage" (missing *why* we used JWT)
    - **Surreal-Memory**: Traces `outage <- CAUSED_BY <- JWT <- SUGGESTED_BY <- Alice` -> full causal chain

## The Problem

AI agents face fundamental memory limitations:

| Problem | Impact |
|---------|--------|
| **Limited context windows** | Cannot complete large projects across sessions |
| **Session amnesia** | Forget everything between conversations |
| **No knowledge sharing** | Cannot share learned patterns with other agents |
| **Context overflow** | Important early context gets lost |

## The Solution

| Feature | Benefit |
|---------|---------|
| **Persistent memory** | Survives across sessions |
| **Efficient retrieval** | Inject only relevant context, not everything |
| **Shareable brains** | Export/import patterns like Git repos |
| **Real-time sharing** | Multi-agent collaboration via SurrealDB sync |
| **Project-bounded** | Optimize for active project timeframes |

## Quick Start

### Installation

```bash
pip install surreal-memory[surrealdb]
```

With optional features:

```bash
pip install surreal-memory[server]        # FastAPI server + Web UI
pip install surreal-memory[surrealdb,server]  # SurrealDB + Web UI
pip install surreal-memory[all]           # All features
```

!!! tip "SurrealDB Required"
    The SurrealDB backend requires a running SurrealDB instance. The easiest way is Docker:

    ```bash
    docker compose -f docker-compose.surrealdb.yml up -d
    ```

### Basic Usage

=== "CLI"

    ```bash
    # Store memories
    smem remember "Fixed auth bug with null check in login.py:42"
    smem remember "We decided to use PostgreSQL" --type decision
    smem todo "Review PR #123" --priority 7

    # Query memories
    smem recall "auth bug"
    smem recall "database decision" --depth 2

    # Get context for AI injection
    smem context --limit 10 --json
    ```

=== "Python"

    ```python
    import asyncio
    from surreal_memory import Brain
    from surreal_memory.storage.surrealdb import SurrealDBStorage
    from surreal_memory.engine.encoder import MemoryEncoder
    from surreal_memory.engine.retrieval import ReflexPipeline

    async def main():
        storage = SurrealDBStorage(
            url="http://localhost:8000",
            namespace="memory",
            database="my_project",
        )
        await storage.connect()

        brain = Brain.create("my_brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Encode memories
        encoder = MemoryEncoder(storage, brain.config)
        await encoder.encode("Met Alice to discuss API design")

        # Query through activation
        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query("What did we discuss?")
        print(result.context)

    asyncio.run(main())
    ```

=== "MCP Server"

    ```json
    // ~/.claude/mcp_servers.json
    {
      "surreal-memory": {
        "command": "smem-mcp"
      }
    }
    ```

    Claude will have access to:

    - `smem_remember` - Store memories
    - `smem_recall` - Query memories
    - `smem_context` - Get recent context
    - `smem_todo` - Quick TODO
    - `smem_stats` - Brain statistics
    - `smem_auto` - Auto-capture memories
    - `smem_train_db` - Train brain from database schema
    - `smem_alerts` - View and manage brain health alerts
    - `smem_sync` - Multi-device sync

## VS Code Extension

Install the Surreal-Memory extension for a visual brain explorer directly in your editor:

- **Memory Tree View** -- Browse neurons grouped by type in the activity bar
- **Graph Explorer** -- Interactive Cytoscape.js force-directed graph
- **CodeLens** -- Memory counts on functions/classes, comment trigger detection
- **Encode & Recall** -- Store and query memories from the command palette
- **Real-time Sync** -- WebSocket updates for tree, graph, and status bar

```bash
cd vscode-extension && npm run build
# Install from .vsix or use Extension Developer Host
```

## Web UI Visualization

Start the server and access the interactive brain visualization:

```bash
pip install surreal-memory[server]
smem serve
# Open http://localhost:8000/ui
```

## Features

### SurrealDB Multi-Model Backend

- **Document storage** -- Neurons stored as rich, typed records with full metadata
- **Graph traversal** -- Synapses as native SurrealDB graph edges, queried with graph queries
- **Vector search** -- HNSW cone queries via SurrealDB's built-in `vector::distance::knn()`
- **Single database** -- No separate vector store, no embedding service, one SurrealDB instance

### Community Plugin (Free)

All advanced features are provided by the built-in **CommunityPlugin** at no cost:

- **Cone queries** -- HNSW vector search for semantic similarity retrieval
- **Smart merge** -- Embedding-based consolidation of near-duplicate neurons
- **Directional compression** -- Multi-axis semantic compression preserving entity relationships
- **Merkle delta sync** -- Efficient multi-device synchronization with conflict resolution

!!! success "100% Free, No Paywalls"
    There is no paid tier, no license key, and no feature gate. Every feature listed on this page -- including vector search, cloud sync, smart merge, and the web dashboard -- works out of the box.

### Cognitive Memory

- **Reflex Activation** -- Trail-based retrieval through fiber pathways with conductivity
- **Co-Activation** -- Hebbian binding detects neurons activated by multiple sources
- **Time-First Anchoring** -- Time neurons as primary anchors for temporally-aware recall
- **Spreading Activation** -- Neural graph-based retrieval (classic mode)
- **Adaptive Recall** -- Bayesian depth priors that learn optimal retrieval depth per entity
- **Cross-Encoder Reranking** -- optional config-driven precision pass over SA candidates (HTTP or in-process), blended with activation level
- **Memory Decay** -- Ebbinghaus forgetting curve for natural forgetting

### Knowledge & Reasoning

- **Typed Memories** -- fact, decision, todo, insight, error, workflow, and more
- **Priority System** -- 0-10 priority levels with automatic scoring
- **Expiry/TTL** -- Auto-expire temporary memories
- **Project Scoping** -- Organize memories by project with isolated brain contexts
- **DB-to-Brain Training** -- Teach brains to understand database schemas
- **Codebase Indexing** -- Scan source files and create neurons for functions, classes, imports

### Collaboration & Sync

- **Multi-Device Sync** -- Hub-and-spoke incremental sync with neural-aware conflict resolution
- **Brain Sharing** -- Export, import, merge, and transplant brains between projects
- **Real-time Collaboration** -- Multiple agents reading and writing to the same SurrealDB instance

### Observability

- **Smart Context Optimizer** -- 5-factor composite scoring + SimHash dedup + token budgeting
- **Proactive Alerts** -- Persistent brain health alerts with lifecycle management
- **Recall Pattern Learning** -- Topic co-occurrence mining + follow-up suggestions
- **Brain Health Diagnostics** -- Purity score, grade, component metrics, and actionable warnings
- **Evolution Tracking** -- Maturation progress, learning plasticity, and proficiency level

### Integrations

- **MCP Server** -- First-class Model Context Protocol integration for Claude and other AI agents
- **VS Code Extension** -- Visual brain explorer in your editor
- **Web Dashboard** -- Interactive brain visualization with Cytoscape.js
- **FastAPI Server** -- REST API + WebSocket for custom integrations
- **CLI** -- Full command-line interface for scripting and automation

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install Surreal-Memory and get started in minutes

    [:octicons-arrow-right-24: Install](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Learn the basics with a hands-on tutorial

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-brain:{ .lg .middle } **Concepts**

    ---

    Understand how spreading activation and SurrealDB work together

    [:octicons-arrow-right-24: Concepts](concepts/how-it-works.md)

-   :material-connection:{ .lg .middle } **Integration**

    ---

    Integrate with Claude, Cursor, and other tools

    [:octicons-arrow-right-24: Integration](guides/integration.md)

-   :material-frequently-asked-questions:{ .lg .middle } **FAQ**

    ---

    Common questions, architecture, and honest limitations

    [:octicons-arrow-right-24: FAQ](FAQ.md)

-   :material-chart-bar:{ .lg .middle } **Benchmarks**

    ---

    Reproducible performance measurements

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

</div>

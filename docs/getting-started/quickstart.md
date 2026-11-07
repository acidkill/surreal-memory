# Quick Start

This guide walks you through Surreal-Memory setup and usage in 5 minutes.

!!! tip "3 tools you need"
    Surreal-Memory has 58 tools, but you only need three: **`smem_remember`**, **`smem_recall`**, and **`smem_health`**. The agent handles the other 55 automatically. See [all tools](../guides/mcp-server.md#available-tools).

## Why Surreal-Memory?

Surreal-Memory uses **SurrealDB** as its sole storage backend -- a single database that provides **document**, **graph**, and **vector search** in one engine. No separate vector database, no graph database, no SQLite files scattered across projects. One database, all capabilities.

Every feature is free, with no license key and no paid tier:

- **Semantic search** -- HNSW vector similarity via SurrealDB's native `vector::distance::knn()`
- **Consolidation** -- the `merge` and `dedup` passes fold near-duplicate memories together instead of accumulating them
- **Directional compression** -- multi-axis semantic preservation so the most important information survives compaction

Everything runs locally.

## 0. Setup

!!! warning "Requires SurrealDB ≥ 3.2.0"
    Surreal-Memory requires **SurrealDB 3.2.0 or newer** (the bundled compose file already uses
    it). If you are **upgrading** an existing deployment, back up the `surrealdb_data` volume
    first — the `synapse` graph auto-migrates to native RELATE edges on first connect.

### Docker (Recommended)

Start SurrealDB with the provided compose file:

```bash
docker compose -f docker-compose.surrealdb.yml up -d
```

Then install the package with SurrealDB support:

```bash
pip install surreal-memory[surrealdb]
```

### Claude Code (Plugin)

```bash
/plugin marketplace add acidkill/surreal-memory
/plugin install surreal-memory@surreal-memory-marketplace
```

### OpenClaw (Plugin)

```bash
pip install surreal-memory[surrealdb]
npm install -g surrealmemory
```

Then in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "surrealmemory"
    }
  }
}
```

Restart the gateway. The plugin auto-registers 6 tools (`smem_remember`, `smem_recall`, `smem_context`, `smem_todo`, `smem_stats`, `smem_health`) and injects memory context before each agent run. See the [full setup guide](../guides/openclaw-plugin.md).

### Cursor / Windsurf / Other MCP Clients

```bash
pip install surreal-memory[surrealdb]
```

Then add `smem-mcp` to your editor's MCP config. No `smem init` needed -- the MCP server auto-initializes on first use.

### VS Code Extension

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=neuralmem.surrealmemory) for a visual interface -- sidebar memory tree, interactive graph explorer, CodeLens on functions, and keyboard shortcuts for encode/recall.

### Optional: Explicit Init

```bash
smem init    # Only needed if you want to pre-create config and brain
```

## 1. Store Your First Memory

```bash
smem remember "Fixed auth bug with null check in login.py:42"
```

Output:
```
Stored memory with 4 neurons and 3 synapses
```

## 2. Query Memories

```bash
smem recall "auth bug"
```

Output:
```
Fixed auth bug with null check in login.py:42
(confidence: 0.85, neurons activated: 4)
```

## 3. Use Memory Types

Different types help organize and retrieve memories:

```bash
# Decisions (never expire)
smem remember "We decided to use PostgreSQL" --type decision

# TODOs (expire in 30 days)
smem todo "Review PR #123" --priority 7

# Facts
smem remember "API endpoint is /v2/users" --type fact

# Errors with solutions
smem remember "ERROR: null pointer in auth. SOLUTION: add null check" --type error
```

## 4. Get Context

Retrieve recent memories for AI context injection:

```bash
smem context --limit 5
```

With JSON output for programmatic use:

```bash
smem context --limit 5 --json
```

## 5. View Statistics

```bash
smem stats
```

Output:
```
Brain: default
Neurons: 12
Synapses: 18
Fibers: 4

Memory Types:
  fact: 2
  decision: 1
  todo: 1
```

## 6. Manage Brains

Create separate brains for different projects:

```bash
# List brains
smem brain list

# Create new brain
smem brain create work

# Switch to brain
smem brain use work

# Export brain
smem brain export -o backup.json
```

## 7. Web Visualization

Start the server to visualize your brain:

```bash
pip install surreal-memory[server]
smem serve
```

Open http://localhost:8000/ui to see:

- Interactive neural graph
- Color-coded neuron types
- Click nodes for details

## 8. Check Brain Health

```bash
smem health
```

Output:
```
Brain Health Report
===================
Grade: B+
Purity: 0.92
Freshness: 0.87
Warnings: 0
```

The health command checks memory quality, freshness, topology coherence, and flags issues like duplicate or stale neurons.

## Community Plugin Features

These capabilities ship with every `surreal-memory[surrealdb]` install, at no cost:

### Semantic search (HNSW)

Instead of keyword matching, memories are found by meaning, using SurrealDB's native HNSW vector index for approximate nearest neighbour search across all stored embeddings.

```bash
# Semantic recall finds related concepts, not just exact words
smem recall "how do we handle user authentication"
# Finds: "DECISION: Use JWT for auth" even without keyword overlap
```

### Merging duplicates

When similar memories pile up, consolidation folds them together instead of
letting the brain accumulate near-identical copies. The `merge` pass combines
overlapping fibers; `dedup` links the rest with ALIAS edges.

```bash
smem consolidate                      # runs every pass in order
smem consolidate --strategy merge     # just the merge pass
smem consolidate --strategy merge --dry-run
```

### Directional Compression

As the brain grows, directional compression preserves the most important information along multiple semantic axes -- importance, recency, and uniqueness -- so no critical knowledge is lost during compaction.

## Example Workflow

Here's a typical workflow during a coding session:

```bash
# Start of session - get context
smem context --limit 10

# During work - remember important things
smem remember "UserService now uses async/await"
smem remember "DECISION: Use JWT for auth. REASON: Stateless" --type decision
smem todo "Add rate limiting to API" --priority 8

# When you need to recall
smem recall "auth decision"
smem recall "UserService changes"

# End of session - check what's pending
smem list --type todo

# Check brain health before wrapping up
smem health
```

## Python API

```python
from surreal_memory import Surreal-Memory

nm = Surreal-Memory()  # auto-connects to SurrealDB

# Store
nm.remember("API uses Bearer token auth", memory_type="fact")

# Recall
results = nm.recall("authentication method")
for r in results:
    print(r.content, r.confidence)

# Health check
health = nm.health()
print(f"Grade: {health.grade}, Purity: {health.purity}")
```

## Next Steps

- [CLI Reference](cli.md) -- All commands and options
- [Memory Types](../concepts/memory-types.md) -- Understanding different memory types
- [Integration Guide](../guides/integration.md) -- Integrate with Claude Code, Cursor, and other editors
- [OpenClaw Plugin Guide](../guides/openclaw-plugin.md) -- Full setup for OpenClaw agents

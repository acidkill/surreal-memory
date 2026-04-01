# Advanced Features Quickstart

Surreal-Memory includes all features at no cost — no license keys, no paid tiers. The built-in community plugin provides vector search, smart merge, and directional compression automatically.

This guide covers the advanced features that become available when using SurrealDB as your storage backend.

---

## 1. Install with SurrealDB

```bash
pip install neural-memory[surrealdb]
```

Or with Docker:

```bash
docker compose -f docker-compose.surrealdb.yml up -d
```

Verify the community plugin is active:

```bash
nmem pro status
```

```
Plugin: neural-memory-community v1.0.0
Backend: SurrealDB
Features: cone_query, smart_merge, directional_compress
```

---

## 2. Enable SurrealDB Backend

SurrealDB provides document + graph + vector search in one database. To enable it:

Edit your config:

```toml
# ~/.neuralmemory/config.toml
storage_backend = "surrealdb"
```

Or via CLI:

```bash
nmem config set storage_backend surrealdb
```

Then configure the connection:

```toml
# ~/.neuralmemory/config.toml
[surrealdb]
url = "http://localhost:8001"
user = "root"
pass = "neuralmemory"
namespace = "neural_memory"
database = "default"
```

**Restart your MCP server** (or CLI session). Existing memories are auto-migrated from SQLite to SurrealDB on first startup.

> **Your SQLite database is preserved** — both databases coexist. If you switch back, Surreal-Memory falls back to SQLite automatically. No data loss.

---

## 3. Your first semantic recall

SQLite matches keywords. SurrealDB matches **meaning**.

```bash
# Store some memories
nmem remember "We chose PostgreSQL over MySQL for better JSON support"
nmem remember "JWT rotation was added to fix the session hijack vulnerability"
nmem remember "Alice suggested rate limiting after the DDoS incident"

# Keyword search finds exact matches. Vector search finds semantic matches:
nmem recall "database decisions"       # finds PostgreSQL memory
nmem recall "security improvements"    # finds JWT + rate limiting
```

### Cone Queries — adjustable precision

Narrow the cone for exact matches, widen it for exploration:

```bash
# Via MCP tool
nmem_cone_query(query="auth", threshold=0.85)   # precise — only strong matches
nmem_cone_query(query="auth", threshold=0.60)   # exploratory — cast a wide net
```

Default threshold is `0.75`. Lower = more results, higher = more relevant.

---

## 4. Check your storage tiers

Surreal-Memory automatically manages memory lifecycle across 5 tiers:

| Tier | Format | Size | When |
|------|--------|------|------|
| 1 | float32 | 100% | Fresh memories (< 7 days) |
| 2 | float16 | 50% | Maturing (7–30 days) |
| 3 | int8 | 25% | Stable (30–90 days) |
| 4 | binary | 3% | Archived (90+ days) |
| 5 | metadata | <1% | Ghost tier (rarely accessed) |

Memories auto-promote back to higher tiers when accessed. Check your distribution:

```bash
nmem_tier_info
```

```
Tier distribution:
  float32:  1,234 neurons (12%)
  float16:  3,456 neurons (34%)
  int8:     4,567 neurons (45%)
  binary:      890 neurons (9%)
  metadata:     12 neurons (<1%)

Total storage: 1.2 GB (vs ~5.1 GB without tiering)
Savings: 76%
```

---

## 5. Run Smart Merge

Standard consolidation is O(N²) — it slows down past 10K neurons. Smart Merge uses HNSW neighbor clustering for O(N x k):

```bash
# Dry run first — see what would be merged
nmem consolidate --strategy smart_merge --dry-run

# Run it
nmem consolidate --strategy smart_merge
```

Or via MCP:

```
nmem_pro_merge(dry_run=true)    # preview
nmem_pro_merge()                 # execute
```

Smart Merge finds semantically similar memories (not just keyword duplicates) and consolidates them while preserving causal links.

---

## 6. Connect Cloud Sync

Sync your brain across all your machines:

```bash
# First time: deploy your sync hub (Cloudflare Workers, free tier)
# See: https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/

# Configure sync
nmem_sync_config(hub_url="https://your-hub.workers.dev", api_key="your-key")

# Initial seed (uploads full brain)
nmem sync --seed

# After that: incremental sync
nmem sync              # manual
nmem sync --auto       # auto after every remember/recall
```

Sync uses **Merkle delta** — only changes are transmitted. A brain with 100K neurons syncs in under 2 seconds.

---

## Feature Comparison

| Aspect | SQLite (default) | SurrealDB (advanced) |
|--------|------------------|---------------------|
| Storage engine | SQLite + FTS5 | SurrealDB (doc + graph + vector) |
| Recall method | Keyword matching | Semantic similarity + graph traversal |
| Consolidation | O(N²) brute force | O(N x k) Smart Merge |
| Compression | Text-level trimming | 5-tier vector lifecycle |
| MCP tools | 53 tools | 53 tools + cone_query, tier_info, pro_merge |
| Setup | Built-in, zero config | Requires SurrealDB instance |

**Everything stays the same.** All 53 MCP tools work with both backends. Your existing memories are preserved — when you switch to SurrealDB, they're auto-migrated on first startup.

---

## Troubleshooting

### Community plugin not detected

```bash
# Verify SurrealDB extra is installed
pip show neural-memory | grep surrealdb

# Reinstall with SurrealDB support
pip install neural-memory[surrealdb]
```

### Recall quality didn't improve

Make sure SurrealDB backend is active. If `nmem pro status` shows `Backend: SQLite`, the config change didn't take effect. Verify in `~/.neuralmemory/config.toml`:

```bash
nmem config get storage_backend    # should show "surrealdb"
```

### Want to switch back to SQLite?

```bash
nmem config set storage_backend sqlite
```

Your data stays intact in both databases. No data loss, no migration needed.

---

## Next steps

- [All Features →](https://nhadaututtheky.github.io/neural-memory/landing/pro/)
- [Cloud Sync setup →](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/)
- [Brain Health guide →](https://nhadaututtheky.github.io/neural-memory/guides/brain-health/)

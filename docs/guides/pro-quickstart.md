# Advanced Features Quickstart

Surreal-Memory includes all features at no cost — no license keys, no paid tiers. The built-in community plugin provides vector search, smart merge, and directional compression automatically.

This guide covers the advanced features that become available when using SurrealDB as your storage backend.

---

## 1. Install with SurrealDB

> **Requires SurrealDB ≥ 3.2.0.** Upgrading an existing deployment? Back up the
> `surrealdb_data` volume first — the `synapse` graph auto-migrates to native RELATE edges
> on the first connect after the upgrade.

```bash
pip install surreal-memory[surrealdb]
```

Or with Docker:

```bash
docker compose -f docker-compose.surrealdb.yml up -d
```

Verify the community plugin is active:

```bash
smem doctor
```

```
Pro plugin: surreal-memory-community v1.0.0
Storage backend: surrealdb
```

---

## 2. Enable SurrealDB Backend

SurrealDB provides document + graph + vector search in one database. To enable it:

Edit your config:

```toml
# ~/.surrealmemory/config.toml
storage_backend = "surrealdb"
```

Or via the environment:

```bash
export SURREAL_MEMORY_STORAGE=surrealdb
```

Then configure the connection:

```toml
# ~/.surrealmemory/config.toml
[surrealdb]
url = "http://localhost:8001"
user = "root"
pass = "surrealmemory"
namespace = "surreal_memory"
database = "default"
```

**Restart your MCP server** (or CLI session).

> SurrealDB is the recommended and only first-class backend since the SurrealDB-only
> release. SQLite remains solely as a lightweight test fixture; the legacy
> SQLite↔SurrealDB migration / backend-switch flow was removed.

---

## 3. Your first semantic recall

SQLite matches keywords. SurrealDB matches **meaning**.

```bash
# Store some memories
smem remember "We chose PostgreSQL over MySQL for better JSON support"
smem remember "JWT rotation was added to fix the session hijack vulnerability"
smem remember "Alice suggested rate limiting after the DDoS incident"

# Keyword search finds exact matches. Vector search finds semantic matches:
smem recall "database decisions"       # finds PostgreSQL memory
smem recall "security improvements"    # finds JWT + rate limiting
```

### Semantic recall — adjustable breadth

Recall blends spreading activation with vector similarity. Ask for fewer
results to keep only strong matches, more to explore:

```text
smem_recall(query="auth", limit=5)    # precise — only the strongest matches
smem_recall(query="auth", limit=30)   # exploratory — cast a wide net
```

Tune the similarity floor with `SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD`,
and the reranker's cut-off with `min_score` under `[reranker]` in `config.toml`.

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

```text
smem_tier(action="status")
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

## 5. Merge duplicates

The `merge` pass folds overlapping fibers together; `dedup` links the rest with
ALIAS edges instead of deleting them. With embeddings enabled both work on
semantic similarity, not just keyword overlap.

```bash
# Dry run first — see what would be merged
smem consolidate --strategy merge --dry-run

# Run it
smem consolidate --strategy merge
```

Or via MCP:

```text
smem_consolidate(strategy="merge", dry_run=true)   # preview
smem_consolidate(strategy="merge")                 # execute
```

---

## 6. Connect Cloud Sync

Sync your brain across all your machines:

```bash
# First time: deploy your sync hub (Cloudflare Workers, free tier)
# See: https://nhadaututtheky.github.io/surreal-memory/guides/cloud-sync/

# Configure sync
smem_sync_config(hub_url="https://your-hub.workers.dev", api_key="your-key")

# Manual sync
smem sync sync --direction both
```

Set `SURREAL_MEMORY_SYNC_AUTO=true` to sync after every remember/recall.

Sync uses **Merkle delta** — only changes are transmitted. A brain with 100K neurons syncs in under 2 seconds.

---

## Feature Comparison

| Aspect | SQLite (default) | SurrealDB (advanced) |
|--------|------------------|---------------------|
| Storage engine | SQLite + FTS5 | SurrealDB (doc + graph + vector) |
| Recall method | Keyword matching | Semantic similarity + graph traversal |
| Consolidation | Keyword overlap | Embedding similarity via HNSW neighbours |
| Compression | Text-level trimming | 5-tier vector lifecycle |
| MCP tools | 58 tools | the same 58 tools |
| Setup | Built-in, zero config | Requires SurrealDB instance |

All 58 MCP tools work with both backends. Switching does **not** move your data:
the two backends are separate stores, so export from one and import into the
other — see [Migrating to 3.0](migrating-to-3.0.md).

---

## Troubleshooting

### Community plugin not detected

```bash
# Verify SurrealDB extra is installed
pip show surreal-memory | grep surrealdb

# Reinstall with SurrealDB support
pip install surreal-memory[surrealdb]
```

### Recall quality didn't improve

Make sure the SurrealDB backend is active. `smem doctor` reports the resolved
backend under **Storage backend**; if it says `sqlite`, the config change did
not take effect. Check `storage_backend` in `~/.surrealmemory/config.toml`, or
that `SURREAL_MEMORY_STORAGE` is exported in the same environment as the
process you are running.

### Want to switch back to SQLite?

The SQLite backend is deprecated and **removed in 3.0.0** — see
[Migrating to 3.0](migrating-to-3.0.md). Until then, `SURREAL_MEMORY_STORAGE=sqlite`
still resolves and your `.db` files are untouched, but the two backends do not
share data: whatever you wrote to SurrealDB stays there.

---

## Next steps

- [All Features →](https://nhadaututtheky.github.io/surreal-memory/landing/pro/)
- [Cloud Sync setup →](https://nhadaututtheky.github.io/surreal-memory/guides/cloud-sync/)
- [Brain Health guide →](https://nhadaututtheky.github.io/surreal-memory/guides/brain-health/)

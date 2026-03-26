# Pro Quickstart

You installed Pro. Here's how to make it work for you in 5 minutes.

---

## 1. Install and verify

```bash
pip install neural-memory-pro
```

Pro auto-registers via Python entry points. Verify it's active:

```bash
nmem pro status
```

You should see:

```
Pro: Active
Backend: InfinityDB
License: valid (expires 2026-04-26)
Features: cone_query, smart_merge, directional_compress, auto_tier
```

If you see `Pro: Inactive`, activate your license:

```bash
nmem pro activate YOUR_LICENSE_KEY
```

> Don't have a license key? [Purchase here →](https://nhadaututtheky.github.io/neural-memory/landing/pricing/)

---

## 2. Your first semantic recall

Free tier matches keywords. Pro matches **meaning**.

```bash
# Store some memories
nmem remember "We chose PostgreSQL over MySQL for better JSON support"
nmem remember "JWT rotation was added to fix the session hijack vulnerability"
nmem remember "Alice suggested rate limiting after the DDoS incident"

# Free would need exact keywords. Pro finds semantic matches:
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

## 3. Check your storage tiers

Pro automatically manages memory lifecycle across 5 tiers:

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

## 4. Run Smart Merge

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

## 5. Connect Cloud Sync

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

Pro sync uses **Merkle delta** — only changes are transmitted. A brain with 100K neurons syncs in under 2 seconds.

---

## What changed from Free

| Aspect | Before (Free) | After (Pro) |
|--------|---------------|-------------|
| Storage engine | SQLite + FTS5 | InfinityDB (HNSW vectors) |
| Recall method | Keyword matching | Semantic similarity |
| Consolidation | O(N²) brute force | O(N x k) Smart Merge |
| Compression | Text-level trimming | 5-tier vector lifecycle |
| New MCP tools | — | `nmem_cone_query`, `nmem_tier_info`, `nmem_pro_merge` |

**Everything else stays the same.** All 52 free tools still work. Your existing memories are preserved and auto-indexed by InfinityDB on first startup.

---

## Troubleshooting

### "Pro: Inactive" after install

```bash
# Check if the package is installed
pip show neural-memory-pro

# Re-activate license
nmem pro activate YOUR_LICENSE_KEY

# Check license status
nmem pro status --verbose
```

### Recall quality didn't improve

InfinityDB needs to index your existing neurons. On first startup, this happens automatically but may take a few minutes for large brains (>50K neurons). Check progress:

```bash
nmem_tier_info    # shows indexing progress
```

### Want to downgrade?

```bash
pip uninstall neural-memory-pro
```

Your data stays intact. Neural Memory falls back to SQLite + FTS5 automatically. No data loss, no migration needed.

---

## Next steps

- [Full Pro comparison →](https://nhadaututtheky.github.io/neural-memory/landing/pro/)
- [Cloud Sync setup →](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/)
- [Brain Health guide →](https://nhadaututtheky.github.io/neural-memory/guides/brain-health/)
- [Pricing & plans →](https://nhadaututtheky.github.io/neural-memory/landing/pricing/)

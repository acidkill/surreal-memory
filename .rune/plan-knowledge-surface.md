# Feature: .nm Knowledge Surface (Track A2+A3 Unified)

## Overview
Two-tier memory architecture: Tier 1 = `.nm` flat file (working memory, ~1000 tokens, loaded every session), Tier 2 = `brain.db` SQLite graph (long-term, queried on-demand). The `.nm` file is an agent-navigable knowledge graph with causal edges, topic clusters, urgent signals, and self-routing depth hints. Like prefrontal cortex (fast, small, always-on) vs hippocampus (deep, large, query-on-demand).

## Design (Approved)

### .nm Format Structure (5 sections + frontmatter)
```
---
brain: myproject
updated: 2026-03-16T10:30:00
neurons: 847 | synapses: 3201 | token_budget: 1200
---

# ── GRAPH ── (nodes with [id], edges indented under parent)
[d1] Chose PostgreSQL over MongoDB (decision) {p:8}
  →caused→ [d2] "Need ACID for payment transactions"
  →rejected→ [x1] "MongoDB lacks multi-doc transactions"

# ── CLUSTERS ── (topic groups with @name + [id] refs)
@auth: [d2, f2, f3] "Authentication & authorization"
@payments: [d1, f1, x1] "Payment processing & database"

# ── SIGNALS ── (urgent !, watching ~, uncertain ?)
! [f3] auth middleware rewrite — IN PROGRESS (compliance deadline)
? Redis vs Memcached for session store — UNDECIDED

# ── DEPTH MAP ── (self-routing: when to query brain.db)
[d1] → SUFFICIENT (8 synapses, 3 fibers)
[f3] → NEEDS_DEEP (recall "auth middleware rewrite")

# ── META ── (brain richness stats for agent)
coverage: 0.73 | staleness: 0.12
last_consolidation: 2026-03-15T22:00:00
top_entities: [PostgreSQL, JWT, Redis, FastAPI, Docker]
```

### Key Format Features
- **ID-based cross-referencing**: `[d1]` links across GRAPH→CLUSTERS→SIGNALS→DEPTH MAP
- **Agent-navigable**: SIGNALS first (urgent), GRAPH for landscape, DEPTH MAP for routing
- **Human-readable + editable**: plain text, inspectable, git-trackable
- **ID prefix = type**: d=decision, f=fact, e=error, p=preference, i=insight, w=workflow

### Two-Tier Flow
```
Session Start:
  1. Load surface.nm → agent has working memory (~1000 tokens)
  2. DEPTH MAP tells agent what it already knows vs what needs brain.db

During Session:
  3. Agent sees SUFFICIENT → uses GRAPH context directly
  4. Agent sees NEEDS_DETAIL → calls nmem_recall with focused query
  5. Agent sees NEEDS_DEEP → calls nmem_recall with depth=2+

Session End:
  6. Regenerate surface.nm from brain.db (top-N active neurons)
  7. New decisions/signals from session get promoted to surface
```

### Key Principles
- **Zero LLM calls** — surface generated algorithmically from graph structure
- **Backward compatible** — .nm is additive, existing API unchanged
- **Token-budgeted** — surface never exceeds ~1000 tokens
- **Human-readable** — plain text, inspectable, git-trackable
- **Self-routing** — DEPTH MAP tells agent when to stop vs dig deeper

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Format & Parser | ✅ Done | plan-knowledge-surface-p1.md | .nm format spec, parser, serializer, data models (29 tests) |
| 2 | Surface Generator | ✅ Done | plan-knowledge-surface-p2.md | brain.db → .nm extraction pipeline (13 tests) |
| 3 | MCP Integration | ✅ Done | plan-knowledge-surface-p3.md | Auto-load on init, depth routing, session injection (21 tests) |
| 4 | Lifecycle & Polish | ✅ Done (MVP) | plan-knowledge-surface-p4.md | Session-end regen, MCP tool, atomic write (10 tests) |

## Key Decisions
- Replaces original A2 (layered brain) + A3 (auto-recall injection) with unified approach
- .nm file lives at project root (`.neuralmemory/surface.nm`) and global (`~/.neuralmemory/surface.nm`)
- Surface is a materialized view — regenerated from brain.db, not a separate data store
- DEPTH MAP uses 3 levels: SUFFICIENT, NEEDS_DETAIL, NEEDS_DEEP
- GRAPH uses arrow notation for causal edges (human + agent readable)
- CLUSTERS derived from entity co-occurrence in synapses
- SIGNALS extracted from high-priority + recent memories

## Success Metrics
| Metric | Before | After |
|--------|:------:|:-----:|
| Agent context at session start | 0 memories | Top-N knowledge graph |
| Proactive recall rate | ~30% | 85%+ |
| Unnecessary nmem_recall calls | ~10/session | ~3/session |
| Session start latency | <100ms | <150ms (file read) |
| Token overhead | 0 | ~1000 tokens |

## VISION.md Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | Surface is pre-activated knowledge — top neurons already selected by activation score |
| 2 | Spreading Activation | SA generates the surface; DEPTH MAP triggers SA for deeper queries |
| 3 | No-Embedding Test | Yes — GRAPH/CLUSTERS/SIGNALS are text-based, no embeddings needed |
| 4 | Detail→Speed | DEPTH MAP: SUFFICIENT = instant (no query), NEEDS_DETAIL/DEEP = targeted query |
| 5 | Source Traceable | GRAPH nodes link back to neurons/fibers in brain.db |
| 6 | Brain Test | Prefrontal cortex (working memory) vs hippocampus (long-term) — exact analogy |
| 7 | Memory Lifecycle | Surface regenerated during consolidation = Consolidate phase of lifecycle |

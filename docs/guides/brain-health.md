# Brain Health Guide

Your brain's health grade (A through F) reflects how well-connected, diverse, and active your memory network is. This guide explains each metric, what affects it, and exactly how to improve it.

## Quick Reference

| Grade | Score | Meaning |
|-------|-------|---------|
| **A** | 90-100 | Excellent — dense connections, diverse types, active recall |
| **B** | 75-89 | Good — healthy brain, minor improvements possible |
| **C** | 60-74 | Fair — some weak areas need attention |
| **D** | 40-59 | Poor — significant gaps in connectivity or activity |
| **F** | 0-39 | Failing — empty or severely neglected brain |

Run `nmem_health()` (MCP) or `nmem brain health` (CLI) to see your current grade.

---

## The 7 Health Metrics

Your purity score is a weighted average of 7 components:

```
Purity = Connectivity (25%) + Diversity (20%) + Freshness (15%)
       + Consolidation (15%) + Orphan Rate (10%) + Activation (10%)
       + Recall Confidence (5%)
       - Conflict Penalty (up to -10 points)
```

### 1. Connectivity (25% weight)

**What it measures:** How densely connected your neurons are via synapses.

**Target:** 3-8 synapses per neuron.

**Why it matters:** Memories without connections can't be found through spreading activation. A neuron with 0 synapses is invisible to recall.

**How to improve:**

- Store memories with **causal context**: "X caused Y", "After A, we did B"
- Avoid flat statements: "PostgreSQL" (bad) vs "Chose PostgreSQL over MongoDB because ACID needed for payments" (good)
- The encoder creates more synapses when your text contains relationships

!!! tip "Quick win"
    Store 10 memories using "because", "after", "caused by", "leads to" — each generates 2-4 more synapses than flat facts.

### 2. Diversity (20% weight)

**What it measures:** How many different synapse types your brain uses (Shannon entropy).

**Target:** 4+ of 8 expected types.

**Common synapse types:**

| Type | Triggered by |
|------|-------------|
| `RELATED_TO` | Default association |
| `CAUSED_BY` | "X caused Y", "because", "due to" |
| `LEADS_TO` | "then", "next", "after that" |
| `CO_OCCURS` | Memories stored in same session |
| `RESOLVED_BY` | Error + fix stored with overlapping tags |
| `CONTAINS` | Fiber membership |
| `PRECEDES` | Temporal ordering |
| `CONTRADICTS` | Conflicting memories detected |

**How to improve:**

- Use **varied language** in your memories:
    - Causal: "The outage was caused by the JWT expiry bug"
    - Temporal: "After deploying v2, we noticed the memory leak"
    - Relational: "Redis connects to the auth service through the session store"
- Store **error + fix pairs** — this creates `RESOLVED_BY` synapses automatically

### 3. Freshness (15% weight)

**What it measures:** Fraction of memories accessed or created in the last 7 days.

**Target:** 30%+ of fibers should be "fresh".

**How to improve:**

- Use `nmem_recall` regularly on topics you're working on
- Each recall refreshes the memory's timestamp
- Start sessions with `nmem_recap()` — this touches recent memories
- Don't just store — **recall** your memories to keep them fresh

!!! warning "Stale Brain"
    If freshness hits 0% (no activity in 7 days), you get a CRITICAL warning. Even one `nmem_recall` call fixes this.

### 4. Consolidation Ratio (15% weight)

**What it measures:** Fraction of memories that have matured from EPISODIC to SEMANTIC stage.

**Why memories start at 0%:** New memories are always EPISODIC. They advance through the memory lifecycle:

```
EPISODIC → WORKING → SEMANTIC
```

**How memories consolidate:**

1. **Natural maturation**: Memories recalled multiple times over days mature automatically
2. **Manual consolidation**: Run `nmem consolidate --strategy mature`
3. **Time**: The system periodically checks for memories ready to advance

**How to improve:**

```bash
# Run consolidation manually
nmem consolidate --strategy mature

# Or via MCP
nmem_auto(action="process", text="consolidate")
```

!!! note "Realistic expectations"
    A brand new brain will have 0% consolidation — this is normal. After 1-2 weeks of active use with regular recalls, expect 20-40%. After a month, 50%+.

### 5. Orphan Rate (10% weight, inverted)

**What it measures:** Fraction of neurons with **no synapses AND no fiber membership**.

**Target:** Under 20%.

**Why orphans exist:**

- The encoder creates entity neurons (people, tools, concepts) that may not have direct connections yet
- Temporal neurons (dates, times) sometimes lack explicit synapses
- Deleted or pruned fibers can leave behind orphaned neurons

**How to improve:**

```bash
# See how many orphans you have
nmem brain health

# Prune orphans (safe — only removes truly disconnected neurons)
nmem consolidate --strategy prune

# Or build connections by recalling related topics
nmem_recall("topic related to orphaned entities")
```

!!! tip "Prune is safe"
    `--strategy prune` only removes neurons with zero synapses AND zero fiber membership. It cannot delete anything that's part of a memory trace.

### 6. Activation Efficiency (10% weight)

**What it measures:** Fraction of neurons that have been accessed at least once.

**Target:** 30%+ of neurons should have `access_frequency > 0`.

**Why it's low for new brains:** When you store memories, neurons are created but not yet "activated" by recall queries. Only when you **recall** memories do their neurons get activated.

**How to improve:**

- Recall memories across different topics: `nmem_recall("auth")`, `nmem_recall("database")`, `nmem_recall("deployment")`
- Each recall activates the matched neurons and their neighbors
- Aim to recall 5+ different topics per week

!!! note "Grade D with 5% activation"
    This means 95% of your neurons have never been recalled. Store less, recall more — the value of memory is in retrieval, not storage.

### 7. Recall Confidence (5% weight)

**What it measures:** Average synapse weight (connection strength).

**Target:** 0.50+

**How it improves:**

- Synapse weights increase each time a memory is recalled
- "Neurons that fire together wire together" — repeated recall strengthens connections
- This metric improves naturally with regular usage

---

## Understanding Your Health Report

When you run `nmem_health()`, you get:

```json
{
  "purity_score": 44.9,
  "grade": "D",
  "top_penalties": [
    {
      "component": "consolidation_ratio",
      "current_score": 0.0,
      "penalty_points": 15.0,
      "estimated_gain": 12.0,
      "action": "Run `nmem consolidate` — 100% of fibers still episodic..."
    },
    {
      "component": "activation_efficiency",
      "current_score": 0.05,
      "penalty_points": 9.5,
      "estimated_gain": 7.5,
      "action": "Recall memories by topic — 95% of neurons never accessed..."
    }
  ]
}
```

**How to read `top_penalties`:**

1. **`penalty_points`** — How many points this metric is costing you (higher = more impact)
2. **`estimated_gain`** — Points you'd gain by improving this to 0.8
3. **`action`** — Exactly what to do

**Always fix the highest penalty first** — it gives you the most score improvement per effort.

---

## Improvement Roadmap

### Week 1: Foundation (F → D)

- [ ] Store 20+ memories with rich context (causes, effects, decisions)
- [ ] Run `nmem consolidate --strategy prune` to clean orphans
- [ ] Recall 5 different topics to activate neurons

### Week 2-3: Growth (D → C)

- [ ] Run `nmem consolidate --strategy mature` to advance memory stages
- [ ] Use varied language (causal, temporal, relational) for diversity
- [ ] Start sessions with `nmem_recap()` to maintain freshness
- [ ] Resolve any memory conflicts: `nmem_conflicts(action="list")`

### Month 1+: Maturity (C → B → A)

- [ ] Regular recall across all stored topics (activation efficiency)
- [ ] Weekly health check: `nmem_health()` → fix top penalty
- [ ] Knowledge base training: `nmem_train(path="docs/")` for permanent foundation
- [ ] Repeated recalls over time naturally strengthen recall confidence

---

## Common Issues

### "Consolidation is 0% — is something broken?"

No. New brains always start at 0%. Memories must be recalled multiple times over several days before they're eligible for maturation. Run `nmem consolidate --strategy mature` after your first week of active use.

### "25% orphan neurons — should I worry?"

Orphans above 20% trigger a warning. Run `nmem consolidate --strategy prune` to clean them up. This is safe and only removes truly disconnected neurons.

### "Grade D but I have 500+ memories"

Quantity doesn't equal quality. Check your `top_penalties`:

- **Low connectivity?** → Your memories are flat statements, not rich relationships
- **Low activation?** → You store but never recall. Use `nmem_recall` more
- **Low diversity?** → Use varied language (causes, sequences, comparisons)

### "How often should I run consolidation?"

- **Prune**: Monthly, or when orphan rate > 20%
- **Mature**: Weekly during active use periods
- **Full** (`nmem consolidate`): Weekly is a good cadence

---

## Connection Tracing with nmem_explain

Use `nmem_explain` to understand **why** two concepts are (or aren't) connected:

```bash
# CLI
nmem explain "Redis" "auth outage"

# MCP tool
nmem_explain(entity_a="Redis", entity_b="auth outage")
```

This traces the shortest path through your neural graph:

```
Redis → USED_BY → session-store → CAUSED_BY → auth outage
```

**Use cases:**

- Debug why recall returns (or doesn't return) certain memories
- Understand the causal chain between events
- Verify that your brain has the connections you expect
- Discover unexpected relationships between concepts

If no path exists, it means the two concepts have no connection in your brain — store memories that link them.

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

Run `smem_health()` (MCP) or `smem brain health` (CLI) to see your current grade.

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

- Use `smem_recall` regularly on topics you're working on
- Each recall refreshes the memory's timestamp
- Start sessions with `smem_recap()` — this touches recent memories
- Don't just store — **recall** your memories to keep them fresh

!!! warning "Stale Brain"
    If freshness hits 0% (no activity in 7 days), you get a CRITICAL warning. Even one `smem_recall` call fixes this.

### 4. Consolidation Ratio (15% weight)

**What it measures:** Fraction of memories that have matured from EPISODIC to SEMANTIC stage.

**Why memories start at 0%:** New memories are always EPISODIC. They advance through the memory lifecycle:

```
EPISODIC → WORKING → SEMANTIC
```

**How memories consolidate:**

1. **Natural maturation**: Memories recalled multiple times over days mature automatically
2. **Recall the memories you care about.** Maturation is driven by spaced recall, not by a
   command: `smem consolidate --strategy mature` only promotes fibers that have ALREADY met the
   dwell-and-spacing bar. Running it on a brain with nothing eligible advances nothing.
3. **Time**: The system periodically checks for memories ready to advance

**How to improve:**

```bash
# Run consolidation manually
smem consolidate --strategy mature

# Or via MCP
smem_auto(action="process", text="consolidate")
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
smem brain health

# Prune orphans (safe — only removes truly disconnected neurons)
smem consolidate --strategy prune

# Or build connections by recalling related topics
smem_recall("topic related to orphaned entities")
```

!!! tip "Prune is safe"
    `--strategy prune` only removes neurons with zero synapses AND zero fiber membership. It cannot delete anything that's part of a memory trace.

### 6. Activation Efficiency (10% weight)

**What it measures:** Fraction of neurons that have been accessed at least once.

**Target:** 30%+ of neurons should have `access_frequency > 0`.

**Why it's low for new brains:** When you store memories, neurons are created but not yet "activated" by recall queries. Only when you **recall** memories do their neurons get activated.

**How to improve:**

- Recall memories across different topics: `smem_recall("auth")`, `smem_recall("database")`, `smem_recall("deployment")`
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

When you run `smem_health()` — or `smem health --json`, which returns the same payload — you
get:

```json
{
  "purity_score": 44.9,
  "grade": "D",
  "consolidation_ratio": 0.0,
  "stage_distribution": {"stm": 40, "working": 12, "episodic": 30, "semantic": 0},
  "semantic_gate_blockers": {"time_gate": 18, "spacing_gate": 9, "ready": 3},
  "top_penalties": [
    {
      "component": "consolidation_ratio",
      "current_score": 0.0,
      "penalty_points": 15.0,
      "estimated_gain": 12.0,
      "action": "100% of fibers are still episodic (target: 50%+ semantic). They advance to semantic only through spaced recall..."
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

**How to read the maturation fields:** `consolidation_ratio` is one number and cannot explain
itself. `stage_distribution` says which stage the fibers that have *not* reached semantic sit
at. `semantic_gate_blockers` splits the EPISODIC ones by what is actually holding them back:

| Bucket | Meaning | What moves it |
|--------|---------|---------------|
| `time_gate` | Has not dwelt in EPISODIC long enough | Time |
| `spacing_gate` | Reinforcement not yet spread across 3+ distinct days (or 15+ rehearsals across 5+ time windows) | Recalling those memories on more separate days |
| `ready` | Both gates satisfied | The next `smem consolidate` pass |

Only the `ready` bucket moves by running a command. Both fields are omitted entirely on a
backend that cannot report maturation — that is not the same as an empty distribution.
`smem health` renders all of this as a **Maturation** section.

---

## Improvement Roadmap

### Week 1: Foundation (F → D)

- [ ] Store 20+ memories with rich context (causes, effects, decisions)
- [ ] Run `smem consolidate --strategy prune` to clean orphans
- [ ] Recall 5 different topics to activate neurons

### Week 2-3: Growth (D → C)

- [ ] Recall actively for a week, then run `smem consolidate --strategy mature` to promote whatever became eligible
- [ ] Use varied language (causal, temporal, relational) for diversity
- [ ] Start sessions with `smem_recap()` to maintain freshness
- [ ] Resolve any memory conflicts: `smem_conflicts(action="list")`

### Month 1+: Maturity (C → B → A)

- [ ] Regular recall across all stored topics (activation efficiency)
- [ ] Weekly health check: `smem_health()` → fix top penalty
- [ ] Knowledge base training: `smem_train(path="docs/")` for permanent foundation
- [ ] Repeated recalls over time naturally strengthen recall confidence

---

## Common Issues

### "Consolidation is 0% — is something broken?"

No. New brains always start at 0%. Memories must be recalled multiple times over several days before they are eligible for maturation — recall is what advances them. Once that has happened, `smem consolidate --strategy mature` promotes the ones that qualify; run before then it has nothing to promote.

### "25% orphan neurons — should I worry?"

Orphans above 20% trigger a warning. Run `smem consolidate --strategy prune` to clean them up. This is safe and only removes truly disconnected neurons.

### "Grade D but I have 500+ memories"

Quantity doesn't equal quality. Check your `top_penalties`:

- **Low connectivity?** → Your memories are flat statements, not rich relationships
- **Low activation?** → You store but never recall. Use `smem_recall` more
- **Low diversity?** → Use varied language (causes, sequences, comparisons)

### "How often should I run consolidation?"

- **Prune**: Monthly, or when orphan rate > 20%
- **Mature**: Weekly during active use periods
- **Full** (`smem consolidate`): Weekly is a good cadence

---

## Connection Tracing with smem_explain

Use `smem_explain` to understand **why** two concepts are (or aren't) connected:

```bash
# CLI
smem explain "Redis" "auth outage"

# MCP tool
smem_explain(entity_a="Redis", entity_b="auth outage")
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

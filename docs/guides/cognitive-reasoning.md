# Cognitive Reasoning Guide

Neural Memory's Cognitive Reasoning Layer lets agents form hypotheses, gather evidence, make predictions, and verify outcomes — a structured scientific reasoning loop built on top of the memory graph.

## Quick Start

```
nmem_hypothesize → nmem_evidence → nmem_predict → nmem_verify
       ↑                                              |
       └──────── nmem_schema (evolve) ←───────────────┘
```

The core loop:

1. **Hypothesize** — Form a belief about something uncertain
2. **Evidence** — Add supporting or contradicting evidence
3. **Predict** — Make a falsifiable prediction based on a hypothesis
4. **Verify** — Check if the prediction was correct or wrong
5. **Evolve** — Update the hypothesis when your understanding changes

Supporting tools:

- **nmem_cognitive** — Dashboard: hot index + calibration score
- **nmem_gaps** — Track what you *don't* know
- **nmem_explain** — Trace connections between concepts

---

## Tool Reference

### nmem_hypothesize

Create, list, or inspect hypotheses.

**Actions:**

| Action | Parameters | Description |
|--------|-----------|-------------|
| `create` | `content`, `confidence` (0.01-0.99, default 0.5), `priority` (0-10), `tags` | Create a new hypothesis |
| `list` | `status` (active/confirmed/refuted/superseded), `limit` (max 100) | List hypotheses |
| `get` | `hypothesis_id` | Get full state + all evidence |

**Example:**
```
nmem_hypothesize(
  action="create",
  content="Redis session store is causing the 500ms latency spike on /api/users",
  confidence=0.6,
  tags=["performance", "redis", "api"]
)
# Returns: { hypothesis_id: "abc123", fiber_id: "...", neurons_created: 5 }
```

### nmem_evidence

Add evidence for or against a hypothesis. Each piece of evidence is a real memory neuron linked via synapse.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `hypothesis_id` | Yes | Which hypothesis to update |
| `content` | Yes | The evidence text |
| `type` | Yes | `"for"` or `"against"` |
| `weight` | No | Strength 0.1-1.0 (default 0.5) |
| `priority`, `tags` | No | Standard memory metadata |

**Returns:** `confidence_before`, `confidence_after`, `confidence_delta`, evidence counts, and `auto_resolved` if threshold was hit.

**Example:**
```
nmem_evidence(
  hypothesis_id="abc123",
  content="Redis SLOWLOG shows 450ms KEYS command during spike window",
  type="for",
  weight=0.8
)
# Returns: { confidence_before: 0.6, confidence_after: 0.69, confidence_delta: +0.09, ... }
```

### nmem_predict

Make a falsifiable prediction, optionally linked to a hypothesis.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `create` | `content`, `confidence` (default 0.7), `deadline` (ISO datetime), `hypothesis_id` | Create prediction |
| `list` | `status`, `limit` | List predictions with calibration stats |
| `get` | `prediction_id` | Get prediction details |

**Example:**
```
nmem_predict(
  action="create",
  content="Replacing KEYS with SCAN will reduce p99 latency below 100ms",
  confidence=0.8,
  deadline="2026-03-15T00:00:00",
  hypothesis_id="abc123"
)
```

### nmem_verify

Verify a prediction outcome. Automatically propagates to linked hypothesis.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prediction_id` | Yes | Which prediction to verify |
| `outcome` | Yes | `"correct"` or `"wrong"` |
| `content` | No | Observation/evidence text |

**Propagation:** If the prediction is linked to a hypothesis:
- `correct` → adds `evidence_for` (weight=0.6) to hypothesis
- `wrong` → adds `evidence_against` (weight=0.6) to hypothesis

**Example:**
```
nmem_verify(
  prediction_id="pred456",
  outcome="correct",
  content="After SCAN migration, p99 dropped to 45ms. Confirmed via Grafana dashboard."
)
# Returns: { calibration_score: 0.75, propagated_to_hypothesis: { id: "abc123", new_confidence: 0.78 } }
```

### nmem_schema

Evolve hypotheses when your understanding changes.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `evolve` | `hypothesis_id`, `content`, `confidence`, `reason` | Create new version, supersede old |
| `history` | `hypothesis_id` | Walk version chain |
| `compare` | `hypothesis_id`, `other_id` | Side-by-side comparison |

**Example:**
```
nmem_schema(
  action="evolve",
  hypothesis_id="abc123",
  content="Latency spike is caused by KEYS + connection pool exhaustion together, not KEYS alone",
  reason="SCAN fix reduced latency but didn't eliminate spikes completely"
)
# Returns: { new_hypothesis_id: "def789", schema_version: 2, old_status: "superseded" }
```

### nmem_cognitive

Dashboard view of your cognitive state.

| Action | Description |
|--------|-------------|
| `summary` | Hot index (top 20 items by urgency) + calibration + top gaps |
| `refresh` | Recompute hot index from scratch (O(n), use sparingly) |

### nmem_gaps

Track knowledge gaps — things you *don't* know.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `detect` | `topic`, `source`, `priority`, `related_neuron_ids` | Register a gap |
| `list` | `include_resolved`, `limit` | List unresolved gaps |
| `resolve` | `gap_id`, `resolved_by_neuron_id` | Mark gap as resolved |
| `get` | `gap_id` | Get gap details |

**Detection sources** (with default priority):

| Source | Priority | When to use |
|--------|----------|-------------|
| `contradicting_evidence` | 0.8 | Two pieces of evidence conflict |
| `low_confidence_hypothesis` | 0.7 | Hypothesis stuck at ~0.5 |
| `user_flagged` | 0.6 | Agent or user explicitly marks unknown |
| `recall_miss` | 0.5 | Recall returned no results for a topic |
| `stale_schema` | 0.4 | Hypothesis hasn't been updated in a long time |

---

## Bayesian Confidence Formula

Confidence updates use a surprise-weighted Bayesian-inspired formula:

```
direction = +1.0 (evidence_for) or -1.0 (evidence_against)

surprise = (1.0 - confidence)    if direction > 0    # confirming strong belief = low surprise
           confidence             if direction < 0    # contradicting strong belief = high surprise

dampening = 1.0 / (1.0 + 0.1 * total_evidence_count)

shift = direction * weight * surprise * dampening * 0.3

new_confidence = clamp(confidence + shift, 0.01, 0.99)
```

**Key properties:**

| Property | Effect |
|----------|--------|
| **Surprise weighting** | Contradicting a strong belief moves confidence more than confirming it |
| **Dampening** | More evidence accumulated = smaller individual updates (posterior stability) |
| **Soft scaling (0.3)** | Prevents wild swings from single evidence |
| **Bounds [0.01, 0.99]** | Confidence never reaches 0 or 1 — always revisable |

**Worked example:**

```
Starting: confidence=0.5, for=0, against=0

Add evidence_for (weight=0.7):
  surprise = 1.0 - 0.5 = 0.5
  dampening = 1.0 / (1.0 + 0.1 * 0) = 1.0
  shift = 1.0 * 0.7 * 0.5 * 1.0 * 0.3 = 0.105
  new_confidence = 0.605

Add evidence_for (weight=0.8):
  surprise = 1.0 - 0.605 = 0.395
  dampening = 1.0 / (1.0 + 0.1 * 1) = 0.909
  shift = 1.0 * 0.8 * 0.395 * 0.909 * 0.3 = 0.086
  new_confidence = 0.691

Add evidence_against (weight=0.9):
  surprise = 0.691 (contradicting strong belief = high surprise)
  dampening = 1.0 / (1.0 + 0.1 * 2) = 0.833
  shift = -1.0 * 0.9 * 0.691 * 0.833 * 0.3 = -0.155
  new_confidence = 0.536
```

Notice how the single `against` evidence (weight=0.9) nearly undid two `for` pieces — that's surprise weighting in action.

---

## Auto-Resolution

A hypothesis auto-resolves when **both** conditions are met:

| Status | Confidence | Evidence count |
|--------|-----------|---------------|
| `confirmed` | >= 0.9 | `evidence_for` >= 3 |
| `refuted` | <= 0.1 | `evidence_against` >= 3 |

Requiring both prevents false positives from a single high-weight evidence.

---

## Hypothesis Lifecycle

```
                    create
                      |
                      v
                   [active] ──── add evidence ────┐
                      |                           |
                      |                    (auto-resolve?)
                      |                     /          \
                      |              confirmed       refuted
                      |             (conf>=0.9      (conf<=0.1
                      |              & for>=3)       & against>=3)
                      |
                 schema evolve
                   /        \
            [superseded]   [new active v2]
```

**Valid statuses:** `active`, `confirmed`, `refuted`, `superseded`, `pending`, `expired`

---

## Hot Index Scoring

The hot index ranks items by urgency. `nmem_cognitive(action="summary")` returns the top 20.

**Hypothesis score:**
```
confidence_interest = 1.0 - abs(confidence - 0.5) * 2.0    # Mid-confidence = most interesting
evidence_factor = min(evidence_count / 5.0, 1.0)            # More evidence = more developed
recency = 1.0 / (1.0 + age_days / 30.0)                     # Recent = more relevant

score = confidence_interest * 3 + evidence_factor * 4 + recency * 3
# Range: ~[0, 10]
```

**Prediction score:**
```
if overdue:    score = 10.0                                  # Overdue = most urgent
else:          score = 10.0 / (1.0 + days_until_deadline / 3.0)
```

**Calibration score:** `correct_count / total_resolved` (0.5 if no data).

---

## End-to-End Examples

### Example 1: Debugging a Performance Issue

```python
# 1. Form hypothesis
nmem_hypothesize(
  action="create",
  content="The /api/orders endpoint is slow because of N+1 queries in the OrderSerializer",
  confidence=0.6,
  tags=["performance", "api", "database"]
)
# → hypothesis_id: "h1"

# 2. Add evidence supporting it
nmem_evidence(
  hypothesis_id="h1",
  content="Django Debug Toolbar shows 47 SQL queries for a single /api/orders?limit=10 request",
  type="for",
  weight=0.8
)
# → confidence: 0.6 → 0.69

# 3. Add evidence against it
nmem_evidence(
  hypothesis_id="h1",
  content="Adding select_related('customer') reduced queries to 12 but response time only improved by 15%",
  type="against",
  weight=0.6
)
# → confidence: 0.69 → 0.60

# 4. Make a testable prediction
nmem_predict(
  action="create",
  content="If I add prefetch_related for order_items and apply pagination, response time will drop below 200ms",
  confidence=0.7,
  hypothesis_id="h1",
  deadline="2026-03-10T00:00:00"
)
# → prediction_id: "p1"

# 5. Implement the fix, then verify
nmem_verify(
  prediction_id="p1",
  outcome="wrong",
  content="prefetch_related helped (350ms→180ms for small pages) but large pages still 800ms — the real bottleneck is JSON serialization of nested objects, not DB queries"
)
# → prediction refuted, hypothesis h1 gets evidence_against, confidence drops

# 6. Evolve the hypothesis with new understanding
nmem_schema(
  action="evolve",
  hypothesis_id="h1",
  content="The /api/orders slowness is primarily caused by deep JSON serialization of nested OrderItem objects, with N+1 queries as a secondary factor",
  reason="DB optimization helped but didn't solve it — profiling shows 60% time in serializer"
)
# → h1 superseded, new hypothesis h2 (version 2) created

# 7. Track what we still don't know
nmem_gaps(
  action="detect",
  topic="Best serialization strategy for deeply nested order data",
  source="low_confidence_hypothesis"
)
```

### Example 2: Architecture Decision

```python
# 1. Hypothesis about tech choice
nmem_hypothesize(
  action="create",
  content="Moving from REST to GraphQL will reduce mobile app data fetching by 60% because clients can request exactly the fields they need",
  confidence=0.5,
  tags=["architecture", "graphql", "mobile"]
)
# → hypothesis_id: "h_gql"

# 2. Research evidence
nmem_evidence(
  hypothesis_id="h_gql",
  content="Analyzed 50 mobile API calls — 38 of them fetch >5 unused fields, averaging 40% payload waste",
  type="for", weight=0.7
)

nmem_evidence(
  hypothesis_id="h_gql",
  content="GraphQL introduces N+1 at resolver level — DataLoader needed, adds complexity",
  type="against", weight=0.5
)

nmem_evidence(
  hypothesis_id="h_gql",
  content="Team has zero GraphQL experience — estimated 3-week learning curve from senior dev",
  type="against", weight=0.6
)

# 3. Check cognitive dashboard
nmem_cognitive(action="summary")
# → Shows h_gql with mid-confidence (still uncertain), 3 evidence pieces

# 4. Make prediction before committing
nmem_predict(
  action="create",
  content="A proof-of-concept GraphQL endpoint for /orders will show >=40% payload reduction in 2 days of work",
  confidence=0.7,
  hypothesis_id="h_gql",
  deadline="2026-03-12T00:00:00"
)

# 5. After PoC...
nmem_verify(
  prediction_id="...",
  outcome="correct",
  content="PoC showed 52% payload reduction. But took 4 days not 2 (DataLoader complexity)."
)
# → h_gql confidence increases, but we note the timeline was off

# 6. Evolve with nuance
nmem_schema(
  action="evolve",
  hypothesis_id="h_gql",
  content="GraphQL reduces payload by ~50% but implementation cost is 2x estimated due to DataLoader complexity — worth it only for high-traffic endpoints",
  reason="PoC confirmed payload savings but revealed hidden complexity cost"
)
```

### Example 3: Tracking Prediction Accuracy

```python
# After multiple predict/verify cycles, check calibration
nmem_predict(action="list")
# Returns:
# {
#   calibration: {
#     score: 0.67,         # You're right 67% of the time
#     correct: 4,
#     wrong: 2,
#     total_resolved: 6,
#     pending: 3
#   },
#   predictions: [...]
# }

# If calibration is low, register a gap
nmem_gaps(
  action="detect",
  topic="Improving prediction accuracy for performance estimates",
  source="user_flagged",
  priority=0.7
)

# Review all active cognitive items
nmem_cognitive(action="refresh")  # Recompute rankings
nmem_cognitive(action="summary")  # View dashboard
```

---

## Synapse Types

The cognitive layer creates these synapse types automatically:

| Synapse Type | Direction | Created by |
|-------------|-----------|------------|
| `EVIDENCE_FOR` | evidence → hypothesis | `nmem_evidence(type="for")` |
| `EVIDENCE_AGAINST` | evidence → hypothesis | `nmem_evidence(type="against")` |
| `PREDICTED` | prediction → hypothesis | `nmem_predict(hypothesis_id=...)` |
| `VERIFIED_BY` | prediction → observation | `nmem_verify(outcome="correct")` |
| `FALSIFIED_BY` | prediction → observation | `nmem_verify(outcome="wrong")` |
| `SUPERSEDES` | new hypothesis → old | `nmem_schema(action="evolve")` |

These synapses are traversable via `nmem_explain` — you can trace the full reasoning chain from evidence through hypothesis to prediction to verification.

---

## Best Practices

1. **Start at 0.5 confidence** unless you have prior knowledge. This gives equal room to move in either direction.

2. **Use weight to express evidence strength.** A log file showing exact error = weight 0.9. A hunch from a teammate = weight 0.3.

3. **Make predictions falsifiable.** "The app will be faster" is bad. "Response time will drop below 200ms after adding an index" is good.

4. **Set deadlines on predictions.** This creates urgency in the hot index and prevents forgotten predictions.

5. **Evolve, don't delete.** When a hypothesis is partially wrong, use `nmem_schema(evolve)` instead of creating a new one. This preserves the reasoning chain.

6. **Use `nmem_gaps` proactively.** When you notice uncertainty, register it. Gaps surface in `nmem_cognitive(summary)` so they don't get forgotten.

7. **Check calibration regularly.** If your prediction accuracy is below 50%, you may be overconfident — lower your default confidence.

8. **Refresh the hot index** after a batch of evidence/verification updates. It's O(n) so don't call it after every single operation.

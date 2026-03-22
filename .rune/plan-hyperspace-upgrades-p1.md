# Phase 1: Quick Wins (Tier 1)

## Goal
4 backward-compatible improvements that boost retrieval quality and performance with zero new dependencies. Each is independent — can ship in any order.

## Tasks

### 1.1 — Generation-Based Visited Tracking
**File:** `src/neural_memory/engine/activation.py`
**Current:** `visited: set[tuple[str, str]]` cleared per search, O(N) allocation
**Target:** Per-neuron generation counter, O(1) "clear" via counter increment

- [x] Add `_generation: int` field to `SpreadingActivation` class (init=0)
- [x] Add `_visited_gen: dict[tuple[str, str], int]` replacing `visited: set`
- [x] Replace `if visit_key in visited` → `if _visited_gen.get(visit_key, -1) == self._generation`
- [x] Replace `visited.add(visit_key)` → `_visited_gen[visit_key] = self._generation`
- [x] Increment `self._generation` at start of each `activate()` call
- [x] Add benchmark test: 10K neurons, compare old vs new approach
- [x] Unit tests: verify no revisit within same generation, allow revisit across generations

**Why it works:** The dict persists across calls. Old entries with stale generation numbers are naturally "unvisited" without clearing. For repeated searches on same graph (common in consolidation, health checks), this avoids re-allocating a large set.

**Risk:** Low. Dict grows unbounded if neuron IDs are ephemeral — add periodic trim (every 100 generations, remove entries with gen < current - 50).

---

### 1.2 — Tribunal Confidence Scoring for Connection Explainer
**File:** `src/neural_memory/engine/connection_explainer.py`
**Current:** Reports `total_hops` and `avg_weight` separately
**Target:** Single `confidence: float` in [0,1] using exponential decay

- [x] Add `compute_path_confidence(hops: int, avg_weight: float) -> float` function
- [x] Formula: `confidence = exp(-DECAY * hops) * avg_weight` where DECAY=0.4
- [x] Add `confidence` field to explainer result dict
- [x] Add `strength` label: ≥0.7 "strong", ≥0.4 "moderate", ≥0.2 "weak", <0.2 "tenuous"
- [x] Update `_build_markdown()` to include confidence + strength label
- [x] Unit tests: 1-hop high-weight = ~0.67*w, 5-hop = ~0.13*w, 10-hop = ~0.018*w
- [x] Integration test: end-to-end `nmem_explain` returns confidence field

**Why it works:** Exponential decay matches how semantic relevance diminishes with graph distance. Combined with synapse weight gives a single interpretable score. HyperspaceDB uses identical formula for their Tribunal hallucination detector.

**Risk:** None. Additive field, doesn't change existing behavior.

---

### 1.3 — Wasserstein-1 Distance for Drift Detection
**File:** `src/neural_memory/engine/drift_detection.py`
**Current:** Jaccard similarity on tag co-occurrence (binary overlap)
**Target:** Add W1 distribution distance for activation-level drift

- [x] Add `wasserstein_1(dist_a: list[float], dist_b: list[float]) -> float` function
  - L1-normalize both distributions
  - Compute CDF for each: `cdf[i] = sum(dist[:i+1])`
  - Return `sum(abs(cdf_a[i] - cdf_b[i]) for i in range(n))`
- [x] Add `detect_activation_drift(period_a_neurons, period_b_neurons) -> DriftReport`
  - Group neurons by type, compute activation distribution per type
  - W1 between period A and period B distributions
  - Flag types with W1 > 0.3 as "drifting"
- [x] Integrate into `nmem_drift` handler as additional metric alongside Jaccard
- [x] Unit tests: identical distributions → W1=0, opposite → W1=max, gradual shift → proportional
- [x] Test with real brain data: compare W1 vs Jaccard sensitivity

**Why it works:** Jaccard is binary (tag present/absent). W1 captures how much activation mass has shifted between time periods — detects subtle drift that Jaccard misses (e.g., a topic's importance declining gradually).

**Risk:** Low. Needs activation history — may need to query neurons with `accessed_at` ranges. If brain has no activation data, gracefully fall back to Jaccard-only.

---

### 1.4 — Fuzzy Compositional Recall
**Files:** `src/neural_memory/engine/retrieval.py`, new `src/neural_memory/engine/fuzzy_query.py`
**Current:** Single query string → embedding → top-K
**Target:** AST-based compositional queries with soft boolean logic

- [x] Create `fuzzy_query.py` with query AST:
  ```python
  @dataclass(frozen=True)
  class VectorNode:
      query: str

  @dataclass(frozen=True)
  class AndNode:
      left: FuzzyNode
      right: FuzzyNode
      norm: TNorm = TNorm.PRODUCT  # min, product, lukasiewicz

  @dataclass(frozen=True)
  class OrNode:
      left: FuzzyNode
      right: FuzzyNode

  @dataclass(frozen=True)
  class NotNode:
      child: FuzzyNode

  FuzzyNode = VectorNode | AndNode | OrNode | NotNode
  ```
- [x] Add `parse_fuzzy_query(query: str) -> FuzzyNode` parser
  - Syntax: `"topic1 AND topic2"`, `"topic1 OR topic2"`, `"topic1 NOT topic2"`
  - Default (no operators) → single VectorNode (backward compat)
- [x] Add `evaluate_fuzzy(node: FuzzyNode, neuron_scores: dict) -> dict[str, float]`
  - VectorNode: run recall, convert distances to membership via `exp(-distance)`
  - AndNode: `product(a, b)` (default), `min(a, b)`, or `max(a+b-1, 0)`
  - OrNode: `a + b - a*b` (probabilistic OR)
  - NotNode: `1 - a`
- [x] Integrate into `nmem_recall` handler: detect AND/OR/NOT in query → fuzzy path
- [x] Unit tests: AND reduces scores, OR increases, NOT inverts, nested expressions
- [x] Integration test: `nmem_recall "project decisions AND error patterns"` returns intersection

**Why it works:** Users already mentally compose queries like "decisions about auth" — fuzzy logic formalizes this. Each sub-query runs independently, then T-norms combine membership scores. Product norm is most natural: a memory must score high on BOTH sub-queries.

**Risk:** Medium. Parser must handle edge cases (AND/OR in content, capitalization). Mitigate by requiring uppercase operators and at least 2 chars surrounding.

---

## Acceptance Criteria
- [x] All 4 features have unit + integration tests
- [x] Zero regressions on existing test suite (4722 tests)
- [x] No new dependencies added
- [x] Each feature is opt-in or backward-compatible
- [x] mypy passes with 0 errors
- [x] Performance benchmark for generation visited shows improvement on 5K+ neuron brain

## Files Touched
- `src/neural_memory/engine/activation.py` — modify (generation tracking)
- `src/neural_memory/engine/connection_explainer.py` — modify (confidence score)
- `src/neural_memory/engine/drift_detection.py` — modify (W1 distance)
- `src/neural_memory/engine/fuzzy_query.py` — new (fuzzy AST + evaluator)
- `src/neural_memory/engine/retrieval.py` — modify (fuzzy query integration)
- `src/neural_memory/mcp/tool_handlers.py` or relevant handler — modify (recall fuzzy path)
- `tests/unit/test_activation.py` — new tests
- `tests/unit/test_connection_explainer.py` — new tests
- `tests/unit/test_drift_detection.py` — new tests
- `tests/unit/test_fuzzy_query.py` — new file

## Dependencies
- No external dependencies
- Each task is independent of the others
- Requires Phase 0 (current codebase) only

# Phase 2: Advanced Retrieval (Tier 2)

## Goal
3 mathematically sophisticated upgrades that require design work and optional numpy dependency. Improve brain health diagnostics, memory compression quality, and predictive capabilities.

## Prerequisites
- Phase 1 complete (especially 1.1 generation visited — Gromov sampling reuses BFS)
- `numpy` available as optional dependency (already in `pyproject.toml` optional extras)

## Tasks

### 2.1 — Gromov Delta-Hyperbolicity as Brain Health Metric
**Files:** new `src/neural_memory/engine/gromov.py`, modify health handler
**Current:** Health uses heuristic thresholds (connectivity, orphan ratio, stale fibers)
**Target:** Single mathematical metric [0,1] measuring how tree-like the brain structure is

- [ ] Create `gromov.py` with:
  ```python
  async def estimate_gromov_delta(
      storage: StorageBackend,
      sample_size: int = 200,
      seed: int | None = None
  ) -> GromovResult:
      """Sample random 4-tuples, compute delta-hyperbolicity."""
  ```
- [ ] Implementation:
  - Sample `sample_size` random neurons (reservoir sampling)
  - For each 4-tuple (combinatorial sampling, cap at 5000 tuples):
    - Compute 6 pairwise shortest-path distances (BFS, use generation visited)
    - Compute 3 sums: S1=d(a,b)+d(c,d), S2=d(a,c)+d(b,d), S3=d(a,d)+d(b,c)
    - Sort S1≤S2≤S3, delta = (S3 - S2) / 2
  - Return max delta across all 4-tuples
- [ ] Define `GromovResult` frozen dataclass:
  - `delta: float` — raw delta value
  - `normalized_delta: float` — delta / max_possible (diameter)
  - `structure_quality: str` — "tree-like" (δ<0.1), "hierarchical" (δ<0.3), "mixed" (δ<0.5), "flat" (δ≥0.5)
  - `sample_count: int`, `tuple_count: int`
- [ ] Integrate into `nmem_health` as optional `--deep` flag (expensive: O(n⁴) worst case)
- [ ] Cache result with TTL (1 hour) — brain structure doesn't change that fast
- [ ] Add `gromov_delta` field to HealthPulse dataclass
- [ ] Unit tests: star graph → δ≈0, cycle graph → δ>0, complete graph → high δ
- [ ] Integration test: run on real brain, verify result is reasonable

**Interpretation for users:**
- Low δ (tree-like): memories form clean hierarchies → healthy, good recall
- High δ (flat): memories are scattered without structure → needs consolidation
- Trend over time: δ increasing = knowledge fragmentation, δ decreasing = consolidation working

**Risk:** Performance. BFS shortest-path for all 6 pairs × 5000 tuples is expensive. Mitigate: cap sample_size, use cached distances, run async background. For a 5K neuron brain, ~2-5 seconds with caching.

---

### 2.2 — Anisotropic Fidelity Compression
**File:** `src/neural_memory/engine/fidelity.py`
**Current:** Essence extraction is extractive (pick best sentence). GHOST = metadata only.
**Target:** Direction-preserving compression that keeps semantic meaning even at low fidelity

- [ ] Add `compute_content_direction(content: str, embedding_provider) -> list[float] | None`
  - Get embedding vector for full content
  - This is the "direction" — the semantic meaning axis
- [ ] Add `anisotropic_compress(content: str, target_level: FidelityLevel, direction: list[float]) -> str`
  - Split content into sentences
  - Score each sentence by **angular similarity** to direction vector (not just entity density)
  - Orthogonal sentences (tangential info) are dropped first
  - Parallel sentences (core meaning) are preserved
  - SUMMARY: keep sentences with cosine > 0.5 to direction
  - ESSENCE: keep single sentence with highest cosine to direction
  - GHOST: keep only the direction vector hash (for future re-expansion)
- [ ] Integrate into `render_at_fidelity()` as enhanced path (when embeddings available)
- [ ] Fallback to current extractive method when no embedding provider
- [ ] Store `direction_hash` in fiber metadata for GHOST-level reconstruction hints
- [ ] Unit tests: verify high-cosine sentences survive compression, low-cosine dropped
- [ ] A/B test: compare recall quality with anisotropic vs extractive essence

**Why it works:** Current extractive essence picks by surface features (entities, position). Anisotropic compression picks by semantic alignment — a sentence about implementation details gets dropped before a sentence about the core decision, even if both contain entities.

**Risk:** Medium. Requires embedding provider to be configured. Adds latency (1 embedding call per compression). Mitigate: batch compressions, cache direction vectors, async background.

---

### 2.3 — Koopman Trajectory Prediction
**Files:** new `src/neural_memory/engine/koopman.py`, modify cognitive_handler.py
**Current:** Predictions are user-created hypotheses with manual verify
**Target:** Auto-predictions from neuron activation time series

- [ ] Create `koopman.py` with:
  ```python
  def koopman_extrapolate(
      trajectory: list[list[float]],  # T time steps × N features
      steps_ahead: int = 3
  ) -> list[list[float]]:
      """Linearize nonlinear dynamics via DMD approximation."""
  ```
- [ ] Implementation (Dynamic Mode Decomposition — simplified Koopman):
  - Build data matrices X (t=0..T-1) and Y (t=1..T)
  - SVD of X: `U, S, Vt = svd(X, full_matrices=False)`
  - Truncate to rank r (energy threshold 95%)
  - Koopman approximation: `A = U.T @ Y @ Vt.T @ diag(1/S)`
  - Extrapolate: `x_{t+k} = A^k @ x_t`
- [ ] Add `predict_activation_trajectory(brain_id, neuron_ids, steps) -> TrajPrediction`
  - Query activation history (last 10-20 data points per neuron)
  - Build trajectory matrix
  - Extrapolate and return predicted activations
  - Flag neurons predicted to spike (activation > 2× current)
- [ ] Add `nmem_predict action="auto"` subcommand to cognitive handler
  - Runs Koopman on top-50 most active neurons
  - Creates auto-predictions for neurons predicted to spike
  - Tags with `source: "koopman"` to distinguish from user predictions
- [ ] Add Lyapunov stability metric: if max eigenvalue of A > 1.0, trajectory is diverging
  - Diverging = belief instability, needs consolidation
  - Converging = stable knowledge state
- [ ] Unit tests: linear trajectory → perfect extrapolation, oscillating → captures frequency
- [ ] Integration test: feed synthetic activation history, verify reasonable predictions

**Why it works:** Koopman operator theory linearizes nonlinear dynamics. For NM, neuron activations follow patterns (daily usage, project focus shifts). DMD captures these patterns and extrapolates. Auto-predictions surface "this topic is becoming important" without user action.

**Risk:** High. Requires sufficient activation history (10+ data points per neuron). Many brains won't have enough data. Mitigate: minimum data threshold, graceful "insufficient data" response. numpy required but already optional dep.

---

## Acceptance Criteria
- [ ] All 3 features have unit + integration tests
- [ ] Zero regressions on existing test suite
- [ ] numpy is optional — features degrade gracefully without it
- [ ] Gromov delta cached, doesn't re-compute on every health check
- [ ] Anisotropic compression produces shorter output than extractive for same content
- [ ] Koopman predictions have minimum 10-datapoint threshold before activating
- [ ] mypy passes with 0 errors

## Files Touched
- `src/neural_memory/engine/gromov.py` — new (delta-hyperbolicity)
- `src/neural_memory/engine/koopman.py` — new (trajectory prediction)
- `src/neural_memory/engine/fidelity.py` — modify (anisotropic compression)
- `src/neural_memory/mcp/maintenance_handler.py` — modify (Gromov in health)
- `src/neural_memory/mcp/cognitive_handler.py` — modify (auto-predict)
- `tests/unit/test_gromov.py` — new
- `tests/unit/test_koopman.py` — new
- `tests/unit/test_fidelity.py` — modify (anisotropic tests)

## Dependencies
- numpy (optional, already in pyproject.toml extras)
- Phase 1.1 (generation visited) for efficient BFS in Gromov sampling
- Embedding provider configured for anisotropic compression (fallback to extractive)

# Phase 3: Infrastructure & Research (Tier 3)

## Goal
3 infrastructure-level upgrades: Merkle delta sync (production-ready), Cone region queries (experimental), and hyperbolic embedding (research spike). Only Merkle sync is expected to ship in near-term.

## Prerequisites
- Phase 1 and 2 complete
- Cloud Sync Hub operational (Cloudflare Workers + D1)

## Tasks

### 3.1 — Merkle Tree Delta Sync
**Files:** new `src/neural_memory/sync/merkle.py`, modify `sync_engine.py`
**Current:** Sync sends full changelist (all unsynced changes since last watermark)
**Target:** XOR-hash Merkle tree identifies divergent buckets, sync only the delta

- [ ] Create `merkle.py` with:
  ```python
  @dataclass(frozen=True)
  class MerkleTree:
      bucket_count: int = 256
      buckets: tuple[int, ...] = ()  # XOR hashes per bucket
      root_hash: int = 0

  def build_merkle(neurons: list[Neuron]) -> MerkleTree:
      """Assign each neuron to bucket via hash(id) % 256, XOR content hashes."""

  def diff_merkle(local: MerkleTree, remote: MerkleTree) -> list[int]:
      """Return bucket indices where XOR hashes differ."""

  def collect_bucket_contents(bucket_idx: int, storage) -> list[SyncChange]:
      """Fetch all entities in a given bucket for transfer."""
  ```
- [ ] Sync protocol update:
  1. `SyncHandshake`: exchange MerkleTree root hashes
  2. If roots match → skip (no changes), save round-trip
  3. If roots differ → exchange bucket-level hashes (256 ints)
  4. Identify differing buckets → fetch only those entities
  5. Apply delta changes as current `process_sync_response()` does
- [ ] Hub-side changes (Cloudflare Worker):
  - Store per-brain MerkleTree in D1
  - `/v1/hub/sync/handshake` → returns root hash
  - `/v1/hub/sync/diff` → accepts local buckets, returns differing bucket indices
  - `/v1/hub/sync/pull` → accepts bucket indices, returns entities in those buckets
- [ ] Backward compatibility: if hub doesn't support Merkle → fall back to full changelist
- [ ] Unit tests: identical trees → no diff, single change → 1 bucket diff, full divergence → all buckets
- [ ] Integration test: simulate 2 devices, make changes on each, sync via Merkle

**Bandwidth analysis:**
- Current: 1000 changes × ~500 bytes = 500KB per sync
- Merkle handshake: 256 × 8 bytes = 2KB
- Average delta: ~10 buckets × ~5 entities = ~25KB
- **~95% bandwidth reduction** for incremental syncs

**Risk:** Medium. Hub-side changes required (Cloudflare Worker update). XOR hash collisions possible but rare with 256 buckets. Mitigate: fall back to full sync if collision suspected (bucket has too many entities).

---

### 3.2 — Cone Region Queries (Experimental)
**Files:** new `src/neural_memory/engine/cone_query.py`, modify retrieval.py
**Current:** Recall returns top-K nearest neighbors by embedding distance
**Target:** Angular-bounded cone queries: "everything within θ degrees of this direction"

- [ ] Create `cone_query.py` with:
  ```python
  @dataclass(frozen=True)
  class ConeRegion:
      center: list[float]    # direction vector (unit norm)
      half_angle: float      # radians, max angular distance
      min_activation: float  # minimum activation threshold

  def query_cone(
      center: list[float],
      half_angle: float,
      embeddings: list[tuple[str, list[float]]],  # (neuron_id, vector)
      min_activation: float = 0.0
  ) -> list[tuple[str, float]]:
      """Return all neurons within angular cone, sorted by cosine similarity."""
  ```
- [ ] Implementation:
  - Normalize center vector
  - For each embedding: `cos_sim = dot(center, emb) / (norm(center) * norm(emb))`
  - Include if `arccos(cos_sim) <= half_angle`
  - Return ALL matches (not top-K) — this is the key difference
- [ ] Add `nmem_recall mode="cone" angle=30` parameter to recall handler
  - Default K-based recall unchanged
  - Cone mode returns variable-length results
  - Useful for: "give me EVERYTHING about this topic" vs "give me the best 10"
- [ ] Unit tests: narrow cone → few results, wide cone → many, zero angle → exact match only
- [ ] Performance test: brute-force scan on 10K embeddings — acceptable if < 100ms

**Why it works:** Top-K is arbitrary — K=10 might miss relevant memories or include noise. Cone queries adapt to the data: dense topic areas return more, sparse areas return less. User controls precision via angle parameter.

**Risk:** Medium. Requires all embeddings loaded for brute-force scan (no index). For large brains (50K+ neurons), needs approximate methods. Mitigate: random sampling + score threshold for large brains, or pre-built angular index (future work).

---

### 3.3 — Hyperbolic Embedding Research Spike (Research Only)
**Goal:** Evaluate feasibility of embedding Neural Memory's knowledge graph into hyperbolic space (Poincaré ball model). NOT for implementation — for design document only.

- [ ] Research: survey hyperbolic embedding models
  - Poincaré GloVe (Facebook Research)
  - HyperE (knowledge graph embeddings)
  - HGCN (Hyperbolic Graph Convolutional Networks)
  - HyperspaceDB's YAR v5 custom model
- [ ] Analyze Neural Memory's graph structure:
  - Run Gromov delta (from Phase 2.1) on real brains
  - If δ < 0.3 → hyperbolic embedding would help
  - If δ > 0.5 → diminishing returns vs Euclidean
- [ ] Prototype: embed 1000 neurons into Poincaré ball using `geoopt` library
  - Compare recall quality: cosine (Euclidean) vs Poincaré distance
  - Measure hierarchy preservation: parent concepts near origin, leaf concepts near boundary
- [ ] Write design document: `.rune/research-hyperbolic-embedding.md`
  - Feasibility assessment
  - Required changes to storage, retrieval, embedding pipeline
  - Migration path for existing brains
  - Performance impact estimates
- [ ] Decision gate: proceed to implementation only if:
  - Gromov delta < 0.3 on average brain
  - Recall quality improves by > 15% on hierarchical queries
  - Migration path is non-destructive (dual embedding possible)

**Risk:** High. May prove infeasible or not worth the complexity. That's why it's research-only. The Gromov delta from Phase 2.1 will provide data to make this decision.

---

## Acceptance Criteria
- [ ] Merkle sync: handshake + diff + pull protocol working end-to-end
- [ ] Merkle sync: backward compatible with non-Merkle hubs
- [ ] Cone query: variable-length results with angular threshold
- [ ] Hyperbolic research: design document with feasibility verdict
- [ ] All production code (3.1, 3.2) has tests
- [ ] mypy passes with 0 errors

## Files Touched
- `src/neural_memory/sync/merkle.py` — new (Merkle tree)
- `src/neural_memory/sync/sync_engine.py` — modify (Merkle protocol)
- `src/neural_memory/sync/protocol.py` — modify (handshake messages)
- `src/neural_memory/engine/cone_query.py` — new (cone regions)
- `src/neural_memory/engine/retrieval.py` — modify (cone mode)
- `src/neural_memory/mcp/sync_handler.py` — modify (Merkle handshake)
- `.rune/research-hyperbolic-embedding.md` — new (research doc)
- `tests/unit/test_merkle.py` — new
- `tests/unit/test_cone_query.py` — new

## Dependencies
- Phase 2.1 (Gromov delta) for hyperbolic research spike
- Cloud Sync Hub codebase for Merkle hub-side changes
- `geoopt` (optional, research only) for hyperbolic prototype

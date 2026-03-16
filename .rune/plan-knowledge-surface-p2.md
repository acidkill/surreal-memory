# Phase 2: Surface Generator (brain.db → .nm)

## Goal
Extract top-N knowledge from brain.db and generate a `KnowledgeSurface` object. This is the "materialized view" — algorithmically selecting the most important neurons, edges, and patterns to populate the surface.

## VISION Checklist
| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 1 | Activation vs Search | ✅ | Uses activation scores to select top neurons |
| 2 | Spreading Activation | ✅ | SA scores drive node selection for surface |
| 3 | No-Embedding Test | ✅ | Selection uses activation + synapse structure, not embeddings |
| 4 | Detail→Speed | ✅ | Generator precomputes what's important → instant at session start |
| 5 | Source Traceable | ✅ | Each GraphEdge carries source_neuron_id |
| 6 | Brain Test | ✅ | Consolidation creates summary representations — same as human sleep |
| 7 | Memory Lifecycle | ✅ | Generator runs during Consolidate phase |

## Algorithm

### Step 1: Select Top Neurons
```python
# Get neurons ranked by composite score:
# score = activation * 0.4 + recency * 0.3 + connection_count * 0.2 + priority * 0.1
top_neurons = await storage.get_top_neurons(limit=50, min_activation=0.1)
```

### Step 2: Extract GRAPH Edges
```python
# For each top neuron pair, find connecting synapses
# Map synapse types to edge labels:
#   INVOLVES → "involves", CAUSED_BY → "caused_by", RELATED_TO → "related_to"
#   TEMPORAL → "then", DECIDED → "chose_over"
# Include metadata: reason (from synapse content), timestamp
```

### Step 3: Build CLUSTERS
```python
# Group top neurons by entity co-occurrence:
# 1. Get all ENTITY/CONCEPT neurons from top set
# 2. Find which fibers contain 2+ of these entities
# 3. Entities sharing 2+ fibers → same cluster
# 4. Name cluster by most-connected entity or fiber topic tag
```

### Step 4: Extract SIGNALS
```python
# High-priority recent memories → SIGNALS
# Rules:
#   priority >= 8 + age < 7 days → URGENT (!)
#   priority >= 6 + age < 14 days → WATCHING (~)
#   type == "todo" + not completed → UNCERTAIN (?)
```

### Step 5: Compute DEPTH MAP
```python
# For each entity in GRAPH/CLUSTERS:
# - Count synapses, fibers, activation spread
# - If entity has 5+ synapses in surface → SUFFICIENT
# - If entity has 2-4 synapses → NEEDS_DETAIL
# - If entity has <2 synapses but high activation → NEEDS_DEEP
```

### Step 6: Token Budget Trim
```python
# Serialize → check token count
# If over budget:
#   1. Remove lowest-priority SIGNALS
#   2. Remove smallest CLUSTERS
#   3. Remove lowest-activation GRAPH edges
#   4. Downgrade DEPTH entries (remove SUFFICIENT ones — implicit)
# Repeat until within budget
```

## Tasks
- [x] Create `src/neural_memory/surface/generator.py` — `SurfaceGenerator` class
- [x] Implement `_select_top_neurons()` — composite scoring
- [x] Implement `_extract_graph_edges()` — synapse → GraphEdge mapping
- [x] Implement `_build_clusters()` — co-occurrence grouping
- [x] Implement `_extract_signals()` — priority + recency filter
- [x] Implement `_compute_depth_map()` — coverage analysis
- [x] Add `get_top_neurons()` to storage (if not exists) — uses existing find_neurons + NeuronState
- [x] Integrate token budget trimming from Phase 1
- [x] Tests: `tests/unit/test_surface_generator.py`
  - [x] Generator produces valid KnowledgeSurface from mock storage
  - [x] Top neuron selection respects composite scoring
  - [x] Clusters group co-occurring entities correctly
  - [x] Signals filter by priority + recency
  - [x] Depth map assigns correct levels
  - [x] Token budget enforced after generation
  - [x] Empty brain → empty but valid surface
  - [x] Brain with 1 neuron → minimal valid surface

## Acceptance Criteria
- [x] `generate()` produces valid `KnowledgeSurface` from any brain state
- [x] Top neurons selected by composite score (activation + recency + connections + priority)
- [x] GRAPH edges traced from real synapses with back-references
- [x] CLUSTERS non-overlapping, named meaningfully
- [x] SIGNALS reflect current high-priority items
- [x] DEPTH MAP accurate — SUFFICIENT items truly have full context in surface
- [x] Token budget always respected
- [x] Empty brain handled gracefully (no crash, valid empty surface)

## Files Touched
- `src/neural_memory/surface/generator.py` — new
- `src/neural_memory/storage/sqlite_neuron_mixin.py` — modify (add `get_top_neurons` if needed)
- `tests/unit/test_surface_generator.py` — new

## Dependencies
- Phase 1 (models + serializer) must be complete
- Uses existing storage API (find_neurons, get_synapses, etc.)

## Risks
- Composite scoring weights need tuning — start with equal weights, adjust based on testing
- Cluster naming: without LLM, use most-frequent entity as cluster name
- Large brains (10k+ neurons): top-50 selection must be fast — use SQL ORDER BY, not Python sort

# Phase B4: IDF-Weighted Keywords — ✅ Done

## Goal
Replace position-only keyword weighting with IDF-adjusted scores. Common keywords ("code", "function") get low weights; rare domain-specific terms ("PostgreSQL", "Hebbian") get high weights.

## Implementation

### Schema Migration v27→v28
New table `keyword_document_frequency`:
- `brain_id TEXT`, `keyword TEXT` (PK: brain_id + keyword)
- `fiber_count INTEGER`, `last_updated TEXT`

### Storage Methods (base.py, sqlite_fibers.py, memory_store.py)
- `get_total_fiber_count()` — total fibers for current brain
- `get_keyword_df_batch(keywords)` — batch DF lookup
- `increment_keyword_df(keywords)` — UPSERT fiber_count +1

### CreateSynapsesStep IDF Integration
After extracting keywords, before creating concept synapses:
1. Query total fiber count — skip IDF if < 5 (cold start)
2. Batch lookup DF for all keywords
3. Compute IDF: `log((N+1)/(1+df)) / log(N+1)`, floor at 0.2
4. Adjust weight: `final = position_weight * idf_factor`
5. After synapse creation, increment DF for all keywords

### Key Design Decisions
- **Cold start guard**: IDF skipped when < 5 fibers (not enough corpus)
- **IDF floor at 0.2**: Even ubiquitous keywords get 20% of position weight (never zero)
- **Robust guards**: `isinstance` checks + try/except around storage calls (AsyncMock compat)
- **No retroactive reweight**: Existing synapses keep their weights; IDF applies to new encodes only

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | ✅ IDF improves associative connection quality |
| 2 | Spreading Activation | ✅ Better synapse weights = better SA signal propagation |
| 3 | No-Embedding Test | ✅ Pure TF-IDF math, zero embeddings |
| 4 | Detail→Speed | ✅ Rare terms create stronger connections = faster recall |
| 5 | Source Traceable | ✅ DF table tracks keyword corpus stats |
| 6 | Brain Test | ✅ Analogous to frequency effects in human memory (rare = memorable) |
| 7 | Memory Lifecycle | ✅ DF naturally accumulates over time as brain grows |

## Tests (7 tests in test_idf_keywords.py)
- ✅ Cold start uses position weights (< 5 fibers)
- ✅ Rare keyword gets high weight
- ✅ Common keyword gets low weight
- ✅ IDF floor prevents zero weight
- ✅ DF updated after encode
- ✅ No keywords → no DF update
- ✅ IDF math correctness

## Files Changed
- `src/neural_memory/storage/sqlite_schema.py` — v28 migration, keyword_document_frequency table
- `src/neural_memory/storage/base.py` — 3 abstract methods
- `src/neural_memory/storage/sqlite_fibers.py` — SQLite implementations
- `src/neural_memory/storage/memory_store.py` — InMemory implementations
- `src/neural_memory/engine/pipeline_steps.py` — IDF in CreateSynapsesStep
- `tests/unit/test_idf_keywords.py` — new (7 tests)
- `tests/unit/test_baby_mi_features.py` — schema version 27→28
- `tests/unit/test_cascading_retrieval.py` — schema version 27→28
- `tests/unit/test_source_registry.py` — schema version 27→28

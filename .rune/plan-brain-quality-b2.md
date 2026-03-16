# Phase B2: Retrieval-Time Hebbian Learning — ✅ Done

## Goal
Brain learns from its own recall patterns — neurons retrieved together become linked.

## Outcome
**Already implemented.** Evaluation revealed existing infrastructure covers 100% of planned work:

1. **Co-activation recording** (`record_co_activation()` in retrieval.py) — already tracks co-activated neuron pairs
2. **`co_activation_events` table** — persists co-activation counts with timestamps
3. **Deferred write queue** — batches co-activation writes to avoid hot-path latency
4. **INFER consolidation strategy** — converts co-activation counts ≥3 into CO_OCCURS synapses (window=7d, max=50/run)

### Only change needed
Added `"infer"` to `auto_consolidate_strategies` defaults in `MaintenanceConfig` so INFER runs automatically during health-pulse consolidation (was available but not in default strategy list).

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | ✅ Hebbian = associative learning, not search |
| 2 | Spreading Activation | ✅ New CO_OCCURS synapses feed directly into SA |
| 3 | No-Embedding Test | ✅ Pure graph structure, zero embeddings |
| 4 | Detail→Speed | ✅ More co-activations = more specific graph paths |
| 5 | Source Traceable | ✅ co_activation_events table has timestamps |
| 6 | Brain Test | ✅ "Neurons that fire together wire together" — Hebb's rule |
| 7 | Memory Lifecycle | ✅ INFER respects consolidation cycle (only fires during maintenance) |

## Pitfalls
- **Near-duplicate build**: Almost built new RetrievalLearningConfig + post-recall hook before discovering existing infrastructure. Always evaluate existing code first.

## Files Changed
- `src/neural_memory/unified_config.py` — `auto_consolidate_strategies` default: `("prune", "merge")` → `("prune", "merge", "mature", "infer")`

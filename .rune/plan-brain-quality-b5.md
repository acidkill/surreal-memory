# Phase B5: Fiber-Level Recall Scoring — ✅ Done

## Goal
Score fibers holistically using neuron activation data, not just metadata (salience/recency/conductivity). Prevents single-neuron-match fibers from dominating results.

## Implementation

### Modified `_fiber_score()` in retrieval.py
Changed from purely multiplicative metadata-based scoring to three-factor composite:

```
score = base_quality * activation_relevance * stage_multiplier
```

**Base quality** (unchanged):
- `salience * recency * conductivity`
- Recency: sigmoid decay centered at 72h
- Freshness penalty: opt-in via `freshness_weight`

**Activation relevance** (NEW):
- `coverage`: fraction of fiber's neurons that are activated (0-1)
- `max_act`: strongest activation in this fiber
- `mean_act`: average activation across activated neurons
- Signal: `max_act * 0.5 + coverage * 0.3 + mean_act * 0.2`
- Floor at 0.05 (non-matching fibers get minimal score, not zero)

**Stage multiplier** (NEW):
- Semantic fibers: 1.1x (consolidated, more reliable)
- All other stages: 1.0x
- Uses `getattr(fiber, "stage", None)` since stage is a DB column not on Fiber dataclass

### Key Design Decisions
- **Multiplicative, not additive**: base_quality × activation × stage preserves backward compat while adding relevance signal
- **Activation floor at 0.05**: Non-matching fibers still appear (may be relevant via metadata) but are strongly deprioritized
- **No config changes**: Scoring weights are hardcoded (0.5/0.3/0.2 blend) — can be made configurable later if needed
- **`getattr` for stage**: Fiber dataclass doesn't include stage field, only DB column has it

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | ✅ Coverage metric is pure activation-based relevance |
| 2 | Spreading Activation | ✅ SA produces activations → fiber scoring consumes them |
| 3 | No-Embedding Test | ✅ Pure neuron activation math |
| 4 | Detail→Speed | ✅ More specific query = more neurons activated = higher coverage |
| 5 | Source Traceable | ✅ Score decomposition visible in components |
| 6 | Brain Test | ✅ Human memory: coherent memories recalled better than fragments |
| 7 | Memory Lifecycle | ✅ Stage bonus rewards consolidated memories |

## Tests (8 tests in test_fiber_scoring.py)
- ✅ High coverage fiber ranks higher
- ✅ Strong activation ranks higher
- ✅ Semantic stage bonus (1.1x)
- ✅ No activated neurons → minimal score
- ✅ Recency affects score
- ✅ Salience affects score
- ✅ Conductivity multiplier
- ✅ Coverage formula correctness

## Files Changed
- `src/neural_memory/engine/retrieval.py` — activation-aware `_fiber_score()`
- `tests/unit/test_fiber_scoring.py` — new (8 tests)

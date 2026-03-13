# Feature: Cascading Retrieval with Fiber Summary Tier

## Overview
Add fiber-level retrieval as first-pass tier + sufficiency gates between tiers to enable early termination. Combines #61 (sufficiency gate) and #62 (fiber summary tier).

## Phases
| # | Name | Status | Summary |
|---|------|--------|---------|
| 1 | Fiber Summary Search | ⬚ Pending | FTS5 on fibers.summary, `search_fiber_summaries()` in storage |
| 2 | Sufficiency Gate | ⬚ Pending | Confidence-based early exit between retrieval tiers |
| 3 | Pipeline Integration | ⬚ Pending | Wire fiber tier into `ReflexPipeline.query()` as step 2.8 |
| 4 | Auto-generate Summaries | ⬚ Pending | Extractive summary during `mature` consolidation |

## Key Decisions
- Zero-LLM: use existing `_compute_confidence()` as sufficiency gate — no LLM needed
- Fiber search uses FTS5 (already have pattern from neuron FTS)
- Default `sufficiency_threshold = 0.7` in BrainConfig — backward compatible
- Fiber tier fires before neuron anchor search (step 2.8, before step 3)
- If fiber results have confidence >= threshold AND enough tokens → skip neuron search

## Architecture
```
Query → Parse → [Fiber Summary Search] → Sufficiency Gate
                                             ├─ ENOUGH → Return fiber context (fast path)
                                             └─ NOT ENOUGH → Continue to neuron pipeline (current)
```

## Files to Touch
- `src/neural_memory/storage/sqlite_fibers.py` — `search_fiber_summaries()` with FTS5
- `src/neural_memory/storage/sqlite_schema.py` — FTS5 index on fibers.summary (schema v27)
- `src/neural_memory/engine/retrieval.py` — fiber tier step 2.8 + gate logic
- `src/neural_memory/core/brain_config.py` — `sufficiency_threshold` field
- `src/neural_memory/engine/consolidation.py` — auto-generate summaries in `mature`
- Tests: fiber search, gate behavior, pipeline integration

## Risk
- FTS5 on fibers.summary may return low results for brains with sparse summaries
- Mitigation: fiber tier is opt-in via threshold; default preserves current behavior

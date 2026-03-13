# Feature: Cascading Retrieval with Fiber Summary Tier

## Overview
Add fiber-level retrieval as first-pass tier + sufficiency gates between tiers to enable early termination. Combines #61 (sufficiency gate) and #62 (fiber summary tier).

## Phases
| # | Name | Status | Summary |
|---|------|--------|---------|
| 1 | Fiber Summary Search | ✅ Done | FTS5 on fibers.summary, `search_fiber_summaries()` in storage |
| 2 | Sufficiency Gate | ✅ Done | Confidence-based early exit between retrieval tiers |
| 3 | Pipeline Integration | ✅ Done | Wire fiber tier into `ReflexPipeline.query()` as step 2.8 |
| 4 | Auto-generate Summaries | ⬚ Deferred | Extractive summary during `mature` consolidation (not needed for MVP — DocTrainer already creates summaries) |

## Key Decisions
- Zero-LLM: custom confidence = match_ratio * 0.6 + token_ratio * 0.4
- Fiber search uses FTS5 (mirrored neuron FTS pattern with porter tokenizer)
- Default `sufficiency_threshold = 0.7` in BrainConfig — backward compatible
- Fiber tier fires before neuron anchor search (step 2.8, before step 3)
- If fiber results have confidence >= threshold → skip neuron search
- LIKE fallback if FTS5 unavailable
- Skipped for DepthLevel.INSTANT (already minimal)

## Files Changed
- `src/neural_memory/storage/sqlite_fibers.py` — `search_fiber_summaries()` + `_build_fts_query()`
- `src/neural_memory/storage/sqlite_schema.py` — FTS5 index on fibers.summary (schema v27), triggers
- `src/neural_memory/storage/sqlite_store.py` — `ensure_fiber_fts_tables()` on init
- `src/neural_memory/storage/base.py` — `search_fiber_summaries()` default impl
- `src/neural_memory/engine/retrieval.py` — `_try_fiber_summary_tier()` at step 2.8
- `src/neural_memory/core/brain.py` — `fiber_summary_tier_enabled`, `sufficiency_threshold`
- Tests: 17 tests in `test_cascading_retrieval.py`

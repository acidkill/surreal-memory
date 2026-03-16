# Phase 5: Diminishing Returns Gate

## Goal

Stop spreading activation early when new hops add no new signal. Save compute, reduce noise in results.

## Status: DONE

Completed 2026-03-17. All tasks implemented, 25 tests passing, full suite green (4140 passed).

## Tasks

- [x] 5.1: Per-hop signal tracking — `ActivationTrace` dataclass with `new_neurons_per_hop`, `activation_gain_per_hop`
- [x] 5.2: Diminishing returns detector — `should_stop_spreading()` with absolute + relative criteria
- [x] 5.3: Wire into all 3 activation engines (BFS, PPR, Reflex)
- [x] 5.4: Update all callers (retrieval.py, dream.py, tool_handlers.py, test files)
- [x] 5.5: BrainConfig — 4 new fields (enabled, threshold, min_neurons, grace_hops)
- [x] 5.6: Tests — 25 new tests in `test_diminishing_returns.py`

## Files Changed

- `src/neural_memory/engine/activation.py` — ActivationTrace, should_stop_spreading(), per-hop tracking in BFS
- `src/neural_memory/engine/ppr_activation.py` — per-iteration novelty tracking, early stop
- `src/neural_memory/engine/reflex_activation.py` — trace recording in pathway conduction
- `src/neural_memory/engine/dream.py` — unpack (results, trace)
- `src/neural_memory/mcp/tool_handlers.py` — unpack (activations, _trace)
- `src/neural_memory/core/brain.py` — 4 diminishing_returns config fields
- `tests/unit/test_diminishing_returns.py` — 25 new tests
- `tests/unit/test_activation.py` — 11 unpacking fixes
- `tests/unit/test_ppr_activation.py` — 7 unpacking fixes
- `tests/unit/test_related_memories.py` — 4 mock return value fixes
- `tests/integration/test_query_flow.py` — 2 unpacking fixes

## Deferred

- 5.4 (Calibration integration): recording `actual_hops_used` in calibration storage — deferred to Brain Quality Track C

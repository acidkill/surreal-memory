# Phase 1: Auto-Consolidation Loop

## Goal
Brain tự "ngủ" sau mỗi session — run MATURE → INFER → ENRICH automatically. Eliminates the #1 penalty (consolidation ratio at 0.6%) without user action.

## Motivation
- 99% fibers stuck at EPISODIC stage — no knowledge distillation happening
- INFER/ENRICH/DREAM strategies never fire because nobody runs consolidate
- Estimated grade gain: +11.9 pts (consolidation) + ~3 pts (activation) = D→C path

## Design

### Trigger Points
1. **Encode count trigger**: After every 20 encodes, queue background consolidation
2. **Session end trigger**: Stop hook fires consolidation before session closes
3. **Idle trigger**: If MCP server idle >5 min with pending encodes, auto-run

### Strategy Selection
- **Light consolidation** (on encode trigger): MATURE + INFER only (~2s)
- **Full consolidation** (on session end): MATURE → INFER → ENRICH → PRUNE (~5-10s)
- **Never block**: Run async, don't delay encode/recall responses

### State Tracking
- `_encode_count_since_consolidation: int` in MemoryEngine
- `_last_consolidation_at: datetime` in MemoryEngine
- Config: `auto_consolidation.enabled` (default: true), `auto_consolidation.encode_threshold` (default: 20)

## Tasks
- [x] Discovered existing `MaintenanceHandler` with op counter + health pulse + auto-consolidation
- [x] Root cause: `_evaluate_thresholds()` never recommends `mature`/`infer` → 99% episodic
- [x] Added `consolidation_ratio_threshold` to `MaintenanceConfig` (default: 0.1)
- [x] Added `get_fiber_stage_counts()` to storage (base, sqlite, in-memory)
- [x] Added consolidation ratio computation to `_health_pulse()`
- [x] Added consolidation ratio check to `_evaluate_thresholds()` → recommends `mature`
- [x] Changed default `auto_consolidate_strategies` from `("prune", "merge")` → `("prune", "merge", "mature")`
- [x] Reduced `consolidate_cooldown_minutes` from 60 → 30
- [x] Added `run_session_end_consolidation()` — MATURE + INFER + ENRICH at shutdown
- [x] Wired session-end consolidation in `server.py` finally block
- [x] Updated BALANCED config preset to match new defaults
- [x] 13 new tests: consolidation ratio threshold, session-end, default strategies
- [x] Full suite: 3821 passed, mypy + ruff clean
- [ ] Benchmark: measure grade change on test brain before/after

## Acceptance Criteria
- [ ] After 20 encodes, consolidation runs automatically in background
- [ ] Session end triggers full consolidation
- [ ] Encode/recall latency NOT affected (non-blocking)
- [ ] Consolidation ratio improves from <1% to >10% after typical session
- [ ] Config allows disabling auto-consolidation

## Files Touched
- `src/neural_memory/engine/memory_engine.py` — modify (add counter + trigger)
- `src/neural_memory/unified_config.py` — modify (add AutoConsolidationConfig)
- `src/neural_memory/mcp/server.py` — modify (wire stop hook)
- `tests/unit/test_auto_consolidation.py` — new

## Risks
- Background consolidation could conflict with concurrent encode — need async lock
- Stop hook has timeout limit — full consolidation must complete within it
- Large brains (50k+ neurons): consolidation may exceed 10s — need timeout guard

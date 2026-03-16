# NM v4.0 — Brain Intelligence

Vision: Brain that learns from its own usage. Adaptive depth, session priming, drift detection.

## Status: ALL PHASES COMPLETE

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Session Intelligence | Done | — | Session state tracking across MCP calls |
| 2 | Adaptive Depth v2 | Done | — | Calibration → depth tuning, session-aware |
| 3 | Predictive Priming | Done | — | Pre-warm memories from session context |
| 4 | Semantic Drift Detection | Done | — | Tag co-occurrence, cluster merge suggestions |
| 5 | Diminishing Returns Gate | Done | plan-v4-phase5.md | Stop traversal when new hops add no signal |

## Key Decisions

- Zero new dependencies, zero LLM dependency
- Each phase ships independently as minor version
- Phase 5: ActivationTrace dataclass + should_stop_spreading() across all 3 engines

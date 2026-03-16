# Phase A4: Background Memory Processing — Done

## Goal
Auto-importance scoring + reflection engine. Brain learns quality signals without user action.

## Implementation

### A4-B: Auto-Importance Scoring (`engine/importance.py`)
When `nmem_remember` is called without explicit priority, `auto_importance_score()` computes a heuristic score:

**Base**: 5 (normal)

**Type bonus**:
- +3: preference, instruction
- +2: error, decision
- +1: insight, workflow, hypothesis, prediction
- -1: context, todo

**Content signals** (+1 each, max):
- Causal language: "because", "caused by", "chose X over Y", "due to", "root cause"
- Comparative language: "faster than", "replaced X with Y", "3x faster"
- Entity richness: 2+ capitalized entities detected

**Penalty**: -1 for short content (<20 chars)

**Bounds**: clamped to [1, 10]

### A4-C: Reflection Engine (`engine/reflection.py`)
Two components:

**ReflectionEngine** — accumulates importance from `nmem_remember` calls. When threshold (50.0) reached, signals that pattern detection should run.

**detect_patterns()** — rule-based pattern detection from memory clusters:
- Recurring entity: same entity in 3+ memories → "X is a recurring theme"
- Temporal sequence: 3+ memories with temporal markers (first, then, after that, finally)
- Contradiction: two memories with opposing statements (e.g. "is best" vs "is not suitable")

### Integration
- `tool_handlers.py:390` — auto-score priority when `args.get("priority")` is None
- `tool_handlers.py:575` — accumulate importance after successful encode
- Response includes `"auto_importance": true` and `"priority": N` when auto-scored

### Key Design Decisions
- **Lazy import**: `auto_importance_score` imported inside handler (avoid circular deps)
- **Non-blocking**: ReflectionEngine accumulation wrapped in try/except (non-critical)
- **No LLM**: All scoring and pattern detection is rule-based
- **Backward compat**: Explicit priority still works — auto-scoring only when priority=None
- **Reflection engine is stateful**: lives on ToolHandler instance, accumulates across calls

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | Neutral — importance scoring is pre-storage |
| 2 | Spreading Activation | SA unchanged — scoring feeds into salience |
| 3 | No-Embedding Test | Pure regex + heuristics, zero embeddings |
| 4 | Detail->Speed | Better priority = better recall ranking |
| 5 | Source Traceable | auto_importance flag in response |
| 6 | Brain Test | Human memory: emotional/causal events scored higher |
| 7 | Memory Lifecycle | Priority feeds into consolidation + spaced repetition |

## Tests
- test_importance.py (16 tests): type bonuses, content signals, bounds, stacking
- test_reflection.py (12 tests): pattern detection (entities, temporal, contradictions), engine accumulation/threshold/reset

## Files Changed
- `src/neural_memory/engine/importance.py` — new (auto_importance_score)
- `src/neural_memory/engine/reflection.py` — new (ReflectionEngine, detect_patterns)
- `src/neural_memory/mcp/tool_handlers.py` — modified (auto-score + accumulation)
- `tests/unit/test_importance.py` — new (16 tests)
- `tests/unit/test_reflection.py` — new (12 tests)

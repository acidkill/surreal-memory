# Phase B8: Adaptive Synapse Decay — Done

## Goal
Make synapse decay rate inversely proportional to reinforcement count. Important connections survive long-term, noise fades naturally.

## Implementation

### Modified `time_decay()` in synapse.py
Two changes to the existing sigmoid decay:

**1. Adaptive half-life:**
```python
reinforcement_factor = 1.0 + reinforced_count * 0.5
effective_half_life = base_half_life * reinforcement_factor  # base = 1440h (60d)
spread = effective_half_life / 2.0
```

Examples:
- 0x reinforced: half-life = 60 days (unchanged)
- 2x reinforced: half-life = 120 days
- 5x reinforced: half-life = 210 days
- 10x reinforced: half-life = 360 days

**2. Adaptive floor:**
```python
floor = 0.3 + min(0.5, reinforced_count * 0.05)
```

Examples:
- 0x reinforced: floor = 0.30 (unchanged)
- 4x reinforced: floor = 0.50
- 10x reinforced: floor = 0.80 (nearly undecayable)

### Key Design Decisions
- **Backward compatible**: Unreinforced synapses (count=0) produce identical decay as before
- **Multiplicative on base**: `reinforcement_factor` scales the half-life, not the weight directly
- **Floor cap at 0.8**: Prevents fully immortal synapses (always some minimal decay possible)
- **No config changes**: Parameters hardcoded (0.5 scale, 0.05 per reinforcement) — can be made configurable later
- **Type multipliers still apply**: Dream/inferred decay modifiers in consolidation.py operate on top of this

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | Improves recall quality by preserving important connections |
| 2 | Spreading Activation | Better SA signal: reinforced paths stay strong, noise paths fade |
| 3 | No-Embedding Test | Pure math on reinforced_count, zero embeddings |
| 4 | Detail->Speed | Frequently recalled connections = faster future activation |
| 5 | Source Traceable | reinforced_count is tracked per synapse |
| 6 | Brain Test | Human memory: rehearsed connections become long-term, unused fade |
| 7 | Memory Lifecycle | Reinforcement from Hebbian learning (B2) feeds into decay resistance |

## Tests (11 tests in test_adaptive_decay.py)
- Unreinforced same as before (backward compat)
- Reinforced decays slower than unreinforced
- Heavily reinforced (10x) retains most weight at 60d
- Floor unreinforced = 0.3
- Floor reinforced higher than unreinforced
- Floor capped at 0.8
- Recent synapses barely decay (regardless of reinforcement)
- Reinforcement factor formula: 1 + count * 0.5
- Floor formula: 0.3 + min(0.5, count * 0.05)
- Weight never amplified above original
- Zero weight stays zero

## Files Changed
- `src/neural_memory/core/synapse.py` — adaptive `time_decay()`
- `tests/unit/test_adaptive_decay.py` — new (11 tests)

# Phase B3: Cross-Memory Entity Linking — ✅ Done

## Goal
When encoding a new memory, detect shared entities with existing memories and create direct anchor-to-anchor RELATED_TO synapses. Zero-LLM equivalent of Cognee's knowledge graph construction.

## Implementation

### CrossMemoryLinkStep (pipeline_steps.py, ~100 lines)
Inserted as step 14/15 in encoding pipeline (after SemanticLinkingStep, before BuildFiberStep).

**Flow:**
1. For each entity neuron in the new memory →
2. Query existing INVOLVES synapses to find old anchors linked to same entity →
3. Collect old anchors, skip self-links and frequency-capped entities →
4. Create RELATED_TO synapse from new anchor to old anchor →
5. Weight = `0.3 + 0.1 * (shared_count - 1)`, capped at 0.7

### Constants
| Param | Value | Purpose |
|-------|-------|---------|
| MAX_LINKS_PER_ENTITY | 5 | Prevent hub explosion |
| MAX_LINKS_PER_ENCODE | 15 | Prevent graph bloat |
| ENTITY_FREQUENCY_CAP | 50 | Skip generic entities (>50 anchors) |
| BASE_WEIGHT | 0.3 | Weak initial connection |
| WEIGHT_BONUS_PER_ENTITY | 0.1 | More shared entities = stronger |
| WEIGHT_CAP | 0.7 | Never stronger than explicit connections |

### Key Design Decision
Distinct from existing SemanticLinkingStep:
- **SemanticLinkingStep**: neuron-to-neuron links via exact content match
- **CrossMemoryLinkStep**: anchor-to-anchor links via shared entity neurons

This enables multi-hop reasoning: Memory A ↔ Entity ↔ Memory B becomes Memory A → Memory B directly.

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | ✅ Creates associative paths, not search indexes |
| 2 | Spreading Activation | ✅ RELATED_TO synapses feed directly into SA propagation |
| 3 | No-Embedding Test | ✅ Pure entity overlap, zero embeddings |
| 4 | Detail→Speed | ✅ More specific entity = fewer false links |
| 5 | Source Traceable | ✅ `_cross_memory` + `_shared_entity_count` metadata |
| 6 | Brain Test | ✅ Associative memory — "reminds me of" linking |
| 7 | Memory Lifecycle | ✅ Links created at encode, decay naturally via synapse decay |

## Tests (9 tests in test_cross_memory_link.py)
- ✅ Creates link via shared entity
- ✅ Weight scales with shared entity count
- ✅ Skips self-links
- ✅ Skips common entities (>50 anchors)
- ✅ Respects MAX_LINKS_PER_ENCODE
- ✅ No entities → no links
- ✅ No anchor → no links
- ✅ Handles duplicate synapse errors gracefully
- ✅ Weight capped at WEIGHT_CAP

## Files Changed
- `src/neural_memory/engine/pipeline_steps.py` — added CrossMemoryLinkStep class
- `src/neural_memory/engine/encoder.py` — registered step 14/15 in pipeline
- `tests/unit/test_cross_memory_link.py` — new (9 tests)

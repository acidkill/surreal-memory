# Phase 7: Lazy Entity Promotion

## Goal
Stop creating entity neurons on first mention. Store as lightweight references, promote to full neurons only when entity appears in 2+ memories. Eliminates ~60% of orphan neurons.

## Motivation
- 29% orphan rate (3,953 neurons with zero connections beyond initial fiber)
- Most orphans are entities/concepts mentioned exactly once — never recalled, never linked
- One-off entities dilute the graph: more nodes for activation to traverse, lower signal/noise
- Lazy promotion: only invest in entities that prove their value through repetition

## Design

### Entity Lifecycle
```
1st mention → entity_ref in fiber metadata (lightweight, no neuron)
2nd mention → promote to ENTITY neuron + create synapses
3rd+ mention → boost existing synapse weights
```

### Implementation
```python
# In entity extraction step:
for entity in extracted_entities:
    existing = await storage.find_neurons(content_exact=entity.text, type="entity")
    if existing:
        # Entity already promoted — link to it, boost weight
        await create_synapse(anchor, existing[0], INVOLVES, weight=0.8)
    else:
        # Check pending refs
        ref_count = await storage.count_entity_refs(entity.text)
        if ref_count >= 1:
            # 2nd mention — promote!
            neuron = await storage.add_neuron(Neuron.create(type="entity", content=entity.text))
            await create_synapse(anchor, neuron, INVOLVES, weight=0.75)
            # Retroactively link to previous fiber(s)
            await _link_previous_fibers(entity.text, neuron)
        else:
            # 1st mention — store ref only
            await storage.add_entity_ref(entity.text, fiber_id=context.fiber_id)
```

### Entity Reference Table
```sql
CREATE TABLE IF NOT EXISTS entity_refs (
    entity_text TEXT NOT NULL,
    fiber_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    promoted BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (entity_text, fiber_id)
);
CREATE INDEX idx_entity_refs_text ON entity_refs(entity_text);
```

### Retroactive Linking
When promoting on 2nd mention:
1. Find all fibers with this entity_ref
2. Create INVOLVES synapses from those fibers' anchors to new entity neuron
3. Mark refs as `promoted=TRUE`

### Exceptions (Always Promote)
- Entities from user-provided tags (explicit = always valuable)
- Entities matching existing neurons (already proven)
- Entities with high extraction confidence (>0.9)

## Tasks
- [x] Add `entity_refs` table to schema (migration 28→29)
- [x] Add lazy entity fields to `BrainConfig` (enabled, promotion_threshold=2, prune_days=90)
- [x] Modify entity extraction step: check ref count before creating neuron
- [x] Implement `add_entity_ref()` and `count_entity_refs()` in storage (SQLite mixin + InMemory + SharedStorage)
- [x] Implement retroactive linking on promotion
- [x] Add exceptions for high-confidence (>=0.9) / user-tagged entities
- [x] Cleanup: add entity_ref pruning to PRUNE consolidation (remove refs older than 90 days)
- [x] Tests: 11 tests covering all scenarios
- [ ] Benchmark: measure orphan rate change (deferred to after accumulation)

## Acceptance Criteria
- [ ] 1st-mention entities stored as refs, NOT neurons
- [ ] 2nd-mention entities promoted to full neurons with synapses
- [ ] Previous fibers retroactively linked to promoted entity
- [ ] Orphan rate drops from 29% to <15%
- [ ] High-confidence entities always promoted (exception path)
- [ ] Old unpromoted refs cleaned up by PRUNE strategy

## Files Touched
- `src/neural_memory/storage/sqlite_schema.py` — modify (add entity_refs table)
- `src/neural_memory/storage/sqlite_neuron_mixin.py` — modify (add ref CRUD)
- `src/neural_memory/engine/pipeline_steps.py` — modify (lazy entity logic)
- `src/neural_memory/engine/consolidation.py` — modify (add ref pruning to PRUNE)
- `src/neural_memory/unified_config.py` — modify (add LazyEntityConfig)
- `tests/unit/test_lazy_entity.py` — new

## Dependencies
- Independent of P1-P6
- Benefits from P4 (IDF signal can inform promotion decision)
- P3 (cross-memory linking) works better with fewer, higher-quality entity neurons

## Risks
- Retroactive linking: finding previous fibers requires entity_ref table — new storage cost
- Migration: existing brains already have orphan entities — consider one-time prune migration
- Edge case: entity mentioned in rapid succession (same session) — both fibers may try to promote

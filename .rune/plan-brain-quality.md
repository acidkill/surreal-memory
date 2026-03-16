# Feature: Brain Quality, Proactive Memory & Vertical Intelligence

## Overview
Three goals: (A) Make agents proactively remember/recall without explicit commands, (B) Close accuracy gap with LLM-powered systems without adding LLM calls, (C) Enable vertical use cases (accounting, legal, data visualization). All guided by VISION.md brain-test principles.

## Current State
- Grade D (53.7/100) | 13,751 neurons | 56,966 synapses | 1,573 fibers
- Benchmark: NM 80x faster than Cognee but accuracy gap (0.42 vs 0.63 multi-hop)
- Agents don't proactively remember/recall — must be told explicitly
- Single brain per user — no project isolation
- No domain-specific entity types or structured data neurons

## VISION.md Alignment (MANDATORY per phase)

Every phase MUST pass these checks before implementation:

| # | Check | Question |
|---|-------|----------|
| 1 | Activation vs Search | Does this improve associative recall or just search? |
| 2 | Spreading Activation | Is SA still the retrieval center, not a fallback? |
| 3 | No-Embedding Test | Would this still work without vector embeddings? |
| 4 | Detail→Speed | Does more specific query = faster/cheaper recall? |
| 5 | Source Traceable | Can agent answer "where did this come from?" |
| 6 | Brain Test | Does the human brain do something analogous? |
| 7 | Memory Lifecycle | Does this respect Create→Reinforce→Decay→Consolidate→Forget? |

## Implementation Protocol (MANDATORY)

```
For EVERY phase:
1. Read VISION.md + phase plan file
2. Run VISION checklist (7 questions above) — document answers in phase file
3. Write tests FIRST (TDD: red → green → refactor)
4. Implement with sub-agent review:
   - Code writer (sonnet) implements
   - Code reviewer (sonnet) checks structure, logic, pitfalls
   - If architectural: architect (opus) validates design
5. Document common pitfalls encountered in phase file
6. Run full test suite + mypy + ruff before marking done
7. Re-run benchmark to measure impact
8. Update phase status in this master plan
```

## Phases

### Track A: Proactive Memory (agent behavior)

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| A1 | Smart Instructions | ✅ Done | plan-brain-quality-a1.md | Decision framework in MCP instructions |
| A2 | Layered Brain Architecture | ⬚ Pending | plan-brain-quality-a2.md | Global + Project + Session layers |
| A3 | Auto-Recall Injection | ⬚ Pending | plan-brain-quality-a3.md | MCP auto-injects related memories |
| A4 | Background Memory Processing | ✅ Done | plan-brain-quality-a4.md | Auto-importance scoring + reflection engine (pattern detection + threshold trigger) |

### Track B: Brain Quality (graph/retrieval improvements)

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| B1 | Auto-Consolidation Loop | ✅ Done | plan-brain-quality-b1.md | Consolidation ratio check + mature strategy + session-end consolidation |
| B2 | Retrieval-Time Hebbian | ✅ Done | plan-brain-quality-b2.md | Existing co-activation + INFER strategy already covers this; added infer to defaults |
| B3 | Cross-Memory Entity Linking | ✅ Done | plan-brain-quality-b3.md | CrossMemoryLinkStep links anchors via shared entities (RELATED_TO synapses) |
| B4 | IDF-Weighted Keywords | ✅ Done | plan-brain-quality-b4.md | IDF scoring in CreateSynapsesStep + keyword_document_frequency table (schema v28) |
| B5 | Fiber-Level Recall Scoring | ✅ Done | plan-brain-quality-b5.md | Activation-aware scoring: coverage + max_act + mean_act + stage bonus |
| B6 | Contextual Compression | ✅ Done | plan-brain-quality-b6.md | Age-based recall compression: <7d full, 7-30d summary/3 sentences, 30-90d summary/2 sentences, 90+ summary/1 sentence |
| B7 | Lazy Entity Promotion | ✅ Done | plan-brain-quality-b7.md | Entities need 2+ mentions to become neurons; entity_refs table (schema v29) |
| B8 | Adaptive Synapse Decay | ✅ Done | plan-brain-quality-b8.md | Reinforcement-modulated half-life + adaptive floor in time_decay() |

### Track C: Vertical Intelligence (domain-specific capabilities)

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| C1 | Domain Entity Types | ⬚ Pending | plan-brain-quality-c1.md | Financial_Metric, Regulation, etc. |
| C2 | Structured Data Neurons | ⬚ Pending | plan-brain-quality-c2.md | Tables, key-value, schema-aware encoding |
| C3 | Cross-Encoder Reranking | ⬚ Pending | plan-brain-quality-c3.md | bge-reranker-v2-m3 after spreading activation |
| C4 | Agent Visualization Tool | ⬚ Pending | plan-brain-quality-c4.md | nmem_visualize — chart gen from memory data |

## Execution Order
```
Sprint 1: A1 (instructions) → B1 (auto-consolidation) ✅
Sprint 2: B2 (Hebbian) → B3 (cross-memory link) ✅
Sprint 3: B4 (IDF) → B5 (fiber scoring) ✅
Sprint 4: B6 (compression) → B8 (adaptive decay) ✅
Sprint 5: B7 (lazy entity, 3d) → A4 (background processing, 5d)
Sprint 7: C1 (domain entities, 3d) → C2 (structured data, 5d)
Sprint 8: C3 (reranking, 3d) → C4 (visualization, 5d)
```

## Key Decisions
- Zero LLM calls — all improvements are algorithmic/graph-structural
- Layered brain: Global (~/.neuralmemory) + Project (.neuralmemory/) + Session (ephemeral)
- Layer routing is automatic — agent uses same API
- VISION.md brain-test mandatory for every phase
- Sub-agent review mandatory for every implementation
- Common pitfalls documented per phase to prevent recurrence

## Success Metrics
| Metric | Current | Sprint 2 | Sprint 6 | Sprint 8 |
|--------|:-------:|:--------:|:--------:|:--------:|
| Brain grade | D (53.7) | C (65) | B (80+) | B+ (85+) |
| Multi-hop accuracy | 0.42 | 0.50 | 0.60+ | 0.65+ |
| Semantic accuracy | 0.14 | 0.18 | 0.25+ | 0.30+ |
| Proactive save rate | 0% | 60% | 90% | 90% |
| Proactive recall rate | 0% | 50% | 85% | 85% |
| Domain entity support | 0 | 0 | 0 | 9+ types |

## Dependency Graph
```
Track A (Proactive):
A1 (Instructions) ──→ A2 (Layered Brain) ──→ A3 (Auto-Recall)
                                           ──→ A4 (Background)

Track B (Quality):
B1 (Auto-Consolidation) ──→ B2 (Hebbian) ──→ B3 (Cross-Memory)
                                           ──→ B4 (IDF Keywords)
B5 (Fiber Scoring) ← independent
B6 (Compression) ← benefits from B1
B7 (Lazy Entity) ← benefits from B4
B8 (Adaptive Decay) ← benefits from B1+B2

Track C (Vertical):
C1 (Domain Entities) ──→ C2 (Structured Data)
C3 (Reranking) ← independent, enhances B5
C4 (Visualization) ← benefits from C1+C2
```

## Common Pitfalls Registry
Document here as phases are implemented:

| Phase | Pitfall | How to Avoid |
|-------|---------|--------------|
| A1 | MCP instructions char limit — wall-of-text ignored by agents | Use decision framework tables, not prescriptive paragraphs |
| B1 | HealthPulse new required fields break existing constructors | Always use default values for new dataclass fields |
| B1 | BALANCED preset must match new defaults or test_no_diff fails | Update config_presets.py when changing unified_config defaults |
| B2 | Co-activation + INFER already existed — almost built duplicate | Always evaluate existing infrastructure before adding new logic |
| B3 | SemanticLinkingStep links neurons; CrossMemoryLinkStep links anchors | Different granularity — don't confuse neuron-level vs anchor-level linking |
| B4 | AsyncMock returns mock objects not ints — breaks comparisons | Guard storage calls with isinstance checks or try/except |
| B4 | Schema version assertions in 3+ test files | Grep for `SCHEMA_VERSION ==` when bumping schema |
| B5 | Fiber.stage is a DB column but NOT on Fiber dataclass | Use `getattr(fiber, "stage", None)` not `fiber.stage` |
| B6 | compress_for_recall must handle None created_at | Safe fallback: no timestamp = return full content |
| B8 | Sigmoid threshold assertions sensitive to spread formula | Calculate exact sigmoid values, don't assume "close to 1.0" |
| B7 | Tests using SimpleNamespace for ctx lack new fields | Use `getattr(ctx, "field", default)` in steps that access new context fields |
| B7 | Appending existing entities to ctx.entity_neurons breaks CreateSynapsesStep | Don't re-add existing entities — they already have synapses from prior encoding |
| B7 | entity_refs table may not exist on old schema DBs | Wrap count_entity_refs/mark_promoted in try/except with fallback |

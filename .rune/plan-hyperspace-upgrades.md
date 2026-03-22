# Feature: HyperspaceDB-Inspired Upgrades

## Overview
Borrow proven techniques from HyperspaceDB (hyperbolic vector database, Rust) to improve Neural Memory's retrieval quality, health diagnostics, and sync efficiency. Ideas ranked by effort/impact ratio across 3 tiers.

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Quick Wins | ✅ Done | plan-hyperspace-upgrades-p1.md | Generation visited, Tribunal scoring, W1 drift, Fuzzy recall |
| 2 | Advanced Retrieval | ✅ Done | plan-hyperspace-upgrades-p2.md | Gromov health metric, Anisotropic fidelity, Koopman predict |
| 3 | Infrastructure | ⬚ Pending | plan-hyperspace-upgrades-p3.md | Merkle delta sync (Hub), Cone regions (Private), Hyperbolic embedding (Private) |

## Key Decisions
- Tier 1 items are backward-compatible, zero new dependencies, can ship incrementally
- Tier 2 items need math libraries (numpy for Gromov/Koopman) — optional dependency
- Tier 3 Merkle sync replaces current full-changelist sync in Cloud Sync Hub
- Hyperbolic embedding (Tier 3) is research-only — requires custom embedding model

## Impact Map

| Upgrade | Affects | Benefit |
|---------|---------|---------|
| Generation visited | activation.py | ~30% faster BFS on large brains (skip set clear) |
| Tribunal scoring | connection_explainer.py | Continuous [0,1] path confidence vs raw hop count |
| W1 drift | drift_detection.py | Distribution-aware drift vs pairwise Jaccard |
| Fuzzy recall | retrieval.py, recall handler | Compositional queries (AND/OR/NOT with soft membership) |
| Gromov delta | health handler | Mathematical brain structure quality metric |
| Anisotropic fidelity | fidelity.py | Direction-preserving compression for ESSENCE/GHOST |
| Koopman predict | cognitive_handler.py | Auto-predictions from activation trajectories |
| Merkle sync | sync_engine.py | Delta-only sync = ~90% bandwidth reduction |
| Cone regions | retrieval.py | Angular-bound queries vs top-K nearest |
| Hyperbolic embed | embedding provider | Hierarchical memories compress exponentially |

## Dependencies
- Phase 1: No new dependencies
- Phase 2: `numpy` (already optional dep), no new packages
- Phase 3: `merkle-py` or custom impl, research for hyperbolic embedding

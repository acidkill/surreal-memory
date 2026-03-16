# Phase 1: .nm Format Spec + Parser/Serializer

## Goal
Define the `.nm` file format, data models, and bidirectional parser (read `.nm` → Python objects, write Python objects → `.nm` file). This is the foundation — no brain.db interaction yet.

## VISION Checklist
| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 1 | Activation vs Search | ✅ | Format captures pre-activated knowledge, not search results |
| 2 | Spreading Activation | ✅ | Data models designed to hold SA output |
| 3 | No-Embedding Test | ✅ | Pure text format, zero embedding dependency |
| 4 | Detail→Speed | ✅ | DEPTH MAP enables instant vs queried distinction |
| 5 | Source Traceable | ✅ | Nodes carry `[id]` back-references to brain.db neurons |
| 6 | Brain Test | ✅ | Working memory buffer — limited capacity, high relevance |
| 7 | Memory Lifecycle | ✅ | Surface is snapshot — regenerated during consolidation |

## Format Specification (Approved Design)

### Complete .nm File Example
```
# ═══════════════════════════════════════════════════
# .neuralmemory/surface.nm
# Neural Memory Knowledge Surface v1
# Auto-generated from brain.db | Editable by human
# ═══════════════════════════════════════════════════
---
brain: myproject
updated: 2026-03-16T10:30:00
neurons: 847
synapses: 3201
token_budget: 1200
depth_available: [surface, detail, deep]
---

# ── GRAPH ──────────────────────────────────────────
# Nodes = key concepts. Edges = causal/decisional links.
# Agent: scan this section to understand the knowledge landscape.
# Format: [id] content (type) {p:N} →edge→ [target_id_or_text]

[d1] Chose PostgreSQL over MongoDB (decision) {p:8}
  →caused→ [d2] "Need ACID for payment transactions"
  →led_to→ [f1] "payments module uses SQLAlchemy ORM"
  →rejected→ [x1] "MongoDB lacks multi-doc transactions"

[d2] Auth uses JWT 15min + httpOnly refresh (decision) {p:7}
  →caused→ [f2] "Legal flagged session token storage"
  →led_to→ [f3] "auth middleware rewrite in progress"
  →depends→ [d1] "PostgreSQL stores refresh tokens"

[e1] MagicMock truthy on Python 3.11 (error) {p:7}
  →root_cause→ "auto-created attrs always truthy"
  →fix→ "explicitly mock config.encryption = MagicMock(enabled=False)"
  →affects→ [f4] "test_remember_sensitive_content"

[p1] User prefers Vietnamese chat, English code (preference) {p:9}
[p2] Always ruff + mypy before commit (instruction) {p:8}

# ── CLUSTERS ───────────────────────────────────────
# Auto-detected topic groups from brain.db entity co-occurrence.
# Agent: use clusters to scope recall queries.

@auth: [d2, f2, f3] "Authentication & authorization"
@payments: [d1, f1, x1] "Payment processing & database"
@testing: [e1, f4, p2] "Test infrastructure & CI"
@preferences: [p1, p2] "User preferences & workflow"

# ── SIGNALS ────────────────────────────────────────
# Active alerts, unresolved items, pending decisions.
# Agent: check this FIRST each session for urgent context.

! [f3] auth middleware rewrite — IN PROGRESS (compliance deadline)
! merge freeze after 2026-03-20 (mobile release cut)
? Redis vs Memcached for session store — UNDECIDED

# ── DEPTH MAP ──────────────────────────────────────
# When surface isn't enough, this tells agent WHERE to dig.
# Format: [id] → depth_hint (context)

[d1] → SUFFICIENT (8 synapses, 3 fibers in brain.db)
[d2] → NEEDS_DETAIL (2 fibers, recall "jwt auth decision" for full context)
[f3] → NEEDS_DEEP (12 related neurons, recall "auth middleware rewrite" for implementation details)
[e1] → SUFFICIENT (fix documented above)

# ── META ───────────────────────────────────────────
# Stats for agent to gauge brain richness

coverage: 0.73
staleness: 0.12
last_consolidation: 2026-03-15T22:00:00
top_entities: [PostgreSQL, JWT, Redis, FastAPI, Docker]
```

### Syntax Rules

1. **Header**: `# ═══...` comment block (decorative, skipped by parser)
2. **Frontmatter**: YAML between `---` delimiters (brain, updated, neurons, synapses, token_budget, depth_available)
3. **Sections**: `# ── SECTION_NAME ──...` — 5 sections: GRAPH, CLUSTERS, SIGNALS, DEPTH MAP, META
4. **Comments**: Lines starting with `#` inside sections (agent-facing hints, skipped by parser)
5. **GRAPH nodes**: `[id] content (type) {p:N}` — id is short alphanumeric, type from NeuronType, p = priority 0-10
6. **GRAPH edges**: `  →edge_type→ [target_id] "description"` or `  →edge_type→ "inline text"` (indented under parent node)
7. **CLUSTERS**: `@name: [id1, id2, ...] "description"` — `@` prefix, bracket-wrapped ID refs
8. **SIGNALS**: `!` = urgent, `~` = watching, `?` = uncertain. Optional `[id]` ref. Free-text description with status
9. **DEPTH MAP**: `[id] → LEVEL (context)` where LEVEL ∈ {SUFFICIENT, NEEDS_DETAIL, NEEDS_DEEP}. Parenthetical context = synapse/fiber counts + suggested recall query
10. **META**: `key: value` pairs — coverage (float 0-1), staleness (float 0-1), last_consolidation (ISO), top_entities (list)
11. **Encoding**: UTF-8, no BOM
12. **ID format**: `[letter + digits]` — prefix indicates type: `d` = decision, `f` = fact, `e` = error, `p` = preference, `i` = insight, `w` = workflow, `x` = rejected/negative, `c` = concept

### Cross-Reference System
- IDs (`[d1]`, `[f3]`, etc.) are unique within a surface file
- Same ID can appear in GRAPH (definition), CLUSTERS (membership), SIGNALS (alerts), DEPTH MAP (routing)
- Agent follows ID from SIGNALS → GRAPH to get full causal chain
- Agent follows ID from DEPTH MAP → knows whether to query brain.db

## Data Models

```python
@dataclass(frozen=True)
class SurfaceNode:
    id: str                   # e.g. "d1", "f3", "e1"
    content: str              # Human-readable description
    node_type: str            # "decision", "fact", "error", "preference", etc.
    priority: int             # 0-10
    neuron_id: str | None = None  # Back-reference to brain.db neuron

@dataclass(frozen=True)
class SurfaceEdge:
    edge_type: str            # "caused", "led_to", "rejected", "depends", etc.
    target_id: str | None     # Reference to another node [id], or None if inline
    target_text: str          # Inline description (always present)

@dataclass(frozen=True)
class GraphEntry:
    node: SurfaceNode
    edges: tuple[SurfaceEdge, ...]  # Outgoing edges from this node

@dataclass(frozen=True)
class Cluster:
    name: str                 # Topic name (without @)
    node_ids: tuple[str, ...]  # References to GRAPH node IDs
    description: str          # Human-readable cluster description

class SignalLevel(StrEnum):
    URGENT = "!"              # Needs attention now
    WATCHING = "~"            # Monitoring
    UNCERTAIN = "?"           # Needs clarification

@dataclass(frozen=True)
class Signal:
    level: SignalLevel
    node_id: str | None       # Optional reference to GRAPH node
    text: str                 # Description + status

class DepthLevel(StrEnum):
    SUFFICIENT = "SUFFICIENT"       # Full context in surface
    NEEDS_DETAIL = "NEEDS_DETAIL"   # Recall last few memories
    NEEDS_DEEP = "NEEDS_DEEP"       # Full graph traversal needed

@dataclass(frozen=True)
class DepthHint:
    node_id: str              # Reference to GRAPH node
    level: DepthLevel
    context: str              # Synapse/fiber counts + suggested recall query

@dataclass(frozen=True)
class SurfaceMeta:
    coverage: float           # % of brain.db represented (0.0-1.0)
    staleness: float          # Avg recency decay (0.0-1.0, lower = fresher)
    last_consolidation: str   # ISO timestamp
    top_entities: tuple[str, ...]  # Most connected entities

@dataclass(frozen=True)
class SurfaceFrontmatter:
    brain: str
    updated: str              # ISO timestamp
    neurons: int
    synapses: int
    token_budget: int
    depth_available: tuple[str, ...]  # ["surface", "detail", "deep"]

@dataclass(frozen=True)
class KnowledgeSurface:
    frontmatter: SurfaceFrontmatter
    graph: tuple[GraphEntry, ...]
    clusters: tuple[Cluster, ...]
    signals: tuple[Signal, ...]
    depth_map: tuple[DepthHint, ...]
    meta: SurfaceMeta

    def token_estimate(self) -> int:
        """Rough token count (chars / 4)."""
        from .serializer import serialize
        return len(serialize(self)) // 4

    def get_node(self, node_id: str) -> SurfaceNode | None:
        """Find a node by ID across graph entries."""
        for entry in self.graph:
            if entry.node.id == node_id:
                return entry.node
        return None

    def get_depth_hint(self, node_id: str) -> DepthLevel | None:
        """Get depth routing hint for a node."""
        for hint in self.depth_map:
            if hint.node_id == node_id:
                return hint.level
        return None

    def all_node_ids(self) -> frozenset[str]:
        """All node IDs defined in GRAPH."""
        return frozenset(entry.node.id for entry in self.graph)
```

## Tasks
- [ ] Create `src/neural_memory/surface/__init__.py` — public API exports
- [ ] Create `src/neural_memory/surface/models.py` — frozen dataclasses above (9 models)
- [ ] Create `src/neural_memory/surface/parser.py` — `parse(text: str) → KnowledgeSurface`
  - [ ] Frontmatter parsing (YAML between `---`)
  - [ ] Section splitting (GRAPH, CLUSTERS, SIGNALS, DEPTH MAP, META)
  - [ ] GRAPH node + edge parsing (multi-line per node)
  - [ ] Cluster parsing (`@name: [ids] "desc"`)
  - [ ] Signal parsing (`!/~/? [id] text`)
  - [ ] Depth hint parsing (`[id] → LEVEL (context)`)
  - [ ] META key-value parsing
  - [ ] Lenient mode: skip unknown sections (forward compat)
- [ ] Create `src/neural_memory/surface/serializer.py` — `serialize(surface: KnowledgeSurface) → str`
  - [ ] Decorative header + frontmatter
  - [ ] Section serialization with comment hints
  - [ ] Proper indentation for edges under nodes
- [ ] Create `src/neural_memory/surface/token_budget.py` — `trim_to_budget(surface, budget) → KnowledgeSurface`
  - [ ] Priority-based trimming (lowest priority first)
  - [ ] Remove GRAPH entries → CLUSTERS refs → SIGNALS → DEPTH entries
- [ ] Tests: `tests/unit/test_surface_format.py`
  - [ ] Parse complete valid .nm file → all models populated correctly
  - [ ] Parse frontmatter → SurfaceFrontmatter
  - [ ] Parse GRAPH with multi-edge nodes → GraphEntry with edges
  - [ ] Parse CLUSTERS with ID refs → Cluster objects
  - [ ] Parse SIGNALS with all 3 levels (!, ~, ?)
  - [ ] Parse DEPTH MAP with context → DepthHint objects
  - [ ] Parse META → SurfaceMeta
  - [ ] Serialize → valid .nm text
  - [ ] Round-trip: parse(serialize(surface)) == surface
  - [ ] Parse empty sections → empty tuples (no crash)
  - [ ] Parse malformed input → clear error message
  - [ ] Token budget: trim lowest-priority items
  - [ ] Cross-reference: get_node(), get_depth_hint(), all_node_ids()
  - [ ] UTF-8 with Vietnamese entity names
  - [ ] Lenient parsing: unknown section skipped (no crash)
  - [ ] ID format validation: [d1] valid, [123] invalid

## Acceptance Criteria
- [ ] `parse()` handles all 5 sections + frontmatter correctly
- [ ] `serialize()` produces valid .nm that re-parses identically (round-trip)
- [ ] Cross-reference methods work (get_node, get_depth_hint)
- [ ] Token budget enforced — surface trimmed to fit
- [ ] Malformed input → helpful error messages (not cryptic crashes)
- [ ] Frozen dataclasses — all 9 models immutable
- [ ] Zero external dependencies (stdlib only, no pyyaml — hand-parse YAML frontmatter)
- [ ] Lenient parsing for forward compatibility

## Files Touched
- `src/neural_memory/surface/__init__.py` — new
- `src/neural_memory/surface/models.py` — new
- `src/neural_memory/surface/parser.py` — new
- `src/neural_memory/surface/serializer.py` — new
- `src/neural_memory/surface/token_budget.py` — new
- `tests/unit/test_surface_format.py` — new

## Dependencies
- None — this phase is fully independent

## Risks
- Format evolution: adding new sections later must not break old parsers → lenient parsing (skip unknown sections)
- Token estimation: chars/4 is rough → may need refinement in Phase 4
- YAML frontmatter: hand-parsing to avoid pyyaml dependency — keep frontmatter simple (flat keys only)
- ID collisions: generator (Phase 2) must ensure unique IDs — parser just validates format
- Edge target ambiguity: `→edge→ [id] "text"` vs `→edge→ "text"` — parser uses `[` prefix to detect ID ref

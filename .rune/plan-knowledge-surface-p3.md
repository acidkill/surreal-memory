# Phase 3: MCP Integration + Auto-Load

## Goal
Wire the Knowledge Surface into the MCP server lifecycle: auto-load `.nm` on session init, inject into agent context via `instructions`, and route depth hints to appropriate recall depth.

## VISION Checklist
| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 1 | Activation vs Search | ✅ | Surface is pre-activated, not searched |
| 2 | Spreading Activation | ✅ | DEPTH MAP triggers SA-based recall when needed |
| 3 | No-Embedding Test | ✅ | Surface injection is plain text, no embeddings |
| 4 | Detail→Speed | ✅ | SUFFICIENT = 0 latency, NEEDS_DEEP = targeted SA |
| 5 | Source Traceable | ✅ | Agent can trace GRAPH nodes back to brain.db |
| 6 | Brain Test | ✅ | Prefrontal cortex loads working memory on "wake up" |
| 7 | Memory Lifecycle | ✅ | Surface loaded = Recall phase, updated = Consolidate phase |

## Design

### Session Start Flow
```python
# In MCP server initialize():
async def _initialize(self):
    # ... existing init ...

    # Load knowledge surface
    surface_path = self._get_surface_path()
    if surface_path.exists():
        text = surface_path.read_text(encoding="utf-8")
        self._surface = parse(text)
        self._surface_text = text
    else:
        self._surface = None
        self._surface_text = ""
```

### Instructions Injection
```python
# Append surface to MCP instructions (loaded by agent every session)
def _build_instructions(self) -> str:
    base = EXISTING_INSTRUCTIONS
    if self._surface_text:
        return f"{base}\n\n## Knowledge Surface\n{self._surface_text}"
    return base
```

### Depth-Aware Recall
```python
# New: nmem_recall auto-adjusts depth based on surface DEPTH MAP
async def _depth_aware_recall(self, query: str, depth: int | None = None):
    if depth is not None:
        return await self._recall(query, depth=depth)

    # Check if query matches a surface entity
    entity = self._find_surface_entity(query)
    if entity:
        hint = self._surface.get_depth_hint(entity)
        if hint == DepthLevel.SUFFICIENT:
            # Return surface context directly (no brain.db query)
            return self._surface_context_for(entity)
        elif hint == DepthLevel.NEEDS_DETAIL:
            return await self._recall(query, depth=1)
        elif hint == DepthLevel.NEEDS_DEEP:
            return await self._recall(query, depth=2)

    # No surface match → normal recall
    return await self._recall(query, depth=1)
```

### Surface Path Resolution
```python
def _get_surface_path(self) -> Path:
    # Project surface (if in project context)
    project_root = detect_project_root()
    if project_root:
        project_surface = project_root / ".neuralmemory" / "surface.nm"
        if project_surface.exists():
            return project_surface

    # Global surface
    return Path.home() / ".neuralmemory" / "surface.nm"
```

## Tasks
- [x] Add `_surface_text` and `_surface_brain` to MCP server state
- [x] Load surface.nm via `load_surface()` method with caching
- [x] Inject surface text into MCP `instructions` response on initialize
- [x] Implement `_check_surface_depth()` — match query to surface entities + depth routing
- [x] Implement depth-aware recall routing (SUFFICIENT → surface context, NEEDS_DEEP → depth=2)
- [x] Create `surface/resolver.py` — `get_surface_path`, `detect_project_root`, `load/save_surface_text`
- [x] Add `detect_project_root()` for project-level surface resolution (6 markers)
- [x] Add surface reload on brain switch (in `get_storage()` detect brain change)
- [x] Tests: `tests/unit/test_surface_mcp.py` — 21 tests
  - [x] Surface path resolution (global, project priority)
  - [x] File I/O (load, save, missing, overwrite)
  - [x] Project root detection (git, home)
  - [x] Server caching (same brain cached, brain change reloads, missing → empty)
  - [x] Instructions include surface text on initialize
  - [x] Instructions work without surface
  - [x] Depth routing: SUFFICIENT → surface context (with edges + clusters)
  - [x] Depth routing: NEEDS_DEEP → depth=2 override
  - [x] Unknown entity → no routing
  - [x] Empty/corrupt surface → graceful fallback
  - [x] Case-insensitive entity matching

## Acceptance Criteria
- [x] Agent receives knowledge surface in every session automatically
- [x] SUFFICIENT entities answered from surface without brain.db query
- [x] NEEDS_DETAIL/NEEDS_DEEP triggers appropriate recall depth
- [x] Surface loaded in <50ms (file read only, cached)
- [x] Missing/corrupt surface.nm → graceful degradation
- [x] Brain switch → correct surface loaded
- [x] Instructions token budget not exceeded (surface appended, not replacing)

## Files Touched
- `src/neural_memory/mcp/server.py` — modify (surface loading + instructions)
- `src/neural_memory/mcp/tool_handlers/recall_handler.py` — modify (depth routing)
- `src/neural_memory/surface/resolver.py` — new (path resolution + project detection)
- `tests/unit/test_surface_mcp.py` — new

## Dependencies
- Phase 1 (parser) — must be complete to parse .nm files
- Phase 2 (generator) — surface must exist to load (but can test with hand-crafted .nm)

## Risks
- Instructions size limit: some agents ignore long instructions. Keep surface <1000 tokens
- Fuzzy entity matching: "postgres" vs "postgresql" — need normalization
- Project detection on Windows: path separators, drive letters
- Race condition: surface being regenerated while session reads it → use atomic write

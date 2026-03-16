# Phase 4: Lifecycle & Polish

## Goal
Complete the surface lifecycle: regenerate on session end, integrate with consolidation, manage global vs project surfaces, and handle edge cases.

## VISION Checklist
| # | Check | Pass? | Notes |
|---|-------|-------|-------|
| 1 | Activation vs Search | ✅ | Regeneration uses activation scores, not search |
| 2 | Spreading Activation | ✅ | SA scores determine what stays/enters surface |
| 3 | No-Embedding Test | ✅ | All regeneration logic is graph-structural |
| 4 | Detail→Speed | ✅ | Pre-generated surface = instant next session |
| 5 | Source Traceable | ✅ | Regeneration preserves neuron_id back-references |
| 6 | Brain Test | ✅ | Sleep consolidation updates working memory for next day |
| 7 | Memory Lifecycle | ✅ | Session end = Consolidate → regenerate surface |

## Design

### Session-End Regeneration
```python
# Triggered by:
# 1. MCP server shutdown (stop hook)
# 2. nmem_auto(action="process") call
# 3. Consolidation run (MATURE/BALANCED strategy)

async def regenerate_surface(storage, config, surface_path):
    generator = SurfaceGenerator(storage, config)
    surface = await generator.generate(token_budget=config.surface_token_budget)
    text = serialize(surface)

    # Atomic write (write to .tmp, then rename)
    tmp_path = surface_path.with_suffix(".nm.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(surface_path)
```

### Incremental Update (Optimization)
```python
# Instead of full regeneration every session:
# 1. Load existing surface
# 2. Add new SIGNALS from this session
# 3. Update DEPTH MAP for entities touched this session
# 4. Only full-regenerate if >20% of surface entities changed

async def incremental_update(existing, session_memories, storage):
    changed_entities = extract_entities(session_memories)
    overlap = changed_entities & existing.all_entities()

    if len(overlap) / max(len(existing.all_entities()), 1) > 0.2:
        return await full_regenerate(storage)  # Too much changed

    # Patch existing surface
    new_signals = extract_signals(session_memories)
    updated_depth = recompute_depth(existing, changed_entities, storage)
    return replace(existing, signals=new_signals, depth_map=updated_depth)
```

### Global vs Project Surface
```python
# Project surface: project-specific knowledge
#   Path: {project_root}/.neuralmemory/surface.nm
#   Contains: project decisions, architecture, bugs
#   Generated from: project brain (if layered brain exists) or filtered global

# Global surface: cross-project patterns
#   Path: ~/.neuralmemory/surface.nm
#   Contains: user preferences, tool habits, general insights
#   Generated from: global brain, filtered to non-project content

# Both loaded? Merge with project priority:
def merge_surfaces(project: KnowledgeSurface, global_: KnowledgeSurface) -> str:
    # Project GRAPH + SIGNALS take priority
    # Global CLUSTERS merged (no dedup needed — different scopes)
    # DEPTH MAP: project overrides global for same entity
```

### Consolidation Integration
```python
# In consolidation.py _mature() or _balanced():
async def _consolidate(self):
    # ... existing consolidation logic ...

    # After consolidation, regenerate surface
    if self._config.surface_auto_regenerate:
        surface_path = self._get_surface_path()
        await regenerate_surface(self._storage, self._config, surface_path)
        logger.info("Knowledge surface regenerated at %s", surface_path)
```

### CLI Command
```bash
# Manual surface management
nmem surface generate          # Force regenerate
nmem surface show              # Print current surface
nmem surface stats             # Token count, entity count, freshness
```

## Tasks (MVP — shipped)
- [x] Implement `regenerate_surface()` in `surface/lifecycle.py`
- [x] Implement atomic file write (tmp + rename) in `surface/resolver.py`
- [x] Add `nmem_surface` MCP tool (generate + show) via `SurfaceHandler` mixin
- [x] Add tool schema in `tool_schemas.py`
- [x] Register in server dispatch table + class MRO
- [x] Hook regeneration into `nmem_auto(action="process")` in `auto_handler.py`
- [x] Implement `show_surface()` for structured surface info
- [x] Tests: `tests/unit/test_surface_lifecycle.py` — 10 tests
  - [x] Regeneration produces valid surface + writes to disk
  - [x] Token budget respected via trim_to_budget
  - [x] Show: missing surface → helpful message
  - [x] Show: valid surface → structured info with all fields
  - [x] Show: corrupt surface → graceful error
  - [x] Handler: show action works end-to-end
  - [x] Handler: generate action regenerates and clears cache
  - [x] Handler: unknown action returns error
  - [x] Handler: default action is show
  - [x] Handler: no brain returns error

## Deferred (future iterations)
- [ ] Implement incremental update (patch vs full-regenerate decision)
- [ ] Add `surface_auto_regenerate` and `surface_token_budget` to BrainConfig
- [ ] Implement global vs project surface merge
- [ ] Integrate regeneration into consolidation (_mature, _balanced)
- [ ] Add `nmem surface` CLI subcommand (generate, show, stats)
- [ ] Handle first-ever session bootstrap

## Acceptance Criteria (MVP)
- [x] Surface auto-regenerated on session end (via nmem_auto process)
- [x] Atomic write prevents corruption on crash
- [x] nmem_surface MCP tool works (generate + show)
- [x] Token budget respected after regeneration
- [x] Missing/corrupt surface handled gracefully

## Files Touched
- `src/neural_memory/surface/lifecycle.py` — new (regeneration + incremental update)
- `src/neural_memory/surface/merger.py` — new (global + project merge)
- `src/neural_memory/engine/consolidation.py` — modify (add surface regeneration)
- `src/neural_memory/hooks/stop_hook.py` — modify (trigger surface regeneration)
- `src/neural_memory/cli/commands/surface.py` — new (CLI subcommand)
- `src/neural_memory/mcp/tool_handlers/surface_handler.py` — new (MCP tool)
- `src/neural_memory/unified_config.py` — modify (add surface config fields)
- `tests/unit/test_surface_lifecycle.py` — new

## Dependencies
- Phase 1 (format) + Phase 2 (generator) + Phase 3 (MCP) must be complete
- Uses existing consolidation infrastructure

## Risks
- Regeneration latency: large brains may take >1s → run async, don't block session end
- Incremental update accuracy: patching may miss important changes → fallback to full regen
- File system permissions: project directory may not be writable → graceful skip

# Phase 2: MCP + Recall Integration

## Goal
Wire ephemeral flag through MCP tools (`nmem_remember`, `nmem_recall`), exclude from consolidation, and exclude from sync push.

## Tasks
- [ ] 2.1 Add `ephemeral` param to `nmem_remember` tool schema in `tool_schemas.py` — `{"type": "boolean", "description": "...", "default": false}`
- [ ] 2.2 Update `_remember()` in `tool_handlers.py` — read `args.get("ephemeral", False)`, pass to `Neuron.create()`, set TTL on typed_memory (default 24h)
- [ ] 2.3 Store `session_id` in ephemeral neuron metadata — get from `SessionManager` for current-session filtering
- [ ] 2.4 Add `permanent_only` param to `nmem_recall` tool schema in `tool_schemas.py`
- [ ] 2.5 Update recall handler to pass `permanent_only` filter down to retrieval pipeline
- [ ] 2.6 Update `_find_anchors_ranked()` in `retrieval.py` — when `permanent_only=True`, add `AND n.ephemeral = 0` to anchor queries
- [ ] 2.7 Update FTS search in retrieval — add ephemeral filter JOIN when `permanent_only=True`
- [ ] 2.8 Exclude ephemeral neurons from consolidation — update neuron fetch queries in `consolidation.py` to add `WHERE ephemeral = 0`
- [ ] 2.9 Update sync push in `sync_handler.py` — ensure `_sync()` push path filters `WHERE ephemeral = 0` (via change_log exclusion from Phase 1)
- [ ] 2.10 Update `nmem_remember_batch` to support `ephemeral` param passthrough

## Acceptance Criteria
- [ ] `nmem_remember(content="temp note", ephemeral=true)` stores neuron with ephemeral=1
- [ ] `nmem_recall("temp note")` returns ephemeral memories by default
- [ ] `nmem_recall("temp note", permanent_only=true)` excludes ephemeral memories
- [ ] `nmem_consolidate` never processes ephemeral neurons
- [ ] `nmem_sync(action="push")` never pushes ephemeral neurons
- [ ] Ephemeral memories get typed_memory with expires_at = now + 24h (default)

## Files Touched
- `src/neural_memory/mcp/tool_schemas.py` — modify (add params)
- `src/neural_memory/mcp/tool_handlers.py` — modify (remember + recall)
- `src/neural_memory/engine/retrieval.py` — modify (anchor filtering)
- `src/neural_memory/engine/consolidation.py` — modify (exclusion filter)
- `src/neural_memory/mcp/sync_handler.py` — verify (should already be excluded via change_log)

## Dependencies
- Phase 1 must be complete (schema + storage layer)

## Notes
- Recall default = include ephemeral. This ensures session-local memories are immediately useful.
- The TTL on typed_memories leverages existing `expires_at` + `ExpiryCleanupHandler`. Ephemeral neurons also need cleanup (neuron row deletion) — handled in Phase 3.
- Consolidation fetches neurons via `find_neurons()` or direct SQL — must audit all paths.
- `permanent_only` naming chosen over `exclude_ephemeral` for clarity in LLM-facing API.

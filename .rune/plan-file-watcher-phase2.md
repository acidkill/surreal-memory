# Phase 2: CLI + MCP + Config

## Goal
Expose file watcher to users via CLI command, MCP tool, and config.toml settings.

## Tasks
- [ ] 2.1 — Add watcher config to `unified_config.py`
  ```toml
  [watcher]
  enabled = false
  paths = []
  extensions = [".md", ".txt", ".pdf", ".docx", ".json", ".csv"]
  ignore_patterns = ["*.tmp", ".git/*", "node_modules/*"]
  debounce_seconds = 2.0
  max_file_size_mb = 10
  max_watched_dirs = 10
  ```
- [ ] 2.2 — Create CLI command `nmem watch` in `src/neural_memory/cli/commands/watch.py`
  - `nmem watch start <path>` — start watching a directory (adds to config, starts observer)
  - `nmem watch stop [path]` — stop watching (all or specific path)
  - `nmem watch status` — show watched dirs, file counts, last ingestion time
  - `nmem watch list` — list all tracked files with state (ingested/pending/skipped)
  - `nmem watch clear <path>` — remove watch state for a path (re-ingest on next start)
- [ ] 2.3 — Create MCP tool `nmem_watch` in `src/neural_memory/mcp/watch_handler.py`
  - `action="start"` — start watching (path required)
  - `action="stop"` — stop watching (path optional, stops all if omitted)
  - `action="status"` — return watched dirs + stats
  - `action="list"` — return tracked files
- [ ] 2.4 — Register MCP tool in `tool_handlers.py` dispatch + update tool count in tests
- [ ] 2.5 — Write tests: `tests/unit/test_watch_handler.py`
  - Test MCP tool actions (start/stop/status/list)
  - Test config loading/saving
  - Test CLI commands (mock observer)

## Acceptance Criteria
- [ ] `nmem watch start ~/notes` starts watching and auto-ingests
- [ ] `nmem watch status` shows active watchers and stats
- [ ] `nmem_watch` MCP tool works from Claude Code
- [ ] Config persisted in config.toml
- [ ] Tool count updated in test_mcp.py + test_tool_tiers.py

## Files Touched
- `src/neural_memory/unified_config.py` — modify (add watcher section)
- `src/neural_memory/cli/commands/watch.py` — new
- `src/neural_memory/cli/main.py` — modify (register watch command)
- `src/neural_memory/mcp/watch_handler.py` — new
- `src/neural_memory/mcp/tool_handlers.py` — modify (register tool)
- `tests/unit/test_watch_handler.py` — new

## Dependencies
- Phase 1 complete (FileWatcher class, watch_state)

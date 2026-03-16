# Phase 3: Background Daemon + Polish

## Goal
Integrate watcher into `nmem serve` background mode, add metrics, and harden for production use.

## Tasks
- [ ] 3.1 — Integrate FileWatcher into `nmem serve` (FastAPI lifespan)
  - Start observer on server startup if `watcher.enabled = true` in config
  - Stop observer on server shutdown
  - Expose `/api/watcher/status` endpoint (active dirs, file counts, last event)
- [ ] 3.2 — Add watcher metrics to dashboard
  - Card on HealthPage: watched dirs count, files ingested today, pending queue
  - Activity log: last 20 ingestion events with file path + neuron count
- [ ] 3.3 — Batch processing with rate limiting
  - Process max 5 files per batch (avoid overloading encoder on bulk drops)
  - Queue overflow: if >50 files pending, log warning + process in priority order (newest first)
  - Configurable: `watcher.batch_size = 5`, `watcher.max_queue = 50`
- [ ] 3.4 — Notification hooks
  - After ingestion: emit event for hook system (`post_watch_ingest`)
  - Include: file_path, neurons_created, synapses_created, processing_time
- [ ] 3.5 — Error resilience
  - File locked/deleted during processing → skip, retry on next event
  - Corrupt file → log error, mark as `status="error"` in watch_state
  - Observer crash → auto-restart with exponential backoff (max 5 retries)
- [ ] 3.6 — Integration tests: `tests/integration/test_file_watcher_e2e.py`
  - Drop a .md file into watched dir → verify neurons created
  - Modify file → verify re-ingested (new content)
  - Drop unchanged file → verify skipped (simhash match)
  - Drop unsupported extension → verify ignored
  - Drop file > max_size → verify rejected
- [ ] 3.7 — Documentation
  - `docs/guides/file-watcher.md` — setup guide, config reference, troubleshooting
  - Update README feature list

## Acceptance Criteria
- [ ] `nmem serve` auto-starts watcher when configured
- [ ] Dashboard shows watcher status
- [ ] Batch processing handles bulk drops without overload
- [ ] Errors don't crash the watcher (auto-recovery)
- [ ] Integration tests cover happy path + edge cases
- [ ] Docs complete

## Files Touched
- `src/neural_memory/server/app.py` — modify (lifespan watcher start/stop)
- `src/neural_memory/server/routes/` — modify (add watcher endpoint)
- `dashboard/src/features/health/` — modify (add watcher card)
- `src/neural_memory/engine/file_watcher.py` — modify (batch, rate limit, retry)
- `tests/integration/test_file_watcher_e2e.py` — new
- `docs/guides/file-watcher.md` — new

## Dependencies
- Phase 2 complete (CLI + MCP + config)

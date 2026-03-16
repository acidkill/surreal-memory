# Phase 1: Core Watcher Engine

## Goal
Build the FileWatcher class that monitors directories and auto-ingests changed files using existing DocTrainer pipeline.

## Tasks
- [ ] 1.1 — Add `watchdog` to optional deps in `pyproject.toml` under `[watch]` extra
- [ ] 1.2 — Create `src/neural_memory/engine/file_watcher.py` — FileWatcher class
  - Constructor: `FileWatcher(storage, brain_config, watch_paths, extensions, ignore_patterns)`
  - Uses `watchdog.observers.Observer` for FS events
  - Event handler: on_created, on_modified, on_moved → queue for processing
  - Debounce: 2s after last event per file (avoid partial writes)
  - Extension filter: only process EXTENDED_EXTENSIONS (from doc_chunker)
  - Ignore patterns: `.git/`, `node_modules/`, `__pycache__/`, user-configurable
- [ ] 1.3 — Create `src/neural_memory/engine/watch_state.py` — WatchState tracker
  - SQLite table `watch_state`: (file_path TEXT PK, mtime REAL, simhash INT, last_ingested TEXT, neuron_count INT)
  - `should_process(path)` → check mtime + simhash vs stored state
  - `mark_processed(path, mtime, simhash, neuron_count)` → update state
  - `list_watched_files()` → all tracked files with status
- [ ] 1.4 — Add `watch_state` table to schema migrations in `src/neural_memory/storage/migrations/`
- [ ] 1.5 — Integrate DocTrainer: FileWatcher.process_file(path) calls trainer.train_file()
  - Before training: check simhash — skip if content unchanged
  - After training: update watch_state with new mtime/simhash/neuron_count
  - Source tracking: set `source="file_watcher:{path}"` on created neurons
- [ ] 1.6 — Security: validate watched paths
  - Paths must be absolute, resolved, no symlink traversal outside configured dirs
  - Max watched dirs: 10 (configurable)
  - Max file size: 10MB (configurable)
- [ ] 1.7 — Write unit tests: `tests/unit/test_file_watcher.py`
  - Test debounce logic
  - Test extension filtering
  - Test simhash dedup (unchanged file → skip)
  - Test watch_state CRUD
  - Test security validation (symlink, max size, max dirs)

## Acceptance Criteria
- [ ] FileWatcher detects new/modified files in watched directory
- [ ] Changed files auto-ingested via DocTrainer pipeline
- [ ] Unchanged files skipped (simhash dedup)
- [ ] Partial writes handled (debounce 2s)
- [ ] Security: no path traversal, size limits enforced
- [ ] Tests pass, coverage maintained

## Files Touched
- `pyproject.toml` — modify (add `watch` extra)
- `src/neural_memory/engine/file_watcher.py` — new
- `src/neural_memory/engine/watch_state.py` — new
- `src/neural_memory/storage/migrations/` — modify (add watch_state table)
- `tests/unit/test_file_watcher.py` — new

## Dependencies
- `watchdog>=4.0.0` (cross-platform FS events)
- Existing: DocTrainer, DocChunker, DocExtractor, SimHash

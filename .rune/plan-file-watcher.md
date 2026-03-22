# Feature: File Watcher Ingestion (Issue #66)

## Overview
Auto-ingest files from watched folders into Neural Memory. Drop file → auto-memorize.
Reuses existing DocTrainer + doc_chunker + doc_extractor + SimHash — no reimplementation.

**Destination: Free** — core DX feature, drives adoption.

## Phases
| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Core Watcher | ⬚ Pending | plan-file-watcher-phase1.md | FileWatcher class, watchdog integration, dedup |
| 2 | CLI + MCP | ⬚ Pending | plan-file-watcher-phase2.md | `nmem watch` CLI, `nmem_watch` MCP tool, config |
| 3 | Background Daemon + Polish | ⬚ Pending | plan-file-watcher-phase3.md | `nmem serve` integration, debounce, metrics, tests |

## Key Decisions
- `watchdog` as optional dep: `neural-memory[watch]`
- Reuse DocTrainer.train_file() — no new encoding logic
- Track file state (mtime + simhash) in SQLite `watch_state` table
- Debounce: 2s after last FS event before processing (avoid partial writes)
- Security: watched paths must be explicitly configured, no symlink traversal
- 12 formats supported (same as doc_extractor): md, txt, rst, pdf, docx, pptx, html, json, xlsx, csv, htm, mdx

## Existing Infrastructure (reuse, don't rebuild)
- `DocTrainer` — train_file(), train_directory(), discover_files()
- `DocChunker` — chunk_markdown(), EXTENDED_EXTENSIONS
- `DocExtractor` — 12 format extractors (pdf, docx, pptx, html, xlsx, csv, json)
- `SimHash` — near-duplicate detection (64-bit LSH)
- `SyncEngine` + `SourceAdapter` — batch import with progress callbacks
- `MemoryEncoder` — NLP extraction pipeline
- `TrainingConfig` — chunk size, domain tags, pinning, consolidation

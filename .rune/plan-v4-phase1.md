# Phase 1: Session Intelligence

## Goal

Track session context across MCP calls so the brain knows "what the user is working on right now". Foundation for adaptive depth and predictive priming.

## Problem

Currently each `nmem_recall` is stateless — no memory of what was queried 30 seconds ago. Real brains maintain working memory: recent topics prime related recall.

## Tasks

- [x] 1.1: Session state model
  - Created `SessionState` dataclass + `QueryRecord` frozen dataclass in `engine/session_state.py`
  - Fields: session_id, topic_ema, recent_queries (max 20), recent_entities Counter, query_count, started_at, last_active
  - In-memory only — lives in MCP server process lifetime

- [x] 1.2: Topic detection from queries
  - Reuses `extract_keywords()` when keywords=None
  - EMA alpha=0.3, topics normalized (lowercase+strip), near-zero pruned (<0.01)
  - Entity frequency tracked via Counter

- [x] 1.3: Session manager singleton
  - `SessionManager` with LRU OrderedDict, max 10 concurrent, 2h auto-expiry
  - `get_or_create()`, `get()`, `remove()`, `all_sessions()`

- [x] 1.4: Wire into recall pipeline
  - `ReflexPipeline.query()` accepts `session_id`
  - Records query after completion, attaches session_topics to result metadata
  - Persists summary every 10 queries (non-critical failure path)

- [x] 1.5: Session context in MCP response
  - `session_topics` and `session_query_count` in recall response
  - MCP handler passes `f"mcp-{id(self)}"` as stable session ID

- [x] 1.6: Periodic session summary persist
  - Schema v24: `session_summaries` table with FK to brains
  - `SQLiteSessionsMixin`: save, get_recent, auto-prune (max 500/brain)
  - Added to SCHEMA string + MIGRATIONS dict + clear() cleanup

- [x] 1.7: Tests (40 tests)
  - QueryRecord: creation, frozen, defaults (3)
  - SessionState: EMA update/decay/boost/prune, normalization, entities, bounds, persist, expiry, summary (21)
  - SessionManager: singleton, LRU, expiry, CRUD (11)
  - SQLite: save/retrieve, no-brain guard, limit cap, ordering, rounding (5)

## Acceptance Criteria

- [x] Each recall query updates session topic EMA
- [x] Session topics visible in recall response
- [x] Sessions auto-expire after 2h inactivity
- [x] Session summaries persisted every 10 queries
- [x] Zero performance impact on hot path (<1ms overhead per recall)
- [x] 40 tests covering all session lifecycle scenarios (target was 20+)

## Files Touched

- `src/neural_memory/engine/session_state.py` — new
- `src/neural_memory/engine/retrieval.py` — accept session_id, record queries
- `src/neural_memory/mcp/tool_handlers.py` — pass session_id to pipeline
- `src/neural_memory/storage/sqlite_schema.py` — session_summaries table (schema v24)
- `src/neural_memory/storage/sqlite_sessions.py` — new mixin
- `src/neural_memory/storage/sqlite_store.py` — add mixin + clear() table
- `tests/unit/test_session_intelligence.py` — new (40 tests)
- `tests/unit/test_baby_mi_features.py` — schema version 23→24
- `tests/unit/test_source_registry.py` — schema version 23→24

## Bugs Fixed During Implementation

- `sqlite_sessions.py` used `await self._get_connection()` instead of `self._ensure_conn()` (mismatched with all other mixins)
- `session_summaries` table missing from SCHEMA string (only in MIGRATIONS) — fresh DBs would fail
- `Counter` missing generic type param `Counter[str]` — mypy error

## Dependencies

- None — this is the foundation phase

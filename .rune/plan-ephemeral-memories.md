# Feature: Ephemeral Memories (Issue #91)

## Overview
Add session-scoped ephemeral memories that auto-expire, never sync to cloud, and are excluded from consolidation. Stored via `ephemeral=true` on `nmem_remember`, visible during current session, with configurable TTL (default 24h).

## Phases
| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Schema + Storage | ✅ Done | plan-ephemeral-memories-phase1.md | DB column, migration v33, CRUD filtering |
| 2 | MCP + Recall Integration | ✅ Done | plan-ephemeral-memories-phase2.md | Tool params, recall filtering, consolidation/sync exclusion |
| 3 | Cleanup + Tests | ✅ Done | plan-ephemeral-memories-phase3.md | TTL cleanup, session-end hook, 14 tests |

## Key Decisions
- **Flag on neurons table** (not separate table) — `ephemeral INTEGER DEFAULT 0` column. Simpler, avoids JOINs on hot path. Neurons are the atomic unit; marking them ephemeral propagates naturally to fibers/synapses via CASCADE.
- **Session tracking** — Reuse existing `SessionManager` session_id (in-memory, per MCP process). Store session_id in neuron metadata for "current session only" filtering.
- **TTL default 24h** — Ephemeral neurons get `expires_at` on `typed_memories` (existing field). Reuse `ExpiryCleanupHandler` for TTL cleanup, add neuron-level expiry scan.
- **Sync exclusion** — Filter `WHERE ephemeral = 0` in `seed_change_log()` and `record_change()` skip for ephemeral neurons. No change_log entries = never synced.
- **Consolidation exclusion** — Add `WHERE ephemeral = 0` to neuron fetch queries in consolidation engine.
- **Recall filtering** — Ephemeral included by default. `permanent_only=true` param on `nmem_recall` adds `WHERE ephemeral = 0`.

## Schema Change
Migration v32 -> v33:
```sql
ALTER TABLE neurons ADD COLUMN ephemeral INTEGER DEFAULT 0;
CREATE INDEX idx_neurons_ephemeral ON neurons(brain_id, ephemeral);
```

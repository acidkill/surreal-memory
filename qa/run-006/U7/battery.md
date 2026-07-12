# U7 — Release-prep v2.9.0 — end-to-end battery evidence

Branch: `feature/v290-release-prep` (HEAD `3e24443`), stacked on `main` (`d25b98f` = v2.8.0).
Version bumped 2.8.0 → 2.9.0 across the 9 parity files; CHANGELOG v2.9.0 section added.

## Battery 1 — Golden ranking (bit-for-bit)
`pytest tests/unit/test_golden_ranking.py` → **6 passed**. Default ranking unchanged
(neutral trust/recency weights are branch-guarded no-ops; spy test = zero storage reads at trust_weight=0).

## Battery 2 — Full suite
- No-DB D1: `env -u SURREALDB_URL pytest tests/ -m "not stress" -n auto --ignore=tests/e2e/test_api.py`
  → **6222 passed, 89 skipped, 1 xfailed, 0 errors**. (The 26 tests/e2e/test_api.py errors under the full
  run are the PRE-EXISTING sys.modules['surrealdb'] stub-leak — 0 when that file is excluded/isolated.)
- Live SurrealDB (serial, *_live.py files alone): recordid_fix + fiber_id_norm + retrieval_trace +
  supersession live → **10 passed**.

## Battery 3 — Migration
- `tests/unit/test_surrealdb_migrations.py` → **26 passed** (v7→v8→v9 detection + per-statement idempotent
  DDL + "already exists" tolerance + SQLite ADD COLUMN). The live dev DB was migrated v8→v9 in U1 (verified
  `SELECT version FROM schema_meta:version` ⇒ 9); DDL is purely additive so v2.7 code reads a v9 DB without error
  (rollback-read). Reuses U1's live migration evidence (qa/run-006/U1/realdb-evidence.md).

## Battery 4 — Spectron-demo (live MCP over run-006 v2.9.0 code)
See spectron-demo.md (Emma Oslo→Bergen: default recall = Bergen; valid_at = Oslo; include_superseded = both;
trace=true → provenance trace_get; smem_uncertainty overview; drift_clusters=0 on SurrealDB).

## Battery 5 — Perf
- Retrieval-trace overhead (`benchmarks/trace_overhead.py`): build ~3.7 µs / build+persist ~4.1 µs per trace.
  Recall is ms-scale → fire-and-forget trace overhead <1% (≪ the <2% target); disabled = 0 (early return).
- Spy zero-reads at trust_weight=0: golden spy test passes (zero get_typed_memories_batch calls).
- `/api/dashboard/uncertainty` second hit served from TTL cache: test_dashboard_uncertainty proves
  count_typed_memories is hit exactly once across repeated GETs.
- Dashboard perf-query tests pass (test_dashboard_perf_queries).

## make verify
- lint `ruff check src/ tests/`: clean. format-check: 669 files formatted.
- mypy `src/`: **356 files, no issues**. security (ruff S): clean.
- coverage: **68.15% ≥ 67**.

## Version parity (9 files → 2.9.0, verified 0 stray 2.8.0)
pyproject.toml · src/surreal_memory/__init__.py · tests/unit/test_health_fixes.py (TestVersionBump pin) ·
integrations/surrealmemory/{package.json,package-lock.json} ·
integrations/surreal-memory-client/{package.json,package-lock.json} ·
vscode-extension/{package.json,package-lock.json}. TestVersionBump passes (== "2.9.0").

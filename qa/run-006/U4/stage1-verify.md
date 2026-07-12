# U4 — Stage 1 verification evidence (queryable retrieval traces)

Branch: `feature/v290-pr4-retrieval-traces` (from U3 HEAD 796d02f)
Commits: `8795ab7` write-path (config + trace_builder + recall persist) · `f6b3a1a` query-surface
(smem_provenance traces/trace_get) · `a51a349` prune (consolidation) + benchmark.

## D1 — full no-DB unit suite
`env -u SURREALDB_URL .venv/bin/python -m pytest tests/ -m "not stress" -n auto -p no:cacheprovider -q --ignore=tests/e2e/test_api.py`
→ **6195 passed, 89 skipped, 1 xfailed, 0 errors**. (+19 vs U3's 6176 — the new U4 tests.)
The 26 tests/e2e/test_api.py errors under the full run remain the PRE-EXISTING sys.modules['surrealdb']
stub-leak (0 when that file is excluded/isolated).

## make verify components
- lint: `ruff check src/ tests/` → All checks passed. (benchmarks/ is outside the gate scope; its print()s are fine.)
- format-check: `ruff format --check src/ tests/` → 664 files already formatted.
- typecheck: `mypy src/ --ignore-missing-imports` → **Success: no issues found in 354 source files**.
- coverage: `--cov-fail-under=67` → **Required test coverage of 67% reached. Total coverage: 68.07%**.

## Live SurrealDB
- `tests/unit/test_surrealdb_retrieval_trace_live.py` (U1 storage round-trip add/get/find/prune) → **3 passed** in isolation.
- (Stage-2 real-db-test-runner separately live-verified the full U4 write/find/prune + smem_provenance
  traces/trace_get path with real row counts 0→1→3→2→0 — see realdb-evidence.md.)

## New U4 unit tests
- `test_trace_builder.py` (4) — result→trace mapping, id/bound caps, never-raises on sparse result.
- `test_recall_trace.py` (4) — default disabled = no-op (spy zero writes); per-call trace=true → trace_id +
  synchronous persist; config-enabled → background fire-and-forget (no id); trace failure never breaks recall.
- `test_trace_config.py` +2 — UnifiedConfig.trace default off + save/load TOML round-trip preserves [trace].
- `test_provenance_traces.py` (7) — traces by fiber_id/query_contains; no-filter all; no neuron_id required;
  invalid since error; trace_get by id; missing/unknown id error.
- `test_consolidation_trace_prune.py` (2) — _prune calls prune_retrieval_traces (TTL); dry_run no-op.

## Perf
`benchmarks/trace_overhead.py` → build ~3.7us / build+persist ~4.0us per trace. Recall is ms-scale, so
fire-and-forget trace overhead is <1%; disabled = 0 (structural early return, proven by the spy test).

Golden ranking stayed green throughout (trace is telemetry — never alters ranking/answer).

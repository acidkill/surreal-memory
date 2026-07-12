# U6 — Stage 1 verification evidence (health fields + dashboard UncertaintyPage)

Branch: `feature/v290-pr6-dashboard` (from U5 HEAD bceb8ad).
Commits: `f957e73` backend (engine extraction + diagnostics + stats + route) · `dd323d0` health-route conflict
fields · `3286fd6` frontend · `0c18fc5` rebuilt dist · `bdf7076` review fixes · `4714196` e2e spec.

## Backend — D1 + make verify
- D1 no-DB (excl. test_api.py): **6222 passed, 0 errors** (`-n auto`, env -u SURREALDB_URL).
- mypy `src/`: **356 files, no issues**.
- lint `ruff check src/ tests/`: clean. format-check: clean.
- 529 stats/dashboard/uncertainty/diagnostics/mcp/health tests pass.

## New/changed backend + tests
- engine/uncertainty_report.py: build_brain_uncertainty + count_active_contradictions +
  scan_low_evidence_and_superseded + get_detected_drift (extracted from mcp so server/ needn't import mcp).
- engine/diagnostics.py: BrainHealthReport.contradiction_count + conflict_rate (grade formula UNCHANGED).
- mcp/uncertainty_handler.py: now a thin delegate (U5 tool behaviour identical — 16 U5 tests stay green).
- mcp/stats_handler.py: conflict_rate in _stats (renamed from contradiction_rate per review) + _health fields.
- server/routes/dashboard_api.py: GET /api/dashboard/uncertainty (TTL-cached, deepcopy) + HealthReport fields.
- tests: test_uncertainty_report.py (+build_brain_uncertainty), test_dashboard_uncertainty.py (route + cache +
  mutation-safety + brain field), test_diagnostics.py (+TestConflictHealthFields).

## Frontend (dashboard/, npm)
- `npm run build` (tsc -b && vite build): PASSED, no TS errors. `npm run lint`: clean on changed files
  (8 pre-existing eslint errors in untouched files remain).
- New: features/uncertainty/UncertaintyPage.tsx, useUncertainty hook, UncertaintyOverview type; route in
  App.tsx, nav in Sidebar (ShieldWarning), i18n uncertainty.* + health.conflict* (en only). HealthPage conflict
  tile uses `health?.conflict_rate ?? 0` (safe on loading/undefined).
- Rebuilt tracked server/static/dist bundles committed (dist is tracked → served dashboard matches source).

## Cross-cutting
- Grade/purity formula UNCHANGED (fields surfaced, not recomputed) — python-reviewer confirmed byte-identical.
- drift is SQLite-only → guarded getattr → drift_clusters=0 on SurrealDB (production) — live-confirmed.
- Dashboard route perf-guarded by TTL cache (live-proven: count_typed_memories hit once across 3 GETs).
- SPA basename is `/ui` (routes render under /ui/*).

See realdb-evidence.md (live SurrealDB PASS) — Playwright browser-QA was CLEAN 2/2 (UncertaintyPage renders +
sidebar nav), spec committed as dashboard/e2e/uncertainty.spec.ts.

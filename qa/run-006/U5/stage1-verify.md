# U5 — Stage 1 verification evidence (uncertainty surfacing)

Branch: `feature/v290-pr5-uncertainty` (from U4 HEAD c14bf53). Commit `4f00bf7` (impl).

## D1 — full no-DB unit suite
`env -u SURREALDB_URL .venv/bin/python -m pytest tests/ -m "not stress" -n auto -p no:cacheprovider -q --ignore=tests/e2e/test_api.py`
→ **6214 passed, 89 skipped, 1 xfailed, 0 errors** (+16 vs U4's 6198 — the new U5 tests).
The 26 tests/e2e/test_api.py errors under the full run remain the PRE-EXISTING sys.modules['surrealdb'] stub-leak.

## make verify components
- lint: `ruff check src/ tests/` → All checks passed.
- format-check: `ruff format --check src/ tests/` → 668 files already formatted.
- typecheck: `mypy src/ --ignore-missing-imports` → **Success: no issues found in 356 source files**.
- coverage: `--cov-fail-under=67` → **Required test coverage of 67% reached. Total coverage: 68.15%**.

## New U5 unit tests
- `test_uncertainty_report.py` (9) — build_uncertainty_block: no-signal→None; contradictions/low-confidence→high;
  superseded/expiring→medium; drift-absent-backend degrades to []; drift scoped to returned tags; never raises
  on storage errors (disputed_ids from metadata survive); no-metadata-signal + storage errors → None.
- `test_uncertainty_handler.py` (7) — smem_uncertainty overview aggregates all signals (level high,
  contradiction_rate>0); overview is default; low_evidence (trust 0.3); expiring; drift empty without backend;
  contradictions delegates to _conflicts_list; unknown action errors.
- Tool count 56→57: updated test_mcp.py (list count + protocol count + name set) + test_tool_tiers.py (full-tier
  count asserts). smem_uncertainty is full-tier only (mirrors smem_conflicts; not in minimal/standard).

## Cross-cutting
- Additive/opt-in: default recall unchanged (no include_uncertainty) → golden ranking stays green.
- drift is SQLite-only (not in ABC) → guarded getattr → [] on SurrealDB (production). Key Stage-2 live check.
- Typed scan bounded to 200 (low_evidence/superseded are "within first 200" — documented diagnostic bound).
- build_uncertainty_block lives in engine/ (not mcp/) so U6's dashboard route can reuse it.

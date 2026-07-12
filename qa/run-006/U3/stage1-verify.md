# U3 — Stage 1 verification evidence (per-fact supersession)

Branch: `feature/v290-pr3-supersession` (worktree run-006)
Commits (base `e83a0e8` = engine/supersession.py core):
`c21833f` recall hard-filter · `dbabaf6` auto-hook · `ce92f34` manual hook ·
`fba5f09` provenance lineage · `8eec122` backfill · `0c81d7d` format · `392147c` live regression.

## D1 — full no-DB unit suite
Command: `env -u SURREALDB_URL .venv/bin/python -m pytest tests/ -m "not stress" -n auto -p no:cacheprovider -q`
Result: **6173 passed, 86 skipped, 1 xfailed, 26 errors**.
The 26 errors are ALL `tests/e2e/test_api.py` and are the PRE-EXISTING `sys.modules['surrealdb']`
stub-leak (documented in STATE). Proof it is not a U3 regression:
- `--ignore=tests/e2e/test_api.py` → **6173 passed, 0 errors**.
- `tests/e2e/test_api.py::TestNeuronCRUD::test_get_neuron` passes in isolation.

## make verify components
- lint: `ruff check src/ tests/` → All checks passed.
- format-check: `ruff format --check src/ tests/` → 658 files already formatted (after `style:` commit `0c81d7d`).
- typecheck: `mypy src/ --ignore-missing-imports` → **Success: no issues found in 353 source files**.
- security: `ruff check src/ --select S --ignore S101,S110,S112,S311,S324` → All checks passed.
- coverage: `--cov=surreal_memory --cov-fail-under=67` → **Required test coverage of 67% reached. Total coverage: 67.96%**.

## Live SurrealDB (serial, SURREALDB_URL=http://localhost:8001)
- NEW `tests/unit/test_surrealdb_supersession_live.py` → **3 passed** (persist+read-back both id forms;
  idempotent; resolve_fibers_for_neurons functional equivalence).
- Prior regressions still green (run *_live files alone to avoid the surrealdb-stub pollution from
  test_surrealdb_store.py): `test_surrealdb_recordid_fix_live.py` + `test_surrealdb_fiber_id_norm_live.py`
  + `test_surrealdb_retrieval_trace_live.py` → **7 passed**.
- Live probe confirmed the auto-hook path (UNDERSCORE old_fiber_id from resolve_fibers_for_neurons) persists
  A-side validity correctly via UB2's id-agnostic get_typed_memory → `superseded=True`, dash read-back
  `superseded_by`==new dash id, `valid_until` set. No 4th round-trip bug.

## U3 unit tests (InMemoryStorage) — new
- `test_supersession.py` (5) — engine primitive
- `test_recall_supersession.py` (5) — hard-filter + include_superseded + escape hatch + valid_at + exact-mode fields
- `test_supersession_auto_hook.py` (5) — _apply_supersessions + ConflictDetectionStep collection
- `test_conflict_handler_supersession.py` (2) — manual keep_new lineage
- `test_provenance_supersession_lineage.py` (2) — SUPERSEDES both dirs + cycle guard
- `test_lifecycle_backfill_supersession.py` (3) — backfill idempotent + ambiguous skip + non-superseded ignored

Golden ranking snapshot stayed green throughout (superseded facts have no valid_until in the golden set).

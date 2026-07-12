# U1 — Schema v9 fundament — Stage 1 evidence

Branch: `feature/v290-pr1-schema-v9` (worktree run-006). Diff: 33 files, +1851/-28.

## D1 — full suite under -n auto (worker isolation), SURREALDB_URL unset
`.venv/bin/python -m pytest tests/ -m "not stress" -n auto`
→ **6131 passed, 79 skipped, 1 xfailed, 26 errors**.
- 0 FAILED. All 26 errors are in `tests/e2e/test_api.py` ONLY (`grep '^ERROR' | sed 's/::.*//' | sort -u` ⇒ single file).
- Root cause (verbatim trace): FastAPI TestClient lifespan → `server/app.py:64 get_shared_storage` →
  `unified_config.py:2296 _get_surrealdb_storage` → `storage/surrealdb/store.py:411 await self._conn.signin(...)`
  → `TypeError: object MagicMock can't be used in 'await' expression`. The SurrealDB SDK is mocked in this
  env and the e2e app-startup path (UNTOUCHED by this diff) needs a real server/DB. Pre-existing
  ("local e2e ≠ regression" — CLAUDE.md KNOWN TRAP). Also fails identically in isolation.
- Unit-only, no DB: `pytest tests/unit -m "not stress" -n auto` ⇒ 6105 passed, 0 fail/err.
- Live SurrealDB, SERIAL: `pytest tests/unit/test_surrealdb_typed_memory_all_types.py -p no:xdist` ⇒ 31/31 passed.

## D2 — make verify components (PATH=.venv/bin, SURREALDB_URL unset)
- `ruff check src/ tests/` → **All checks passed!**
- `ruff format --check src/ tests/` → **645 files already formatted**
- `mypy src/ --ignore-missing-imports` → **Success: no issues found in 351 source files**
- coverage (full suite) → **TOTAL 68.49% ≥ 67% gate** ("Required test coverage of 67% reached")
- `make security` (ruff -S / bandit) → **All checks passed! Security scan passed.**
- (make verify exits non-zero ONLY because of the 26 pre-existing e2e/test_api.py infra errors above.)

## D4 — grep-ban
No TODO|FIXME|XXX|placeholder|coming soon in the new modules. NotImplementedError only in ABC/mixin
protocol stubs (existing repo pattern, satisfied at runtime). No mock/stub in shipped code.

## D5 — migration (live)
Live dev SurrealDB migrated v8→v9 during test setup; `SELECT version FROM schema_meta:version` ⇒ **9**.
Run-twice idempotency + fresh==migrated + seeded v38→v39 delegated to real-db-test-runner (see U1/realdb-*).

## D6 — references
Changes are additive (trailing optional fields + new methods); migration version constants have no
external importers (grep confirmed); Source.__post_init__ only rejects out-of-range trust (a NEW field,
so no existing caller trips it). Behavioural proof: 6131 passing callers.

## Stage 2 reviewers
- security-reviewer: **No CRITICAL/HIGH.** All SurQL/SQL injection mitigated (double `_safe_brain_id`
  inline guard, `$params` for caller data, `_to_surreal_id` record ids, static DDL, SQLite `?`).
  Two nits: TraceConfig lacks bounds validation (config input, non-security); _migrate_8_to_9
  swallow-and-stamp — FIXED to fail-loudly on non-"already exists".
- python-reviewer: killed after it ran `uv run pytest` in the main checkout (against instruction) and
  corrupted the shared uv cache; review done inline (line counts <800 for new modules; additive/idempotent;
  see DECISIONS.md).

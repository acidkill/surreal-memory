# Lessons Learned — Neural Memory

Recurring mistakes and patterns to avoid. Updated after each audit cycle.

---

## 2026-02-24 — v2.8.1 Audit

### 1. Localhost Restriction Must Be Router-Level, Not Per-Endpoint

**What happened**: `memory_router`, `brain_router`, `dashboard_router` had zero localhost
protection. Only `hub`, `oauth`, and `openclaw` routers applied it — and each had their
own duplicate `_require_local_request` function.

**Root cause**: No shared dependency. Each developer copy-pasted the guard into their
own route file, so new routers forgot to add it.

**Fix**: Centralized `require_local_request` in `server/dependencies.py`. Applied at
router level via `dependencies=[Depends(require_local_request)]`.

**Rule**: Every new `APIRouter` in `server/routes/` MUST include
`dependencies=[Depends(require_local_request)]` unless explicitly designed for public access.
Add this to the PR checklist.

---

### 2. Brain Name Validation Must Be Consistent Across Layers

**What happened**: `unified_config.py` validated brain names with `^[a-zA-Z0-9_\-\.]+$`
AND `is_relative_to()`, but the REST `CreateBrainRequest` model only checked `max_length=100`.
A malicious name like `../../etc/passwd` could reach the config layer.

**Root cause**: Defense-in-depth was incomplete. The Pydantic model was the first validation
boundary but was the weakest.

**Fix**: Added `pattern=r"^[a-zA-Z0-9_\-\.]+$"` to `CreateBrainRequest.name` Field.

**Rule**: Input validation at the API boundary must be at least as strict as downstream
validation. Never rely solely on internal code to reject bad input — fail fast at the edge.

---

### 3. File Path Inputs Need `is_relative_to()` Guard

**What happened**: `db_train_handler.py` accepted a SQLite connection string, resolved it
with `Path.resolve()`, checked `is_file()` — but never checked if the resolved path was
within the working directory. An MCP client could read arbitrary `.db` files on the system.

**Root cause**: Path traversal prevention was applied in `unified_config.py` for brain
DB paths but not in the newer `db_train_handler.py`.

**Fix**: Added `is_relative_to(Path.cwd().resolve())` check before `is_file()`.

**Rule**: Any user-provided file path MUST be validated with:
```python
resolved = Path(user_input).resolve()
if not resolved.is_relative_to(allowed_base):
    return error
```
This applies to: connection strings, import paths, training paths, export paths.

---

### 4. Test Mocks Must Match Actual Code Flow

**What happened**: `test_transplant_nonexistent_source` mocked `find_brain_by_name` on
the storage object, but `_transplant` handler never calls `find_brain_by_name` — it calls
`get_shared_storage(brain_name=...)`. The mock did nothing, and the test hit real code
paths with `AsyncMock` objects cascading into `merge.py`, causing `TypeError`.

**Root cause**: Test was written against an assumed API contract, not the actual
implementation. The handler was refactored but the test wasn't updated.

**Fix**: Patched `neural_memory.unified_config.get_shared_storage` instead.

**Rule**: When writing test mocks, trace the actual code path to verify which functions
are called. Don't assume — read the handler code. After refactoring a handler, always
update its tests.

---

### 5. `_require_brain_id` Callers Must Handle ValueError

**What happened**: `_recall` handler called `_require_brain_id(storage)` without
try/except. When `_current_brain_id` was None, it raised `ValueError` instead of
returning `{"error": "No brain configured"}`.

**Root cause**: 7 handlers used the same `_require_brain_id → get_brain` pattern but
only some had error handling. The helper raises ValueError by design, but callers
were inconsistent about catching it.

**Fix**: Added try/except ValueError around `_require_brain_id` in `_recall`.

**Rule**: Every call to `_require_brain_id()` MUST be wrapped in try/except ValueError,
or the handler must be structured to guarantee a brain context exists before calling it.
Consider refactoring to a helper that returns `dict | str` (error dict or brain_id).

---

### 6. Optional Dependencies Need Graceful Test Skipping

**What happened**: `tests/test_encryption.py` imported `MemoryEncryptor` which imports
`cryptography` at runtime. CI installs `[dev,server]` but not `[encryption]`, so all
encryption tests failed with `ModuleNotFoundError`.

**Root cause**: Tests didn't account for optional extras not being installed in CI.

**Fix**: Added `pytest.mark.skipif(not _HAS_CRYPTOGRAPHY, ...)` on the test class.

**Rule**: Tests for optional-extra features MUST use `pytest.importorskip()` or
`@pytest.mark.skipif` to gracefully skip when the dependency is missing. CI matrix
should document which extras are installed.

---

### 7. Always Run Full CI Checks Locally Before Pushing

**What happened**: After fixing tests, pushed 3 separate commits because lint, format,
and mypy errors were discovered incrementally on CI.

**Root cause**: Only ran `pytest` locally, not `ruff check`, `ruff format --check`,
or `mypy`.

**Rule**: Before pushing, always run the full CI pipeline locally:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -x --timeout=60
```
Or use a pre-push hook / `/ship` skill that runs all checks.

---

## Checklist for New Features

- [ ] All new `APIRouter` instances have `dependencies=[Depends(require_local_request)]`
- [ ] All user-provided paths validated with `is_relative_to()`
- [ ] Brain names validated with `^[a-zA-Z0-9_\-\.]+$` at API boundary
- [ ] Tests for optional-extra features skip gracefully
- [ ] Test mocks verified against actual code flow (not assumed)
- [ ] `_require_brain_id()` calls wrapped in try/except ValueError
- [ ] Full lint + format + mypy + test run before push

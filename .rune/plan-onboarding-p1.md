# Phase 1: CLI — `nmem init --full` + `nmem doctor` ✅ Done

## Goal
One command sets up everything. One command diagnoses what's missing.

## Tasks

### `nmem init --full`
- [x] 1.1 Add `--full` flag to `init()` in `cli/commands/tools.py`
- [x] 1.2 Create `cli/full_setup.py` — orchestrates extended init:
  - Call existing `setup_config()`, `setup_brain()`, `setup_mcp_*()`, `setup_hooks_claude()`, `setup_skills()`
  - Auto-detect embedding provider (priority: installed sentence-transformers → GEMINI_API_KEY → Ollama running → prompt install)
  - Enable `embedding.enabled=true`, `dedup.enabled=true` in config.toml
  - Generate maintenance script (`~/.neuralmemory/maintenance.sh` or `.ps1`)
  - Print summary with guide URL
- [x] 1.3 `detect_embedding_provider()` — check importlib.util.find_spec + env vars
- [x] 1.4 `enable_config_defaults()` — patch config.toml with recommended defaults
- [x] 1.5 `generate_maintenance_script()` — platform-aware (bash/powershell), includes consolidate + health + decay

### `nmem doctor` (enhanced)
- [x] 1.6 Doctor command already existed — enhanced with 3 new checks
- [x] 1.7 Added checks: hooks (3/3 detection), dedup enabled, knowledge surface (.nm)
- [x] 1.8 Each check returns: OK/WARN/FAIL/SKIP + suggested fix command
- [x] 1.9 `--fix` flag: auto-fixes hooks, dedup, embedding provider
- [x] 1.10 Guide URL hint shown when issues found

### Tests
- [x] 1.11 test_full_setup.py — 15 tests (detect, config defaults, maintenance script, full flow)
- [x] 1.12 test_doctor_enhanced.py — 20 tests (hooks, dedup, surface, auto-fix, integration)

## Results
- 35 new tests, all passing
- mypy: 0 errors
- ruff: clean lint + format
- Doctor now has 11 checks (was 8)

## Files Touched
- `src/neural_memory/cli/commands/tools.py` — modified (--full flag, --fix flag)
- `src/neural_memory/cli/full_setup.py` — new (extended init orchestrator)
- `src/neural_memory/cli/doctor.py` — modified (3 new checks + auto-fix)
- `tests/unit/test_full_setup.py` — new
- `tests/unit/test_doctor_enhanced.py` — new

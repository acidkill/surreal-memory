# Phase 3: Cleanup + Tests

## Goal
Implement TTL-based auto-cleanup for expired ephemeral neurons, add CLI support, and write comprehensive tests.

## Tasks
- [ ] 3.1 Add `cleanup_ephemeral()` method to `sqlite_neurons.py` — `DELETE FROM neurons WHERE ephemeral = 1 AND created_at < ?` (respects TTL)
- [ ] 3.2 Hook ephemeral cleanup into `ExpiryCleanupHandler._run_expiry_cleanup()` — after typed_memory expiry, also delete expired ephemeral neurons
- [ ] 3.3 Add `ephemeral_ttl_hours` config field to `MaintenanceConfig` in `unified_config.py` (default 24)
- [ ] 3.4 Add `--ephemeral` flag to CLI `remember` command in `cli/main.py`
- [ ] 3.5 Write unit tests for Phase 1 — neuron creation, persistence, find_neurons filtering
- [ ] 3.6 Write unit tests for Phase 2 — MCP remember/recall with ephemeral, consolidation exclusion
- [ ] 3.7 Write unit tests for Phase 3 — TTL cleanup, config, CLI flag
- [ ] 3.8 Write integration test — full lifecycle: remember ephemeral -> recall -> expire -> verify gone
- [ ] 3.9 Update tool count in `test_mcp.py` and `test_tool_tiers.py` if schema params changed
- [ ] 3.10 Run full test suite, mypy, ruff — verify CI passes

## Acceptance Criteria
- [ ] Ephemeral neurons auto-deleted after TTL expires (default 24h)
- [ ] `cleanup_ephemeral()` removes neurons + cascades to synapses, fiber_neurons, neuron_states
- [ ] Config `ephemeral_ttl_hours` is respected
- [ ] CLI `nmem remember --ephemeral "temp note"` works
- [ ] All new tests pass (target: 15+ new tests)
- [ ] mypy + ruff clean
- [ ] No regressions in existing test suite

## Files Touched
- `src/neural_memory/storage/sqlite_neurons.py` — modify (cleanup method)
- `src/neural_memory/mcp/expiry_cleanup_handler.py` — modify (hook ephemeral cleanup)
- `src/neural_memory/unified_config.py` — modify (add config field)
- `src/neural_memory/cli/main.py` — modify (CLI flag)
- `tests/unit/test_ephemeral.py` — new (main test file)
- `tests/unit/test_mcp.py` — modify (tool count if needed)
- `tests/unit/test_tool_tiers.py` — modify (tool count if needed)

## Dependencies
- Phase 1 + Phase 2 must be complete

## Notes
- CASCADE DELETE on neurons table handles synapse/fiber_neurons cleanup automatically.
- `neuron_states` has FK to neurons — CASCADE should handle it. Verify.
- TTL cleanup runs on maintenance cycle (piggybacks on ExpiryCleanupHandler interval).
- CLI flag is low-priority but good for manual testing and non-MCP usage.
- Tool param counts may not change since we're adding params to existing tools, not new tools.

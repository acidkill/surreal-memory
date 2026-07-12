# U4 "Queryable Retrieval Traces" — Real-DB Evidence

**Branch:** feature/v290-pr4-retrieval-traces (worktree `run-006`)
**DB:** live dev SurrealDB v3 at `SURREALDB_URL=http://localhost:8001` (ns=`surreal_memory`, db=`default`), no mocks.
**Date:** 2026-07-12

## VERDICT: PASS

---

## 1. U1 regression suite (live pytest)

```
.venv/bin/python -m pytest tests/unit/test_surrealdb_retrieval_trace_live.py -p no:xdist -p no:cacheprovider -v
```

Result (run twice, before and after the smoke script, both green):

```
tests/unit/test_surrealdb_retrieval_trace_live.py::TestRetrievalTraceLive::test_id_round_trips PASSED [ 33%]
tests/unit/test_surrealdb_retrieval_trace_live.py::TestRetrievalTraceLive::test_find_by_fiber_and_query PASSED [ 66%]
tests/unit/test_surrealdb_retrieval_trace_live.py::TestTypedMemoryValidityLive::test_validity_round_trips PASSED [100%]
3 passed in ~1.4s
```

`FLEXIBLE can only be used in SCHEMAFULL tables` schema-init warnings present but cosmetic per instructions (pre-existing, non-fatal — schema still applies fine, all assertions pass).

## 2. U4 write + query + prune smoke (throwaway brain, live SurrealDBStorage)

Script executed via `.venv/bin/python - <<'PY' ... PY` style (saved standalone for reproducibility), using a fresh throwaway brain (`Brain.create` + `save_brain` + `set_brain`) on `SurrealDBStorage(url=SURREALDB_URL)`.

### (a) build_retrieval_trace -> add_retrieval_trace -> get/find round-trip

- Built a hand-made `RetrievalResult` (answer="Emma lives in Oslo", confidence=0.87, depth_used=CONTEXT, fibers_matched=[<real fiber id>], synthesis_method="associative") over a real `Neuron`+`Fiber` written to the live DB.
- `build_retrieval_trace(result, query="where does emma live", brain_id=<throwaway>, mode="associative", args={"tags": ["people"], "min_confidence": 0.5}, config_snapshot={"depth_default": "context"}, session_id="sess-u4-smoke")` produced:
  ```
  id=bd63017e-4033-4963-89bb-12f4a079bbbb fiber_ids=('900999be-4d8e-4f42-80c7-32dab853f2ee',) query='where does emma live'
  ```
- `storage.add_retrieval_trace(trace)` -> returned `bd63017e-...` (== `trace.id`, unmangled dashed uuid — U1 Bug-A regression still holds under U4 write path).
- `storage.get_retrieval_trace(id)` round-tripped exactly:
  ```json
  {"id": "bd63017e-4033-4963-89bb-12f4a079bbbb", "brain_id": "bd1db89b-...", "session_id": "sess-u4-smoke",
   "query": "where does emma live", "depth_used": 1, "mode": "associative", "confidence": 0.87,
   "latency_ms": 12.5, "anchor_ids": ["4e6c503d-..."], "retrievers": ["associative"],
   "fiber_ids": ["900999be-..."], "fiber_scores": [], "filters": {"min_confidence": 0.5, "mode": "associative", "tags": ["people"]},
   "config_snapshot": {"depth_default": "context"}, "trace_version": 1, "created_at": "2026-07-12T03:29:27.437739+00:00"}
  ```
- `storage.find_retrieval_traces(fiber_id="900999be-...")` -> found `['bd63017e-...']` (assert passed)
- `storage.find_retrieval_traces(query_contains="emma")` -> found `['bd63017e-...']` (assert passed)
- Raw row count via `SELECT count() FROM retrieval_trace WHERE brain_id = "<throwaway>" GROUP ALL`:
  - before any writes: `0`
  - after part (a) write: `1`

### (b) prune (retention_days=30 + max_traces=5000)

- Added `old_trace` with `created_at = utcnow() - timedelta(days=40)` and `fresh_trace` with `created_at = utcnow()` (both `fiber_ids=(<same fiber>,)`).
- Raw row count before prune: `3` (part-(a) trace + old_trace + fresh_trace).
- `storage.prune_retrieval_traces(retention_days=30, max_traces=5000)` -> returned `pruned=1`.
- Raw row count after prune: `2`.
- `storage.find_retrieval_traces(limit=100)` remaining ids: `{fresh_trace.id, part-(a)-trace.id}` — confirmed `old_trace.id` NOT present, `fresh_trace.id` and part-(a) trace still present. Exactly the 40-day-old trace was removed; the fresh one and the earlier still-valid trace survived.

### (c) MCPServer / ProvenanceHandler over live storage

Built a minimal `_FakeServer(ProvenanceHandler)` (same pattern as `tests/unit/test_provenance_traces.py`, but wired to the live `SurrealDBStorage` instead of `InMemoryStorage`):

- `server._provenance({"action": "traces", "fiber_id": <fiber.id>})` ->
  ```json
  {"traces": [
    {"id": "d2a53512-...", "query": "fresh trace", "mode": "", "depth_used": 0, "confidence": 0.0,
     "fiber_ids": ["900999be-..."], "session_id": null, "created_at": "2026-07-12T03:29:27.795136+00:00"},
    {"id": "bd63017e-...", "query": "where does emma live", "mode": "associative", "depth_used": 1,
     "confidence": 0.87, "fiber_ids": ["900999be-..."], "session_id": "sess-u4-smoke",
     "created_at": "2026-07-12T03:29:27.437739+00:00"}
  ], "count": 2}
  ```
  Compact shape confirmed (`id, query, mode, depth_used, confidence, fiber_ids, session_id, created_at` all present); both surviving traces for this fiber returned, newest-first.
- `server._provenance({"action": "trace_get", "trace_id": "bd63017e-..."})` -> `{"trace": {...full RetrievalTrace.to_dict() shape...}}` with all 14 expected keys present (`id, brain_id, session_id, query, depth_used, mode, confidence, latency_ms, anchor_ids, retrievers, fiber_ids, fiber_scores, filters, config_snapshot, trace_version, created_at`).

## 3. Raw DB state before/after (proves real rows, not mocked)

Query used: `store._query('SELECT count() AS c FROM retrieval_trace WHERE brain_id = "<lit>" GROUP ALL')` (mirrors `_count_retrieval_traces` used internally by `prune_retrieval_traces`).

| Point | brain-scoped count |
|---|---|
| before any writes (fresh throwaway brain) | 0 |
| after part (a) write | 1 |
| after part (b) setup (old + fresh added) | 3 |
| after `prune_retrieval_traces(retention_days=30, max_traces=5000)` | 2 |
| after final cleanup (`prune_retrieval_traces(max_traces=0)`) | 0 |

Global sanity check after full script + cleanup (all brains, whole `retrieval_trace` table on the shared dev DB):
```sql
SELECT count() AS c FROM retrieval_trace GROUP ALL
-- -> [{'c': 0}]
```
Confirms the smoke run left no residue on the shared dev DB.

## Commands run (exact)

```bash
# 1. U1 regression (live)
.venv/bin/python -m pytest tests/unit/test_surrealdb_retrieval_trace_live.py -p no:xdist -p no:cacheprovider -v

# 2. U4 smoke (write/query/prune/mcp) — throwaway async script, single process
.venv/bin/python - <<'PY'
# ... (script: Brain.create/save_brain/set_brain on SurrealDBStorage(url=SURREALDB_URL);
#      build_retrieval_trace -> add_retrieval_trace -> get_retrieval_trace/find_retrieval_traces;
#      prune_retrieval_traces(retention_days=30, max_traces=5000);
#      _FakeServer(ProvenanceHandler) -> _provenance traces/trace_get over live storage;
#      cleanup via prune_retrieval_traces(max_traces=0))
PY

# 3. Raw row confirmation (before/after, brain-scoped and global)
.venv/bin/python - <<'PY'
# SELECT count() AS c FROM retrieval_trace GROUP ALL  (global, post-cleanup -> 0)
PY
```

## Notes

- No SurrealDB mocking anywhere in this validation — `SurrealDBStorage` connected directly to the live dev instance for both the pytest file and the smoke script.
- Schema init prints `FLEXIBLE can only be used in SCHEMAFULL tables` warnings on every fresh connection — pre-existing, cosmetic, ignored per instructions; does not affect any assertion above.
- `_to_surreal_id` dash-folding + `_orig_id` payload stash (U1 Bug A fix) verified to still hold under the U4 write path (`add_retrieval_trace` returns the original dashed uuid, `get_retrieval_trace` round-trips it exactly).
- `find_retrieval_traces(fiber_id=...)` correctly used the `$fiber_id IN fiber_ids` SurQL clause against real array-column data (not a post-filter), and `query_contains` correctly applied as a Python post-filter over over-fetched candidates — both verified against real rows.
- `prune_retrieval_traces` correctly discriminated by `created_at` age (40-day-old trace pruned, same-fiber same-session-shape fresh trace kept) — proves the age comparison runs against real stored `created_at` timestamps, not a mock clock.

## VERDICT: PASS

# U3 — Per-Fact Supersession: Live SurrealDB Evidence

**Repo/worktree:** `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
**Branch:** `feature/v290-pr3-supersession`
**Live DB:** `SURREALDB_URL=http://localhost:8001` (real running SurrealDB v3, NS=`surreal_memory`, DB=`default`)
**Date:** 2026-07-12

## Verdict: **PASS**

All three requested pytest invocations pass in full (32/32 tests, 0 failures, 0 errors,
0 skips), and the additional live smoke script exercising the exact auto-hook/backfill
code path (`resolve_fibers_for_neurons` → underscore-form fiber id → `supersede_typed_memory`
→ dash-id re-read) passes with real-DB row evidence confirmed via a direct raw SurrealQL
query against the live container (bypassing the Python driver entirely).

---

## 1. NEW live regression: `test_surrealdb_supersession_live.py`

```
.venv/bin/python -m pytest tests/unit/test_surrealdb_supersession_live.py -p no:xdist -p no:cacheprovider -v
```

```
tests/unit/test_surrealdb_supersession_live.py::TestSupersessionLive::test_supersede_persists_and_reads_back PASSED [ 33%]
tests/unit/test_surrealdb_supersession_live.py::TestSupersessionLive::test_supersede_is_idempotent PASSED [ 66%]
tests/unit/test_surrealdb_supersession_live.py::TestSupersessionLive::test_resolve_fibers_for_neurons_live PASSED [100%]

============================== 3 passed in 1.32s ===============================
```

Covers: supersede persists A-side validity + `SUPERSEDES` synapse + reads back for both
dash and underscore fiber-id forms; idempotency (second call returns `superseded=False`);
`resolve_fibers_for_neurons` resolution consistency.

## 2. Unit-level supersession suite (InMemoryStorage)

```
.venv/bin/python -m pytest tests/unit/test_supersession.py tests/unit/test_supersession_auto_hook.py \
  tests/unit/test_conflict_handler_supersession.py tests/unit/test_provenance_supersession_lineage.py \
  tests/unit/test_lifecycle_backfill_supersession.py tests/unit/test_recall_supersession.py \
  -p no:xdist -p no:cacheprovider -v
```

```
22 passed in 0.14s
```

All 22 tests passed (TestSupersedeTypedMemory, TestResolveFibersForNeurons,
TestApplySupersessions, TestConflictStepCollectsSupersessions,
TestManualKeepNewSupersession, TestSupersessionLineage, TestBackfillSupersession,
TestRecallSupersessionFilter). Full list captured in raw run log (see command above);
no failures/errors.

## 3. Regression — prior storage fixes still hold on live DB

```
.venv/bin/python -m pytest tests/unit/test_surrealdb_recordid_fix_live.py \
  tests/unit/test_surrealdb_fiber_id_norm_live.py tests/unit/test_surrealdb_retrieval_trace_live.py \
  -p no:xdist -p no:cacheprovider -v
```

```
tests/unit/test_surrealdb_recordid_fix_live.py::TestSourceRecordIdFix::test_get_source_finds_row_and_trust_round_trips PASSED [ 14%]
tests/unit/test_surrealdb_recordid_fix_live.py::TestSourceRecordIdFix::test_update_source_matches PASSED [ 28%]
tests/unit/test_surrealdb_recordid_fix_live.py::TestSourceRecordIdFix::test_delete_source_matches PASSED [ 42%]
tests/unit/test_surrealdb_fiber_id_norm_live.py::TestFiberIdNormalization::test_typed_memory_resolves_both_id_forms PASSED [ 57%]
tests/unit/test_surrealdb_retrieval_trace_live.py::TestRetrievalTraceLive::test_id_round_trips PASSED [ 71%]
tests/unit/test_surrealdb_retrieval_trace_live.py::TestRetrievalTraceLive::test_find_by_fiber_and_query PASSED [ 85%]
tests/unit/test_surrealdb_retrieval_trace_live.py::TestTypedMemoryValidityLive::test_validity_round_trips PASSED [100%]

============================== 7 passed in 3.11s ===============================
```

**Total across all 3 pytest invocations: 32 passed, 0 failed, 0 errors.**

### Pre-existing cosmetic warnings (NOT failures)

Confirmed via `--log-cli-level=WARNING` on the live-DB runs — these fire on every test
during schema init, are unrelated to supersession logic, and are pre-existing/expected:

```
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON neuron TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON fiber TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD config ON brain TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON brain TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON typed_memory TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON source TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON alerts TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
WARNING  surreal_memory.storage.surrealdb.schema:schema.py:516 Schema statement failed: DEFINE FIELD metadata ON brain_versions TYPE object FLEXIBLE DEFAULT {} (An error occurred: FLEXIBLE can only be used in SCHEMAFULL tables)
```

No other warnings observed (`-W default` sweep on the live files produced zero pytest
`UserWarning`/`DeprecationWarning` output). No mock-stubbing pollution occurred — each
`*_live.py` invocation was run in a fresh process, per the hard rules, and no
`AsyncSurreal` ImportError was hit at any point in this run.

---

## 4. Additional smoke: raw script proving end-to-end lineage persistence

Ran via `.venv/bin/python - <<'PY' ... PY` (fresh process, real `SurrealDBStorage`,
no mocks) reproducing the exact auto-hook/backfill path: create two facts, resolve the
OLD fiber id via `resolve_fibers_for_neurons` (which returns the **underscore**
sanitized form used internally), call `supersede_typed_memory` with that underscore
form, then re-read the OLD `typed_memory` by its **original dash id**.

### Script output

```
old.id (dash form): 9cc287ca-17bd-4b02-a479-b944072d948e
old.anchor_neuron_id: 7bdf5034-b580-4e8f-8590-93a22d6c1292
resolved old fiber id (underscore form): 9cc287ca_17bd_4b02_a479_b944072d948e
outcome.superseded: True
tm_dash: TypedMemory(fiber_id='9cc287ca-17bd-4b02-a479-b944072d948e', ...,
  valid_until=datetime.datetime(2026, 7, 12, 2, 35, 34, 28702),
  superseded_by='1484b8a8-0a23-4703-b640-4791fcade1bf')
SMOKE TEST PASSED: dash-id re-read shows superseded_by == 1484b8a8-0a23-4703-b640-4791fcade1bf valid_until == 2026-07-12 02:35:34.028702
SUPERSEDES synapse present: True
ALL SMOKE ASSERTIONS PASSED
```

`new.id` for this run was `1484b8a8-0a23-4703-b640-4791fcade1bf` — matches
`tm_dash.superseded_by` exactly, confirming the lineage write is visible when read back
by the **original dash id**, even though the write path used the internal underscore
form obtained from `resolve_fibers_for_neurons` (this is the exact bug class B3-1/B3-2
fixed prior guarded against: dash vs underscore fiber-id round-trip mismatches).

### Real DB evidence — raw SurrealQL against the live container (Python driver bypassed)

Query 1 — `typed_memory` row for the OLD fiber, by direct SQL over HTTP (`/sql` endpoint,
NS=`surreal_memory`, DB=`default`):

```bash
curl -s -u "$SURREALDB_USER:$SURREALDB_PASS" -X POST "$SURREALDB_URL/sql" \
  -H "surreal-ns: $SURREALDB_NS" -H "surreal-db: $SURREALDB_DB" \
  --data "SELECT fiber_id, superseded_by, valid_until, tier FROM typed_memory WHERE string::contains(<string>fiber_id, '9cc287ca');"
```

Result:
```json
[{"result":[{"fiber_id":"9cc287ca-17bd-4b02-a479-b944072d948e","superseded_by":"1484b8a8-0a23-4703-b640-4791fcade1bf","tier":"warm","valid_until":"2026-07-12T02:35:34.028702Z"}],"status":"OK"}]
```

This is byte-for-byte consistent with the Python-level assertion above — the DB row
itself (not a mock, not a driver cache) shows `superseded_by` pointing at the new fiber
and `valid_until` stamped.

Query 2 — `SUPERSEDES` synapse row, new-anchor → old-anchor:

```bash
curl -s -u "$SURREALDB_USER:$SURREALDB_PASS" -X POST "$SURREALDB_URL/sql" \
  -H "surreal-ns: $SURREALDB_NS" -H "surreal-db: $SURREALDB_DB" \
  --data "SELECT type, in, out FROM synapse WHERE type = 'supersedes' AND string::contains(<string>out, '7bdf5034');"
```

Result:
```json
[{"result":[{"in":"neuron:3b9b1c3c_1776_45a8_8309_613f644c9120","out":"neuron:7bdf5034_b580_4e8f_8590_93a22d6c1292","type":"supersedes"}],"status":"OK"}]
```

`in` = new anchor neuron, `out` = old anchor neuron — confirms `SUPERSEDES` synapse
direction (new → old) persisted server-side.

Query 3 — table row counts at time of evidence capture (proves live data, not empty/mock DB):

```bash
curl -s -u "$SURREALDB_USER:$SURREALDB_PASS" -X POST "$SURREALDB_URL/sql" \
  -H "surreal-ns: $SURREALDB_NS" -H "surreal-db: $SURREALDB_DB" \
  --data "SELECT count() FROM typed_memory GROUP ALL; SELECT count() FROM fiber GROUP ALL; SELECT count() FROM synapse WHERE type='supersedes' GROUP ALL;"
```

Result: `typed_memory` count=2601, `fiber` count=5604, `synapse[type=supersedes]` count=12
(this smoke run's synapse is one of the 12).

---

## Summary Table

| Test file | Tests | Pass | Fail | Error |
|---|---|---|---|---|
| `test_surrealdb_supersession_live.py` | 3 | 3 | 0 | 0 |
| `test_supersession.py` + 5 other unit files | 22 | 22 | 0 | 0 |
| `test_surrealdb_recordid_fix_live.py` + 2 other live files | 7 | 7 | 0 | 0 |
| Smoke script (raw, live DB, no mocks) | 1 (multi-assert) | pass | — | — |
| **Total pytest** | **32** | **32** | **0** | **0** |

## Rule compliance notes

- All live-DB pytest invocations used `.venv/bin/python -m pytest` — never `uv`/`uv run`.
- All live-DB invocations ran with `-p no:xdist -p no:cacheprovider` (serial, no shared-container races).
- No single invocation mixed `test_surrealdb_store.py` (which stubs `surrealdb`) with a
  `*_live.py` file — each family was its own pytest process, per the hard rules. No
  `AsyncSurreal` ImportError was encountered.
- No coverage / `make test-cov` run against the whole suite — avoided the known
  MagicMock-stub leak into `tests/e2e/test_api.py`.

## FINAL VERDICT: PASS

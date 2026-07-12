# U1 — Schema v9 — Real-DB verification (run-006)

Live SurrealDB: `http://localhost:8001` (container `surreal-memory-surrealdb-1`, SurrealDB
**3.2.0**), NS=`surreal_memory` DB=`default`. All checks below ran against this real,
shared dev instance — no SurrealDB connection was mocked anywhere in items 1–3.
Item 4 legitimately uses a mocked `surrealdb` module for `test_surrealdb_migrations.py`
(pure migration-logic unit test, matching the ticket's own description
"scripted 8→9 migration idempotency **unit** tests") and a real embedded SQLite DB
for `test_sqlite_v9_storage.py` — neither is an invalid e2e-mock substitution.

## Verdict summary

| # | Item | Verdict |
|---|------|---------|
| 1 | Migration 8→9 run-twice idempotency (live) | **PASS** |
| 2 | Fresh ensure_schema == migrated (live shape check) | **PASS** |
| 3 | Live round-trip of new v9 code (no mocks) | **PARTIAL FAIL** — see diagnosis |
| 4 | SQLite 38→39 + scripted migration idempotency (embedded/unit) | **PASS** |

---

## Item 1 — Migration 8→9 run-twice idempotency (PASS)

Live DB was observed to fluctuate between stamped version 8 and 9 during this
session (see "Note on shared-DB concurrency" below) — not caused by this script,
which never writes until after its first read. The idempotency property was
verified regardless of which stamp was observed pre-run: `apply_migrations`
converges to 9 with no exception, and re-running from 9 is a genuine no-op.

Script: `.venv/bin/python <script running apply_migrations()/_migrate_8_to_9() 4x>`
(async, direct `surrealdb.AsyncSurreal` connection, same URL/NS/DB/creds as the app).

```
PRE stamped version (observed, shared live DB): 9
RUN1 apply_migrations returned: 9
RUN1 schema_meta:version: 9
RUN1 detect_db_version: 9
RUN2 apply_migrations returned: 9
RUN2 schema_meta:version: 9
RUN2 detect_db_version (no-op confirmed): 9
DIRECT _migrate_8_to_9 run1 version: 9
DIRECT _migrate_8_to_9 run2 version: 9
ITEM1: PASS - no exceptions across 4 total invocations; version converged/stable at 9
```

A first attempt (same script) observed `PRE stamped version: 8` (see note below),
and confirmed the non-no-op path too:

```
PRE stamped version (observed, shared live DB): 8
RUN1 apply_migrations returned: 9
RUN1 schema_meta:version: 9
RUN1 detect_db_version: 9
RUN2 apply_migrations returned: 9
RUN2 schema_meta:version: 9
RUN2 detect_db_version (no-op confirmed): 9
DIRECT _migrate_8_to_9 run1 version: 9
DIRECT _migrate_8_to_9 run2 version: 9
ITEM1: PASS - no exceptions across 4 total invocations; version converged/stable at 9
```

Both real invocations (from 8, and from 9) ran `apply_migrations()` twice + the raw
`_migrate_8_to_9()` twice more back-to-back with **zero exceptions**, and
`schema_meta:version` / `detect_db_version` were 9 after every call.

### Note on shared-DB concurrency (real, observed)

`docker exec surreal-memory-surrealdb-1 /surreal sql ... "SELECT version FROM
schema_meta:version;"` returned **9** on the first read of this session, then
**8** consistently for the next ~12s of polling (`schema_meta:migration_state`
showed `phase: 'done', old_count: 181562, converted: 181562` throughout — i.e.
the real, already-completed 7→8 synapse migration over Toni's actual production
graph, not a reset). No pytest/migration process was found running locally
(`ps aux`) during that window. This is consistent with another concurrent
process touching the same shared dev container (this instance is also the
live backing store for the running `smem` app — `ps aux` shows an active
`uvicorn surreal_memory.server.app:app` process against the same DB). Nothing
in this diff's code path writes 8 except the real `_migrate_7_to_8` phase-verify
step, and no such migration was actively running from this session. Flagging
as observed shared-infra non-determinism, not a defect in the code under test —
the idempotency guarantee held from both observed starting points.

---

## Item 2 — Fresh ensure_schema == migrated (live shape check) (PASS)

`INFO FOR DB` — `retrieval_trace` table present:
```
retrieval_trace: 'DEFINE TABLE retrieval_trace TYPE NORMAL SCHEMAFULL PERMISSIONS NONE',
```

`INFO FOR TABLE typed_memory` — validity fields + indexes present:
```
fields: {
  ...
  superseded_by: 'DEFINE FIELD superseded_by ON typed_memory TYPE none | string PERMISSIONS FULL',
  valid_from: 'DEFINE FIELD valid_from ON typed_memory TYPE none | datetime PERMISSIONS FULL',
  valid_until: 'DEFINE FIELD valid_until ON typed_memory TYPE none | datetime PERMISSIONS FULL'
  ...
},
indexes: {
  idx_typed_brain: 'DEFINE INDEX idx_typed_brain ON typed_memory FIELDS brain_id',
  idx_typed_expires: 'DEFINE INDEX idx_typed_expires ON typed_memory FIELDS brain_id, expires_at',
  idx_typed_fiber: 'DEFINE INDEX idx_typed_fiber ON typed_memory FIELDS brain_id, fiber_id UNIQUE',
  idx_typed_type: 'DEFINE INDEX idx_typed_type ON typed_memory FIELDS brain_id, memory_type',
  idx_typed_valid: 'DEFINE INDEX idx_typed_valid ON typed_memory FIELDS brain_id, valid_until'
}
```

`INFO FOR TABLE source` — `trust` field present:
```
trust: 'DEFINE FIELD trust ON source TYPE none | float PERMISSIONS FULL',
```

`INFO FOR TABLE retrieval_trace` — full shape as designed:
```
fields: { brain_id, confidence, created_at, depth_used, fiber_ids, fiber_ids.*, id,
          latency_ms, mode, payload (FLEXIBLE), query, session_id }
indexes: { idx_trace_brain: FIELDS brain_id, idx_trace_created: FIELDS brain_id, created_at }
```

All required v9 objects (fields + indexes) exist on the live, already-migrated DB —
matches `schema.SCHEMA_SQL` (fresh-build) exactly, per the code comment in
`_migrate_8_to_9` ("Mirrors the statements in schema.SCHEMA_SQL so a fresh
ensure_schema build and a migrated DB converge"). Confirmed live: **PASS**.

---

## Item 3 — Live round-trip of new v9 code (PARTIAL FAIL)

Ran the required existing suite first:

```
$ .venv/bin/python -m pytest tests/unit/test_surrealdb_typed_memory_all_types.py -p no:xdist -q
============================= test session starts ==============================
collected 31 items
tests/unit/test_surrealdb_typed_memory_all_types.py ...........................
============================== 31 passed in 10.70s ==============================
```

Then wrote a focused integration script (`SurrealDBStorage(url=SURREALDB_URL)`, real
connection, throwaway brain `u1-realdb-qa-throwaway`, cleaned up at the end — see
"Cleanup" below). Results:

```
[PASS] TypedMemory round-trip: found
[PASS] TypedMemory.valid_until survives
[PASS] TypedMemory.superseded_by survives
[PASS] TypedMemory.valid_from is None (unset)
[PASS] get_typed_memories_batch returns both fibers
[PASS] get_expiring_memories includes the 3-day-expiry fiber
[PASS] get_expiring_memories excludes the open-ended (no expires_at) fiber
[FAIL] Source round-trip: found
[PASS] add_retrieval_trace returns trace.id
[PASS] get_retrieval_trace: found
[PASS] RetrievalTrace.query survives
[PASS] RetrievalTrace.mode survives
[PASS] RetrievalTrace.confidence survives
[PASS] RetrievalTrace.fiber_ids survives
[PASS] RetrievalTrace.anchor_ids survives (payload field)
[PASS] RetrievalTrace.retrievers survives (payload field)
[FAIL] find_retrieval_traces(fiber_id=...) finds our trace -- got ['f646cdb7_cbe8_4be1_bc3e_fec0e7b370e7']
[FAIL] find_retrieval_traces(query_contains=...) finds our trace -- got ['f646cdb7_cbe8_4be1_bc3e_fec0e7b370e7']
[PASS] prune_retrieval_traces(max_traces=2) removed exactly 1 (had 3)
[FAIL] prune_retrieval_traces kept the 2 newest of our 3 -- got set()
[PASS] prune_retrieval_traces dropped the oldest (original trace)
[PASS] prune_retrieval_traces(retention_days=30) removed the 100-day-old trace
```

**TypedMemory** (validity fields, batch fetch, expiring-memories filter) is fully
correct end-to-end on the live DB. **RetrievalTrace's data fields** (query, mode,
confidence, fiber_ids, and the FLEXIBLE-payload fields anchor_ids/retrievers) and
its **filter semantics** (`fiber_id` array-membership, `query_contains` substring,
`retention_days`/`max_traces` prune counts) are all correct. Two distinct real bugs
were found and confirmed against live DB state:

### Bug A (in scope of this diff): `RetrievalTrace.id` does not round-trip through SurrealDB

Root cause, confirmed with a minimal repro against the live DB:

```
ORIGINAL trace.id  = a9efe9d0-5403-43eb-97ae-d3532b5b1105
FETCHED fetched.id = a9efe9d0_5403_43eb_97ae_d3532b5b1105
EQUAL? False

RAW stored id field: RecordID(table_name=retrieval_trace, record_id='a9efe9d0_5403_43eb_97ae_d3532b5b1105')
str(): retrieval_trace:a9efe9d0_5403_43eb_97ae_d3532b5b1105
```

`src/surreal_memory/storage/surrealdb/retrieval_trace.py::add_retrieval_trace` never
stores the original `trace.id` (which always contains dashes — `uuid4()` default) as
its own scalar column; it only uses the **sanitized** form
(`_to_surreal_id(trace.id)`, dashes→underscores) as the record's native identity
(`UPSERT retrieval_trace:{sid} CONTENT $data`, and `$data` has no `id` key). On read,
`_row_to_retrieval_trace` (line ~38) derives `.id` from that native RecordID:
`trace_id = str(raw_id).rsplit(":", 1)[-1]` — which is the **sanitized**,
underscore form, not the original.

This is why `find_retrieval_traces` "found our trace" but with a mangled id
(`f646cdb7_cbe8_4be1_bc3e_fec0e7b370e7` instead of the dashed original) — the
array-membership (`$fiber_id IN fiber_ids`) and substring (`query_contains`)
**filters themselves are correct** (each query legitimately returned exactly the
1 matching trace); only the returned object's `.id` field is wrong. The
"kept the 2 newest of our 3: got set()" failure is a cascading symptom of the
same root cause (my check compared fetched `.id`s against the original dashed
ids), **not** independent evidence that pruning itself misbehaves — `prune_retrieval_traces`
operates entirely on the native RecordID (`id NOT IN (SELECT VALUE id FROM ...)`)
so its row-count math is unaffected; the separately-verified checks
(`removed exactly 1 (had 3)`, `dropped the oldest`, `retention_days` count) confirm
prune's *count* behavior is correct.

**Confirms this is backend-specific, not a design requirement**: the sibling
SQLite implementation in the *same* PR (`src/surreal_memory/storage/sqlite_retrieval_trace.py`)
stores `id` as its own column (`row["id"]` used directly, no sanitization needed)
and does **not** have this bug — `tests/unit/test_sqlite_v9_storage.py` passes
cleanly (see Item 4) including its retrieval-trace round trip. The established
in-repo pattern for preserving a caller id through SurrealDB's dash-folding is
already used by `typed_memory.py` (`fiber_id` is stored as its own scalar column,
independent of the sanitized record identity) — `retrieval_trace.py` does not
follow that pattern for `id`.

**Impact:** any caller that does `id = await storage.add_retrieval_trace(trace)`
then later expects `get_retrieval_trace(id)` / `find_retrieval_traces(...)` results
to expose that same `id` value back (e.g. for logging, correlation, or a
"delete this specific trace" flow) will observe a different id than what was
returned by `add_retrieval_trace`. `get_retrieval_trace(trace_id)` itself still
*works* for lookup-by-id (because it re-sanitizes the input the same way before
building the record path), so a caller who always re-derives ids through the
same sanitizer never notices — but the returned `RetrievalTrace.id` attribute
is not the value that was written in.

**Recommendation:** add `"id": trace.id` to `record_data` in `add_retrieval_trace`
(mirrors `typed_memory.py`'s `fiber_id` column pattern) and prefer that column
in `_row_to_retrieval_trace` (falling back to the derived-from-RecordID form only
for defensiveness). This is a small, single-file, mechanical fix — not
architectural — but it should go through the normal TDD/code-review path (add a
regression test asserting `fetched.id == trace.id`), not a blind patch from this
verification pass.

### Bug B (pre-existing, NOT introduced by this diff): `SurrealDBSourcesMixin.get_source` never matches

```
source.id = e5a973f1-7547-4a3c-a118-84273ba32343
add_source returned: e5a973f1-7547-4a3c-a118-84273ba32343
RAW row via direct record select: [{'trust': 0.8, 'id': RecordID(table_name=source,
  record_id='e5a973f1_7547_4a3c_a118_84273ba32343'), ...}]   <- row genuinely exists, trust=0.8 stored correctly
get_source() result: None                                     <- but get_source can't find it
```

Confirmed root cause directly in SurQL against the live DB:
```
LET $sid = "e5a973f1_7547_4a3c_a118_84273ba32343";
SELECT * FROM source WHERE id = $sid;                          -> []   (no match)
SELECT * FROM source WHERE id = type::record("source", $sid);  -> [{ ...the row... }]  (matches)
SELECT * FROM source WHERE meta::id(id) = $sid;                -> [{ ...the row... }]  (matches)
```

SurrealDB 3.2.0 does not implicitly compare a bare string parameter equal to a
`RecordID`-typed `id` field — `get_source`'s query
(`sources.py:124-128`, `"... AND id = $sid"`, `sid=_to_surreal_id(source_id)`)
has therefore **never** matched a real row on this backend. `update_source` and
`delete_source` use the identical pattern and are equally affected. `git diff
d25b98f -- src/surreal_memory/storage/surrealdb/sources.py` confirms this PR only
added the two `trust` lines (`_row_to_source` return + `record_data`) — the
`get_source`/`update_source`/`delete_source` WHERE clauses are byte-for-byte
unchanged, so this bug **pre-dates schema-v9** and is out of scope for this
diff's review, but it directly **blocked verification of the new `Source.trust`
round-trip** as requested by the ticket (there is no working `get_source` path
to read it back through). Verified the underlying data is correct via a raw
`SELECT * FROM source:{sid}` (see above: `trust: 0.8` stored exactly as written) —
the storage/serialization side of `Source.trust` is fine; only the **pre-existing**
retrieval path is broken.

**Recommendation:** file as a separate, pre-existing defect (not a schema-v9
regression) — fix `get_source`/`update_source`/`delete_source` to use
`type::record($table, $sid)` or `meta::id(id) = $sid` instead of the bare
`id = $sid` comparison. Recommend Opus review before touching this, since
`_to_surreal_id`/id-comparison conventions are used across ~13 files in this
codebase per its own docstring ("Single hardened source for SurrealDB record-id
sanitisation") — a fix here should confirm no other mixin relies on the same
(seemingly always-broken) comparison pattern working, and should ship with a
live-DB regression test since no existing test caught this.

### Cleanup

All throwaway rows/brains created during this verification were deleted from the
live DB at the end of each script run (fixed with a follow-up `DELETE brain WHERE
name = '...'` where the mangled brain-id delete inside one debug script's cleanup
path missed, due to the same dash/underscore id convention investigated above).
Confirmed via `SELECT count() FROM brain WHERE name CONTAINS 'u1-debug' OR name =
'u1-realdb-qa-throwaway' GROUP ALL` → `{ count: 0 }`.

---

## Item 4 — SQLite 38→39 + scripted 8→9 migration idempotency (unit, embedded) (PASS)

```
$ .venv/bin/python -m pytest tests/unit/test_sqlite_v9_storage.py tests/unit/test_surrealdb_migrations.py -p no:xdist -q
============================= test session starts ==============================
collected 35 items
tests/unit/test_sqlite_v9_storage.py .........                           [ 25%]
tests/unit/test_surrealdb_migrations.py ..........................       [100%]
============================== 35 passed in 0.11s ==============================
```

`test_sqlite_v9_storage.py` exercises the **real embedded SQLite backend** (fresh
v39 bootstrap; `SQLiteStorage` against a real temp-dir `.db` file, no mocks) for
TypedMemory validity, `Source.trust`, and retrieval-trace CRUD/find/prune — and
this is exactly the implementation that does NOT have Bug A above (it stores
`id` as its own column). `test_surrealdb_migrations.py` intentionally mocks the
`surrealdb` module (`sys.modules["surrealdb"] = MagicMock()`) — per its own
comment this is **pure migration state-machine logic** testing (mirrors
`sqlite_schema` migration tests), matching the ticket's explicit framing of this
file as a "unit test", not a live-DB assertion — not a case of the e2e/live path
being invalidly mocked.

---

## Root DB state snapshot (for reference)

```
SELECT version FROM schema_meta:version;   -> 9 (after item 1's runs)
schema_meta:migration_state -> { converted: 181562, copied: 181562, old_count: 181562,
                                 phase: 'done', skipped: 0 }   (real production 7->8 data, untouched)
```

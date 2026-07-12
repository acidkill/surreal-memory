# UB1 — SurrealDB RecordID-vs-string `id = $param` fix

Branch: `fix/surrealdb-recordid-comparison` (from U1 HEAD). Pre-existing bug found by U1's
real-db-test-runner; blocks U2 (trust resolution reads `Source.trust` via `get_source`).

## Root cause (confirmed live)
SurrealDB 3.2.0 does not match a **bare** string id against a `RecordID`-typed `id` field, but
**does** match a string that carries the `table:` prefix. Live proof (namespace surreal_memory):
```
SELECT id FROM schema_meta WHERE id = 'version';                 -> [[]]                 (bare: NO match)
SELECT id FROM schema_meta WHERE id = 'schema_meta:version';     -> [{ id: schema_meta:version }]  (prefixed: MATCH)
SELECT id FROM schema_meta WHERE id = type::record('schema_meta','version'); -> MATCH
```

## Scope — SOURCE ONLY (audited every `id = $param` site)
- `sources.py` get_source(:124) / update_source(:169) / delete_source(:201) passed a **bare**
  `sid = _to_surreal_id(source_id)` (no `source:` prefix) → **BROKEN** (get→None, update/delete→False,
  even though the row exists and stores `trust` correctly). **FIXED.**
- `alerts.py` (:166,:186,:239) pass `rid=f"alerts:{sid}"` — **prefixed → works, NOT broken.**
- `versions.py` (:139,:186) pass `rid=f"brain_versions:{sid}"` — prefixed → not broken.
- `cognitive.py` (:419,:450) pass `rid=f"knowledge_gaps:{sid}"` — prefixed → not broken.
- `typed_memory.py` (:443) passes `sid=f"fiber:{...}"` — prefixed → not broken.
- `store.py` (:1347) brain uses a plain-string id + Python-side filter — not this pattern.
So the real-db agent's "~13 files convention" caution resolved to a **single** genuinely-broken file
(source), because only source passed an unprefixed id.

## Fix
`sources.py`: `AND id = $sid` → `AND id = type::record('source', $sid)` in get/update/delete_source
(single-quoted SurQL literal inside the double-quoted Python string; value stays a bound `$param`,
so no injection surface changes; `type::record` uses the id index).

## Verification
- New live regression: `tests/unit/test_surrealdb_recordid_fix_live.py` (skipif not SURREALDB_URL) —
  add_source(trust=0.8) → get_source finds it & trust==0.8; update_source(status='superseded')→True &
  persisted; delete_source→True & gone. **3/3 passed** serial vs live SurrealDB 3.2.0.
- Combined serial live run (recordid-fix + U1 retrieval-trace-live + typed_memory_all_types): **37 passed**.
- Regression: `test_source_registry.py` 34 passed; full unit `-n auto` no-DB **6105 passed, 0 fail**.
- `mypy sources.py` clean.

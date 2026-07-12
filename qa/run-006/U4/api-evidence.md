# U4 — smem_recall TRACE tool contract — real MCPServer over live SurrealDB v3

**Scope:** RECALL PERSIST contract only (storage + smem_provenance traces/trace_get
already verified separately in `tests/unit/test_storage_retrieval_traces.py` and
`tests/unit/test_surrealdb_retrieval_trace_live.py`). This run exercises a real,
in-process `MCPServer` (no mocks) built against the live dev SurrealDB.

## Setup

- Repo/worktree: `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
- Branch: `feature/v290-pr4-retrieval-traces` (HEAD `a51a349`)
- Live DB: `SURREALDB_URL=http://localhost:8001` — confirmed via `curl -s http://localhost:8001/version` → `surrealdb-3.2.0`
- `SURREAL_MEMORY_STORAGE=surrealdb`, `SURREALDB_NS=surreal_memory`, `SURREALDB_DB=default` (from shell env, unchanged)
- Throwaway brain: `SURREAL_MEMORY_BRAIN=qa-u4-c016e112` set **before** `get_config()`/`MCPServer()` construction
- Disk config `~/.surrealmemory/config.toml` has **no** `[trace]` section → `TraceConfig.enabled` defaults to `False`.
  Confirmed at runtime: `cfg.trace.enabled == False`, `cfg.trace.sample_rate == 1.0` — no forcing needed, item 1 uses the
  real neutral default end-to-end.
- Harness: real `MCPServer()` (no `get_config`/`get_storage` patching) built via `.venv/bin/python` throwaway script,
  piped over stdin per hard rules. Pattern for live `SurrealDBStorage`/brain setup adapted from
  `tests/unit/test_surrealdb_supersession_live.py` and `tests/unit/test_surrealdb_retrieval_trace_live.py`; `call_tool`
  invocation pattern from `tests/unit/test_mcp.py`.
- Cosmetic `FLEXIBLE ... SCHEMAFULL` schema-statement warnings on `SurrealDBStorage.initialize()` — ignored per instructions
  (pre-existing, unrelated to this feature).

## Seed data

```
server.call_tool("smem_remember", {"content": "Emma lives in Bergen, Norway.", "type": "fact"})
  → fiber_id 452e7297-5f93-4109-b262-97786fe860c2
server.call_tool("smem_remember", {"content": "Emma works as a solutions architect at AI-Flow.", "type": "fact"})
  → fiber_id e4bf1aeb-a150-4256-ac7e-dbddb94ec06d
```
Both `"success": true`.

---

## Item 1 — DEFAULT (tracing off globally, no per-call flag)

**Call:**
```python
count_before = await storage._count_retrieval_traces('"qa-u4-c016e112"')   # -> 0
res = await server.call_tool("smem_recall", {"query": "where does emma live"})
for task in list(getattr(server, "_trace_tasks", set())): await task       # drain (none expected)
count_after = await storage._count_retrieval_traces('"qa-u4-c016e112"')    # -> 0
```

**Real DB evidence:**
- `retrieval_trace` row count for brain `qa-u4-c016e112`: **before = 0, after = 0** (unchanged)
- Response keys: `['answer', 'confidence', 'depth_used', 'fibers_matched', 'neurons_activated', 'score_breakdown', 'session_query_count', 'session_topics', 'tokens_used']`
- `"trace_id" not in res` → **True** (no trace_id key present)
- `server._trace_tasks` was empty (no fire-and-forget task was even scheduled) — confirms no build/no task/no storage
  write occurred, not just a hidden no-op write.
- `res["answer"]` = `"## Relevant Memories\n\n- Emma lives in Bergen, Norway.\n..."`, `fibers_matched = ["452e7297_5f93_4109_b262_97786fe860c2"]` — recall itself worked correctly.

**RESULT: PASS**

---

## Item 2 — PER-CALL `trace=true`

**Call:**
```python
count_before = await storage._count_retrieval_traces('"qa-u4-c016e112"')   # -> 0
res = await server.call_tool("smem_recall", {
    "query": "where does emma live", "trace": True, "session_id": "qa-sess"
})
count_after = await storage._count_retrieval_traces('"qa-u4-c016e112"')    # -> 1
```

**Real DB evidence:**
- Response contains `"trace_id": "eebc041b-1e64-4151-bd25-a3e432cf9b71"`.
- `retrieval_trace` row count for brain `qa-u4-c016e112`: **before = 0, after = 1** (exactly one new row).
- Live raw `SELECT count() AS c FROM retrieval_trace WHERE brain_id = "qa-u4-c016e112" GROUP ALL` → `[{"c": 1}]`.
- `storage.get_retrieval_trace(trace_id)` returned a record whose `.id == "eebc041b-1e64-4151-bd25-a3e432cf9b71"`
  (matches the returned `trace_id` exactly — this is the U1 dash-vs-underscore id round-trip regression guard),
  `.brain_id == "qa-u4-c016e112"`, `.session_id == "qa-sess"`, `.query == "where does emma live"`,
  `.fiber_ids == ["452e7297_5f93_4109_b262_97786fe860c2"]`.

**RESULT: PASS**

---

## Item 3 — `smem_provenance` trace_get + traces query surface

**Call A:**
```python
prov_get = await server.call_tool("smem_provenance", {"action": "trace_get", "trace_id": "eebc041b-1e64-4151-bd25-a3e432cf9b71"})
```
**Real DB evidence:** full record returned under `"trace"` key —
`id`, `brain_id="qa-u4-c016e112"`, `session_id="qa-sess"`, `query="where does emma live"`, `depth_used=1`,
`mode="associative"`, `confidence=1.0`, `latency_ms=39.75`, `anchor_ids=[2 ids]`, `retrievers=["multi_neuron"]`,
`fiber_ids=["452e7297_5f93_4109_b262_97786fe860c2"]`, `config_snapshot`, `trace_version=1`, `created_at`.

**Call B:**
```python
prov_traces = await server.call_tool("smem_provenance", {"action": "traces", "query_contains": "emma live"})
```
**Real DB evidence:** `{"count": 1, "traces": [{"id": "eebc041b-1e64-4151-bd25-a3e432cf9b71", "query": "where does emma live", "mode": "associative", "depth_used": 1, "confidence": 1.0, "fiber_ids": [...], "session_id": "qa-sess", "created_at": ...}]}` — the item-2 trace appears in the compact list, matched via the substring `"emma live"` of its query.

**RESULT: PASS**

---

## Cleanup

```python
pruned = await storage.prune_retrieval_traces(max_traces=0)   # -> 1
after  = await storage._count_retrieval_traces('"qa-u4-c016e112"')  # -> 0
```
Confirmed: `retrieval_trace` rows for brain `qa-u4-c016e112` = 0 after cleanup.

Additionally (best-effort, beyond the stated requirement) deleted the throwaway brain's seed data via raw SurQL
(`DELETE <table> WHERE brain_id = "qa-u4-c016e112"` for `typed_memory`, `fiber`, `neuron`, `synapse`):

| table | before | after |
|---|---|---|
| retrieval_trace | 0 | 0 |
| typed_memory | 2 | 0 |
| fiber | 2 | 0 |
| neuron | 9 | 0 |
| synapse | 0 | 0 |

Only the empty `brain:qa-u4-c016e112` row itself remains (a single metadata row, no content) — same residue accepted
by the project's own live-DB fixtures (e.g. `test_surrealdb_supersession_live.py` leaves its throwaway brain row in
place too; only `test_surrealdb_retrieval_trace_live.py` prunes traces, matching what this run did).

---

## VERDICT: PASS

All three items of the U4 RECALL PERSIST contract verified end-to-end against the live SurrealDB v3 (`surrealdb-3.2.0`
at `http://localhost:8001`) through a real, unmocked `MCPServer`:

1. Neutral default (`trace.enabled=False`, no per-call flag) is a true no-op — no `trace_id` in the response, zero
   `retrieval_trace` rows written, and no background task even scheduled.
2. `trace=true` per-call forces exactly one synchronous trace write, returns its `trace_id` in the response, and the
   persisted row round-trips with matching `id`, `session_id`, and `query`.
3. `smem_provenance` `trace_get` and `traces` (query_contains) both surface that same persisted trace correctly
   through the same server instance.

No database mocking was used at any point — every assertion is backed by a live SurrealDB read (`_count_retrieval_traces`,
raw `SELECT ... GROUP ALL`, `get_retrieval_trace`) shown above.

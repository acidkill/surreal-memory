# U6 Backend Validation — Real SurrealDB v3 Evidence

**Repo/worktree:** `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
**Branch:** `feature/v290-pr6-dashboard`
**Live DB:** `SURREALDB_URL=http://localhost:8001` (confirmed live: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/status` → `200`)
**Python:** `.venv/bin/python` only (no `uv`/`uv run` used anywhere in this validation)
**Throwaway brains used:** `qa-u6-0ff5c427` (first probe run) and `qa-u6-812a5752` (final combined run) — both fully deleted at the end (see Cleanup).

Cosmetic `Schema statement failed: DEFINE FIELD ... FLEXIBLE ... SCHEMAFULL` warnings emitted by SurrealDB on every `storage.initialize()` are pre-existing/expected and were ignored per instructions.

---

## 0. Unit tests (baseline, must pass before live-DB work)

```bash
.venv/bin/python -m pytest tests/unit/test_uncertainty_report.py tests/unit/test_uncertainty_handler.py \
  tests/unit/test_dashboard_uncertainty.py tests/unit/test_diagnostics.py -p no:xdist -p no:cacheprovider -q
```

**Result:** `61 passed, 1 warning in 0.56s` — PASS.

---

## 1. `engine.uncertainty_report.build_brain_uncertainty` on live SurrealDB

**Setup:** fresh brain `qa-u6-812a5752` via `SurrealDBStorage(url=SURREALDB_URL)` (same pattern as
`tests/unit/test_surrealdb_supersession_live.py`), seeded with 2 neurons+fibers+typed FACT memories
plus 1 extra neuron and 1 unresolved `CONTRADICTS` synapse (source=new "contradiction" neuron,
target=neuron 1).

**Command:** throwaway script (`.venv/bin/python - <<'PY' ... PY` pattern), core call:
```python
overview = await build_brain_uncertainty(storage, within_days=14)
```

**Real DB evidence (actual returned dict):**
```json
{
  "level": "high",
  "counts": {
    "contradictions": 1, "low_evidence": 0, "superseded": 0,
    "expiring": 0, "drift_clusters": 0
  },
  "contradiction_rate": 0.5,
  "total_memories": 2,
  "scan": {"typed_scanned": 2, "typed_scan_truncated": false, "contradictions_capped": false},
  "samples": {"low_evidence": [], "superseded": [], "drift_clusters": []}
}
```

- Did not raise (`raised: null`).
- All required keys present: `level`, `counts`, `contradiction_rate`, `total_memories`, `scan`, `samples` — confirmed (`has_all_keys: true`).
- **CRITICAL check:** `hasattr(storage, "get_drift_clusters")` on the live `SurrealDBStorage` instance → **`False`**. `counts.drift_clusters == 0` — confirmed. This proves the `getattr(storage, "get_drift_clusters", None)` guard in `engine/uncertainty_report.py::_drift`/`get_detected_drift` degrades cleanly to `[]`/`0` on SurrealDB rather than raising `AttributeError`.

**VERDICT: PASS**

---

## 2. Dashboard route `GET /api/dashboard/uncertainty` (real FastAPI TestClient + live storage)

**Setup:** `FastAPI()` + `app.include_router(dashboard_api.router)` +
`app.dependency_overrides[get_storage] = lambda: storage` (the same live `SurrealDBStorage`
instance from item 1, same brain/data). `fastapi.testclient.TestClient`.

**Calls:**
```python
resp1 = client.get("/api/dashboard/uncertainty")
resp2 = client.get("/api/dashboard/uncertainty")
resp3 = client.get("/api/dashboard/uncertainty")
resp_bad_low  = client.get("/api/dashboard/uncertainty?within_days=0")
resp_bad_high = client.get("/api/dashboard/uncertainty?within_days=400")
```

**Real DB evidence:**
- `resp1.status_code == 200`, body identical to item 1's `build_brain_uncertainty` output (same
  brain/data, confirms the route wraps the engine function faithfully):
  ```json
  {"level": "high", "counts": {"contradictions": 1, "low_evidence": 0, "superseded": 0,
   "expiring": 0, "drift_clusters": 0}, "contradiction_rate": 0.5, "total_memories": 2,
   "scan": {"typed_scanned": 2, "typed_scan_truncated": false, "contradictions_capped": false},
   "samples": {"low_evidence": [], "superseded": [], "drift_clusters": []}}
  ```
- `resp2.status_code == 200`, `resp3.status_code == 200`, all three bodies byte-identical
  (`bodies_identical_across_3_calls: true`).
- **TTL-cache proof (not just coincidental equality):** `storage.count_typed_memories` was wrapped
  with a call counter before the 3 requests. `storage_hit_count_across_3_calls == 1`
  (`ttl_cached_confirmed: true`) — the aggregation query hit the live DB only once across 3 GETs,
  proving `_UNCERTAINTY_CACHE` (TTLCache) served requests 2 and 3 from cache.
- `resp_within_days_0_status == 422` (violates `ge=1` in `Query(14, ge=1, le=365)`).
- `resp_within_days_400_status == 422` (violates `le=365`).

**VERDICT: PASS**

---

## 3. `DiagnosticsEngine(storage).analyze(brain_id)` on live SurrealDB

**Command:**
```python
report = await DiagnosticsEngine(storage).analyze("qa-u6-812a5752")
```

**Real DB evidence (actual field values, same brain/data as items 1–2):**
```json
{
  "raised": null,
  "contradiction_count": 1,
  "contradiction_count_is_int": true,
  "neuron_count": 3,
  "conflict_rate": 0.3333,
  "expected_conflict_rate": 0.3333,
  "conflict_rate_matches_formula": true,
  "grade": "F",
  "purity_score": 19.4
}
```

- `report.contradiction_count` is `int` (`1`) — confirmed.
- `report.conflict_rate (0.3333) == round(contradiction_count / max(neuron_count, 1), 4) == round(1/3, 4) == 0.3333` — confirmed, formula unchanged (`src/surreal_memory/engine/diagnostics.py:395` `conflict_rate = contradicts_count / max(neuron_count, 1)`, surfaced unrounded-computed-then-rounded at line 457 `conflict_rate=round(conflict_rate, 4)`).
- `grade` (`"F"`) and `purity_score` (`19.4`) still produced by the unchanged purity formula
  (connectivity/diversity/freshness/consolidation/orphan/activation/recall_confidence weighted sum,
  minus the pre-existing conflict penalty at `diagnostics.py:396-397`) — a 3-neuron/2-fiber brain
  with a contradiction correctly grades F/low purity; U6 did not touch this formula, only added the
  `contradiction_count`/`conflict_rate` fields to the report dataclass for surfacing.

**VERDICT: PASS**

---

## 4. `smem` tools via a real `MCPServer` over the live DB (optional, done)

**Setup:** same brain (`qa-u6-812a5752`, same seeded data) via `os.environ["SURREAL_MEMORY_BRAIN"]`
+ `get_config(reload=True)` + `MCPServer()` (uses `unified_config.get_shared_storage()` → its own
`SurrealDBStorage` connection to the same live DB/brain — confirmed values match items 1–3 exactly,
proving it resolved the identical brain via the `brain_id == name` convention in
`_get_surrealdb_storage`).

**Real DB evidence:**
```python
stats       = await server.call_tool("smem_stats", {})
health      = await server.call_tool("smem_health", {})
uncertainty = await server.call_tool("smem_uncertainty", {"action": "overview"})
```
- `"contradiction_rate" in stats` → `True`; `stats["contradiction_rate"] == 0.3333` (matches item 3's `conflict_rate`).
- `"conflict_rate" in health` → `True`, `health["conflict_rate"] == 0.3333`.
- `"contradiction_count" in health` → `True`, `health["contradiction_count"] == 1`.
- `smem_uncertainty(action="overview")` returned:
  ```json
  {"brain": "qa-u6-812a5752", "level": "high",
   "counts": {"contradictions": 1, "low_evidence": 0, "superseded": 0, "expiring": 0, "drift_clusters": 0},
   "contradiction_rate": 0.5, "total_memories": 2,
   "scan": {"typed_scanned": 2, "typed_scan_truncated": false, "contradictions_capped": false},
   "samples": {"low_evidence": [], "superseded": [], "drift_clusters": []}}
  ```
  — `drift_clusters == 0`, identical shape/values to items 1–2 (same brain data), no raise.

**VERDICT: PASS**

---

## Cleanup

Both throwaway brains were fully purged from the live SurrealDB after validation:

```bash
.venv/bin/python u6_cleanup.py qa-u6-0ff5c427 qa-u6-812a5752
```
```
cleared brain: qa-u6-0ff5c427
cleared brain: qa-u6-812a5752
verify qa-u6-0ff5c427: neuron_rows=[{'count': 0}] brain_row_exists=False
verify qa-u6-812a5752: neuron_rows=[{'count': 0}] brain_row_exists=False
```
(`clear(brain_id)` removed neuron/neuron_state/synapse/fiber/change_log/device/merkle_hash/typed_memory
rows; explicit `DELETE brain WHERE id=... / name=...` removed the brain records themselves. Verified
0 neuron rows and no surviving brain record for both brains — no residue left in the live DB, and
no real/production brain data was touched.)

No files were modified in the repo; no git commit/push performed.

---

## FINAL VERDICT: **PASS** (all 4 items, plus the 61-test unit baseline)

Key takeaways for the reviewer:

1. **SurrealDB drift-degradation (the CRITICAL item) is confirmed live**: `SurrealDBStorage` has no
   `get_drift_clusters` method (`hasattr(...) == False` on the real, live-connected instance), and
   both `engine.uncertainty_report.build_brain_uncertainty` (item 1), the dashboard route (item 2),
   and the `smem_uncertainty` MCP tool (item 4) all returned `drift_clusters: 0` without raising —
   the `getattr(storage, "get_drift_clusters", None)` guard degrades cleanly on the real v3
   SurrealDB backend, exactly as designed.
2. **The diagnostics grade/purity formula is unchanged**: `conflict_rate` on the live report exactly
   equals `round(contradiction_count / max(neuron_count, 1), 4)` computed independently in this
   validation, and `grade`/`purity_score` are still produced from the same weighted-component
   formula (verified against `src/surreal_memory/engine/diagnostics.py` source) — U6 only adds
   `contradiction_count`/`conflict_rate` as new surfaced fields on `BrainHealthReport`, it does not
   alter how `purity`/`grade` are computed.
3. **Dashboard TTL caching is real, not coincidental**: a call-counter wrapped around
   `storage.count_typed_memories` proved the underlying aggregation hit the live DB exactly once
   across 3 consecutive `GET /api/dashboard/uncertainty` calls.
4. No mocked SurrealDB anywhere in this validation — every item exercised a real, live-connected
   `SurrealDBStorage` against `http://localhost:8001` (SurrealDB v3), using a fresh, now-deleted
   throwaway brain.

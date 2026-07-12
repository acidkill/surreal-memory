# GitHub issue draft — TypeScript client calls endpoints the server does not expose

> NEEDS-HUMAN: file this as a GitHub issue. Discovered during U9 (LangChain adapter);
> intentionally NOT fixed there — the adapter is in-process Python and does not touch the
> REST client. Separate concern, separate PR.

---

**Title:** `surreal-memory-client` (TS) calls `/api/remember` + `/api/recall` — endpoints that don't exist on the server

**Labels:** bug, typescript-client, api

## Summary

The TypeScript client's `remember`/`recall` methods POST to `/api/remember` and
`/api/recall`, but the FastAPI server exposes neither path. The real memory endpoints are
`encode` and `query` under the `/memory` router (mounted at both `/api/v1/memory/...` and
`/memory/...`). Any app using the published TS client for remember/recall gets a 404.

## Evidence

Client (`integrations/surreal-memory-client/src/client.ts`):

- L99  → `this.request<RememberResponse>("POST", "/api/remember", req, options)`
- L103 → `this.request<RecallResponse>("POST", "/api/recall", req, options)`

Server (`src/surreal_memory/server/routes/memory.py`, router `prefix="/memory"`):

- L40 → `@router.post("/encode", ...)`  → real path `POST /api/v1/memory/encode` (and `/memory/encode`)
- L86 → `@router.post("/query", ...)`   → real path `POST /api/v1/memory/query` (and `/memory/query`)

Mount (`src/surreal_memory/server/app.py`): the memory router is included under an
`/api/v1` parent router *and* at the app root — so there is no `/api/remember` or
`/api/recall` anywhere, and the operation names differ (`remember`→`encode`,
`recall`→`query`).

## Impact

- The published TS client is non-functional for its two core operations against a current
  server. This is a correctness/compatibility bug, not cosmetic.

## Proposed fix (for the follow-up PR — decide direction)

Two directions; pick one deliberately:

1. **Fix the client** to call `/api/v1/memory/encode` and `/api/v1/memory/query`, and map
   its `Remember`/`Recall` request/response shapes to the server's `EncodeRequest`/
   `EncodeResponse` and `QueryRequest`/`QueryResponse`.
2. **Add server aliases** `POST /api/remember` → encode and `POST /api/recall` → query
   (thin adapters) if `/api/remember|recall` is considered the intended public contract.

Direction 1 is lower-surface (no new public API) and matches the existing server contract;
direction 2 preserves the client's naming but widens the API. Recommend 1 unless the
`remember`/`recall` REST names are a committed external contract.

## Not in scope

- The U9 LangChain adapter (`src/surreal_memory/adapters/langchain.py`) is in-process
  Python and does not use this client; it is unaffected by this bug.

# Feature: Cloudflare Sync Hub

## Overview

Cloud-hosted sync hub for Neural Memory on Cloudflare (Workers + D1 + R2 + Pages). Multi-tenant service with API key auth. MVP sync protocol deployed and working.

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Worker + D1 + R2 Core | ✅ Done | — | Sync API, change_log, Hono framework |
| 2 | Auth + API Keys | ✅ Done | — | User table, `nmk_` keys, SHA-256 hashed |
| 3 | Landing + Payment UI | ⬚ Pending | plan-sync-hub-phase3.md | Landing page + SePay/Polar billing | **Hub (closed)** |
| 4 | Team Sharing | ⬚ Pending | plan-sync-hub-phase4.md | Multi-user brain access, roles, audit | **Hub (closed)** |
| 5 | NM Client Update | ✅ Done | — | API key in config, hub_url to CF |

## Architecture

```
NM Client (Python) → aiohttp POST → Cloudflare Worker (Hono)
  ├── D1 (users, api_keys, change_log, brains, teams, orders)
  ├── R2 (brain snapshots for backup/restore)
  └── KV (rate limit counters)
```

## Key Decisions

- D1 as primary store (IS SQLite, mirrors NM model)
- No conflict resolution on hub — client-side merge
- API key auth (not OAuth) — simpler for CLI/MCP
- Dual payment: SePay (Vietnam, 0% fee) + Polar (international) via pay.theio.vn
- Free tier handles 100 users comfortably (100x D1 read headroom)

# Neural Memory Commercialization Plan

## Nguyên tắc cốt lõi

1. **Free users KHÔNG bị mất gì** — tất cả features 4.x giữ nguyên free mãi mãi
2. **Pro users được thêm value** — features MỚI từ giờ, không "mở khóa" cái cũ
3. **Open source vẫn open source** — repo GitHub không thay đổi license
4. **Gate đúng chỗ** — Server-side trên Hub (closed), Private package trên GitHub (closed)
5. **Python client 100% free** — không có license check nào trong open source code

---

## Architecture: Free vs Pro

### Free (public repo — neural-memory)

| Feature | Status |
|---------|--------|
| Local brain (SQLite, unlimited) | Free forever |
| Remember / Recall / Context (50 MCP tools) | Free forever |
| Cognitive layer (hypothesize, predict, verify) | Free forever |
| HyperspaceDB Phase 1+2 (Gromov, Koopman, anisotropic) | Free forever |
| Consolidation, fidelity decay | Free forever |
| Cloud Sync (full changelist) | Free forever |
| Dashboard + Brain Oracle | Free forever |
| CLI | Free forever |
| nmem_visualize (C4) | Free — building |
| File Watcher | Free — building |

**Rule: Bất kỳ feature nào đã ship free → KHÔNG BAO GIỜ chuyển sang paid.**

### Pro features — 2 gates

#### Gate 1: Sync Hub (Cloudflare Worker — closed source)

| Feature | Free | Pro |
|---------|------|-----|
| Sync protocol | Full changelist | Merkle delta (~95% faster) |
| Brain sync | 1 brain | Unlimited |
| Sync history | Không | 30 ngày |
| Priority queue | Best-effort | Priority |
| Webhook alerts (Team) | Không | POST on sync |

#### Gate 2: Private Package (AIVN-Foundation/neural-memory-pro)

| Feature | Description |
|---------|-------------|
| Cone queries | Exhaustive recall — ALL memories in similarity cone |
| Directional compression | Multi-axis semantic compression (3 reference axes) |
| Smart merge consolidation | Priority-aware clustering + temporal coherence |
| Cross-Encoder reranking (C3) | bge-reranker post-SA refinement |
| Hyperbolic embedding (P3.3) | Poincaré disk encoding — research |

**Plugin architecture**: `pip install` → auto-discover via `entry_points` → Pro features active.
Delivery: GitHub private repo invite on purchase.

---

## Pricing (confirmed)

| Tier | Monthly | Yearly | Includes |
|------|---------|--------|----------|
| **Free** | $0 | $0 | Everything in public repo |
| **Pro** | $9 (219K VND) | $89 (2.19M VND) | Hub Pro + Private package |
| **Team** | $29 (719K VND) | $249 (6.19M VND) | Pro + shared brains + team sync + webhooks |

---

## Payment (confirmed)

- **Provider**: SePay (VN bank transfer, 0% fee) + Polar (international)
- **Worker**: `pay.theio.vn` (shared Cloudflare Worker — same as Rune, Companion)
- **Delivery**: Dual — license key (KV, TTL-based) + GitHub repo invite
- **Email**: `claude@theio.vn`
- **Products**: NM-PRO-MONTHLY, NM-PRO-YEARLY, NM-TEAM-MONTHLY, NM-TEAM-YEARLY

### Activation flow

```
User buys → pay.theio.vn generates license key + invites to GitHub repo
  ├── License key: nmem_sync_config(action='activate', license_key='nm_pro_xxx')
  ├── Private package: pip install git+https://github.com/AIVN-Foundation/neural-memory-pro.git
  └── Done — Pro features active on next session
```

---

## Implementation Status

### Phase A: License Infrastructure ✅ Done
- [x] D1 table `licenses` on sync hub (migration 0003)
- [x] `attachLicense` middleware — enriches context, never blocks
- [x] `isPro` helper function
- [x] License key generator (`nm_pro_XXXX_XXXX_XXXX`)
- [x] pay.theio.vn: 4 NM products configured
- [x] Dual delivery: license key + GitHub invite
- [x] Email template with activation instructions
- [x] Upsell hints in sync (>50 changes), recall (>10 fibers), health (>500 entities)
- [x] `/v1/hub/activate` endpoint on sync hub

### Phase B: Pro Features — Mixed
- [ ] Merkle delta sync on Hub (3.1) — **Hub, not built yet**
- [x] Cone queries — **Private pkg, scaffolded**
- [x] Directional compression — **Private pkg, scaffolded**
- [x] Smart merge — **Private pkg, scaffolded**
- [ ] Cross-Encoder reranking (C3) — **Private pkg, not built yet**
- [ ] Multi-brain sync on Hub — **Hub, not built yet**

### Phase C: Selling
- [ ] Landing page (need domain — NOT neuralmemory.dev)
- [ ] Create Polar products (run script)
- [ ] README badge: "Free forever · Pro available"
- [ ] Deploy updated pay.theio.vn worker
- [ ] Test end-to-end payment flow

---

## Cái gì KHÔNG làm

- ❌ Không lock features hiện tại sau paywall
- ❌ Không thêm license check vào Python open source code
- ❌ Không rate-limit free users
- ❌ Không bán data, không tracking
- ❌ Không force cloud — local brain luôn hoạt động offline
- ❌ Không dùng Stripe/Lemon — chỉ SePay + Polar

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Không ai mua | $9/mo rất thấp, focus marketing |
| User copy private pkg | DMCA + real value = updates + support |
| Hub downtime | Free users không affected. Pro graceful fallback |
| Maintenance 2 repos | Plugin interface nhỏ, ít thay đổi |

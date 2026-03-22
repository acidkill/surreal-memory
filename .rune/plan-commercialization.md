# Neural Memory Commercialization Plan

## Nguyên tắc cốt lõi

1. **Free users KHÔNG bị mất gì** — tất cả features hiện tại giữ nguyên free mãi mãi
2. **Pro users được thêm value** — không phải "mở khóa" cái đã có, mà là features MỚI
3. **Open source vẫn open source** — repo GitHub không thay đổi license
4. **Không cần LLM** — NM dùng spreading activation, giá transparent, không hidden cost

---

## Hiện tại: NM cho gì miễn phí?

| Feature | Status |
|---------|--------|
| Local brain (SQLite, unlimited) | Free forever |
| Remember / Recall / Context | Free forever |
| Cognitive layer (hypothesize, predict, verify) | Free forever |
| Consolidation, fidelity decay | Free forever |
| Cloud Sync (full changelist) | Free forever |
| Dashboard + Brain Oracle | Free forever |
| 50 MCP tools | Free forever |
| CLI | Free forever |

**Rule: Bất kỳ feature nào đã ship free → KHÔNG BAO GIỜ chuyển sang paid.**

---

## Pro tier: Features MỚI chưa tồn tại

| Pro Feature | Tại sao đáng tiền | Gate mechanism |
|---|---|---|
| **Merkle delta sync** | Sync nhanh hơn 95%, tiết kiệm bandwidth | Hub-side gate (Cloudflare Worker check license) |
| **Cone queries** | Recall exhaustive "mọi thứ về topic X" thay vì top-10 | Client-side license check |
| **Koopman auto-predict enhanced** | Auto-detect topic trends + email/webhook alerts | Hub-side (alert delivery qua cloud) |
| **Multi-brain sync** | Sync nhiều brains cùng lúc | Hub-side (free = 1 brain sync, pro = unlimited) |
| **Priority sync** | Sync queue ưu tiên (free users chờ, pro users sync ngay) | Hub-side queue priority |
| **Sync history** | Xem lịch sử sync, rollback | Hub-side (cần storage trên D1) |

### Tại sao gate ở Hub?

```
Free user:  Client → Hub → Full changelist sync (chậm nhưng VẪN HOẠT ĐỘNG)
Pro user:   Client → Hub (license verified) → Merkle delta sync (nhanh 95%)

Free user KHÔNG bị block. Chỉ chậm hơn.
```

- Client code 100% open source, không có license check trong Python code
- Gate nằm trên Cloudflare Worker (server bro control)
- Không ai bypass được vì Merkle protocol CẦN hub cooperation
- Free user vẫn sync bình thường qua full changelist

---

## Pricing (draft)

| Tier | Price | Includes |
|------|-------|----------|
| **Free** | $0 | Everything current + future local features |
| **Pro** | $9/mo hoặc $89/year | Merkle sync + Cone queries + Multi-brain sync + Sync history |
| **Team** | $29/mo hoặc $249/year | Pro + Shared brains + Team sync + Webhook alerts |

### So sánh với Mem0

| | Mem0 | NM Free | NM Pro |
|---|---|---|---|
| Price | Free → $249/mo (cliff) | $0 | $9/mo |
| Memory model | LLM-powered (hidden cost) | Graph (deterministic) | Graph (deterministic) |
| Sync | Cloud only | Local + Cloud | Local + Merkle Cloud |
| Self-host | No | Yes | Yes + Cloud boost |

**Key advantage**: Mem0 nhảy từ free → $249. NM có middle tier $9 capture được users bị Mem0 bỏ rơi.

---

## Implementation plan

### Phase A: License infrastructure (cần làm trước)

1. D1 table `licenses` trên Cloudflare Hub
2. License key generation (format: `nm_pro_XXXX_XXXX_XXXX`)
3. Payment integration (LemonSqueezy — đơn giản nhất, hỗ trợ VN)
4. Webhook: payment success → tạo license key → email cho user
5. `nmem config set license_key=xxx` command trong CLI
6. Client gửi license key trong `X-License-Key` header khi sync

### Phase B: Pro features (Phase 3 của Hyperspace)

1. Merkle delta sync (3.1) — gate ở hub middleware
2. Cone queries (3.2) — gate ở client (check license key locally)
3. Multi-brain sync — gate ở hub (count brains per license)

### Phase C: Selling

1. Landing page (LemonSqueezy checkout embed)
2. README badge: "Free forever · Pro available"
3. In-app upsell: khi sync chậm, hint "Upgrade to Pro for 95% faster sync"
4. GitHub Sponsors link

---

## Cái gì KHÔNG làm

- ❌ Không lock features hiện tại sau paywall
- ❌ Không thêm license check vào Python open source code (chỉ gate ở hub)
- ❌ Không rate-limit free users (chỉ khác tốc độ sync, không block)
- ❌ Không bán data, không tracking
- ❌ Không force cloud — local brain luôn hoạt động offline

---

## Timeline đề xuất

| When | What |
|------|------|
| Tuần 1 | Phase A: License infra trên Hub |
| Tuần 2-3 | Phase B: Merkle sync + Cone queries |
| Tuần 4 | Phase C: Landing page + LemonSqueezy |
| Ongoing | Marketing: blog posts, Twitter, GitHub discussions |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Không ai mua | $9/mo rất thấp, barrier nhỏ. Focus marketing |
| User fork bypass client gate | Client gate chỉ cho cone queries. Merkle CẦN hub → không bypass được |
| Hub downtime | Free users không affected (local works). Pro users graceful fallback to full sync |
| Complexity | Chỉ thêm 1 middleware trên hub + 1 D1 table. Minimal code |

---

## Decision needed from bro

1. **Pricing**: $9/mo ok hay muốn khác?
2. **Payment provider**: LemonSqueezy, Stripe, hay Gumroad?
3. **Timeline**: Bắt đầu Phase A luôn hay cần thêm features trước?
4. **Landing page**: Tự build hay dùng template?

# Pricing

> Everything is free. No tiers, no keys, no catch.

---

## Why Free?

Surreal-Memory is **MIT licensed, fully open source**. Every feature -- vector search, semantic recall, smart consolidation, cloud sync -- ships at no cost. There are no license keys, no paywalled backends, no "Pro" upgrade.

We believe a memory system for AI agents should be universally accessible. Locking core capabilities behind a paywall hurts adoption and fragments the ecosystem.

---

## How Every Feature Ships Free

Every advanced capability is available out of the box:

| Feature | How It's Delivered |
|---------|--------------------|
| HNSW vector search | SurrealDB vector index |
| Semantic recall | Spreading activation blended with vector similarity |
| Consolidation (`merge`, `dedup`) | Consolidation engine |
| Directional compression (multi-axis) | Built-in CommunityPlugin |
| 5-tier vector lifecycle | Tier engine + SurrealDB |
| Cloud sync | Optional self-hosted Cloudflare Worker |

No separate package to install. No activation step. No expiration date.

---

## Optional Cloud Sync

Sync brains across machines with a self-hosted hub:

- **Cloudflare Workers free tier** -- 100K requests/day, zero cost
- Merkle-tree delta sync -- only changed neurons are transferred
- End-to-end encryption -- your data is unreadable in transit and at rest

Deploy it in 5 minutes with the provided Worker template. No vendor lock-in, no managed service dependency.

---

## How the Project Sustains

Surreal-Memory is sustained the same way most open-source projects are:

- **Community contributions** -- bug reports, patches, plugins, documentation
- **Optional sponsorships and donations** -- GitHub Sponsors, one-time contributions
- **Adjacent services** -- consulting, custom integration work, hosted deployments (future)

We do not monetize through feature restrictions. The code you clone is the complete product.

---

## FAQ

**Is there really no paid tier?**
No paid tier. No license keys. No feature gates. Everything is MIT licensed.

**What about the old "Pro" package?**
Deprecated. The CommunityPlugin replaces it entirely. All Pro-tier features are now built in.

**Can I use this commercially?**
Yes. The MIT license permits commercial use without restrictions.

**How do I get cloud sync working?**
Follow the self-hosted Cloudflare Worker guide in the docs. It runs on Cloudflare's free tier.

**Who maintains this?**
The open-source community. Contributions welcome -- see the repository for guidelines.

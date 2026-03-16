# Feature: Onboarding Overhaul (Issue #82)

## Overview
Reduce 26 manual setup steps to 1 command (`nmem init --full`) + interactive guide page.
Two tracks: CLI improvements + MkDocs guide page with animated terminal demo.

## Phases
| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | CLI: init --full + doctor | ✅ Done | plan-onboarding-p1.md | One-command setup + health diagnostic |
| 2 | Guide Page | ✅ Done | plan-onboarding-p2.md | MkDocs interactive quickstart with animated terminal |
| 3 | CLI → Guide Link | ✅ Done | plan-onboarding-p3.md | Post-init banner pointing to guide URL |
| 4 | Dashboard Guide Card | ✅ Done | plan-onboarding-p4.md | Small card in dashboard linking to guide |

## Key Decisions
- MkDocs Material + custom JS (not standalone React) — reuses existing docs site
- Auto-detect embedding provider, don't force install without consent
- `nmem doctor` = diagnostic tool, not just health check — suggests fixes
- Guide page URL: `/guides/quickstart/` on existing GitHub Pages docs site
- Dashboard card: small banner, not a full page — link out to docs

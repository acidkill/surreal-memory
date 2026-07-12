# U10 — NEEDS-HUMAN: v2.10.0 push / PR / tag (ready-to-copy)

The whole run (U1–U10 + UB1/UB2) is stacked linearly. The **v2.10.0 tip
`feature/v2100-release-prep` (HEAD `70cf1a1`)** contains the ENTIRE history on top of
`main` (`d25b98f` = v2.8.0) — including all of v2.9.0. Nothing below has been executed;
these are for Toni to run/verify.

## Branch chain (stacked, the v2.10.0 tip contains everything)
```
main d25b98f (v2.8.0)
 └─ …v2.9.0 stack (U1–U7)… → feature/v290-release-prep b92856f   (v2.9.0 tip)
     └─ feature/v2100-pr7-geo        b893604   (U8 geospatial recall)
         └─ feature/v2100-pr8-langchain 6dba78f (U9 LangChain adapter)
             └─ feature/v2100-release-prep 70cf1a1  ← v2.10.0 TIP (push this)
```

## Option A — ship v2.9.0 then v2.10.0 as two releases (recommended; matches the two CHANGELOG sections)
```bash
cd <the run-006 worktree>

# 1. v2.9.0 first
git push -u origin feature/v290-release-prep
gh pr create --base main --head feature/v290-release-prep \
  --title "v2.9.0 — Memory you can trust" \
  --body "Trust release: schema v9, trust/recency, per-fact supersession, retrieval traces, uncertainty, dashboard. See CHANGELOG.md [2.9.0]."
# … merge that PR, then:
git checkout main && git pull
git tag -a v2.9.0 -m "v2.9.0 — Memory you can trust" && git push origin v2.9.0

# 2. v2.10.0 on top (rebases cleanly — linear history)
git push -u origin feature/v2100-release-prep
gh pr create --base main --head feature/v2100-release-prep \
  --title "v2.10.0 — Ecosystem" \
  --body "Geospatial recall (near filter + location metadata) + LangChain adapter (retriever + chat history, optional extra). No schema change. See CHANGELOG.md [2.10.0]."
# … merge, then:
git checkout main && git pull
git tag -a v2.10.0 -m "v2.10.0 — Ecosystem" && git push origin v2.10.0
```

## Option B — one merge, two tags (faster; v2.10.0 subsumes v2.9.0)
```bash
git push -u origin feature/v2100-release-prep
gh pr create --base main --head feature/v2100-release-prep \
  --title "v2.9.0 + v2.10.0 — trust + ecosystem" --body "See CHANGELOG.md [2.9.0] and [2.10.0]."
# … merge, then tag BOTH points on main (v2.9.0 at the U7 tip SHA once on main, v2.10.0 at the head):
git checkout main && git pull
git tag -a v2.9.0  <SHA-of-U7-release-prep-on-main> -m "v2.9.0 — Memory you can trust"
git tag -a v2.10.0 -m "v2.10.0 — Ecosystem"
git push origin v2.9.0 v2.10.0
```

## Post-publish registry verification (release.yml MASKS publish failures — a green job ≠ published)
Each tag triggers release.yml. Verify each artifact actually reached its registry:
- **PyPI**: `pip index versions surreal-memory` (or the pypi.org project page shows 2.10.0).
- **npm ×2**: `npm view surrealmemory@2.10.0 version` and the `surreal-memory-client` package at 2.10.0.
- **VS Code Marketplace**: the vscode-extension shows 2.10.0.
- **ClawHub**: the published entry shows 2.10.0.
- Repeat for 2.9.0 if shipping both tags.

## Also NEEDS-HUMAN (from U9)
- Create the "stale TS client" GitHub issue — draft ready at `qa/run-006/U9/stale-ts-client-issue.md`
  (`gh issue create --title "..." --body-file qa/run-006/U9/stale-ts-client-issue.md`).

## Notes
- No push / PR / merge / tag / publish was executed by the agent (per the run's human-gated policy).
- The tinybench `2.9.0` dependency pins were deliberately NOT bumped; only the package version is 2.10.0.

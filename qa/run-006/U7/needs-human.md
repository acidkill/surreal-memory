# U7 — NEEDS-HUMAN: v2.9.0 push / PR / tag (ready-to-copy)

Everything for v2.9.0 (U1–U6 + UB1/UB2 + U7 release-prep) is stacked linearly on ONE branch tip:
**`feature/v290-release-prep`** (HEAD `3e24443`) on top of **`main`** (`d25b98f` = v2.8.0). That tip contains
the full history, so a single push + PR captures the whole release. Nothing below has been executed — these
are for Toni to run/verify.

## Branch chain (stacked, tip contains all)
```
main d25b98f (v2.8.0)
 └─ feature/v290-pr1-schema-v9 474afd5
     └─ fix/surrealdb-recordid-comparison 78cb61e   (UB1)
         └─ feature/v290-pr2-trust-recency 6f95b83
             └─ fix/surrealdb-fiber-id-normalization 9e32a85   (UB2)
                 └─ feature/v290-pr3-supersession dea3ebc
                     └─ feature/v290-pr4-retrieval-traces c14bf53
                         └─ feature/v290-pr5-uncertainty bceb8ad
                             └─ feature/v290-pr6-dashboard df47e4c
                                 └─ feature/v290-release-prep 3e24443  ← TIP (push this)
```

## 1. Push + PR (single-PR path, recommended)
```bash
cd <the run-006 worktree>
git push -u origin feature/v290-release-prep
gh pr create --base main --head feature/v290-release-prep \
  --title "v2.9.0 — Memory you can trust" \
  --body "Trust release: schema v9, trust/recency, per-fact supersession (default superseded hard-filter + escape hatch), queryable retrieval traces, uncertainty surfacing, dashboard Uncertainty page. See CHANGELOG.md [2.9.0]."
```
(Optional: if per-PR review is preferred, push each feature/v290-* branch and open stacked PRs bottom-up;
the linear history makes this a rebase-free stack.)

## 2. Tag (only AFTER the PR is merged to main) — this triggers release.yml → publish
```bash
git checkout main && git pull
git tag -a v2.9.0 -m "v2.9.0 — Memory you can trust"
git push origin v2.9.0
```

## 3. Post-publish registry verification (release.yml MASKS publish failures — green job != published)
Verify each artifact actually reached its registry at version 2.9.0:
- PyPI: `pip index versions surreal-memory` (or check the pypi.org project page for 2.9.0).
- npm (x2): `npm view surrealmemory@2.9.0 version` and the surreal-memory-client package.
- VS Code Marketplace: the vscode-extension shows 2.9.0.
- ClawHub: the published entry shows 2.9.0.

## Notes
- No push/PR/merge/tag/publish was executed by the agent (per the run's human-gated policy).
- The run continues on U8–U10 (v2.10.0) stacked above this tip.

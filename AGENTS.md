# AGENTS.md — Rules for AI-Assisted Contributions

This file is for **anyone using an AI agent** (Claude Code, Cursor, Copilot, Codex, Aider, custom Anthropic/OpenAI SDK harness, …) to contribute to `surreal-memory`.

Humans without agents: skim it anyway, then read `CONTRIBUTING.md`.

> **You are responsible for what your agent ships.** Your GitHub account is the responsible party. If your agent submits AI-slop, *you* take the strike (see `CONTRIBUTING.md` § AI-slop policy).

---

## Hard Rule #1 — Execute Instructions Fully (No Shortcuts)

When a maintainer or reviewer gives you instructions, execute them in **full scope**. Never:

- Pick the subset that is easier or faster.
- Skip steps because "the existing pattern already covers it."
- Defer rename / refactor / cleanup steps to "later" if the instruction names them.
- Propose configs or implementations that contradict the user's stated intent.
- Substitute a smaller deliverable for the one that was requested.

If an instruction is **ambiguous**, ask **before** acting. If the **scope is larger than one PR can hold**, propose a sequence of PRs that cumulatively complete the instruction, and get the maintainer's nod on the sequence before you start.

This rule overrides every other heuristic in this file.

---

## Hard Rule #2 — No AI Attribution

**Do not put AI attribution in commits, PR descriptions, or the CHANGELOG.** No
`Co-Authored-By: Claude <noreply@anthropic.com>` (or any other agent) trailer, no
`Built with: …` footer, no "Generated with …" line.

If your tooling adds such a trailer by default, strip it before pushing:

```bash
git commit --amend  # delete the trailer line, save
```

A commit or PR describes **the change**, not the process that produced it. The GitHub
account on the commit is the responsible human either way — that is the accountability
record, and it is already there without a trailer. See Hard Rule #1: you own what you
ship, regardless of what typed it.

> Earlier revisions of this file asked for the opposite. That guidance is withdrawn: it
> conflicted with the project's own release rules and repeatedly leaked agent trailers
> into `main`'s permanent history.

---

## Gitflow

- `main` is protected. No direct pushes. No force-pushes ever.
- Branch names:
  - `feature/<short-kebab>` for new functionality.
  - `fix/<short-kebab>` for bug fixes.
  - `docs/<short-kebab>` for docs-only changes.
  - `ci/<short-kebab>` for pipeline changes.
- One logical change per PR. Long migrations (like the v2.0.0 rebrand) get split into a planned sequence — each piece its own PR.
- Merge to `main` requires **≥ 1 human reviewer approval** plus green CI.
- Never self-merge your own PR.

---

## Before You Push — Local Quality Gate

Run these and make them pass. Don't push red.

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -m "not stress" -n auto
```

Hooks blocked, CI bypassed, `--no-verify` — **forbidden**. If a hook fails, fix the underlying issue.

---

## Before You Touch a File — Facts Check

Run the same check Toni's own agents run for him. Before each `Write` / `Edit`, state these four:

1. **Which files import / invoke this file** (`grep`/`Grep`).
2. **Whether anything in the repo already serves the same purpose** (`Glob`).
3. **Data files written by this change** — exact field names, structure, date format. Synthetic values, not raw data.
4. **The maintainer's instruction, verbatim**.

If you can't answer all four for the file you're about to change, you don't yet understand the change. Investigate first.

---

## PR Template

Title: conventional commits.

- `feat: …` — new functionality
- `fix: …` — bug fix
- `refactor: …` — internal cleanup, no behavior change
- `refactor!: …` — breaking refactor (bumps major)
- `docs: …` — documentation only
- `test: …` — tests only
- `ci: …` — pipeline only
- `chore: …` — housekeeping

Body **must** contain:

```markdown
## Summary

- Bullet 1: what this PR does (not how).
- Bullet 2: …

## Why

One paragraph: the problem this PR solves and why it solves it this way.

## Test plan

- [ ] `pytest tests/ -m "not stress" -n auto` passes locally.
- [ ] `ruff check src/ tests/` clean.
- [ ] `mypy src/ --ignore-missing-imports` clean.
- [ ] (manual checks specific to this PR)

## Verified by

@your-github-handle.
```

The handle is the accountability record — Hard Rule #2 forbids naming the tool
next to it, so this section stays attribution-free.

---

## What Agents Are NEVER Allowed To Do

- Modify `LICENSE`. Verbatim upstream MIT, period.
- Modify `NOTICE`, `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md` without first opening a discussion issue and getting a maintainer's explicit OK.
- `git push --force` on `main` or `release/*` branches. On your own feature branch only if no one else has pulled it.
- Skip git hooks (`--no-verify`) or strip GPG signing.
- Commit secrets, API keys, `.env` files, raw credentials. If you see one in a diff — stop, alert the maintainer, rotate.
- Self-merge your own PR.
- Add new top-level Python or npm dependencies without a justification paragraph in the PR body.
- Edit code in `.claude/`, `.taskmaster/`, `.cursor/`, `.aider/`, `~/.local/...` — those are per-user directories and not part of the project.

---

## Style Reminders Specific to This Repo

- **Python**: 3.11+, ruff (project config), mypy strict-ish (`--ignore-missing-imports`). Type hints on every public function.
- **TypeScript** (dashboard / vscode-extension / integrations): strict mode on, no `any` without justification.
- **Branding**: package is **`surreal-memory`**, Python module **`surreal_memory`**, npm package **`surrealmemory`**, CLI binary **`smem`**. These replaced the upstream `neural-memory` / `neural_memory` / `nm` names in the v2.0.0 rebrand — don't reintroduce the old ones (except where upstream attribution requires it).
- **Upstream attribution**: the only places where `nhadaututtheky/neural-memory` URLs are allowed are `LICENSE`, `NOTICE`, and `README.md` § Acknowledgments. Don't introduce new ones elsewhere.

---

## Recommended Tooling for Agents

| Task | Tool |
|------|------|
| Lint / format | `ruff` |
| Type check | `mypy` |
| Test runner | `pytest` (with `pytest-xdist -n auto`) |
| PR / repo ops | `gh` CLI |
| Up-to-date library docs | Context7 MCP or vendor docs |
| Live web checks | Playwright MCP (don't use `WebFetch` for SPA frontends — it returns the SSR shell only) |

---

## See Also

- `CONTRIBUTING.md` — full contributor guide + AI-slop policy details.
- `NOTICE` — upstream attribution.
- `LICENSE` — MIT.
- `pyproject.toml` — package config, entry points (`smem`, `smem-mcp`, `smem-hook-*`).

---
title: Quickstart Guide
description: Get Neural Memory running in under 2 minutes with a single command.
---

# Quickstart Guide

Get a fully functional AI memory system in **one command**. No 26-step manual setup.

## See It In Action

<div class="qs-terminal" data-lines="init">
  <div class="qs-terminal__bar">
    <span class="qs-terminal__dot qs-terminal__dot--red"></span>
    <span class="qs-terminal__dot qs-terminal__dot--yellow"></span>
    <span class="qs-terminal__dot qs-terminal__dot--green"></span>
    <span class="qs-terminal__title">Terminal</span>
  </div>
  <div class="qs-terminal__body"></div>
  <button class="qs-terminal__replay">&#x21bb; Replay</button>
</div>

## 3 Steps — That's It

<div class="qs-steps">
  <div class="qs-step qs-reveal">
    <div class="qs-step__number">1</div>
    <div class="qs-step__title">Install</div>
    <div class="qs-step__desc">
      <code>pip install neural-memory</code><br>
      Python 3.11+, works on Windows, Linux, macOS.
    </div>
  </div>
  <div class="qs-step qs-reveal">
    <div class="qs-step__number">2</div>
    <div class="qs-step__title">Set Up Everything</div>
    <div class="qs-step__desc">
      <code>nmem init --full</code><br>
      Config, brain, MCP, hooks, embeddings, dedup — all automatic.
    </div>
  </div>
  <div class="qs-step qs-reveal">
    <div class="qs-step__number">3</div>
    <div class="qs-step__title">Restart Your IDE</div>
    <div class="qs-step__desc">
      Restart Claude Code, Cursor, or your MCP client.<br>
      Your brain is live. Start talking.
    </div>
  </div>
</div>

## What You Get

<div class="qs-stats qs-reveal">
  <div class="qs-stat"><span class="qs-stat__value">46</span> MCP tools</div>
  <div class="qs-stat"><span class="qs-stat__value">11</span> diagnostic checks</div>
  <div class="qs-stat"><span class="qs-stat__value">3</span> auto-installed hooks</div>
  <div class="qs-stat">Semantic search via embeddings</div>
  <div class="qs-stat">Duplicate detection</div>
</div>

## Verify Your Setup

Run `nmem doctor` to see if everything is healthy:

<div class="qs-terminal" data-lines="doctor">
  <div class="qs-terminal__bar">
    <span class="qs-terminal__dot qs-terminal__dot--red"></span>
    <span class="qs-terminal__dot qs-terminal__dot--yellow"></span>
    <span class="qs-terminal__dot qs-terminal__dot--green"></span>
    <span class="qs-terminal__title">Terminal</span>
  </div>
  <div class="qs-terminal__body"></div>
  <button class="qs-terminal__replay">&#x21bb; Replay</button>
</div>

Something wrong? Run `nmem doctor --fix` to auto-remediate hooks, dedup, and embedding issues.

## Core Features

<div class="qs-features qs-reveal">
  <div class="qs-feature">
    <div class="qs-feature__icon">🧠</div>
    <div class="qs-feature__title">Remember</div>
    <div class="qs-feature__desc">
      Store memories with type, priority, and tags.
      Rich context creates strong neural connections.
      <br><code>nmem_remember("chose Redis over Memcached because...")</code>
    </div>
  </div>
  <div class="qs-feature">
    <div class="qs-feature__icon">🔍</div>
    <div class="qs-feature__title">Recall</div>
    <div class="qs-feature__desc">
      Retrieve by meaning, not keywords.
      Spreading activation finds related memories across your graph.
      <br><code>nmem_recall("caching strategy")</code>
    </div>
  </div>
  <div class="qs-feature">
    <div class="qs-feature__icon">🩺</div>
    <div class="qs-feature__title">Doctor</div>
    <div class="qs-feature__desc">
      11 diagnostic checks. Auto-fix with <code>--fix</code>.
      Know exactly what's configured and what's missing.
    </div>
  </div>
  <div class="qs-feature">
    <div class="qs-feature__icon">📄</div>
    <div class="qs-feature__title">Knowledge Surface</div>
    <div class="qs-feature__desc">
      A <code>.nm</code> file loaded every session — working memory.
      Depth-aware routing skips the database when the surface is sufficient.
    </div>
  </div>
</div>

## What `--full` Does (Under The Hood)

| Step | What | Why |
|------|------|-----|
| Config | Creates `~/.neuralmemory/config.toml` | Central settings for brain, embedding, dedup |
| Brain | Creates `default.db` SQLite database | Your memory graph lives here |
| MCP | Registers in Claude Code + Cursor | AI tools can use your brain |
| Hooks | Installs PreCompact, Stop, PostToolUse | Captures memories automatically |
| Embeddings | Auto-detects best provider | Semantic search (find by meaning) |
| Dedup | Enables duplicate detection | Prevents memory bloat |
| Maintenance | Generates `maintenance.sh`/`.ps1` | Scheduled decay + consolidation |

Without `--full`, run `nmem init` for the basics, then `nmem setup embeddings` separately.

## Power User Features

### Cognitive Reasoning

Build and test hypotheses across sessions:

```
nmem_hypothesize("Redis is better than Memcached for our read pattern")
nmem_evidence("mem_id", "supports", confidence=0.8)
nmem_predict("Latency will drop 40% after Redis migration", deadline="2026-04-01")
nmem_verify("prediction_id", outcome="correct")
```

### Cloud Sync

Sync your brain across devices:

```bash
nmem sync push    # Upload to sync hub
nmem sync pull    # Download from sync hub
```

### Background Maintenance

Keep your brain healthy automatically:

```bash
nmem serve -p 8765    # Start API server with auto-consolidation
```

Or use the generated maintenance script:

```bash
# Linux/macOS — add to crontab
crontab -e
0 3 * * * ~/.neuralmemory/maintenance.sh

# Windows — run via Task Scheduler
~\.neuralmemory\maintenance.ps1
```

## Need Help?

- **Issues?** Run `nmem doctor --fix` first
- **Questions?** [GitHub Discussions](https://github.com/nhadaututtheky/neural-memory/discussions)
- **Bugs?** [Open an issue](https://github.com/nhadaututtheky/neural-memory/issues)
- **Full docs?** You're already here — check the sidebar

<script src="../assets/js/quickstart.js"></script>

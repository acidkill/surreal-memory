"""System prompt for AI tools using Surreal-Memory.

This prompt instructs AI assistants on when and how to use Surreal-Memory
for persistent memory across sessions.
"""

SYSTEM_PROMPT = """# Surreal-Memory - Persistent Memory System

You have access to Surreal-Memory, a persistent memory system that survives across sessions.
Use it to remember important information and recall past context.

## When to REMEMBER (smem_remember)

Automatically save these to memory:
- **Decisions**: "We decided to use PostgreSQL" -> remember as decision
- **User preferences**: "I prefer dark mode" -> remember as preference
- **Project context**: "This is a React app using TypeScript" -> remember as context
- **Important facts**: "The API key is stored in .env" -> remember as fact
- **Errors & solutions**: "Fixed by adding await" -> remember as error
- **TODOs**: "Need to add tests later" -> remember as todo
- **Workflows**: "Deploy process: build -> test -> push" -> remember as workflow

## When to RECALL (smem_recall)

Query memory when:
- Starting a new session on an existing project
- User asks about past decisions or context
- You need information from previous conversations
- Before making decisions that might conflict with past choices

## When to get CONTEXT (smem_context)

Use at session start to:
- Load recent memories relevant to current task
- Understand project state from previous sessions
- Avoid asking questions already answered before

## Auto-Capture (smem_auto)

After important conversations, call smem_auto to automatically capture memories:

```
# Simple: process and save in one call
smem_auto(action="process", text="<conversation or response text>")

# Preview first: see what would be captured
smem_auto(action="analyze", text="<text>")

# Force save (even if auto-capture disabled)
smem_auto(action="analyze", text="<text>", save=true)
```

Auto-capture detects:
- **Decisions**: "We decided...", "Let's use...", "Going with..."
- **Errors**: "Error:", "The issue was...", "Bug:", "Failed to..."
- **TODOs**: "TODO:", "Need to...", "Remember to...", "Later:"
- **Facts**: "The solution is...", "It works because...", "Learned that..."

**When to call smem_auto(action="process")**:
- After making important decisions
- After solving bugs or errors
- After learning something new about the project
- At the end of a productive session

## Session State (smem_session)

Track your current working session:
- **Session start**: `smem_session(action="get")` to resume where you left off
- **During work**: `smem_session(action="set", feature="auth", task="login form", progress=0.5)`
- **Session end**: `smem_session(action="end")` to save summary

This helps you resume exactly where you left off in the next session.

## System Behaviors (automatic — no action needed)

- **Session-aware recall**: When you call smem_recall with a short query (<8 words),
  the system automatically injects your active session's feature/task as context.
  No need to manually add session info to queries.
- **Passive learning**: Every smem_recall call with >=50 characters automatically
  analyzes the query for capturable patterns (decisions, errors, insights).
  You do NOT need to call smem_auto after recalls — it happens automatically.
- **Recall reinforcement**: Retrieved memories become easier to find next time
  (neurons that fire together wire together).
- **Priority impact**: Higher priority (7-10) memories get boosted in retrieval
  ranking through neuron state. Use 7+ for decisions and errors you'll need again.

## Depth Guide (for smem_recall)

- **0 (instant)**: Direct lookup, 1 hop. Use for: "What's Alice's email?"
- **1 (context)**: Spreading activation, 3 hops. Use for: "What happened with auth?"
- **2 (habit)**: Cross-time patterns, 4 hops. Use for: "What do I usually do on deploy?"
- **3 (deep)**: Full graph traversal. Use for: "Why did the outage happen?"

Leave depth unset for auto-detection (recommended).

## Best Practices

1. **Be proactive**: Don't wait for user to ask - remember important info automatically
2. **Be concise**: Store essence, not full conversations
3. **Use types**: Categorize memories (fact/decision/todo/error/etc.)
4. **Set priority**: Critical info = high priority (7-10), routine = normal (5)
5. **Add tags**: Help organize memories by project/topic
6. **Check first**: Recall before asking questions user may have answered before

## Examples

```
# User mentions a preference
User: "I always use 4-space indentation"
-> smem_remember(content="User prefers 4-space indentation", type="preference", priority=6, tags=["coding-style"])

# Starting work on existing project
-> smem_context(limit=10)
-> smem_recall(query="project setup and decisions")

# Made an important decision
"Let's use Redis for caching"
-> smem_remember(content="Chose Redis for caching — low latency, team familiarity", type="decision", priority=7, tags=["myapp", "infrastructure"])

# Found a bug fix
"The issue was missing await - fixed by adding await before fetch()"
-> smem_remember(content="Bug fix: Missing await before fetch() caused race condition", type="error", priority=7, tags=["myapp", "async"])

# Temporary scratch note (auto-expires, never synced)
-> smem_remember(content="Debugging: auth token expires at step 3", ephemeral=true)

# Error Resolution: store the fix normally. System auto-detects contradiction,
# creates RESOLVED_BY synapse, demotes error activation by >=50%.
"Actually the race condition was in the websocket handler, not fetch()"
-> smem_remember(content="Fix: Race condition was in websocket handler, not fetch(). Use asyncio.Lock().", type="insight", priority=7, tags=["myapp", "async"])
```

## Codebase Indexing (smem_index)

Index code for code-aware recall. Supports Python (AST), JS/TS, Go, Rust, Java/Kotlin, and C/C++ (regex):
- **First time**: `smem_index(action="scan", path="./src")` to index codebase
- **Check status**: `smem_index(action="status")` to see what's indexed
- **Custom extensions**: `smem_index(action="scan", extensions=[".py", ".ts", ".go"])`
- **After indexing**: `smem_recall(query="authentication")` finds related files, functions, classes

Indexed code becomes neurons in the memory graph. Queries activate related code through spreading activation — no keyword search needed.

## Eternal Context (smem_eternal + smem_recap)

Context is **automatically saved** on these events:
- Workflow completion ("done", "finished", "xong")
- Key decisions ("decided to use...", "going with...")
- Error fixes ("fixed by...", "resolved")
- User leaving ("bye", "tam nghi")
- Every 15 messages (background checkpoint)
- Context > 80% full → call `smem_auto(action="flush")` for emergency capture

### Emergency Flush (Pre-Compaction)
Before `/compact`, `/new`, or when context is nearly full, call:
```
smem_auto(action="flush", text="<paste recent conversation>")
```
This captures ALL memory types with a lower threshold (0.5), skips dedup, and boosts priority. Use it to prevent post-compaction amnesia.

### Session Gap Detection
When `smem_session(action="get")` returns `gap_detected: true`, it means content may have been lost between sessions (e.g. user ran `/new` without saving). Run `smem_auto(action="flush")` with recent conversation to recover.

### Session Start
Always call `smem_recap()` to resume where you left off:
```
smem_recap()             # Quick: project + current task (~500 tokens)
smem_recap(level=2)      # Detailed: + decisions, errors, progress
smem_recap(level=3)      # Full: + conversation history, files
smem_recap(topic="auth") # Search: find context about a topic
```

### Manual Save
Use `smem_eternal(action="save")` to persist project context into the neural graph:
```
smem_eternal(action="save", project_name="MyApp", tech_stack=["Next.js", "Prisma"])
smem_eternal(action="save", decision="Use Redis for caching", reason="Low latency")
smem_eternal(action="status")   # View memory counts and session state
```

## Edit & Forget (smem_edit + smem_forget)

Correct or remove memories without breaking the neural graph:

### Edit (smem_edit)
```
# Change memory type (was auto-detected wrong)
smem_edit(memory_id="fiber-abc", type="insight")

# Fix content (typo, wrong info)
smem_edit(memory_id="fiber-abc", content="Corrected: the bug was in auth.py, not login.py")

# Adjust priority
smem_edit(memory_id="fiber-abc", priority=9)

# Multiple changes at once
smem_edit(memory_id="fiber-abc", type="decision", priority=8, content="Updated decision text")
```

### Forget (smem_forget)
```
# Soft delete — sets expiry, memory decays naturally (recommended)
smem_forget(memory_id="fiber-abc", reason="outdated info")

# Hard delete — permanent removal, cascades to fiber + typed_memory
smem_forget(memory_id="fiber-abc", hard=true)

# Delete orphan neuron directly
smem_forget(memory_id="neuron-xyz", hard=true)
```

**When to use:**
- **smem_edit**: Wrong type assigned, content needs correction, priority adjustment
- **smem_forget (soft)**: Info is outdated but deletion trail wanted (default — sets expires_at)
- **smem_forget (hard)**: Sensitive data, test garbage, or duplicates that must be permanently removed

## Memory Types

- `fact`: Objective information
- `decision`: Choices made
- `preference`: User preferences
- `todo`: Tasks to do
- `insight`: Learned patterns
- `context`: Project/session context
- `instruction`: User instructions
- `error`: Bugs and fixes
- `workflow`: Processes/procedures
- `reference`: Links/resources

## Knowledge Base Training (smem_train + smem_pin)

Train permanent knowledge from documentation files into the brain:

```
# Train from a directory (supports .md, .txt, .rst, .pdf, .docx, .pptx, .html, .json, .xlsx, .csv)
smem_train(action="train", path="docs/", domain_tag="react")

# Train a single file
smem_train(action="train", path="guide.pdf", domain_tag="onboarding")

# Check training status
smem_train(action="status")
```

Trained knowledge is **pinned** by default — it never decays, never gets pruned, never gets compressed.
This creates a permanent knowledge base foundation that enriches organic (conversational) memories.

**Pin/Unpin memories manually:**
```
smem_pin(fiber_ids=["fiber-id-1", "fiber-id-2"], pinned=true)   # Pin
smem_pin(fiber_ids=["fiber-id-1"], pinned=false)                  # Unpin (lifecycle resumes)
```

**Re-training same file is idempotent** — files are tracked by SHA-256 hash. Already-trained files are skipped.

Install optional extraction dependencies for non-text formats:
```
pip install surreal-memory[extract]   # PDF, DOCX, PPTX, HTML, XLSX support
```

## Health & Diagnostics

- `smem_health()` — Brain health: purity score, grade (A-F), warnings, top_penalties
- `smem_evolution()` — Brain evolution: maturation, plasticity, coherence
- `smem_alerts(action="list")` — View active health alerts
- `smem_stats()` — Memory counts, type distribution, freshness
- `smem_conflicts(action="list")` — View conflicting memories

### Reading Health Reports

`smem_health()` returns `top_penalties` — a ranked list of what's hurting the score most.
**Always fix the highest penalty first** for maximum improvement.

7 components (weighted): Connectivity 25%, Diversity 20%, Freshness 15%,
Consolidation 15%, Orphan Rate 10%, Activation 10%, Recall Confidence 5%.

**Common fixes:**
- Consolidation 0% → normal for new brains; it rises only via spaced recall — reinforcement
  spread across 3+ distinct days (or 15+ rehearsals across 5+ time windows), not `smem consolidate`
- Orphan rate > 20% → Run `smem consolidate --strategy prune`
- Activation < 10% → Recall stored topics: `smem_recall('topic')` for 5+ topics
- Low connectivity → Store memories with context: "X because Y", "after A then B"
- Low diversity → Use causal/temporal/relational language in memories

### Maintenance Schedule
- **Every session**: `smem_recap()` at start (maintains freshness)
- **Weekly**: `smem_health()` → fix top penalty → `smem consolidate`
- **Monthly**: `smem consolidate --strategy prune` to clean orphans

## Connection Tracing (smem_explain)

Trace the shortest path between two concepts in your neural graph:
```
smem_explain(entity_a="Redis", entity_b="auth outage")
```
Returns the path with evidence: `Redis → USED_BY → session-store → CAUSED_BY → auth outage`.
Use this to debug recall results, verify brain connections, or understand causal chains.
If no path exists, the concepts are disconnected — store memories that link them.

## Spaced Repetition (smem_review)

- `smem_review(action="queue")` — Get memories due for review (Leitner box system)
- `smem_review(action="mark", fiber_id="...", success=true)` — Record review result
- `smem_review(action="stats")` — Review statistics

## Brain Management

- `smem_version(action="create", name="v1")` — Snapshot current brain state
- `smem_version(action="list")` — List all snapshots
- `smem_version(action="rollback", version_id="...")` — Restore a snapshot
- `smem_transplant(source_brain="other-brain", tags=["react"])` — Import memories from another brain
- `smem_narrative(action="topic", topic="auth")` — Generate narrative about a topic

## Cognitive Reasoning

The cognitive layer lets the brain reason about what it knows and doesn't know:

```
# Hypothesize + Evidence (Bayesian confidence tracking)
smem_hypothesize(action="create", content="Redis is the bottleneck", confidence=0.6)
smem_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency 200ms")
# Auto-resolution: confidence ≥0.9 + 3 for → confirmed. ≤0.1 + 3 against → refuted.

# Predict + Verify (propagates to linked hypothesis)
smem_predict(action="create", content="Fix will drop latency 50%", hypothesis_id="h-1", deadline="2026-04-01")
smem_verify(prediction_id="p-1", outcome="correct")  # or "wrong"

# Schema Evolution (SUPERSEDES chain)
smem_schema(action="evolve", hypothesis_id="h-1", content="Network config was root cause", reason="New evidence")
smem_schema(action="history", hypothesis_id="h-1")

# Knowledge Gaps
smem_gaps(action="detect", topic="Why 3am latency spike?", source="recall_miss")
smem_gaps(action="resolve", gap_id="g-1", resolved_by_neuron_id="n-42")

# Cognitive Dashboard
smem_cognitive(action="summary")   # Hot index: ranked active hypotheses + predictions
```

## Telegram Backup (smem_telegram_backup)

```
smem_telegram_backup()                        # Backup current brain
smem_telegram_backup(brain_name="work")       # Backup specific brain
```

Requires: `SURREAL_MEMORY_TELEGRAM_BOT_TOKEN` env var + `[telegram] chat_ids` in config.toml.

## Import External Data (smem_import)

```
smem_import(source="chromadb", connection="/path/to/chroma")
smem_import(source="mem0", user_id="user123")
smem_import(source="llamaindex", connection="/path/to/index")
```

## Sync Engine vs Git Backup

Use **smem_sync** for real-time multi-device memory synchronization:
- Works across devices (laptop, desktop, server) via hub server
- Automatic conflict resolution (prefer_recent, prefer_local, prefer_remote, prefer_stronger)
- Granular per-fiber sync — only changed memories are transferred
- Bi-directional: push local changes, pull remote, or full sync

Use **git backup** for version-controlled snapshots:
- Better for single-device users who want history/rollback
- Commit the `~/.surrealmemory/` data directory to a private repo
- No conflict resolution — just point-in-time snapshots
- Manual process (commit/push when you want)

**When to use which:**
- Single device, want history → git backup
- Multiple devices, want auto-sync → smem_sync
- Both → use smem_sync for real-time + git for disaster recovery
"""

COMPACT_PROMPT = """You have Surreal-Memory for persistent memory across sessions.

**Core:**
- **Remember** (smem_remember): Save decisions, preferences, facts, errors, todos, workflows.
- **Recall** (smem_recall): Query past context. Depth: 0=direct, 1=context, 2=patterns, 3=deep (auto if unset).
- **Context** (smem_context): Load recent memories at session start.
- **Recap** (smem_recap): Resume session. `smem_recap()` quick, `level=2` detailed, `topic="X"` search.

**Workflow:**
- **Auto-capture** (smem_auto): `process` after conversations, `flush` before compaction.
- **Session** (smem_session): `get` at start, `set` during work, `end` when done.
- **Eternal** (smem_eternal): Persist project context, decisions, instructions.
- **Index** (smem_index): Scan codebase into memory graph. `scan` once, then recall finds code.

**Knowledge Base:**
- **Train** (smem_train): Train docs into permanent memory. Supports PDF/DOCX/PPTX/HTML/JSON/XLSX/CSV.
- **Pin** (smem_pin): Pin/unpin memories to prevent decay. Trained KB is auto-pinned.

**Edit & Forget:**
- **Edit** (smem_edit): Fix memory type/content/priority by fiber_id. Preserves all connections.
- **Forget** (smem_forget): Soft delete (expires) or hard delete (permanent). Use for outdated/wrong memories.

**Advanced:**
- **Health** (smem_health): Brain health score, grade, top_penalties. Fix highest penalty first.
- **Explain** (smem_explain): Trace shortest path between two concepts. Debug why recall works/doesn't.
- **Review** (smem_review): Spaced repetition queue (Leitner boxes).
- **Sync** (smem_sync): Multi-device memory synchronization.
- **Version** (smem_version): Brain snapshots, rollback.
- **Transplant** (smem_transplant): Import memories from other brains.
- **Import** (smem_import): Import from ChromaDB, Mem0, LlamaIndex.
- **Conflicts** (smem_conflicts): View and resolve conflicting memories.
- **Narrative** (smem_narrative): Generate topic/timeline/causal narratives.
- **Telegram** (smem_telegram_backup): Send brain .db backup to Telegram chats.

**Cognitive Reasoning:**
- **Hypothesize** (smem_hypothesize): Create hypotheses with Bayesian confidence tracking.
- **Evidence** (smem_evidence): Submit for/against evidence — auto-updates confidence.
- **Predict** (smem_predict): Falsifiable predictions with deadlines, linked to hypotheses.
- **Verify** (smem_verify): Verify predictions correct/wrong — propagates to hypotheses.
- **Cognitive** (smem_cognitive): Hot index of active hypotheses and predictions.
- **Gaps** (smem_gaps): Track knowledge gaps — what the brain doesn't know.
- **Schema** (smem_schema): Evolve hypotheses into new versions (SUPERSEDES chain).

Be proactive: remember important info without being asked. Call smem_recap() at session start."""


def get_system_prompt(compact: bool = False) -> str:
    """Get the system prompt for AI tools.

    Args:
        compact: If True, return shorter version for limited context

    Returns:
        System prompt string
    """
    return COMPACT_PROMPT if compact else SYSTEM_PROMPT


MCP_INSTRUCTIONS = """\
Surreal-Memory gives you persistent memory across sessions. Use it proactively — \
each session starts fresh, so without explicit saves ALL discoveries are lost.

## WHEN TO RECALL (before responding)

| Trigger | Action |
|---------|--------|
| New session starts | smem_recall("current project context") |
| User references past event/decision | smem_recall("<that topic>") |
| Task involves tech/pattern discussed before | smem_recall("<project> <tech>") |
| Purely new, self-contained question | Skip recall |

Query tips: Be specific ("auth bug fix March 2026"), prefix with project name, \
avoid vague queries ("stuff", "what happened").

## WHEN TO SAVE (after completing work)

After each task, check: did I just...

| Signal | Type | Priority |
|--------|------|----------|
| Choose between alternatives | decision | 7 |
| Fix a bug (root cause + fix) | error | 7 |
| Discover a pattern/insight | insight | 6 |
| Learn a user preference | preference | 8 |
| Establish a workflow | workflow | 6 |
| Find a reusable fact | fact | 5 |
| Receive explicit instruction | instruction | 8 |

Priority scale: 9-10 critical (security, data loss), 7-8 important (decisions, preferences), \
5-6 normal (patterns, facts), 1-4 minor.

## EPHEMERAL MEMORIES

For scratch notes, debugging context, or temporary reasoning that should NOT persist:
`smem_remember(content="...", ephemeral=true)` — auto-expires after 24h, never synced, \
excluded from consolidation. Use `smem_recall(permanent_only=true)` to filter them out.

## DO NOT SAVE (as permanent)

- Routine file reads/writes — use `ephemeral=true` or skip entirely
- Things already in code or git history (derivable)
- Temporary debugging steps — use `ephemeral=true`
- Content already stored (check with smem_recall first)

## CONTENT QUALITY

1. Max 1-3 sentences. Never dump file structures or full implementation details.
2. Use causal language: "Chose X over Y because Z", "Root cause was X, fixed by Y".
3. Always include project name + topic in tags (lowercase).

## SESSION END

Call smem_auto(action="process", text="<brief session summary>") to capture remaining context.

## COMPACT MODE

All tools support `compact=true` to reduce response tokens by 60-80%. Use it for list queries \
when you don't need full details. Use `token_budget=N` to cap response size. \
Full details always available via smem_show(memory_id). Responses with >20 list items are \
auto-compacted.\
"""


def get_mcp_instructions() -> str:
    """Get concise behavioral instructions for MCP InitializeResult.

    These instructions are injected into the agent's system context
    automatically by MCP clients that support the `instructions` field.
    Keep under ~200 words — behavioral directives, not documentation.

    Returns:
        Concise instruction string for proactive memory usage.
    """
    return MCP_INSTRUCTIONS


def get_prompt_for_mcp() -> dict[str, str]:
    """Get prompt formatted for MCP resources."""
    return {
        "uri": "surrealmemory://prompt/system",
        "name": "Surreal-Memory System Prompt",
        "description": "Instructions for AI assistants on using Surreal-Memory",
        "mimeType": "text/plain",
        "text": SYSTEM_PROMPT,
    }

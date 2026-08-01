# Memory Types

Surreal-Memory supports typed memories for better organization and automatic lifecycle management.

## Available Types

| Type | Description | Default Expiry | Use Case |
|------|-------------|----------------|----------|
| `fact` | Objective information | Never | API endpoints, configuration values |
| `decision` | Choices made | 90 days | Architectural decisions, tool choices |
| `preference` | User preferences | Never | Coding style, naming conventions |
| `todo` | Action items | 30 days | Tasks, reminders, follow-ups |
| `insight` | Learned patterns | 180 days | Debugging tricks, optimization tips |
| `context` | Situational info | Never | Meeting notes, temporary context |
| `instruction` | User guidelines | Never | Project rules, conventions |
| `error` | Error patterns | 30 days | Bug fixes, error solutions |
| `workflow` | Process patterns | 365 days | Deployment steps, review processes |
| `reference` | External references | Never | Documentation links, resources |
| `tool` | Tool/CLI usage patterns | 90 days | CLI flags, commands, invocation tips |
| `boundary` | Safety rules | Never | "Never use eval()", "Always confirm before X" |
| `hypothesis` | Evolving beliefs | 180 days | Cognitive layer — produced by `smem_hypothesize` |
| `prediction` | Falsifiable claims | 30 days | Cognitive layer — produced by `smem_predict` |
| `schema` | Mental model versions | Never | Cognitive layer — produced by knowledge-gap flows |

## Classifier Coverage Matrix

`suggest_memory_type(content)` auto-detects 12 of the 15 types from raw
text. The remaining 3 are intentionally excluded — they belong to the
cognitive layer and are produced by structured handlers, not free-form
content.

| Type | Auto-classified | Explicit `--type` | Cognitive flow only |
|------|:---------------:|:------------------:|:--------------------:|
| `fact` (default) | ✓ | ✓ | |
| `todo` | ✓ | ✓ | |
| `decision` | ✓ | ✓ | |
| `error` | ✓ | ✓ | |
| `insight` | ✓ | ✓ | |
| `instruction` | ✓ | ✓ | |
| `preference` | ✓ | ✓ | |
| `workflow` | ✓ | ✓ | |
| `reference` | ✓ | ✓ | |
| `boundary` | ✓ | ✓ | |
| `tool` | ✓ | ✓ | |
| `context` | ✓ | ✓ | |
| `hypothesis` | | ✓ | ✓ (`smem_hypothesize`) |
| `prediction` | | ✓ | ✓ (`smem_predict`) |
| `schema` | | ✓ | ✓ (knowledge-gap flow) |

**Branch precedence** (top wins on collisions):
`boundary → todo → decision → error → insight → instruction → preference
→ workflow → tool → reference → context → fact (default)`.

Notable collisions:

- `"Never use eval()"` → **boundary**, not instruction. Safety wins.
- `"Must not log credentials"` → **boundary**, not todo (`must` matches both).
- `"Deploy pipeline runs build then push command"` → **workflow**, not tool.
- `"Currently working on the SurrealDB fork"` → **context**, not fact.

The 6 types that are **not** auto-classified from free-form content
(`tool`, `boundary` are auto-classified now; `hypothesis`, `prediction`,
`schema` are cognitive-only and require their dedicated handlers; the
others are auto-classifiable) must be created via explicit `--type` or
the appropriate cognitive command:

```bash
# Cognitive layer — MCP tools, no CLI equivalent. These create HYPOTHESIS and
# PREDICTION via their handlers:
#   smem_hypothesize(statement="The slow path is gated by a global lock")
#   smem_predict(statement="Cache hit rate will drop after the migration")
# SCHEMA is produced by knowledge-gap detection during smem_remember /
# consolidate flows, not directly authored.

# Explicit override — supports any of the 15 types
smem remember "rg supports PCRE2 with -P" --type tool
smem remember "Currently working on classifier expansion" --type context
smem remember "Never delete the production database" --type boundary
```

## Using Memory Types

### Explicit Type

```bash
smem remember "We decided to use PostgreSQL" --type decision
smem remember "API endpoint: /v2/users" --type fact
smem remember "Review PR before merge" --type instruction
```

### Auto-Detection

Surreal-Memory can detect types from content:

```bash
# Detected as TODO
smem remember "TODO: fix the login bug"

# Detected as ERROR
smem remember "ERROR: null pointer in auth module"

# Detected as DECISION
smem remember "We chose FastAPI over Flask"
```

## Type-Specific Features

### fact

Facts are objective, verifiable information.

```bash
smem remember "Database host is db.example.com" --type fact
smem remember "Max file size is 10MB" --type fact
```

**Behavior:**

- Never expires
- High priority in retrieval for technical queries
- Good for configuration, endpoints, specifications

### decision

Architectural and strategic decisions.

```bash
smem remember "DECISION: Use JWT for auth. REASON: Stateless, scales better." --type decision
```

**Best Practice:** Include rationale

```bash
smem remember "DECISION: PostgreSQL over MongoDB. REASON: Strong consistency needed. ALTERNATIVE: Considered MongoDB for flexibility." --type decision
```

**Behavior:**

- Expires after 90 days by default
- Searchable by decision keywords
- Critical for understanding project history

### preference

User and team preferences.

```bash
smem remember "User prefers tabs over spaces" --type preference
smem remember "Team uses camelCase for JS" --type preference
```

**Behavior:**

- Never expires
- Lower activation weight (preferences are contextual)
- Used for personalization

### todo

Action items and tasks.

```bash
smem todo "Fix the login bug"
smem todo "Review PR #123" --priority 8
smem todo "Deploy to production" --priority 10 --expires 1
```

**Behavior:**

- Expires in 30 days by default
- Supports priority 0-10
- Listed with `smem list --type todo`

### insight

Learned patterns and tips.

```bash
smem remember "Cache invalidation causes 90% of our bugs" --type insight
smem remember "Always check for null before array access" --type insight
```

**Behavior:**

- Expires after 180 days by default
- High value for similar problem-solving
- Good for documenting "lessons learned"

### context

Temporary, situational information.

```bash
smem remember "Currently working on auth module" --type context
smem remember "Sprint 5 focus: performance" --type context --expires 14
```

**Behavior:**

- Never expires by default
- Lower retrieval priority for older queries
- Good for session-specific context

### instruction

Rules and guidelines.

```bash
smem remember "Always run tests before committing" --type instruction
smem remember "Use semantic commit messages" --type instruction
```

**Behavior:**

- Never expires
- High priority in retrieval
- Good for enforcing conventions

### error

Error patterns and solutions.

```bash
smem remember "ERROR: 'Cannot read id of undefined'. SOLUTION: Add null check before user.id" --type error
```

**Best Practice:** Include both error and solution

```bash
smem remember "ERROR: CORS blocked request. SOLUTION: Add origin to allowed list in cors.config.ts" --type error --tag cors --tag api
```

**Behavior:**

- Expires after 30 days by default
- Highly relevant for debugging queries
- Pairs well with tags for categorization

### workflow

Process documentation.

```bash
smem remember "Deploy process: 1. Run tests 2. Build 3. Push to staging 4. Verify 5. Push to prod" --type workflow
```

**Behavior:**

- Expires after 365 days by default
- Good for recurring processes
- Can be broken into steps

### reference

External links and resources.

```bash
smem remember "FastAPI docs: https://fastapi.tiangolo.com" --type reference
smem remember "Design doc: notion.so/design-v2" --type reference
```

**Behavior:**

- Never expires
- Lower activation weight (supplementary info)
- Good for documentation links

### tool

CLI/tool usage patterns and invocation knowledge.

```bash
smem remember "ruff supports --fix and --unsafe-fixes flags" --type tool
smem remember "Use 'cargo build --release' for prod binaries" --type tool
```

**Behavior:**

- Expires in 90 days (tool patterns become stale as workflows change)
- Auto-classified from keywords like `command`, `flag`, `cli`, `invoke`,
  `run with`, `subcommand`
- Checked after `workflow` so deploy/process descriptions stay workflow

### boundary

Safety rules — directives whose violation has high cost.

```bash
smem remember "Never run rm -rf without --dry-run first" --type boundary
smem remember "Always confirm before pushing to main" --type boundary
```

**Behavior:**

- **Never expires** (safety-critical)
- **Auto-promoted to HOT tier** — always loaded into context
- Slowest decay rate of any type (0.01)
- Auto-classified from phrases: `must not`, `must never`, `never use`,
  `don't ever`, `always ask before`, `always confirm`
- Checked **first** in the classifier so safety rules cannot be
  silently downgraded to `instruction`

### hypothesis

Evolving beliefs with evidence-based confidence. Cognitive layer.

```text
smem_hypothesize(statement="The encoder is the throughput bottleneck")
```

**Behavior:**

- Default expiry: 180 days
- Tracked in the `cognitive_state` table with evidence_for / against counts
- **Not auto-classified** from free-form content — must be authored via
  the `smem_hypothesize` MCP tool
- Produces `confidence` updates as evidence accumulates

### prediction

Falsifiable claims about future observations. Cognitive layer.

```text
smem_predict(statement="Migration will take more than 8 hours", resolves_at="2026-06-01")
```

**Behavior:**

- Default expiry: 30 days (predictions should be verified soon)
- Tracked in `cognitive_state` with `predicted_at` and `resolved_at`
- **Not auto-classified** — produced by the `smem_predict` MCP tool only
- Calibration stats are aggregated by `get_calibration_stats`

### schema

Mental model versions — explicit knowledge-structure snapshots.

**Behavior:**

- Never expires (older versions are superseded, not deleted)
- Versioned via `parent_schema_id` chain in `cognitive_state`
- **Not directly authored** — produced by knowledge-gap detection
  during `smem_remember` / `consolidate` flows
- Walked newest-first by `get_schema_history(neuron_id)`

## Priority System

All types support priority 0-10:

| Priority | Meaning | Use Case |
|----------|---------|----------|
| 0-2 | Low | Nice to have, minor notes |
| 3-4 | Below normal | Useful but not critical |
| 5 | Normal (default) | Standard importance |
| 6-7 | Above normal | Important items |
| 8-9 | High | Critical information |
| 10 | Critical | Must not forget |

```bash
smem remember "API key rotation needed" --priority 9
smem todo "Update dependencies" --priority 3
```

## Expiry

Set custom expiry in days:

```bash
smem remember "Sprint goal" --type context --expires 14
smem todo "Review before Friday" --expires 5
```

Check expired memories:

```bash
smem list --expired
smem cleanup --expired --dry-run
```

## Querying by Type

```bash
# List all TODOs
smem list --type todo

# List high-priority decisions
smem list --type decision --min-priority 7

# Get only facts about auth
smem recall "auth configuration" --type fact
```

## Cleanup by Type

```bash
# Clean expired context
smem cleanup --type context --expired

# Preview cleanup
smem cleanup --type todo --expired --dry-run
```

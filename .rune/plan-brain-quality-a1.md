# Phase A1: Smart Instructions

## Goal
Give agents a clear decision framework for WHEN and WHAT to remember/recall — via MCP `instructions` field. Zero code change, immediate impact.

## Motivation
- Competitors (ChatGPT, Letta) achieve proactive memory through **prompted behavior** — the model is told when to save/recall
- NM's current instructions say "MUST use proactively" but don't explain WHEN or HOW
- Missing: decision tree for save triggers, importance scoring guide, recall triggers
- This is the #1 lowest-effort, highest-impact change

## Design

### Current Instructions (problems)
```
"SESSION START: Call nmem_recall to load past context"        ← Too vague, no query guidance
"AFTER EVERY TASK: Call nmem_remember to save what you did"  ← Too aggressive, saves noise
"NEVER skip remembering after completing a feature"          ← No importance filter
```

### Proposed Instructions (structured decision framework)

```markdown
## RECALL Protocol (BEFORE responding)
When user asks a question or gives a task:
1. Does it reference a past event, decision, or person? → RECALL that topic
2. Does it involve a technology/pattern you may have discussed before? → RECALL that tech
3. Is this a new session? → RECALL "current project context" + "recent decisions"
4. Is it a purely new, self-contained question? → Skip recall

Query construction:
- GOOD: "PostgreSQL migration decision", "auth bug March 2026"
- BAD: "what happened", "stuff" (too vague → noisy results)
- Always prefix with project name for project-specific recall

## SAVE Protocol (AFTER completing work)
Evaluate each completed task against this checklist:

| Signal | Type | Priority | Example |
|--------|------|----------|---------|
| Made a choice between alternatives | decision | 7-8 | "Chose Redis over Memcached because..." |
| Fixed a bug (root cause + fix) | error | 7 | "Auth failed because token expired..." |
| Discovered a pattern/insight | insight | 6-7 | "This codebase uses X pattern for Y" |
| Learned a user preference | preference | 8 | "User prefers Vietnamese communication" |
| Established a workflow | workflow | 6 | "Deploy process: build → test → push" |
| Found a reusable fact | fact | 5 | "API endpoint is /v2/users" |
| Received explicit instruction | instruction | 8 | "Always run linter before commit" |

DO NOT SAVE:
- Routine file reads/writes (ephemeral)
- Things already in code/git (derivable)
- Temporary debugging steps (transient)
- Exact same content already stored (duplicate)

## IMPORTANCE SCALE
- 9-10: Security issues, data loss incidents, critical user corrections
- 7-8: Architecture decisions, user preferences, bug root causes
- 5-6: Patterns, workflows, reusable facts
- 3-4: Minor observations, routine notes
- 1-2: Temporary context (use session memory instead)

## Content Quality Rules
- Max 1-3 sentences. NEVER dump file structures or implementation details
- Use causal language: "X because Y", "chose X over Y because Z"
- Include project name in tags for layer routing
```

### Implementation
- Update `MCP_INSTRUCTIONS` constant in `src/neural_memory/mcp/server.py`
- Update `.claude-plugin/plugin.json` instructions field
- Update SKILL.md behavioral directives

## Tasks
- [x] Draft new instructions with decision framework
- [x] Update `MCP_INSTRUCTIONS` in prompt.py (70→45 lines, 36% reduction)
- [x] Update plugin.json instructions field (added `instructions` key)
- [x] Update SKILL.md with behavioral directives (RECALL/SAVE/FLUSH pattern)
- [x] Test: 135 MCP tests pass, import verified
- [ ] Validate: run 3 test sessions, measure proactive save/recall rate

## Acceptance Criteria
- [ ] Agent recalls relevant context before responding (>50% of applicable turns)
- [ ] Agent saves meaningful memories after tasks (not noise)
- [ ] Agent uses appropriate priority levels (not all 5s)
- [ ] Agent skips saving ephemeral/derivable information
- [ ] Instructions fit within MCP instructions size limit

## Files Touched
- `src/neural_memory/mcp/server.py` — modify (MCP_INSTRUCTIONS)
- `.claude-plugin/plugin.json` — modify (instructions field)
- `.claude-plugin/SKILL.md` — modify (behavioral section)

## Risks
- Instructions too long → exceeds MCP field limit or wastes context
- Too prescriptive → agent follows robotically without judgment
- Model-dependent → works well with Opus/Sonnet but may fail with smaller models

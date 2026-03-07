# Learning Habits Guide

Neural Memory automatically detects repetitive workflow patterns from your tool usage and materializes them as learned habits. This guide explains how habit detection works, what the output looks like, and how to tune it.

## How It Works

The habits system uses a 3-stage pipeline:

```
Action Events (auto-recorded per tool call)
  → Sequential Pair Mining (find repeated A→B patterns)
  → Habit Candidate Extraction (bigrams + trigrams above threshold)
  → WORKFLOW Fiber Materialization (stored in memory graph)
```

### Stage 1: Action Event Recording

Every MCP tool call is automatically recorded as a lightweight action event in a hippocampal buffer (the `action_events` table). Each event captures:

- **action_type**: The tool name (e.g., `recall`, `remember`, `context`)
- **session_id**: Groups events by conversation session
- **timestamp**: When the action happened

These events are *not* neural graph nodes — they're a fast append-only log used only for pattern mining.

### Stage 2: Sequential Pair Mining

Events are grouped by session and sorted by time. Consecutive pairs within a **time window** are counted:

```
Session 1: recall (10:00:00) → edit (10:00:10) → test (10:00:25)
Session 2: recall (14:00:00) → edit (14:00:08) → test (14:00:20)
Session 3: recall (09:00:00) → edit (09:00:12) → test (09:00:30)

Pairs found:
  recall → edit:  3 occurrences, avg gap 10s
  edit → test:    3 occurrences, avg gap 18.3s
```

The **time window** (`sequential_window_seconds`, default 30s) filters out unrelated actions. If the gap between two consecutive actions exceeds 30 seconds, they're not counted as a sequential pair.

### Stage 3: Habit Candidate Extraction

Pairs meeting the **minimum frequency** (`habit_min_frequency`, default 3) become habit candidates:

- **Bigrams**: Direct A→B pairs (e.g., `recall→edit`)
- **Trigrams**: Combined A→B + B→C chains (e.g., `recall→edit→test`)

**Confidence** is calculated as:

```
confidence = frequency / total_sessions
```

Example: `recall→edit→test` occurs 3 times across 5 sessions → confidence = 0.6

Cycle patterns (A→B→A) are automatically skipped.

## Configuration

All thresholds are configurable per brain:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequential_window_seconds` | 30.0 | Max gap between consecutive events to count as sequential |
| `habit_min_frequency` | 3 | Minimum repetitions before a pattern becomes a habit |
| `habit_suggestion_min_weight` | 0.8 | Synapse weight threshold for next-action suggestions |
| `habit_suggestion_min_count` | 5 | Sequential count threshold for suggestions |

Note: Habit **detection** uses `habit_min_frequency` (3), while **suggestions** use stricter thresholds (`min_weight` 0.8 + `min_count` 5). This means a pattern can be detected as a habit but not yet trigger suggestions until it's observed more frequently.

## When Habits Are Learned

Habit learning runs during **consolidation** as a Tier 1 strategy:

```bash
nmem consolidate --strategy learn_habits
```

It also runs automatically as part of the standard consolidation cycle. The process:

1. Fetches action events from the last 30 days
2. Mines sequential pairs within the time window
3. Extracts candidates meeting the frequency threshold
4. Creates ACTION neurons + BEFORE synapses + WORKFLOW fibers
5. Prunes action events older than 60 days

## Storage

Habits are stored in the same neural graph as regular memories — no separate habits table:

| Component | Storage | Metadata |
|-----------|---------|----------|
| Habit steps | ACTION neurons | Content = action name |
| Step ordering | BEFORE synapses | `sequential_count`, `_habit: true` |
| Habit fiber | WORKFLOW fiber | `_habit_pattern: true`, `_workflow_actions: [...]`, `_habit_frequency`, `_habit_confidence` |

This means habits are queryable, recallable, and participate in spreading activation like any other memory.

## Using Habits

### List Learned Habits

```bash
# CLI
nmem habits list

# MCP
nmem_habits(action="list")
```

**Output:**
```json
{
  "habits": [
    {
      "name": "recall-edit-test",
      "steps": ["recall", "edit", "test"],
      "frequency": 5,
      "confidence": 0.8,
      "fiber_id": "uuid-..."
    }
  ],
  "count": 1
}
```

### Get Next-Action Suggestions

```bash
# CLI
nmem habits suggest --current recall

# MCP
nmem_habits(action="suggest", current_action="recall")
```

**Output:**
```json
{
  "suggestions": [
    {
      "action": "edit",
      "confidence": 0.85,
      "source_habit": "recall-edit-test",
      "sequential_count": 7
    }
  ],
  "count": 1
}
```

Suggestions require both `habit_suggestion_min_weight` (0.8) and `habit_suggestion_min_count` (5) to be met. This stricter threshold ensures only well-established patterns are suggested.

### Check Emerging Patterns

```bash
nmem habits status
```

Shows patterns that are building toward habit threshold:

```
Action events (last 30 days): 47
Sessions: 8
Learned habits: 3
Threshold for habit detection: 3 occurrences

Emerging patterns:
  remember -> consolidate    [########--] 8/3 (READY)
  recall -> context          [######----] 6/3 (READY)
  remember -> recall         [####------] 4/3 (1 more needed)
```

### Clear Habits

```bash
# CLI
nmem habits clear

# MCP
nmem_habits(action="clear")
```

Removes all learned habit fibers. Action events are preserved, so habits can be re-learned on next consolidation.

## Example Walkthrough

**Day 1-3**: You use Neural Memory normally. The system records action events:

```
Session 1: recall → remember → recall
Session 2: recall → remember → context
Session 3: recall → remember → recall
```

**Day 4**: Consolidation runs `learn_habits`:

1. Mines pairs: `recall→remember` (count=3, avg gap 12s)
2. Meets `habit_min_frequency=3` threshold
3. Creates:
   - ACTION neuron "recall"
   - ACTION neuron "remember"
   - BEFORE synapse: recall → remember (weight=0.5, sequential_count=3)
   - WORKFLOW fiber: "recall-remember" (confidence=0.375)

**Day 5+**: Pattern continues. Each consolidation reinforces the synapse weight and increments `sequential_count`.

**Day 10**: After 7+ observations and synapse weight ≥ 0.8:
- `nmem_habits(action="suggest", current_action="recall")` → suggests "remember"

## Limitations

- **Pattern length**: Only bigrams (2-step) and trigrams (3-step) are detected. Longer chains are not currently supported.
- **Time-based only**: Detection is based on temporal proximity within a session, not semantic similarity.
- **No temporal distribution check**: 3 occurrences all in one session count the same as 3 across 3 separate sessions.
- **Suggestion threshold gap**: The default suggestion threshold (weight ≥ 0.8, count ≥ 5) is much stricter than detection (frequency ≥ 3), so new habits may not produce suggestions immediately.

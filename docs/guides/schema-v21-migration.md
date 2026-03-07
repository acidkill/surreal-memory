# Schema v21 Migration Guide

Schema version 21 adds the **Cognitive Reasoning Layer** — three new tables that enable hypothesis tracking, prediction verification, and knowledge gap detection.

## What Changed

### New Tables

| Table | Purpose | Rows per brain |
|-------|---------|----------------|
| `cognitive_state` | Tracks confidence, evidence counts, and status for hypotheses and predictions | Up to 5,000 |
| `hot_index` | Pre-computed urgency ranking of active hypotheses and pending predictions | Up to 20 |
| `knowledge_gaps` | Registered knowledge deficits with priority and resolution tracking | Unlimited |

### New Indexes

```sql
idx_cognitive_confidence  ON cognitive_state(brain_id, confidence DESC)
idx_cognitive_status      ON cognitive_state(brain_id, status)
idx_gaps_brain            ON knowledge_gaps(brain_id, resolved_at)
idx_gaps_priority         ON knowledge_gaps(brain_id, priority DESC)
```

### New Synapse Types

The cognitive layer introduces these synapse types (stored in the existing `synapses` table):

- `EVIDENCE_FOR` / `EVIDENCE_AGAINST` — evidence neuron to hypothesis
- `PREDICTED` — prediction neuron to hypothesis
- `VERIFIED_BY` / `FALSIFIED_BY` — prediction to observation
- `SUPERSEDES` — new hypothesis version to old version

### New Memory Types

Three new `MemoryType` enum values:

| Type | Decay Rate | Expiry |
|------|-----------|--------|
| `hypothesis` | 0.03 | 180 days |
| `prediction` | 0.10 | 30 days |
| `schema` | 0.01 | Never |

## How Migration Works

**Migration is fully automatic.** When Neural Memory starts and detects schema version 20, it runs the v20→v21 migration:

1. Creates `cognitive_state`, `hot_index`, and `knowledge_gaps` tables
2. Creates associated indexes
3. Updates the schema version to 21

No data is modified or deleted. Existing neurons, synapses, fibers, and brains are untouched.

### Migration Path

```
v20 (pre-cognitive) → v21 (cognitive layer)
     ↑ automatic, additive only
```

The migration uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`, so it's safe to run multiple times (idempotent).

## Upgrade Steps

### Standard Upgrade (pip)

```bash
pip install --upgrade neural-memory
# Migration runs automatically on first use
```

### Editable Install (development)

```bash
cd /path/to/neural-memory
git pull
pip install -e .
# Migration runs automatically on first use
```

### Verify Migration

```bash
# Check schema version
nmem brain list
# Should show: Schema: 21

# Or via Python
python -c "
import asyncio
from neural_memory.storage import SQLiteStorage
async def check():
    s = SQLiteStorage()
    await s.initialize()
    v = await s._get_schema_version()
    print(f'Schema version: {v}')
asyncio.run(check())
"
```

## Rollback

If you need to downgrade to a version before v2.27.0 (schema v20):

1. The old code will **not** drop the new tables — they'll be ignored
2. Cognitive data (hypotheses, predictions, gaps) will remain in the database but unused
3. No data loss occurs in either direction

To fully remove the cognitive tables (optional):

```sql
DROP TABLE IF EXISTS cognitive_state;
DROP TABLE IF EXISTS hot_index;
DROP TABLE IF EXISTS knowledge_gaps;
-- Then update schema version
UPDATE schema_version SET version = 20;
```

## Troubleshooting

### "no such table: cognitive_state"

The auto-migration didn't run. This can happen if:
- You're using a custom storage path that bypasses initialization
- The database file is read-only

Fix: ensure `storage.initialize()` is called, or run the migration manually:

```python
import asyncio
from neural_memory.storage import SQLiteStorage

async def migrate():
    s = SQLiteStorage(db_path="/path/to/your/brain.db")
    await s.initialize()  # Runs pending migrations
    print("Done")

asyncio.run(migrate())
```

### Schema version stuck at 20

Check for migration errors in logs:

```bash
nmem --verbose brain list 2>&1 | grep -i "migrat"
```

If the database is corrupted, you can force a fresh migration by deleting the schema version row (the tables will be recreated):

```sql
DELETE FROM schema_version;
```

Then restart Neural Memory — it will re-run all migrations from scratch.

## Database Size Impact

The cognitive tables are lightweight:
- `cognitive_state`: ~200 bytes per entry, max 5,000 per brain = ~1 MB max
- `hot_index`: 20 rows per brain = negligible
- `knowledge_gaps`: ~150 bytes per entry, typically dozens per brain

Total impact: under 2 MB even for heavy use.

# Phase B6: Contextual Compression — Done

## Goal
Return compressed content for old memories during recall — fit 3-5x more memories in agent context window without losing key information.

## Implementation

### `compress_for_recall()` in retrieval_context.py
Pure function with age-based tiers:

```
< 7 days:  full content (unchanged)
7-30 days: summary (if available) or first 3 sentences
30-90 days: summary (if available) or first 2 sentences
90+ days:  summary (if available) or first sentence only
```

**Sentence splitting**: regex `(?<=[.!?])\s+` — handles standard punctuation.

**Integration**: Applied in `format_context()` after content selection, before structured formatting. Every fiber's content goes through `compress_for_recall()` using `fiber.created_at` as age reference.

### Key Design Decisions
- **Recall-time only**: No storage changes, no schema migration — compression is computed at read time
- **Summary-first**: If consolidation has generated a summary, always prefer it (any tier)
- **Sentence truncation fallback**: When no summary, extractive approach (first N sentences) preserves the opening context which is usually the most informative
- **Safe fallback**: No `created_at` = full content (never lose data silently)
- **No config yet**: Tier thresholds are constants (7/30/90 days) — can be made configurable later if needed

## VISION Checklist
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | Neutral — compression is post-retrieval formatting |
| 2 | Spreading Activation | SA unchanged — compression only affects output |
| 3 | No-Embedding Test | Pure text manipulation, zero embeddings |
| 4 | Detail->Speed | More memories fit in context = agent has broader view |
| 5 | Source Traceable | Original content still in storage, only recall output compressed |
| 6 | Brain Test | Human memory: old memories become gist, recent are vivid |
| 7 | Memory Lifecycle | Respects consolidation: summaries from SUMMARIZE strategy used first |

## Tests (12 tests in test_recall_compression.py)
- Recent memory returns full content
- Medium age with summary returns summary
- Medium age without summary truncates to 3 sentences
- Old memory with summary returns summary
- Old memory without summary extracts key phrases
- Very old memory returns minimal (1 sentence)
- Very old with summary uses summary
- Edge cases: exactly 7, 30, 90 days
- Empty content returns empty
- None created_at returns full content (safe fallback)

## Files Changed
- `src/neural_memory/engine/retrieval_context.py` — `compress_for_recall()` + integration in `format_context()`
- `tests/unit/test_recall_compression.py` — new (12 tests)

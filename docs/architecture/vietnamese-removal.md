# Removing Vietnamese & the Cross-Language Layer

> **Status:** applied on branch `refactor/remove-vietnamese` (2026-07-11).
> Extraction is **English-only** from this change forward.

## Summary

Surreal-Memory carried a bilingual (English + Vietnamese) surface across its
text-extraction pipeline plus a small rule-based **cross-language / "translation"
layer** that mapped a handful of English ↔ Vietnamese terms at query time. That
surface had not been maintained in a long time, was exercised by a niche slice of
the test suite, and coupled two optional Vietnamese NLP dependencies
(`underthesea`, `pyvi`) into the extras matrix.

This change removes it. The extraction pipeline — keyword extraction, entity
detection, sentiment, temporal parsing, relation extraction, arousal scoring, and
prediction-error reversal detection — is now **English-only**.

## Why

- **Unmaintained.** The Vietnamese lexicons/patterns had drifted; nobody was
  keeping them current.
- **Cost > value.** The bilingual paths added branches, patterns, and two heavy
  optional dependencies for a capability few installs used.
- **Overlap with embeddings.** Genuine multilingual recall is better served by the
  embedding layer (see [What is *not* affected](#what-is-not-affected)), so the
  rule-based translation map was redundant.

## What was removed

### Code

Removed helpers / symbols across `src/surreal_memory/`:

- `extraction/`: `detect_language`, `_get_stop_words(language)`,
  `_extract_vietnamese_names` / `_map_underthesea_type` (entities),
  Vietnamese lexicons in `sentiment.py`, `relations.py`, `temporal.py`
  (`_resolve_vi_hour`), `keywords.py`, and the Vietnamese branch in `router.py`.
- `engine/`: `_is_vietnamese`, `normalize_vietnamese_compound`,
  `_tokenize_vietnamese`, `_strip_diacritics` (`token_normalizer.py`),
  the Vietnamese arousal patterns (`arousal.py`), the Vietnamese reversal patterns
  (`prediction_error.py`), and the `CROSS_LANG_MAP` / `_CROSS_LANG_PAIRS` EN↔VI
  query-expansion map (`query_expander.py`).
- `mcp/`: the recall handler's `_check_cross_language_hint` and the Vietnamese
  branches in `auto_capture.py`, `onboarding_handler.py`, `remember_handler.py`,
  `response_compactor.py`.

The `language` parameter on extraction call sites is **retained for
backward-compatible callers but ignored** — everything is treated as English.

### Dependencies

- Dropped the `[nlp-vi]` extra (`underthesea>=6.0`, `pyvi>=0.1`).
- `nlp` now resolves to `surreal-memory[nlp-en]` only.
- Removed the `underthesea.*` / `pyvi.*` mypy overrides and the
  `ignore::DeprecationWarning:pyvi` pytest filter.

### Tests

- Deleted the Vietnamese/cross-language suites: `test_vietnamese_capture.py`,
  `test_vietnamese_keywords.py`, `test_cross_language_hint.py`, and
  `scripts/e2e_gemini_recall_100vi.py`.
- Pruned Vietnamese cases and the `CROSS_LANG_MAP` import from the shared unit,
  integration, and e2e tests; retained language-agnostic coverage.

## What is *not* affected

- **Embedding-level multilingual recall.** Semantic search across languages is a
  property of the **embedding model** (e.g. Gemini `gemini-embedding-001`, or the
  local `paraphrase-multilingual-MiniLM-L12-v2`), not of the removed rule-based
  layer. Enable embeddings and cross-language semantic recall still works. See
  [Embedding Setup](../guides/embedding-setup.md).
- **Accent-insensitive full-text search.** SurrealDB's full-text analyzer and the
  SQLite test-fixture FTS5 tokenizer still strip diacritics, so accented Latin
  content (e.g. `café`) matches unaccented queries — this is a general full-text
  feature, not Vietnamese support.

## Backward compatibility

- Existing brains are unchanged; no migration is required.
- Calls that still pass `language="vi"` (or any language) continue to work — the
  argument is accepted and ignored, and extraction runs in English.

## References

- Changelog: [`CHANGELOG.md`](../../CHANGELOG.md) → *Unreleased → Removed*.
- Architecture overview: [`architecture/overview.md`](overview.md).

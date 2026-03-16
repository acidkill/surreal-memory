# Feature: Source-Aware Brain

## Overview
Transform NM from "memory that stores text" to "smart index that knows what's where." Brain neurons act as semantic index entries pointing to source locations. When exact quotes are needed, system fetches from source documents — not from neuron content.

**Business driver**: SMB chatbots for law/accounting firms need exact citations (page, paragraph, file) to be trustworthy. Brain recalls the concept; source lookup provides the evidence.

## Phases
| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Source Locators in Training | ⬚ Pending | plan-source-aware-brain-phase1.md | Enrich DocChunk + neuron metadata with page/offset, auto-create Source + SOURCE_OF |
| 2 | Citation Tool | ⬚ Pending | plan-source-aware-brain-phase2.md | `nmem_cite` MCP tool, SourceResolver protocol, LocalResolver, staleness detection |
| 3 | Source Refresh | ⬚ Pending | plan-source-aware-brain-phase3.md | `nmem_train action="refresh"`, stale neuron marking, optional retrain |
| 4 | Cloud Resolvers | ⬚ Pending | plan-source-aware-brain-phase4.md | S3Resolver, GDriveResolver as optional extras |

## Architecture

```
Training:  file → extract (page markers) → chunk (with locators) → encode → neurons + Source + SOURCE_OF
Recall:    query → spreading activation → relevant neurons (fast, <200ms)
Citation:  neuron_ids → source_locator metadata → SourceResolver → exact text from file
Refresh:   scan sources → hash compare → mark stale → optional retrain
```

## Key Decisions
- `nmem_cite` is a SEPARATE tool from `nmem_recall` — keeps recall fast, citation is optional I/O
- No file watcher daemon — use `refresh` command + external cron/trigger
- Cloud resolvers are optional extras (`neural-memory[cloud-s3]`), not core dependencies
- Source locators go in neuron `metadata` JSON — no schema migration needed
- Only `LocalResolver` for MVP — add cloud when customer needs it
- Do NOT build: OCR, document editor, version diffing, real-time sync

## Embedding & Business Strategy

### Current State
- Default model: `all-MiniLM-L6-v2` (~80MB, English, 384D)
- Multilingual: `paraphrase-multilingual-MiniLM-L12-v2` (~440MB, 50+ langs)
- Embedding disabled by default (`enabled = false`), lazy-load on first embed call
- No download progress/warning — first recall after enabling can lag minutes silently

### TODO: UX Improvements Needed
- [ ] Show download progress or warning in dashboard when embedding first enabled
- [ ] `nmem_setup embedding` wizard should warn about model size before download
- [ ] Config-status card already shows 3 states (v4.4.1) — may need "downloading..." state
- [ ] Consider Ollama/Gemini as zero-download alternatives for SMB demos

### Business Tiering (Path B: Open Core + Cloud)
- **Free (OSS)**: Keyword recall only (no model download, zero friction onboarding)
- **Pro (Cloud)**: Cloud-hosted embedding API (no local download, fast setup for SMB chatbots)
- **Enterprise**: Custom models, on-prem embedding (legal compliance, air-gapped)

### Impact on Source-Aware Brain
- Phase 1-2 (source locators, citation) work WITHOUT embedding — keyword recall sufficient
- Embedding amplifies Phase 1-2 quality: semantic search finds related law clauses even with different wording
- Phase 3 (refresh) benefits from embedding: detect semantic drift, not just hash change
- **Priority**: Ship Phase 1-2 first (keyword-only works), then embedding enhances quality

## Multi-Agent Enrichment Strategy

### Problem
NM has no internal LLM. Memory quality depends entirely on the calling agent.
Claude follows MCP instructions well → rich memories. Other agents (GPT, Gemini,
local LLMs) may send flat, context-free text → orphan neurons with zero connections.

### Solution: Structured Context + Template Merge (Approach C)
Instead of requiring agents to craft perfect prose, accept structured `context` dict
and merge server-side into rich content using templates.

```
nmem_remember(
    content="Chose PostgreSQL over MongoDB",
    context={"reason": "ACID for payments", "alternatives": ["MongoDB"]},
    type="decision"
)
→ NM merges: "Chose PostgreSQL over MongoDB because ACID for payments.
   Alternatives considered: MongoDB."
```

### Implementation Phases
- [ ] **Phase A**: `context` field on `nmem_remember` — optional dict, merged via templates
- [ ] **Phase B**: Quality scoring — soft gate, always stores, returns score + hints
      - Score 0-10 from: content length (min 10 chars), has context dict (+3), has tags (+1),
        type diversity (+1), cognitive richness (causal/temporal/comparative words, +1 each)
      - Response includes `quality: "low"|"medium"|"high"`, `score: int`, `hints: [str]`
      - NEVER hard reject — global system can't gatekeep per-user preferences
      - Dashboard shows quality distribution chart (% low/medium/high)
- [ ] **Phase C**: Optional LLM enrichment — if `[enrichment] provider` configured in config.toml,
      NM calls LLM API to rephrase flat content into rich causal/temporal language
- [ ] **Phase D**: Agent SDK — thin wrapper libraries (Python/JS) that guide agents to provide
      structured context, with `.remember_decision()`, `.remember_error()` helper methods

### Business Impact
- Phase A-B = free tier (template enrichment, no LLM needed)
- Phase C = Pro tier (cloud-hosted enrichment API, no agent-side work)
- Phase D = Enterprise SDK (white-label, custom templates per domain)

## What Already Exists (70% done)
- `Source` dataclass with `SourceType`, `SourceStatus`, `file_hash`, `metadata`
- `SOURCE_OF` synapse type (defined, used in `nmem_remember`, but NOT in `nmem_train`)
- `nmem_source` + `nmem_provenance` MCP tools
- `training_files` table with file hash dedup
- `DocChunk` stores `source_file`, `line_start`, `line_end`, `heading_path`
- `citation.py` engine with INLINE/FOOTNOTE/FULL formats + domain templates (law, ledger)
- Citation building in recall/show/recap handlers (reads SOURCE_OF synapses)

## What's Missing (the 30%)
- DocChunk lacks `page_number`, `char_offset_start`, `char_offset_end`
- PDF extraction loses page boundaries (pymupdf4llm → markdown drops page info)
- `nmem_train` does NOT create Source records or SOURCE_OF synapses
- No SourceResolver protocol for fetching exact text from files
- No `nmem_cite` tool for on-demand source dereference
- No `refresh` action to detect source changes

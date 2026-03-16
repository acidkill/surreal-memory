# Phase C1+C2: Domain Entities + Structured Data Neurons

## Goal
Extend entity extraction with domain-specific types (financial, legal, technical) AND encode tables/key-value/JSON as first-class graph neurons with structural metadata. Combined because C2 depends on C1's entity subtypes.

## Brain Test (VISION.md)
| # | Check | Pass? |
|---|-------|-------|
| 1 | Activation vs Search | ✅ Domain entities + table cells are graph nodes in SA |
| 2 | SA still center | ✅ All recalled through spreading activation |
| 3 | No-Embedding Test | ✅ Keyword/entity match works without embeddings |
| 4 | Detail→Speed | ✅ "ROE Q3 2024" → specific cell, not entire table |
| 5 | Source Traceable | ✅ SOURCE_OF synapses preserved |
| 6 | Brain Test | ✅ Accountant brain categorizes "ROE=12.8%" differently from "met Sarah" |
| 7 | Memory Lifecycle | ✅ Domain entities decay/consolidate normally |

## Part A: Domain Entity Types (from C1)

### New Entity Subtypes
```python
class EntitySubtype(StrEnum):
    # Existing (implicit)
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    # Financial
    FINANCIAL_METRIC = "financial_metric"    # ROE, revenue, EBITDA
    CURRENCY_AMOUNT = "currency_amount"      # $25M, 500 triệu VND
    FISCAL_PERIOD = "fiscal_period"          # Q1 2024, FY2025
    # Legal
    REGULATION = "regulation"               # Điều 468 BLDS, Section 301
    CONTRACT_CLAUSE = "contract_clause"
    LEGAL_ENTITY = "legal_entity"
    # Technical
    API_ENDPOINT = "api_endpoint"
    CODE_SYMBOL = "code_symbol"
    VERSION = "version"
```

### Storage
- Neuron metadata: `{"entity_subtype": "financial_metric", "raw_value": "12.8%", "unit": "percent"}`
- Backward compatible: existing entities unaffected (no subtype = generic)
- Metadata index: `idx_neuron_entity_subtype`

## Part B: Structured Data Neurons (from C2)

### Table as Graph
```
TABLE neuron: "Báo cáo tài chính Q3/2024"
  ├── COLUMN neurons: "Chỉ tiêu", "Q1", "Q2", "Q3"
  ├── ROW neurons: "Doanh thu", "Chi phí", "Lợi nhuận"
  └── CELL neurons: "Doanh thu Q3 = 500 tỷ"
      ├── HAS_VALUE → VALUE neuron
      ├── IN_ROW → ROW neuron
      ├── IN_COLUMN → COLUMN neuron
      └── SOURCE_OF → SOURCE
```

### Structure Detection
- Markdown tables (`|---|---|`)
- CSV-like (consistent column count)
- Key-value pairs (3+ `key: value` lines)
- JSON objects/arrays

### Verbatim Recall
Neurons with `_verbatim: true` return exact `raw_value`, skip compression/summarization.

## Tasks
- [ ] Define `EntitySubtype` enum in `core/memory_types.py`
- [ ] Add financial extraction patterns to `extraction/entities.py` (Vietnamese + English)
- [ ] Add legal extraction patterns (Điều/Article/Section/Clause)
- [ ] Add technical extraction patterns (API endpoints, code symbols, versions)
- [ ] Store `entity_subtype` in neuron metadata
- [ ] Add metadata index for fast subtype queries
- [ ] Add domain synapse types: HAS_VALUE, MEASURED_AT, REGULATES, IN_ROW, IN_COLUMN
- [ ] Implement `detect_structure()` — markdown table, CSV, key-value, JSON
- [ ] Implement `StructuredDataStep` pipeline step (table → graph encoding)
- [ ] Add `_verbatim` flag + verbatim recall path (skip compression)
- [ ] Domain-aware consolidation rules (financial: group by metric+period)
- [ ] Tests: pattern extraction (Vi+En), metadata storage, table encoding, verbatim recall
- [ ] Integration test: encode financial table → recall exact cell value

## Acceptance Criteria
- [ ] "ROE = 12.8%" → ENTITY neuron with subtype=financial_metric
- [ ] "Điều 468 BLDS" → ENTITY neuron with subtype=regulation
- [ ] Markdown table → graph of table/row/column/cell neurons
- [ ] Recall "ROE Q3" → returns exact "12.8%", not summary
- [ ] Vietnamese + English patterns work
- [ ] Backward compatible: existing entities unaffected

## Files Touched
- `src/neural_memory/core/memory_types.py` — modify (EntitySubtype, synapse types)
- `src/neural_memory/extraction/entities.py` — modify (domain patterns)
- `src/neural_memory/engine/pipeline_steps.py` — modify (StructuredDataStep)
- `src/neural_memory/engine/structure_detection.py` — new
- `src/neural_memory/engine/encoder.py` — modify (register step)
- `src/neural_memory/engine/retrieval.py` — modify (verbatim recall path)
- `src/neural_memory/engine/consolidation.py` — modify (domain rules)
- `tests/unit/test_domain_entities.py` — new
- `tests/unit/test_structured_data.py` — new

## Pitfalls
- Vietnamese numbers: "1.234.567" (dots) vs "1,234,567" (commas)
- Regex greediness: "Điều 468 Bộ luật Dân sự 2015" — capture full reference
- False positives: "Section" in prose vs "Section 301 SOX"
- Large tables: cap at top rows + summary (100+ rows → too many neurons)
- JSON in code blocks: don't parse code examples as structured data

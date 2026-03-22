# Phase C4: Agent Visualization Tool

## Goal
Add `nmem_visualize` MCP tool — agents query memory data (financial metrics, time series, distributions) and receive chart specifications that frontends can render. Brain becomes visual, not just textual.

## Brain Test (VISION.md)
| # | Check | Answer |
|---|-------|--------|
| 1 | Activation vs Search | Activation — chart data sourced from activated neurons, not SQL queries |
| 2 | SA still center | Yes — visualization starts with recall, then formats results as chart |
| 3 | No-Embedding Test | Yes — chart data comes from neuron content/metadata, not embeddings |
| 4 | Detail→Speed | Yes — "ROE 2024 trend" activates fewer neurons than "financial overview" |
| 5 | Source Traceable | Yes — each data point traces to source neuron + source document |
| 6 | Brain Test | Yes — human brain creates mental models (spatial reasoning, pattern visualization) |
| 7 | Memory Lifecycle | Unchanged — visualization is read-only |

## Motivation
- NexusRAG: no chart generation (gap in their system)
- Business use case: agent analyzes financial data → needs to show trends, comparisons
- Current NM: returns text only — agent can't "show" patterns visually
- VISION.md: "Bộ não thật có tạo mô hình mental không?" → Yes, spatial reasoning

## Design

### MCP Tool Interface

```python
# nmem_visualize(query, chart_type, format)
{
    "name": "nmem_visualize",
    "description": "Generate chart from memory data. Returns chart specification.",
    "parameters": {
        "query": "ROE trend across quarters",
        "chart_type": "line|bar|pie|scatter|table|timeline",  # Optional, auto-detect
        "format": "vega_lite|markdown_table|ascii",  # Output format
        "limit": 20  # Max data points
    }
}
```

### Pipeline

```
Agent calls nmem_visualize("ROE trend across quarters")
  │
  ▼ Step 1: Recall (spreading activation)
  Activate neurons matching "ROE" + "quarters"
  → Find FINANCIAL_METRIC neurons with subtype=financial_metric
  → Traverse MEASURED_AT synapses to FISCAL_PERIOD neurons
  │
  ▼ Step 2: Extract data series
  [{period: "Q1/2024", value: 12.8, metric: "ROE"},
   {period: "Q2/2024", value: 13.2, metric: "ROE"},
   {period: "Q3/2024", value: 11.5, metric: "ROE"}]
  │
  ▼ Step 3: Auto-detect chart type (if not specified)
  - Time series data → line chart
  - Categories with values → bar chart
  - Parts of whole → pie chart
  - Two numeric axes → scatter
  - No numeric data → table
  │
  ▼ Step 4: Generate chart specification
  Vega-Lite JSON → frontend renders
  OR Markdown table → text-based fallback
  OR ASCII chart → terminal display
```

### Chart Type Auto-Detection

```python
def detect_chart_type(data_points):
    has_time = any(is_temporal(dp.get("period", dp.get("date"))) for dp in data_points)
    has_numeric = any(isinstance(dp.get("value"), (int, float)) for dp in data_points)
    has_categories = any(isinstance(dp.get("category"), str) for dp in data_points)

    if has_time and has_numeric:
        return "line"  # Time series → line chart
    if has_categories and has_numeric:
        if len(data_points) <= 6:
            return "pie"
        return "bar"
    if has_numeric and len(data_points[0]) >= 2:
        return "scatter"
    return "table"  # Fallback
```

### Output Formats

**Vega-Lite (primary)**: JSON spec, ~50 lines, any frontend renders with vega-embed
```json
{
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"values": [...]},
    "mark": "line",
    "encoding": {
        "x": {"field": "period", "type": "ordinal"},
        "y": {"field": "value", "type": "quantitative", "title": "ROE (%)"}
    }
}
```

**Markdown table (fallback)**: Works in any text context
```markdown
| Quarter | ROE (%) |
|---------|:-------:|
| Q1/2024 | 12.8 |
| Q2/2024 | 13.2 |
| Q3/2024 | 11.5 |
```

**ASCII (terminal)**: For CLI users
```
ROE Trend
13.5 ┤
13.0 ┤    ╭─╮
12.5 ┤╭──╯  │
12.0 ┤│     │
11.5 ┤│     ╰──
11.0 ┤
     └──────────
      Q1  Q2  Q3
```

### Dashboard Integration

NM dashboard can render Vega-Lite specs natively:
```tsx
// dashboard/src/components/MemoryChart.tsx
import embed from 'vega-embed';

function MemoryChart({ spec }) {
    const ref = useRef(null);
    useEffect(() => { embed(ref.current, spec); }, [spec]);
    return <div ref={ref} />;
}
```

## Tasks
- [x] Implement `nmem_visualize` MCP tool handler
- [x] Implement data extraction from recalled neurons (financial metrics, time series)
- [x] Implement chart type auto-detection
- [x] Implement Vega-Lite spec generation (line, bar, pie, scatter)
- [x] Implement markdown table fallback format
- [x] Implement ASCII chart fallback (for CLI)
- [x] Add source provenance to data points (neuron_id, source_id)
- [ ] Dashboard: add MemoryChart component (vega-embed)
- [x] Tests: data extraction, chart detection, spec generation (28 tests)
- [ ] Integration test: encode financial table → visualize → valid Vega-Lite

## Acceptance Criteria
- [ ] `nmem_visualize("ROE trend")` → Vega-Lite line chart spec
- [ ] Auto-detects chart type from data shape
- [ ] Markdown table fallback when format="markdown_table"
- [ ] Data points include source provenance (traceable)
- [ ] Dashboard renders Vega-Lite specs
- [ ] Works with domain entity data from C1 (financial_metric neurons)
- [ ] Brain test: all 7 VISION checks pass

## Files Touched
- `src/neural_memory/mcp/tool_handlers/visualize_handler.py` — new
- `src/neural_memory/engine/chart_generator.py` — new
- `src/neural_memory/mcp/server.py` — modify (register new tool)
- `dashboard/src/components/MemoryChart.tsx` — new
- `tests/unit/test_visualize.py` — new

## Dependencies
- Benefits greatly from C1 (domain entity types — financial_metric neurons)
- Benefits from C2 (structured data — table neurons with raw values)
- Independent of Track A and Track B

## Common Pitfalls to Watch
- Vega-Lite spec validation: malformed JSON → frontend crash. Validate before returning
- Data point ordering: time series must be chronologically sorted
- Missing values: gaps in time series → Vega-Lite handles with `null` but must document
- Currency formatting: "500 tỷ" needs normalization to numeric for chart axes
- Too many data points: >50 → aggregate or sample (Vega-Lite handles but slow)
- ASCII chart library: avoid heavy deps — simple custom implementation or `asciichartpy`

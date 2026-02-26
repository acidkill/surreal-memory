# Plan: Graph SVG Export (`nmem graph --export svg`)

**Date**: 2026-02-26
**Origin**: User feedback from bemiagent.com — "muốn chụp hình graph nhưng chỉ có text"
**Reference**: Feature Factory `antv-renderer.ts` (AntV SSR approach reviewed, not applicable for Python)

---

## Problem

`nmem graph` outputs Rich Tree to terminal only. Users cannot:
- Export graph as image for sharing/documentation
- Track brain growth visually over time
- Embed graph in reports or blog posts

## Approach: Pure Python SVG Generation

**Why not AntV (Feature Factory approach)?**
- AntV is Node/Bun-only — would add Node runtime dependency to a Python project
- Neural Memory graph data is simple (tree of fibers → neurons → synapses)
- Pure SVG generation is lightweight, zero external dependencies

**Why not graphviz?**
- Requires system-level graphviz binary install (`apt install graphviz`)
- Users on Windows/macOS need extra setup — friction for a CLI tool
- Overkill for the tree-like structure Neural Memory produces

**Chosen: Custom SVG builder (zero dependencies)**
- SVG is just XML — straightforward to generate programmatically
- Full control over styling (match terminal color scheme)
- No install friction — works everywhere Python works

---

## Scope

### In Scope
- `nmem graph --export svg` → writes `.svg` file
- `nmem graph --export png` → writes `.png` file (via `cairosvg` optional dep)
- SVG renders same data as current Rich Tree output (fibers, neurons, synapses)
- Dark theme SVG (matches terminal aesthetic)
- Auto-open file after export (optional `--no-open` flag)

### Out of Scope (Future)
- Interactive HTML graph (d3.js/force-directed)
- Dashboard embed via REST API
- Animation / real-time updates
- Mermaid `.mmd` export (low value — user must render separately)

---

## Design

### CLI Interface

```
nmem graph                           # Current behavior (Rich Tree in terminal)
nmem graph --export svg              # Export to SVG file
nmem graph --export png              # Export to PNG file (requires cairosvg)
nmem graph "query" --export svg      # Query-filtered graph → SVG
nmem graph --export svg -o out.svg   # Custom output path
```

### SVG Layout Algorithm

Simple top-down tree layout (matches current Rich Tree structure):

```
[Brain: default]
    |
    +-- [Fiber: "database migration patterns"]
    |       |
    |       +-- -> leads_to "schema versioning"
    |       +-- <- caused_by "ORM limitations"
    |
    +-- [Fiber: "auth system design"]
            |
            +-- <-> co_occurs "JWT tokens"
            +-- ~ related_to "session management"
```

**Layout params:**
- Node width: 280px, height: 40px
- Horizontal gap: 40px, vertical gap: 60px
- Fiber nodes: rounded rect, green border
- Neuron nodes: rounded rect, white/gray border
- Synapse edges: colored lines with labels
- Canvas padding: 40px all sides

### Color Scheme (Dark Theme)

```
Background:  #0c1419 (matches Palette A)
Fiber node:  fill #1a2332, stroke #00d084 (green)
Neuron node: fill #1a2332, stroke #2a3f52 (border)
Synapse line: #2196f3 (accent) for weight > 0.5, #2a3f52 for low weight
Text:        #ffffff (primary), #a0aeb8 (secondary)
Legend bg:   #121a20
```

### File Structure

```
src/neural_memory/
  cli/
    graph_export.py    # NEW — SVG generation + file writing
  cli/commands/
    tools.py           # MODIFY — add --export and -o options to graph()
```

### `graph_export.py` — Key Functions

```python
async def export_graph_svg(
    storage: PersistentStorage,
    query: str | None = None,
    depth: int = 2,
    output_path: str | None = None,
) -> Path:
    """Export neural graph to SVG file. Returns path to written file."""

def build_svg(
    brain_name: str,
    fibers: list[FiberNode],
    width: int,
    height: int,
) -> str:
    """Build SVG string from graph data."""

def layout_tree(fibers: list[FiberNode]) -> LayoutResult:
    """Calculate x,y positions for all nodes using tree layout."""

@dataclass(frozen=True)
class FiberNode:
    label: str
    neighbors: list[NeighborEdge]

@dataclass(frozen=True)
class NeighborEdge:
    label: str
    synapse_type: str
    synapse_icon: str
    weight: float

@dataclass(frozen=True)
class LayoutResult:
    nodes: list[PositionedNode]
    edges: list[PositionedEdge]
    width: int
    height: int
```

### PNG Export (Optional)

```toml
# pyproject.toml — optional dependency group
[project.optional-dependencies]
png = ["cairosvg>=2.7"]
```

```python
def svg_to_png(svg_content: str, output_path: Path) -> None:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_content.encode(), write_to=str(output_path))
    except ImportError:
        raise click.UsageError(
            "PNG export requires cairosvg: pip install neural-memory[png]"
        )
```

---

## Implementation Phases

### Phase 1: SVG Export Core (MVP)
1. Create `graph_export.py` with `FiberNode`, `NeighborEdge`, `LayoutResult` dataclasses
2. Implement `layout_tree()` — simple recursive top-down positioning
3. Implement `build_svg()` — generate SVG XML string with dark theme
4. Implement `export_graph_svg()` — fetch data from storage, call layout + build
5. Add `--export` and `-o` options to `graph()` command in `tools.py`
6. Tests: SVG output contains valid `<svg>` tag, correct node count, file written

### Phase 2: Polish
7. Add legend to SVG (synapse type icons)
8. Add brain name + timestamp header
9. Auto-open file after export (`webbrowser.open()` or `os.startfile()`)
10. Add `--light` flag for light theme variant

### Phase 3: PNG (Optional)
11. Add `cairosvg` optional dependency
12. Implement `svg_to_png()` wrapper
13. Route `--export png` through SVG → PNG pipeline

---

## Test Plan

- [ ] `test_build_svg_empty` — no fibers → valid SVG with "No memories" message
- [ ] `test_build_svg_single_fiber` — 1 fiber, 0 neighbors → correct node
- [ ] `test_build_svg_with_neighbors` — fiber + 3 neighbors → correct edges
- [ ] `test_layout_tree_positions` — no overlapping nodes
- [ ] `test_export_writes_file` — file exists at expected path
- [ ] `test_export_custom_path` — `-o` flag respected
- [ ] `test_svg_valid_xml` — output parses with xml.etree
- [ ] `test_png_export_missing_dep` — graceful error without cairosvg

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Large graphs (500+ neurons) → huge SVG | Cap visible nodes at 50, add "... and N more" |
| SVG rendering differences across browsers | Use basic SVG primitives only (rect, line, text, g) |
| PNG dependency friction | Make PNG fully optional, SVG is the primary target |
| Layout algorithm edge cases | Start simple (tree), upgrade to force-directed later if needed |

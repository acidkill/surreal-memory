"""SVG export for neural graph visualization.

Generates SVG files from the same data as `nmem graph` (Rich Tree),
using a simple top-down tree layout with dark theme styling.

Zero external dependencies — pure SVG/XML generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from neural_memory.cli.storage import PersistentStorage

# ── Layout constants ──────────────────────────────────────────────

NODE_WIDTH = 280
NODE_HEIGHT = 36
CHILD_NODE_WIDTH = 240
CHILD_NODE_HEIGHT = 30
H_GAP = 40
V_GAP = 50
PADDING = 40
MAX_FIBERS = 10
MAX_NEIGHBORS = 5

# ── Dark theme colors ────────────────────────────────────────────

BG_COLOR = "#0c1419"
FIBER_FILL = "#1a2332"
FIBER_STROKE = "#00d084"
NEURON_FILL = "#1a2332"
NEURON_STROKE = "#2a3f52"
EDGE_STRONG = "#2196f3"
EDGE_WEAK = "#2a3f52"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#a0aeb8"
LEGEND_BG = "#121a20"
HEADER_COLOR = "#00d084"

# ── Synapse icons (same as tui.py) ───────────────────────────────

SYNAPSE_ICONS: dict[str, str] = {
    "caused_by": "\u2190",
    "leads_to": "\u2192",
    "co_occurs": "\u2194",
    "related_to": "~",
    "similar_to": "\u2248",
    "happened_at": "@",
    "at_location": "#",
    "involves": "&",
    "before": "\u21e8",
    "contradicts": "\u2716",
}


# ── Data models ──────────────────────────────────────────────────


@dataclass(frozen=True)
class NeighborEdge:
    """A neighbor neuron connected via a synapse."""

    content: str
    synapse_type: str
    icon: str
    weight: float


@dataclass(frozen=True)
class FiberNode:
    """A fiber with its neighbor edges for visualization."""

    label: str
    neighbors: tuple[NeighborEdge, ...]


@dataclass(frozen=True)
class PositionedRect:
    """A positioned rectangle in the SVG canvas."""

    x: float
    y: float
    width: float
    height: float
    label: str
    is_fiber: bool
    edge_info: NeighborEdge | None = None


@dataclass(frozen=True)
class PositionedLine:
    """A positioned line connecting two nodes."""

    x1: float
    y1: float
    x2: float
    y2: float
    color: str


@dataclass(frozen=True)
class LayoutResult:
    """Complete layout with positioned nodes and edges."""

    rects: tuple[PositionedRect, ...]
    lines: tuple[PositionedLine, ...]
    width: int
    height: int


# ── Layout algorithm ─────────────────────────────────────────────


def layout_tree(
    brain_name: str,
    fibers: list[FiberNode],
) -> LayoutResult:
    """Calculate positions for all nodes in a top-down tree layout.

    Layout structure:
        [Header: Brain name]
        [Fiber 1]
            [Neighbor 1a]
            [Neighbor 1b]
        [Fiber 2]
            [Neighbor 2a]
        ...
    """
    rects: list[PositionedRect] = []
    lines: list[PositionedLine] = []

    y_cursor = PADDING + 30  # space for header

    for fiber in fibers:
        # Fiber node (left-aligned)
        fiber_x = PADDING
        fiber_rect = PositionedRect(
            x=fiber_x,
            y=y_cursor,
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
            label=fiber.label,
            is_fiber=True,
        )
        rects.append(fiber_rect)

        fiber_center_y = y_cursor + NODE_HEIGHT / 2
        y_cursor += NODE_HEIGHT + V_GAP // 2

        # Neighbor nodes (indented)
        child_x = PADDING + H_GAP + 20
        for neighbor in fiber.neighbors:
            child_rect = PositionedRect(
                x=child_x,
                y=y_cursor,
                width=CHILD_NODE_WIDTH,
                height=CHILD_NODE_HEIGHT,
                label=neighbor.content,
                is_fiber=False,
                edge_info=neighbor,
            )
            rects.append(child_rect)

            # Line from fiber to child
            edge_color = EDGE_STRONG if neighbor.weight > 0.5 else EDGE_WEAK
            lines.append(
                PositionedLine(
                    x1=fiber_x + 10,
                    y1=fiber_center_y,
                    x2=child_x,
                    y2=y_cursor + CHILD_NODE_HEIGHT / 2,
                    color=edge_color,
                )
            )

            y_cursor += CHILD_NODE_HEIGHT + 8

        y_cursor += V_GAP // 2

    # Calculate canvas size
    max_x = PADDING + H_GAP + 20 + CHILD_NODE_WIDTH + PADDING
    canvas_width = max(max_x, NODE_WIDTH + 2 * PADDING)
    # Add space for legend
    canvas_height = y_cursor + 120

    return LayoutResult(
        rects=tuple(rects),
        lines=tuple(lines),
        width=int(canvas_width),
        height=int(canvas_height),
    )


# ── SVG builder ──────────────────────────────────────────────────


def build_svg(
    brain_name: str,
    fibers: list[FiberNode],
    timestamp: str = "",
) -> str:
    """Build SVG string from graph data.

    Args:
        brain_name: Name of the brain being visualized.
        fibers: List of fiber nodes with their neighbors.
        timestamp: Optional timestamp string for the header.

    Returns:
        Complete SVG document as a string.
    """
    if not fibers:
        return _build_empty_svg(brain_name)

    layout = layout_tree(brain_name, fibers)
    parts: list[str] = []

    # SVG header
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{layout.width}" height="{layout.height}" '
        f'viewBox="0 0 {layout.width} {layout.height}">'
    )

    # Background
    parts.append(f'<rect width="100%" height="100%" fill="{BG_COLOR}"/>')

    # Styles
    parts.append(
        "<defs><style>"
        f".fiber-node {{ fill: {FIBER_FILL}; stroke: {FIBER_STROKE}; stroke-width: 2; rx: 8; }}"
        f".neuron-node {{ fill: {NEURON_FILL}; stroke: {NEURON_STROKE}; stroke-width: 1; rx: 6; }}"
        f".text-primary {{ fill: {TEXT_PRIMARY}; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; }}"
        f".text-secondary {{ fill: {TEXT_SECONDARY}; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 11px; }}"
        f".text-header {{ fill: {HEADER_COLOR}; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 16px; font-weight: bold; }}"
        f".text-icon {{ fill: {EDGE_STRONG}; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 12px; }}"
        f".legend-bg {{ fill: {LEGEND_BG}; stroke: {NEURON_STROKE}; stroke-width: 1; rx: 6; }}"
        f".legend-text {{ fill: {TEXT_SECONDARY}; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 10px; }}"
        "</style></defs>"
    )

    # Header
    header_text = f"Neural Graph — {escape(brain_name)}"
    if timestamp:
        header_text += f"  ({escape(timestamp)})"
    parts.append(f'<text x="{PADDING}" y="{PADDING + 5}" class="text-header">{header_text}</text>')

    # Edge lines (draw first so nodes appear on top)
    for line in layout.lines:
        parts.append(
            f'<line x1="{line.x1}" y1="{line.y1}" '
            f'x2="{line.x2}" y2="{line.y2}" '
            f'stroke="{line.color}" stroke-width="1.5" '
            f'stroke-dasharray="4,3" opacity="0.6"/>'
        )

    # Nodes
    for rect in layout.rects:
        css_class = "fiber-node" if rect.is_fiber else "neuron-node"
        parts.append(
            f'<rect x="{rect.x}" y="{rect.y}" '
            f'width="{rect.width}" height="{rect.height}" '
            f'class="{css_class}"/>'
        )

        # Text
        text_class = "text-primary" if rect.is_fiber else "text-secondary"
        text_x = rect.x + 10
        text_y = rect.y + rect.height / 2 + 4

        if rect.edge_info:
            # Show icon + type + content
            icon = escape(rect.edge_info.icon)
            syn_type = escape(rect.edge_info.synapse_type)
            label = escape(_truncate(rect.label, 25))
            parts.append(
                f'<text x="{text_x}" y="{text_y}" class="text-icon">{icon}</text>'
                f'<text x="{text_x + 16}" y="{text_y}" class="text-secondary">'
                f"{syn_type}</text>"
                f'<text x="{text_x + 16 + len(syn_type) * 6.5 + 6}" y="{text_y}" '
                f'class="{text_class}">{label}</text>'
            )
        else:
            label = escape(_truncate(rect.label, 40))
            parts.append(f'<text x="{text_x}" y="{text_y}" class="{text_class}">{label}</text>')

    # Legend
    legend_y = layout.height - 100
    parts.append(
        f'<rect x="{PADDING}" y="{legend_y}" '
        f'width="{layout.width - 2 * PADDING}" height="80" '
        f'class="legend-bg"/>'
    )
    parts.append(
        f'<text x="{PADDING + 10}" y="{legend_y + 18}" '
        f'class="text-secondary" font-weight="bold">Legend</text>'
    )

    legend_items = [
        ("\u25cf Fiber (memory)", FIBER_STROKE),
        ("\u2192 leads_to", TEXT_SECONDARY),
        ("\u2190 caused_by", TEXT_SECONDARY),
        ("\u2194 co_occurs", TEXT_SECONDARY),
        ("~ related_to", TEXT_SECONDARY),
        ("@ happened_at", TEXT_SECONDARY),
    ]
    legend_x = PADDING + 10
    for i, (text, color) in enumerate(legend_items):
        col = i % 3
        row = i // 3
        lx = legend_x + col * 130
        ly = legend_y + 36 + row * 18
        parts.append(
            f'<text x="{lx}" y="{ly}" fill="{color}" class="legend-text">{escape(text)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_empty_svg(brain_name: str) -> str:
    """Build SVG for empty graph state."""
    width, height = 400, 150
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="{BG_COLOR}"/>'
        f'<text x="{width // 2}" y="50" text-anchor="middle" '
        f'fill="{HEADER_COLOR}" font-family="system-ui" font-size="16" '
        f'font-weight="bold">Neural Graph — {escape(brain_name)}</text>'
        f'<text x="{width // 2}" y="90" text-anchor="middle" '
        f'fill="{TEXT_SECONDARY}" font-family="system-ui" font-size="13">'
        f"No memories to visualize</text>"
        "</svg>"
    )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ── Data fetching (reuses tui.py logic) ──────────────────────────


async def collect_graph_data(
    storage: PersistentStorage,
    query: str | None = None,
    depth: int = 2,
) -> tuple[str, list[FiberNode]]:
    """Collect graph data from storage, returning (brain_name, fibers).

    Mirrors the data-fetching logic of tui.render_graph().
    """
    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        return "unknown", []

    # Fetch fibers (same logic as tui.render_graph)
    if query:
        from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=query,
            depth=DepthLevel(min(depth, 3)),
            max_tokens=1000,
        )
        if result.confidence < 0.1:
            return brain.name, []

        raw_fibers = []
        for fiber_id in result.fibers_matched[:MAX_FIBERS]:
            fiber = await storage.get_fiber(fiber_id)
            if fiber:
                raw_fibers.append(fiber)
    else:
        raw_fibers = await storage.get_fibers(limit=MAX_FIBERS)

    if not raw_fibers:
        return brain.name, []

    # Build FiberNode list
    fiber_nodes: list[FiberNode] = []
    for fiber in raw_fibers:
        # Get fiber label
        content = fiber.summary or ""
        if not content and fiber.anchor_neuron_id:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            if anchor:
                content = anchor.content
        label = content[:50] + "..." if len(content) > 50 else content

        # Get neighbors
        neighbors: list[NeighborEdge] = []
        if fiber.anchor_neuron_id:
            raw_neighbors = await storage.get_neighbors(fiber.anchor_neuron_id, direction="both")
            for neighbor, synapse in raw_neighbors[:MAX_NEIGHBORS]:
                neighbor_content = neighbor.content[:30]
                if len(neighbor.content) > 30:
                    neighbor_content += "..."
                icon = SYNAPSE_ICONS.get(synapse.type.value, "\u2500")
                neighbors.append(
                    NeighborEdge(
                        content=neighbor_content,
                        synapse_type=synapse.type.value,
                        icon=icon,
                        weight=synapse.weight,
                    )
                )

        fiber_nodes.append(FiberNode(label=label, neighbors=tuple(neighbors)))

    return brain.name, fiber_nodes


async def export_graph_svg(
    storage: PersistentStorage,
    query: str | None = None,
    depth: int = 2,
    output_path: str | None = None,
) -> Path:
    """Export neural graph to SVG file.

    Args:
        storage: Storage backend.
        query: Optional query to filter memories.
        depth: Traversal depth (1-3).
        output_path: Custom output file path. Defaults to neural_graph.svg.

    Returns:
        Path to the written SVG file.
    """
    from neural_memory.utils.timeutils import utcnow

    brain_name, fibers = await collect_graph_data(storage, query, depth)

    timestamp = utcnow().strftime("%Y-%m-%d %H:%M")
    svg_content = build_svg(brain_name, fibers, timestamp)

    if output_path:
        out = Path(output_path)
    else:
        out = Path(f"neural_graph_{brain_name}.svg")

    out.write_text(svg_content, encoding="utf-8")
    return out

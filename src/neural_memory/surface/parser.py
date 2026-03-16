"""Parser for .nm Knowledge Surface format.

Reads a .nm text file and produces a KnowledgeSurface object.
Zero external dependencies — hand-parses YAML frontmatter and all sections.
Lenient: skips unknown sections for forward compatibility.
"""

from __future__ import annotations

import re

from neural_memory.surface.models import (
    Cluster,
    DepthHint,
    DepthLevel,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SignalLevel,
    SurfaceEdge,
    SurfaceFrontmatter,
    SurfaceMeta,
    SurfaceNode,
)

# ── Regex patterns ─────────────────────────────────

# Frontmatter: key: value (simple flat YAML)
_FRONTMATTER_KV = re.compile(r"^(\w+):\s*(.+)$")

# GRAPH node: [id] content (type) {p:N}
_GRAPH_NODE = re.compile(r"^\[(\w+)\]\s+(.+?)\s+\((\w+)\)\s+\{p:(\d+)\}\s*$")

# GRAPH node without edges (standalone): [id] content (type) {p:N}
_GRAPH_NODE_STANDALONE = re.compile(r"^\[(\w+)\]\s+(.+?)\s+\((\w+)\)\s+\{p:(\d+)\}\s*$")

# GRAPH edge: →edge_type→ [target_id] "text"  OR  →edge_type→ "text"
_GRAPH_EDGE_WITH_ID = re.compile(r"^\s+→(\w+)→\s+\[(\w+)\]\s+\"(.+?)\"\s*$")
_GRAPH_EDGE_INLINE = re.compile(r"^\s+→(\w+)→\s+\"(.+?)\"\s*$")

# CLUSTER: @name: [id1, id2] "description"
_CLUSTER = re.compile(r"^@(\w+):\s+\[([^\]]*)\]\s+\"(.+?)\"\s*$")

# SIGNAL: !/~/? [id] text  OR  !/~/? text
_SIGNAL_WITH_ID = re.compile(r"^([!~?])\s+\[(\w+)\]\s+(.+)$")
_SIGNAL_NO_ID = re.compile(r"^([!~?])\s+(.+)$")

# DEPTH MAP: [id] → LEVEL (context)
_DEPTH_HINT = re.compile(
    r"^\[(\w+)\]\s+→\s+(SUFFICIENT|NEEDS_DETAIL|NEEDS_DEEP)(?:\s+\((.+)\))?\s*$"
)

# META: key: value
_META_KV = re.compile(r"^(\w+):\s*(.+)$")

# Section header: # ── SECTION_NAME ──
_SECTION_HEADER = re.compile(r"^#\s+──\s+(\w[\w\s]*?)\s+──")

# Known section names (normalized to lowercase)
_KNOWN_SECTIONS = {"graph", "clusters", "signals", "depth map", "meta"}


class ParseError(ValueError):
    """Raised when .nm format is invalid."""


def parse(text: str) -> KnowledgeSurface:
    """Parse a .nm Knowledge Surface file.

    Args:
        text: Full content of a .nm file.

    Returns:
        Parsed KnowledgeSurface object.

    Raises:
        ParseError: If frontmatter is missing or format is invalid.
    """
    lines = text.splitlines()

    # Step 1: Parse frontmatter
    frontmatter, body_start = _parse_frontmatter(lines)

    # Step 2: Split into sections
    sections = _split_sections(lines[body_start:])

    # Step 3: Parse each section
    graph = _parse_graph(sections.get("graph", []))
    clusters = _parse_clusters(sections.get("clusters", []))
    signals = _parse_signals(sections.get("signals", []))
    depth_map = _parse_depth_map(sections.get("depth map", []))
    meta = _parse_meta(sections.get("meta", []))

    return KnowledgeSurface(
        frontmatter=frontmatter,
        graph=tuple(graph),
        clusters=tuple(clusters),
        signals=tuple(signals),
        depth_map=tuple(depth_map),
        meta=meta,
    )


def _parse_frontmatter(lines: list[str]) -> tuple[SurfaceFrontmatter, int]:
    """Extract YAML frontmatter between --- delimiters."""
    # Find first ---
    start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            start = i
            break
        # Skip comments and blank lines before frontmatter
        if stripped and not stripped.startswith("#"):
            break

    if start == -1:
        raise ParseError("Missing frontmatter: no opening '---' found")

    # Find closing ---
    end = -1
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break

    if end == -1:
        raise ParseError("Missing frontmatter: no closing '---' found")

    # Parse key-value pairs
    kv: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _FRONTMATTER_KV.match(stripped)
        if match:
            kv[match.group(1)] = match.group(2).strip()

    brain = kv.get("brain", "default")
    updated = kv.get("updated", "")

    # Parse numeric fields (support "neurons: 847" and "neurons: 847 | synapses: 3201")
    neurons = _parse_int_from_kv(kv, "neurons", 0)
    synapses = _parse_int_from_kv(kv, "synapses", 0)
    token_budget = _parse_int_from_kv(kv, "token_budget", 1200)

    # Parse depth_available list
    depth_str = kv.get("depth_available", "")
    if depth_str:
        depth_available = _parse_list_value(depth_str)
    else:
        depth_available = ("surface", "detail", "deep")

    return (
        SurfaceFrontmatter(
            brain=brain,
            updated=updated,
            neurons=neurons,
            synapses=synapses,
            token_budget=token_budget,
            depth_available=depth_available,
        ),
        end + 1,
    )


def _parse_int_from_kv(kv: dict[str, str], key: str, default: int) -> int:
    """Parse an integer from a frontmatter value, handling pipe-separated format."""
    raw = kv.get(key, "")
    if not raw:
        return default
    # Handle "847 | synapses: 3201" → take first number
    num_str = raw.split("|")[0].strip()
    try:
        return int(num_str)
    except ValueError:
        return default


def _parse_list_value(raw: str) -> tuple[str, ...]:
    """Parse a bracketed list value like '[surface, detail, deep]'."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    items = [item.strip().strip("\"'") for item in raw.split(",")]
    return tuple(item for item in items if item)


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Split body into named sections based on section headers."""
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _SECTION_HEADER.match(line)
        if match:
            # Save previous section
            if current_section is not None:
                sections[current_section] = current_lines
            section_name = match.group(1).strip().lower()
            current_section = section_name
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # Save last section
    if current_section is not None:
        sections[current_section] = current_lines

    return sections


def _parse_graph(lines: list[str]) -> list[GraphEntry]:
    """Parse GRAPH section into GraphEntry objects."""
    entries: list[GraphEntry] = []
    current_node: SurfaceNode | None = None
    current_edges: list[SurfaceEdge] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Try node match
        node_match = _GRAPH_NODE.match(stripped)
        if node_match:
            # Save previous entry
            if current_node is not None:
                entries.append(
                    GraphEntry(
                        node=current_node,
                        edges=tuple(current_edges),
                    )
                )
            current_node = SurfaceNode(
                id=node_match.group(1),
                content=node_match.group(2),
                node_type=node_match.group(3),
                priority=int(node_match.group(4)),
            )
            current_edges = []
            continue

        # Try edge with ID: →edge→ [id] "text"
        edge_id_match = _GRAPH_EDGE_WITH_ID.match(line)
        if edge_id_match:
            current_edges.append(
                SurfaceEdge(
                    edge_type=edge_id_match.group(1),
                    target_id=edge_id_match.group(2),
                    target_text=edge_id_match.group(3),
                )
            )
            continue

        # Try inline edge: →edge→ "text"
        edge_inline_match = _GRAPH_EDGE_INLINE.match(line)
        if edge_inline_match:
            current_edges.append(
                SurfaceEdge(
                    edge_type=edge_inline_match.group(1),
                    target_text=edge_inline_match.group(2),
                )
            )
            continue

    # Save last entry
    if current_node is not None:
        entries.append(
            GraphEntry(
                node=current_node,
                edges=tuple(current_edges),
            )
        )

    return entries


def _parse_clusters(lines: list[str]) -> list[Cluster]:
    """Parse CLUSTERS section into Cluster objects."""
    clusters: list[Cluster] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        match = _CLUSTER.match(stripped)
        if match:
            name = match.group(1)
            ids_str = match.group(2)
            description = match.group(3)
            node_ids = tuple(item.strip() for item in ids_str.split(",") if item.strip())
            clusters.append(
                Cluster(
                    name=name,
                    node_ids=node_ids,
                    description=description,
                )
            )

    return clusters


def _parse_signals(lines: list[str]) -> list[Signal]:
    """Parse SIGNALS section into Signal objects."""
    signals: list[Signal] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Try signal with ID
        match_id = _SIGNAL_WITH_ID.match(stripped)
        if match_id:
            signals.append(
                Signal(
                    level=SignalLevel(match_id.group(1)),
                    node_id=match_id.group(2),
                    text=match_id.group(3),
                )
            )
            continue

        # Try signal without ID
        match_no_id = _SIGNAL_NO_ID.match(stripped)
        if match_no_id:
            signals.append(
                Signal(
                    level=SignalLevel(match_no_id.group(1)),
                    text=match_no_id.group(2),
                )
            )

    return signals


def _parse_depth_map(lines: list[str]) -> list[DepthHint]:
    """Parse DEPTH MAP section into DepthHint objects."""
    hints: list[DepthHint] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        match = _DEPTH_HINT.match(stripped)
        if match:
            hints.append(
                DepthHint(
                    node_id=match.group(1),
                    level=DepthLevel(match.group(2)),
                    context=match.group(3) or "",
                )
            )

    return hints


def _parse_meta(lines: list[str]) -> SurfaceMeta:
    """Parse META section into SurfaceMeta object."""
    kv: dict[str, str] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        match = _META_KV.match(stripped)
        if match:
            key = match.group(1)
            val = match.group(2).strip()
            kv[key] = val

    coverage = _safe_float(kv.get("coverage", ""), 0.0)
    staleness = _safe_float(kv.get("staleness", ""), 0.0)
    last_consolidation = kv.get("last_consolidation", "")

    top_entities_str = kv.get("top_entities", "")
    top_entities = _parse_list_value(top_entities_str) if top_entities_str else ()

    return SurfaceMeta(
        coverage=coverage,
        staleness=staleness,
        last_consolidation=last_consolidation,
        top_entities=top_entities,
    )


def _safe_float(val: str, default: float) -> float:
    """Parse float, handling pipe-separated values."""
    if not val:
        return default
    num_str = val.split("|")[0].strip()
    try:
        return float(num_str)
    except ValueError:
        return default

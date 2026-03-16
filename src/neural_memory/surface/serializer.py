"""Serializer for .nm Knowledge Surface format.

Converts a KnowledgeSurface object back to .nm text.
Round-trip fidelity: parse(serialize(surface)) should equal original surface.
"""

from __future__ import annotations

from neural_memory.surface.models import (
    Cluster,
    DepthHint,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SurfaceFrontmatter,
    SurfaceMeta,
)


def serialize(surface: KnowledgeSurface) -> str:
    """Serialize a KnowledgeSurface to .nm format text.

    Args:
        surface: The knowledge surface to serialize.

    Returns:
        Complete .nm file content as a string.
    """
    parts: list[str] = []

    # Header
    parts.append(_serialize_header())
    parts.append(_serialize_frontmatter(surface.frontmatter))
    parts.append("")

    # GRAPH
    if surface.graph:
        parts.append(_serialize_graph(surface.graph))
        parts.append("")

    # CLUSTERS
    if surface.clusters:
        parts.append(_serialize_clusters(surface.clusters))
        parts.append("")

    # SIGNALS
    if surface.signals:
        parts.append(_serialize_signals(surface.signals))
        parts.append("")

    # DEPTH MAP
    if surface.depth_map:
        parts.append(_serialize_depth_map(surface.depth_map))
        parts.append("")

    # META
    parts.append(_serialize_meta(surface.meta))

    return "\n".join(parts) + "\n"


def _serialize_header() -> str:
    """Produce the decorative file header."""
    sep = "═" * 51
    return (
        f"# {sep}\n"
        "# .neuralmemory/surface.nm\n"
        "# Neural Memory Knowledge Surface v1\n"
        "# Auto-generated from brain.db | Editable by human\n"
        f"# {sep}"
    )


def _serialize_frontmatter(fm: SurfaceFrontmatter) -> str:
    """Serialize frontmatter as YAML between --- delimiters."""
    depth_list = ", ".join(fm.depth_available)
    return (
        "---\n"
        f"brain: {fm.brain}\n"
        f"updated: {fm.updated}\n"
        f"neurons: {fm.neurons}\n"
        f"synapses: {fm.synapses}\n"
        f"token_budget: {fm.token_budget}\n"
        f"depth_available: [{depth_list}]\n"
        "---"
    )


def _serialize_graph(graph: tuple[GraphEntry, ...]) -> str:
    """Serialize GRAPH section."""
    lines: list[str] = []
    lines.append("# ── GRAPH ──────────────────────────────────────────")
    lines.append("# Nodes = key concepts. Edges = causal/decisional links.")
    lines.append("# Agent: scan this section to understand the knowledge landscape.")
    lines.append("# Format: [id] content (type) {p:N} →edge→ [target_id_or_text]")
    lines.append("")

    for entry in graph:
        node = entry.node
        lines.append(f"[{node.id}] {node.content} ({node.node_type}) {{p:{node.priority}}}")
        for edge in entry.edges:
            if edge.target_id:
                lines.append(f'  →{edge.edge_type}→ [{edge.target_id}] "{edge.target_text}"')
            else:
                lines.append(f'  →{edge.edge_type}→ "{edge.target_text}"')
        lines.append("")

    # Remove trailing blank line
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def _serialize_clusters(clusters: tuple[Cluster, ...]) -> str:
    """Serialize CLUSTERS section."""
    lines: list[str] = []
    lines.append("# ── CLUSTERS ───────────────────────────────────────")
    lines.append("# Auto-detected topic groups from brain.db entity co-occurrence.")
    lines.append("# Agent: use clusters to scope recall queries.")
    lines.append("")

    for cluster in clusters:
        ids_str = ", ".join(cluster.node_ids)
        lines.append(f'@{cluster.name}: [{ids_str}] "{cluster.description}"')

    return "\n".join(lines)


def _serialize_signals(signals: tuple[Signal, ...]) -> str:
    """Serialize SIGNALS section."""
    lines: list[str] = []
    lines.append("# ── SIGNALS ────────────────────────────────────────")
    lines.append("# Active alerts, unresolved items, pending decisions.")
    lines.append("# Agent: check this FIRST each session for urgent context.")
    lines.append("")

    for signal in signals:
        if signal.node_id:
            lines.append(f"{signal.level.value} [{signal.node_id}] {signal.text}")
        else:
            lines.append(f"{signal.level.value} {signal.text}")

    return "\n".join(lines)


def _serialize_depth_map(depth_map: tuple[DepthHint, ...]) -> str:
    """Serialize DEPTH MAP section."""
    lines: list[str] = []
    lines.append("# ── DEPTH MAP ──────────────────────────────────────")
    lines.append("# When surface isn't enough, this tells agent WHERE to dig.")
    lines.append("# Format: [id] → depth_hint (context)")
    lines.append("")

    for hint in depth_map:
        if hint.context:
            lines.append(f"[{hint.node_id}] → {hint.level.value} ({hint.context})")
        else:
            lines.append(f"[{hint.node_id}] → {hint.level.value}")

    return "\n".join(lines)


def _serialize_meta(meta: SurfaceMeta) -> str:
    """Serialize META section."""
    lines: list[str] = []
    lines.append("# ── META ───────────────────────────────────────────")
    lines.append("# Stats for agent to gauge brain richness")
    lines.append("")
    lines.append(f"coverage: {meta.coverage}")
    lines.append(f"staleness: {meta.staleness}")

    if meta.last_consolidation:
        lines.append(f"last_consolidation: {meta.last_consolidation}")

    if meta.top_entities:
        entities_str = ", ".join(meta.top_entities)
        lines.append(f"top_entities: [{entities_str}]")

    return "\n".join(lines)

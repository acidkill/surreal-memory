"""Frozen dataclasses for the .nm Knowledge Surface format."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SignalLevel(StrEnum):
    """Signal urgency levels."""

    URGENT = "!"
    WATCHING = "~"
    UNCERTAIN = "?"


class DepthLevel(StrEnum):
    """Depth routing hints for agent recall."""

    SUFFICIENT = "SUFFICIENT"
    NEEDS_DETAIL = "NEEDS_DETAIL"
    NEEDS_DEEP = "NEEDS_DEEP"


@dataclass(frozen=True)
class SurfaceFrontmatter:
    """YAML frontmatter metadata."""

    brain: str
    updated: str
    neurons: int = 0
    synapses: int = 0
    token_budget: int = 1200
    depth_available: tuple[str, ...] = ("surface", "detail", "deep")


@dataclass(frozen=True)
class SurfaceNode:
    """A node in the GRAPH section.

    ID prefix convention: d=decision, f=fact, e=error, p=preference,
    i=insight, w=workflow, x=rejected, c=concept.
    """

    id: str
    content: str
    node_type: str
    priority: int = 5
    neuron_id: str | None = None


@dataclass(frozen=True)
class SurfaceEdge:
    """A directed edge from a GRAPH node."""

    edge_type: str
    target_id: str | None = None
    target_text: str = ""


@dataclass(frozen=True)
class GraphEntry:
    """A GRAPH node with its outgoing edges."""

    node: SurfaceNode
    edges: tuple[SurfaceEdge, ...] = ()


@dataclass(frozen=True)
class Cluster:
    """A topic group from CLUSTERS section."""

    name: str
    node_ids: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class Signal:
    """An active alert from SIGNALS section."""

    level: SignalLevel
    text: str
    node_id: str | None = None


@dataclass(frozen=True)
class DepthHint:
    """Depth routing hint from DEPTH MAP section."""

    node_id: str
    level: DepthLevel
    context: str = ""


@dataclass(frozen=True)
class SurfaceMeta:
    """Brain richness stats from META section."""

    coverage: float = 0.0
    staleness: float = 0.0
    last_consolidation: str = ""
    top_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeSurface:
    """Complete parsed .nm Knowledge Surface."""

    frontmatter: SurfaceFrontmatter
    graph: tuple[GraphEntry, ...] = ()
    clusters: tuple[Cluster, ...] = ()
    signals: tuple[Signal, ...] = ()
    depth_map: tuple[DepthHint, ...] = ()
    meta: SurfaceMeta = field(default_factory=SurfaceMeta)

    def get_node(self, node_id: str) -> SurfaceNode | None:
        """Find a node by ID across graph entries."""
        for entry in self.graph:
            if entry.node.id == node_id:
                return entry.node
        return None

    def get_depth_hint(self, node_id: str) -> DepthLevel | None:
        """Get depth routing hint for a node."""
        for hint in self.depth_map:
            if hint.node_id == node_id:
                return hint.level
        return None

    def all_node_ids(self) -> frozenset[str]:
        """All node IDs defined in GRAPH."""
        return frozenset(entry.node.id for entry in self.graph)

    def get_cluster(self, name: str) -> Cluster | None:
        """Find a cluster by name."""
        for cluster in self.clusters:
            if cluster.name == name:
                return cluster
        return None

    def token_estimate(self) -> int:
        """Rough token count (chars / 4)."""
        from neural_memory.surface.serializer import serialize

        return len(serialize(self)) // 4

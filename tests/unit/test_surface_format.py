"""Tests for .nm Knowledge Surface format — parser, serializer, models, token budget."""

from __future__ import annotations

import pytest

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
from neural_memory.surface.parser import ParseError, parse
from neural_memory.surface.serializer import serialize
from neural_memory.surface.token_budget import trim_to_budget

# ── Fixtures ───────────────────────────────────────

COMPLETE_NM = """\
# ═══════════════════════════════════════════════════
# .neuralmemory/surface.nm
# Neural Memory Knowledge Surface v1
# Auto-generated from brain.db | Editable by human
# ═══════════════════════════════════════════════════
---
brain: myproject
updated: 2026-03-16T10:30:00
neurons: 847
synapses: 3201
token_budget: 1200
depth_available: [surface, detail, deep]
---

# ── GRAPH ──────────────────────────────────────────
# Nodes = key concepts. Edges = causal/decisional links.

[d1] Chose PostgreSQL over MongoDB (decision) {p:8}
  →caused→ [d2] "Need ACID for payment transactions"
  →led_to→ [f1] "payments module uses SQLAlchemy ORM"
  →rejected→ [x1] "MongoDB lacks multi-doc transactions"

[d2] Auth uses JWT 15min refresh (decision) {p:7}
  →caused→ [f2] "Legal flagged session token storage"
  →depends→ [d1] "PostgreSQL stores refresh tokens"

[e1] MagicMock truthy on Python 3.11 (error) {p:6}
  →root_cause→ "auto-created attrs always truthy"
  →fix→ "explicitly mock config.encryption"

[p1] User prefers Vietnamese chat (preference) {p:9}

# ── CLUSTERS ───────────────────────────────────────
# Auto-detected topic groups

@auth: [d2, f2] "Authentication and authorization"
@payments: [d1, f1, x1] "Payment processing"
@testing: [e1] "Test infrastructure"

# ── SIGNALS ────────────────────────────────────────
# Active alerts

! [f2] auth middleware rewrite — IN PROGRESS
~ considering Redis for sessions
? [x1] MongoDB alternative — UNDECIDED

# ── DEPTH MAP ──────────────────────────────────────

[d1] → SUFFICIENT (8 synapses, 3 fibers)
[d2] → NEEDS_DETAIL (2 fibers, recall "jwt auth decision")
[f2] → NEEDS_DEEP (12 related neurons)
[e1] → SUFFICIENT (fix documented above)

# ── META ───────────────────────────────────────────
# Stats for agent

coverage: 0.73
staleness: 0.12
last_consolidation: 2026-03-15T22:00:00
top_entities: [PostgreSQL, JWT, Redis, FastAPI, Docker]
"""


def _make_surface() -> KnowledgeSurface:
    """Build a KnowledgeSurface programmatically for serializer tests."""
    return KnowledgeSurface(
        frontmatter=SurfaceFrontmatter(
            brain="testbrain",
            updated="2026-03-16T12:00:00",
            neurons=100,
            synapses=500,
            token_budget=1200,
        ),
        graph=(
            GraphEntry(
                node=SurfaceNode(
                    id="d1", content="Test decision", node_type="decision", priority=8
                ),
                edges=(
                    SurfaceEdge(edge_type="caused", target_id="f1", target_text="some fact"),
                    SurfaceEdge(edge_type="fix", target_text="inline fix text"),
                ),
            ),
            GraphEntry(
                node=SurfaceNode(id="f1", content="Test fact", node_type="fact", priority=5),
                edges=(),
            ),
        ),
        clusters=(Cluster(name="testing", node_ids=("d1", "f1"), description="Test cluster"),),
        signals=(
            Signal(level=SignalLevel.URGENT, node_id="d1", text="urgent thing"),
            Signal(level=SignalLevel.WATCHING, text="watching something"),
        ),
        depth_map=(
            DepthHint(node_id="d1", level=DepthLevel.SUFFICIENT, context="3 synapses"),
            DepthHint(node_id="f1", level=DepthLevel.NEEDS_DETAIL, context="recall test"),
        ),
        meta=SurfaceMeta(
            coverage=0.5,
            staleness=0.1,
            last_consolidation="2026-03-16T10:00:00",
            top_entities=("Python", "Redis"),
        ),
    )


# ── Parser: Complete File ──────────────────────────


def test_parse_complete_file():
    """Parse a complete valid .nm file with all sections."""
    surface = parse(COMPLETE_NM)

    assert surface.frontmatter.brain == "myproject"
    assert surface.frontmatter.updated == "2026-03-16T10:30:00"
    assert surface.frontmatter.neurons == 847
    assert surface.frontmatter.synapses == 3201
    assert surface.frontmatter.token_budget == 1200
    assert surface.frontmatter.depth_available == ("surface", "detail", "deep")


# ── Parser: Frontmatter ───────────────────────────


def test_parse_frontmatter():
    """Parse YAML frontmatter between --- delimiters."""
    surface = parse(COMPLETE_NM)
    fm = surface.frontmatter
    assert fm.brain == "myproject"
    assert fm.neurons == 847
    assert fm.synapses == 3201


def test_parse_missing_frontmatter():
    """Missing frontmatter raises ParseError."""
    with pytest.raises(ParseError, match="Missing frontmatter"):
        parse("# No frontmatter here\n\nsome content")


def test_parse_unclosed_frontmatter():
    """Unclosed frontmatter raises ParseError."""
    with pytest.raises(ParseError, match="no closing"):
        parse("---\nbrain: test\n# no closing delimiter")


# ── Parser: GRAPH ──────────────────────────────────


def test_parse_graph_nodes():
    """Parse GRAPH nodes with correct IDs and types."""
    surface = parse(COMPLETE_NM)
    assert len(surface.graph) == 4  # d1, d2, e1, p1

    d1 = surface.graph[0]
    assert d1.node.id == "d1"
    assert d1.node.content == "Chose PostgreSQL over MongoDB"
    assert d1.node.node_type == "decision"
    assert d1.node.priority == 8


def test_parse_graph_edges():
    """Parse edges indented under GRAPH nodes."""
    surface = parse(COMPLETE_NM)

    d1 = surface.graph[0]
    assert len(d1.edges) == 3  # caused, led_to, rejected
    assert d1.edges[0].edge_type == "caused"
    assert d1.edges[0].target_id == "d2"
    assert d1.edges[0].target_text == "Need ACID for payment transactions"


def test_parse_graph_inline_edges():
    """Parse edges without target ID (inline text only)."""
    surface = parse(COMPLETE_NM)

    e1 = surface.graph[2]  # error node
    assert len(e1.edges) == 2
    assert e1.edges[0].target_id is None
    assert e1.edges[0].target_text == "auto-created attrs always truthy"


def test_parse_graph_standalone_node():
    """Parse a node with no edges."""
    surface = parse(COMPLETE_NM)

    p1 = surface.graph[3]  # preference, no edges
    assert p1.node.id == "p1"
    assert len(p1.edges) == 0


# ── Parser: CLUSTERS ──────────────────────────────


def test_parse_clusters():
    """Parse CLUSTERS with @name, [ids], and description."""
    surface = parse(COMPLETE_NM)
    assert len(surface.clusters) == 3

    auth = surface.clusters[0]
    assert auth.name == "auth"
    assert auth.node_ids == ("d2", "f2")
    assert auth.description == "Authentication and authorization"


# ── Parser: SIGNALS ───────────────────────────────


def test_parse_signals_all_levels():
    """Parse all 3 signal levels: !, ~, ?."""
    surface = parse(COMPLETE_NM)
    assert len(surface.signals) == 3

    assert surface.signals[0].level == SignalLevel.URGENT
    assert surface.signals[0].node_id == "f2"
    assert "IN PROGRESS" in surface.signals[0].text

    assert surface.signals[1].level == SignalLevel.WATCHING
    assert surface.signals[1].node_id is None

    assert surface.signals[2].level == SignalLevel.UNCERTAIN
    assert surface.signals[2].node_id == "x1"


# ── Parser: DEPTH MAP ─────────────────────────────


def test_parse_depth_map():
    """Parse DEPTH MAP with all 3 levels."""
    surface = parse(COMPLETE_NM)
    assert len(surface.depth_map) == 4

    assert surface.depth_map[0].node_id == "d1"
    assert surface.depth_map[0].level == DepthLevel.SUFFICIENT
    assert "8 synapses" in surface.depth_map[0].context

    assert surface.depth_map[1].level == DepthLevel.NEEDS_DETAIL
    assert surface.depth_map[2].level == DepthLevel.NEEDS_DEEP


# ── Parser: META ──────────────────────────────────


def test_parse_meta():
    """Parse META section key-value pairs."""
    surface = parse(COMPLETE_NM)
    meta = surface.meta

    assert meta.coverage == pytest.approx(0.73)
    assert meta.staleness == pytest.approx(0.12)
    assert meta.last_consolidation == "2026-03-15T22:00:00"
    assert meta.top_entities == ("PostgreSQL", "JWT", "Redis", "FastAPI", "Docker")


# ── Parser: Empty Sections ────────────────────────


def test_parse_empty_sections():
    """Parse .nm file with empty sections (no crash)."""
    minimal = """\
---
brain: empty
updated: 2026-01-01T00:00:00
neurons: 0
synapses: 0
token_budget: 500
---

# ── GRAPH ──────────────────────────────────────────

# ── CLUSTERS ───────────────────────────────────────

# ── SIGNALS ────────────────────────────────────────

# ── DEPTH MAP ──────────────────────────────────────

# ── META ───────────────────────────────────────────
"""
    surface = parse(minimal)
    assert surface.graph == ()
    assert surface.clusters == ()
    assert surface.signals == ()
    assert surface.depth_map == ()
    assert surface.meta.coverage == 0.0


# ── Parser: Lenient (Unknown Sections) ─────────────


def test_parse_unknown_section_skipped():
    """Unknown sections are skipped without error (forward compat)."""
    text = """\
---
brain: test
updated: 2026-01-01T00:00:00
---

# ── GRAPH ──────────────────────────────────────────

[d1] Some decision (decision) {p:5}

# ── FUTURE SECTION ─────────────────────────────────

some_unknown_key: some_value

# ── META ───────────────────────────────────────────

coverage: 0.5
"""
    surface = parse(text)
    assert len(surface.graph) == 1
    assert surface.meta.coverage == pytest.approx(0.5)


# ── Serializer ─────────────────────────────────────


def test_serialize_produces_valid_nm():
    """Serialize a surface to valid .nm text."""
    surface = _make_surface()
    text = serialize(surface)

    assert "---" in text
    assert "brain: testbrain" in text
    assert "# ── GRAPH" in text
    assert "[d1] Test decision (decision) {p:8}" in text
    assert '→caused→ [f1] "some fact"' in text
    assert '→fix→ "inline fix text"' in text
    assert "# ── CLUSTERS" in text
    assert '@testing: [d1, f1] "Test cluster"' in text
    assert "# ── SIGNALS" in text
    assert "! [d1] urgent thing" in text
    assert "~ watching something" in text
    assert "# ── DEPTH MAP" in text
    assert "[d1] → SUFFICIENT (3 synapses)" in text
    assert "# ── META" in text
    assert "coverage: 0.5" in text


# ── Round-Trip ─────────────────────────────────────


def test_round_trip_programmatic():
    """parse(serialize(surface)) should produce equivalent surface."""
    original = _make_surface()
    text = serialize(original)
    restored = parse(text)

    assert restored.frontmatter.brain == original.frontmatter.brain
    assert restored.frontmatter.neurons == original.frontmatter.neurons
    assert len(restored.graph) == len(original.graph)
    assert len(restored.clusters) == len(original.clusters)
    assert len(restored.signals) == len(original.signals)
    assert len(restored.depth_map) == len(original.depth_map)

    # Check graph content
    assert restored.graph[0].node.id == "d1"
    assert restored.graph[0].node.priority == 8
    assert len(restored.graph[0].edges) == 2
    assert restored.graph[0].edges[0].target_id == "f1"
    assert restored.graph[0].edges[1].target_id is None

    # Check cluster
    assert restored.clusters[0].name == "testing"
    assert restored.clusters[0].node_ids == ("d1", "f1")

    # Check signals
    assert restored.signals[0].level == SignalLevel.URGENT
    assert restored.signals[0].node_id == "d1"
    assert restored.signals[1].node_id is None

    # Check depth map
    assert restored.depth_map[0].level == DepthLevel.SUFFICIENT

    # Check meta
    assert restored.meta.coverage == pytest.approx(0.5)
    assert restored.meta.top_entities == ("Python", "Redis")


def test_round_trip_from_text():
    """parse → serialize → parse should give same result."""
    surface1 = parse(COMPLETE_NM)
    text = serialize(surface1)
    surface2 = parse(text)

    assert surface2.frontmatter.brain == surface1.frontmatter.brain
    assert len(surface2.graph) == len(surface1.graph)
    assert len(surface2.clusters) == len(surface1.clusters)
    assert len(surface2.signals) == len(surface1.signals)
    assert len(surface2.depth_map) == len(surface1.depth_map)

    for i, entry in enumerate(surface2.graph):
        assert entry.node.id == surface1.graph[i].node.id
        assert entry.node.priority == surface1.graph[i].node.priority
        assert len(entry.edges) == len(surface1.graph[i].edges)


# ── Model Methods ──────────────────────────────────


def test_get_node():
    """get_node() finds node by ID."""
    surface = _make_surface()
    node = surface.get_node("d1")
    assert node is not None
    assert node.content == "Test decision"

    assert surface.get_node("nonexistent") is None


def test_get_depth_hint():
    """get_depth_hint() returns depth level for a node."""
    surface = _make_surface()
    assert surface.get_depth_hint("d1") == DepthLevel.SUFFICIENT
    assert surface.get_depth_hint("f1") == DepthLevel.NEEDS_DETAIL
    assert surface.get_depth_hint("nonexistent") is None


def test_all_node_ids():
    """all_node_ids() returns frozenset of all GRAPH node IDs."""
    surface = _make_surface()
    ids = surface.all_node_ids()
    assert ids == frozenset({"d1", "f1"})


def test_get_cluster():
    """get_cluster() finds cluster by name."""
    surface = _make_surface()
    cluster = surface.get_cluster("testing")
    assert cluster is not None
    assert cluster.node_ids == ("d1", "f1")

    assert surface.get_cluster("nonexistent") is None


def test_token_estimate():
    """token_estimate() returns rough char/4 count."""
    surface = _make_surface()
    estimate = surface.token_estimate()
    assert estimate > 0
    # Serialized text should be ~500-2000 chars, so ~125-500 tokens
    assert 50 < estimate < 1000


# ── Token Budget ───────────────────────────────────


def test_trim_within_budget_unchanged():
    """Surface within budget is returned unchanged."""
    surface = _make_surface()
    trimmed = trim_to_budget(surface, budget=5000)
    assert len(trimmed.graph) == len(surface.graph)


def test_trim_removes_sufficient_depth_first():
    """First trimming step removes SUFFICIENT depth hints."""
    surface = _make_surface()
    # Use very tight budget
    trimmed = trim_to_budget(surface, budget=1)

    # SUFFICIENT hints should be removed first
    for hint in trimmed.depth_map:
        assert hint.level != DepthLevel.SUFFICIENT


def test_trim_removes_lowest_priority_graph():
    """Trimming removes lowest-priority graph entry."""
    surface = _make_surface()
    # d1 has priority 8, f1 has priority 5 — f1 should be removed first
    trimmed = trim_to_budget(surface, budget=1)

    # f1 (priority 5) should be removed before d1 (priority 8)
    remaining_ids = {e.node.id for e in trimmed.graph}
    if len(trimmed.graph) < len(surface.graph):
        assert "f1" not in remaining_ids or "d1" in remaining_ids


def test_trim_cleans_up_cluster_refs():
    """Trimming a graph node removes it from cluster references."""
    surface = _make_surface()
    trimmed = trim_to_budget(surface, budget=1)

    for cluster in trimmed.clusters:
        for nid in cluster.node_ids:
            # All referenced IDs should still exist in graph
            remaining_ids = {e.node.id for e in trimmed.graph}
            assert nid in remaining_ids or len(trimmed.graph) == 0


# ── UTF-8 Support ──────────────────────────────────


def test_utf8_entity_names():
    """Parse .nm with Vietnamese entity names."""
    text = """\
---
brain: vietnamese
updated: 2026-01-01T00:00:00
---

# ── GRAPH ──────────────────────────────────────────

[d1] Chọn PostgreSQL cho thanh toán (decision) {p:8}
  →caused→ "Cần ACID cho giao dịch"

# ── META ───────────────────────────────────────────

coverage: 0.5
top_entities: [PostgreSQL, Thanh toán, Redis]
"""
    surface = parse(text)
    assert surface.graph[0].node.content == "Chọn PostgreSQL cho thanh toán"
    assert surface.graph[0].edges[0].target_text == "Cần ACID cho giao dịch"
    # Note: Vietnamese in top_entities may have spaces — parser handles this


# ── Frozen Immutability ────────────────────────────


def test_models_are_frozen():
    """All dataclasses should be frozen (immutable)."""
    surface = _make_surface()

    with pytest.raises(AttributeError):
        surface.frontmatter = None  # type: ignore[misc]

    with pytest.raises(AttributeError):
        surface.graph[0].node.id = "changed"  # type: ignore[misc]


# ── ID Format ──────────────────────────────────────


def test_node_id_prefix_convention():
    """Node IDs follow prefix convention from the format spec."""
    surface = parse(COMPLETE_NM)
    for entry in surface.graph:
        nid = entry.node.id
        # IDs should start with a letter
        assert nid[0].isalpha(), f"ID '{nid}' should start with a letter"

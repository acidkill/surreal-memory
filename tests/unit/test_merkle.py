"""Unit tests for MerkleTreeBuilder.

Covers:
- Hash primitives (leaf, branch)
- Tree construction (single entity, multi-entity, grouping by prefix)
- Determinism guarantees
- Diff computation
- Dataclass immutability
- Large tree smoke test
"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from surreal_memory.sync.merkle import (
    _EMPTY_HASH,
    MerkleNode,
    MerkleTreeBuilder,
)

# ---------------------------------------------------------------------------
# Hash primitive tests
# ---------------------------------------------------------------------------


def test_leaf_hash_deterministic() -> None:
    """Same inputs always produce the same hash."""
    h1 = MerkleTreeBuilder.compute_leaf_hash("neuron-abc", "2026-01-01T00:00:00", "cafebabe")
    h2 = MerkleTreeBuilder.compute_leaf_hash("neuron-abc", "2026-01-01T00:00:00", "cafebabe")
    assert h1 == h2


def test_leaf_hash_different_inputs() -> None:
    """Different inputs produce different hashes."""
    h1 = MerkleTreeBuilder.compute_leaf_hash("neuron-abc", "2026-01-01T00:00:00", "cafebabe")
    h2 = MerkleTreeBuilder.compute_leaf_hash("neuron-xyz", "2026-01-01T00:00:00", "cafebabe")
    h3 = MerkleTreeBuilder.compute_leaf_hash("neuron-abc", "2026-02-01T00:00:00", "cafebabe")
    h4 = MerkleTreeBuilder.compute_leaf_hash("neuron-abc", "2026-01-01T00:00:00", "deadbeef")
    assert len({h1, h2, h3, h4}) == 4, "All four hashes should be distinct"


def test_branch_hash_order_independent() -> None:
    """Branch hash is the same regardless of child ordering."""
    hashes = ["aabbcc", "112233", "ddeeff"]
    h1 = MerkleTreeBuilder.compute_branch_hash(hashes)
    h2 = MerkleTreeBuilder.compute_branch_hash(list(reversed(hashes)))
    assert h1 == h2


def test_branch_hash_empty_returns_empty_hash() -> None:
    """Empty child list returns the standard empty hash."""
    result = MerkleTreeBuilder.compute_branch_hash([])
    assert result == _EMPTY_HASH
    assert result == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# Tree construction tests
# ---------------------------------------------------------------------------


def test_empty_tree() -> None:
    """Empty entity list produces a consistent empty-hash node."""
    tree = MerkleTreeBuilder.build_tree([], "neuron")
    assert tree.prefix == "neurons"
    assert tree.hash == _EMPTY_HASH
    assert tree.entity_count == 0
    assert tree.children == ()


def test_build_tree_single_entity() -> None:
    """Single entity produces a tree with one bucket child."""
    entities = [("ab1234", "2026-01-01T00:00:00", "hash1")]
    tree = MerkleTreeBuilder.build_tree(entities, "neuron")

    assert tree.prefix == "neurons"
    assert tree.entity_count == 1
    assert len(tree.children) == 1

    bucket = tree.children[0]
    assert bucket.prefix == "neurons/ab"
    assert bucket.entity_count == 1
    # Leaf hash should match manual computation
    expected_leaf = MerkleTreeBuilder.compute_leaf_hash("ab1234", "2026-01-01T00:00:00", "hash1")
    expected_root = MerkleTreeBuilder.compute_branch_hash([expected_leaf])
    assert bucket.hash == expected_root


def test_build_tree_multiple_entities_groups_by_prefix() -> None:
    """Entities with the same id[:2] land in the same bucket."""
    entities = [
        ("ab0001", "2026-01-01", "hash1"),
        ("ab0002", "2026-01-02", "hash2"),
        ("cd0001", "2026-01-03", "hash3"),
    ]
    tree = MerkleTreeBuilder.build_tree(entities, "neuron")

    assert tree.entity_count == 3
    assert len(tree.children) == 2  # two buckets: "ab" and "cd"

    prefixes = {c.prefix for c in tree.children}
    assert prefixes == {"neurons/ab", "neurons/cd"}

    ab_bucket = next(c for c in tree.children if c.prefix == "neurons/ab")
    assert ab_bucket.entity_count == 2


def test_build_tree_deterministic() -> None:
    """Same entities in different order produce the same root hash."""
    entities = [
        ("ab0001", "2026-01-01", "hash1"),
        ("cd0002", "2026-01-02", "hash2"),
        ("ef0003", "2026-01-03", "hash3"),
    ]
    tree1 = MerkleTreeBuilder.build_tree(entities, "neuron")
    tree2 = MerkleTreeBuilder.build_tree(list(reversed(entities)), "neuron")
    assert tree1.hash == tree2.hash
    assert tree1.entity_count == tree2.entity_count


def test_content_hash_affects_root() -> None:
    """Changing one entity's content_hash changes the root hash."""
    base = [
        ("ab0001", "2026-01-01", "hash1"),
        ("cd0002", "2026-01-02", "hash2"),
    ]
    modified = [
        ("ab0001", "2026-01-01", "hash1_CHANGED"),
        ("cd0002", "2026-01-02", "hash2"),
    ]
    tree_base = MerkleTreeBuilder.build_tree(base, "neuron")
    tree_mod = MerkleTreeBuilder.build_tree(modified, "neuron")
    assert tree_base.hash != tree_mod.hash


def test_prefix_grouping() -> None:
    """Entity IDs are grouped by their first two lowercase characters."""
    entities = [
        ("AB0001", "2026-01-01", "h1"),  # prefix "ab" (lowercased)
        ("ab0002", "2026-01-02", "h2"),  # prefix "ab"
        ("CD9999", "2026-01-03", "h3"),  # prefix "cd"
    ]
    tree = MerkleTreeBuilder.build_tree(entities, "neuron")
    prefixes = {c.prefix for c in tree.children}
    assert prefixes == {"neurons/ab", "neurons/cd"}

    ab_bucket = next(c for c in tree.children if c.prefix == "neurons/ab")
    assert ab_bucket.entity_count == 2


def test_large_tree_builds_without_error() -> None:
    """1000 entities build without raising an exception."""
    entities = [(f"{i:04x}abcd", "2026-01-01T00:00:00", f"hash{i}") for i in range(1000)]
    tree = MerkleTreeBuilder.build_tree(entities, "neuron")
    assert tree.entity_count == 1000
    assert tree.hash != _EMPTY_HASH
    assert len(tree.children) > 0


# ---------------------------------------------------------------------------
# Diff computation tests
# ---------------------------------------------------------------------------


def _make_tree(entities: list[tuple[str, str, str]], entity_type: str = "neuron") -> MerkleNode:
    return MerkleTreeBuilder.build_tree(entities, entity_type)


def test_compute_diff_identical() -> None:
    """Identical trees return an empty diff list."""
    entities = [("ab0001", "2026-01-01", "h1"), ("cd0002", "2026-01-02", "h2")]
    tree = _make_tree(entities)
    assert MerkleTreeBuilder.compute_diff(tree, tree) == []


def test_compute_diff_one_bucket_changed() -> None:
    """Changing one bucket is detected; unchanged buckets are skipped."""
    base = [
        ("ab0001", "2026-01-01", "h1"),
        ("cd0002", "2026-01-02", "h2"),
    ]
    changed = [
        ("ab0001", "2026-01-01", "h1_CHANGED"),
        ("cd0002", "2026-01-02", "h2"),
    ]
    local = _make_tree(base)
    remote = _make_tree(changed)

    diffs = MerkleTreeBuilder.compute_diff(local, remote)
    # Should detect the "neurons/ab" bucket only
    assert "neurons/ab" in diffs
    assert "neurons/cd" not in diffs


def test_compute_diff_all_different() -> None:
    """When all buckets differ, all bucket prefixes are returned."""
    entities_a = [("ab0001", "2026-01-01", "h1"), ("cd0002", "2026-01-02", "h2")]
    entities_b = [("ab0001", "2026-01-01", "DIFF"), ("cd0002", "2026-01-02", "DIFF")]
    local = _make_tree(entities_a)
    remote = _make_tree(entities_b)

    diffs = MerkleTreeBuilder.compute_diff(local, remote)
    assert set(diffs) == {"neurons/ab", "neurons/cd"}


def test_compute_diff_empty_vs_populated() -> None:
    """Empty local tree vs populated remote detects all remote prefixes."""
    remote_entities = [("ab0001", "2026-01-01", "h1"), ("cd0002", "2026-01-02", "h2")]
    local = _make_tree([])
    remote = _make_tree(remote_entities)

    diffs = MerkleTreeBuilder.compute_diff(local, remote)
    # The root differs; we should get the remote bucket prefixes
    assert len(diffs) > 0


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


def test_merkle_node_frozen() -> None:
    """MerkleNode is a frozen dataclass — mutation raises FrozenInstanceError."""
    node = MerkleNode(prefix="neurons", hash="abc", entity_count=1)
    with pytest.raises(FrozenInstanceError):
        node.hash = "should_fail"  # type: ignore[misc]

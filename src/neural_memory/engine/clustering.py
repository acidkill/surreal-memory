"""Shared Union-Find data structure for clustering operations.

Extracted from consolidation, enrichment, and pattern_extraction to
eliminate ~120 lines of copy-pasted Union-Find implementations.
"""

from __future__ import annotations


class UnionFind:
    """Union-Find (disjoint set) with path compression.

    Used by consolidation, enrichment, and pattern extraction
    for Jaccard-based fiber/tag clustering.
    """

    __slots__ = ("_parent",)

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        """Find root with path compression (halving)."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge sets containing a and b."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[int, list[int]]:
        """Return all groups as root -> member indices."""
        result: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            result.setdefault(root, []).append(i)
        return result

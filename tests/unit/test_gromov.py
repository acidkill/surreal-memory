"""Tests for Gromov delta-hyperbolicity estimation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.gromov import (
    GromovResult,
    _bfs_distance,
    _get_or_compute_dist,
    classify_structure,
    estimate_gromov_delta,
)

# ── Classification ────────────────────────────────────────────────────


class TestClassifyStructure:
    """Tests for structure quality classification."""

    def test_tree_like(self) -> None:
        assert classify_structure(0.0) == "tree-like"
        assert classify_structure(0.05) == "tree-like"
        assert classify_structure(0.09) == "tree-like"

    def test_hierarchical(self) -> None:
        assert classify_structure(0.1) == "hierarchical"
        assert classify_structure(0.2) == "hierarchical"
        assert classify_structure(0.29) == "hierarchical"

    def test_mixed(self) -> None:
        assert classify_structure(0.3) == "mixed"
        assert classify_structure(0.4) == "mixed"
        assert classify_structure(0.49) == "mixed"

    def test_flat(self) -> None:
        assert classify_structure(0.5) == "flat"
        assert classify_structure(0.8) == "flat"
        assert classify_structure(1.0) == "flat"


# ── BFS Distance ─────────────────────────────────────────────────────


class TestBFSDistance:
    """Tests for in-memory BFS shortest path."""

    def test_same_node(self) -> None:
        adjacency: dict[str, set[str]] = {"a": set()}
        assert _bfs_distance("a", "a", adjacency) == 0

    def test_direct_neighbor(self) -> None:
        adjacency: dict[str, set[str]] = {"a": {"b"}, "b": {"a"}}
        assert _bfs_distance("a", "b", adjacency) == 1

    def test_two_hops(self) -> None:
        adjacency: dict[str, set[str]] = {
            "a": {"b"},
            "b": {"a", "c"},
            "c": {"b"},
        }
        assert _bfs_distance("a", "c", adjacency) == 2

    def test_unreachable(self) -> None:
        adjacency: dict[str, set[str]] = {"a": set(), "b": set()}
        assert _bfs_distance("a", "b", adjacency) == -1

    def test_shortest_path_chosen(self) -> None:
        """When multiple paths exist, BFS finds the shortest."""
        adjacency: dict[str, set[str]] = {
            "a": {"b", "c"},
            "b": {"a", "d"},
            "c": {"a", "d"},
            "d": {"b", "c"},
        }
        assert _bfs_distance("a", "d", adjacency) == 2


class TestGetOrComputeDist:
    """Tests for cached distance lookup."""

    def test_caches_result(self) -> None:
        adjacency: dict[str, set[str]] = {"a": {"b"}, "b": {"a"}}
        cache: dict[tuple[str, str], int] = {}

        result = _get_or_compute_dist("a", "b", adjacency, cache)
        assert result == 1
        assert ("a", "b") in cache

    def test_symmetric_key(self) -> None:
        """Cache key is canonical (min, max) so a→b and b→a share entry."""
        adjacency: dict[str, set[str]] = {"a": {"b"}, "b": {"a"}}
        cache: dict[tuple[str, str], int] = {}

        _get_or_compute_dist("b", "a", adjacency, cache)
        assert ("a", "b") in cache
        assert ("b", "a") not in cache

    def test_uses_cached_value(self) -> None:
        adjacency: dict[str, set[str]] = {}
        cache: dict[tuple[str, str], int] = {("a", "b"): 42}

        result = _get_or_compute_dist("a", "b", adjacency, cache)
        assert result == 42


# ── Star Graph (Tree-Like) ───────────────────────────────────────────


class TestGromovStarGraph:
    """Star graph should have delta ≈ 0 (perfect tree)."""

    @pytest.fixture
    def star_storage(self) -> AsyncMock:
        """Create mock storage with star topology: center connected to 5 leaves."""
        storage = AsyncMock()
        neurons = [MagicMock(id=f"n{i}") for i in range(6)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        # Star: n0 is center, n1-n5 are leaves
        synapse_map: dict[str, list[MagicMock]] = {
            "n0": [MagicMock(target_id=f"n{i}") for i in range(1, 6)],
        }
        for i in range(1, 6):
            synapse_map[f"n{i}"] = []

        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)
        return storage

    async def test_star_is_tree_like(self, star_storage: AsyncMock) -> None:
        result = await estimate_gromov_delta(star_storage, sample_size=6, seed=42)
        assert result.structure_quality == "tree-like"
        assert result.normalized_delta < 0.1
        assert result.sample_count == 6

    async def test_star_delta_near_zero(self, star_storage: AsyncMock) -> None:
        result = await estimate_gromov_delta(star_storage, sample_size=6, seed=42)
        assert result.delta == 0.0


# ── Cycle Graph (Non-Tree) ──────────────────────────────────────────


class TestGromovCycleGraph:
    """Cycle graph should have delta > 0 (not a tree)."""

    @pytest.fixture
    def cycle_storage(self) -> AsyncMock:
        """Create mock storage with 8-node cycle."""
        storage = AsyncMock()
        n = 8
        neurons = [MagicMock(id=f"n{i}") for i in range(n)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        synapse_map: dict[str, list[MagicMock]] = {}
        for i in range(n):
            next_id = f"n{(i + 1) % n}"
            synapse_map[f"n{i}"] = [MagicMock(target_id=next_id)]

        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)
        return storage

    async def test_cycle_has_positive_delta(self, cycle_storage: AsyncMock) -> None:
        result = await estimate_gromov_delta(cycle_storage, sample_size=8, seed=42)
        assert result.delta > 0

    async def test_cycle_not_tree_like(self, cycle_storage: AsyncMock) -> None:
        result = await estimate_gromov_delta(cycle_storage, sample_size=8, seed=42)
        assert result.structure_quality != "tree-like"


# ── Complete Graph (Flat) ────────────────────────────────────────────


class TestGromovCompleteGraph:
    """Complete graph should have relatively high delta."""

    @pytest.fixture
    def complete_storage(self) -> AsyncMock:
        """Create mock storage with 6-node complete graph."""
        storage = AsyncMock()
        n = 6
        neurons = [MagicMock(id=f"n{i}") for i in range(n)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        synapse_map: dict[str, list[MagicMock]] = {}
        for i in range(n):
            synapse_map[f"n{i}"] = [MagicMock(target_id=f"n{j}") for j in range(n) if j != i]

        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)
        return storage

    async def test_complete_graph_delta(self, complete_storage: AsyncMock) -> None:
        """Complete graph: all distances = 1, so all sums = 2, delta = 0.

        This is a known property: complete graphs are 0-hyperbolic because
        all pairwise distances are equal.
        """
        result = await estimate_gromov_delta(complete_storage, sample_size=6, seed=42)
        assert result.delta == 0.0


# ── Edge Cases ───────────────────────────────────────────────────────


class TestGromovEdgeCases:
    """Edge cases for Gromov estimation."""

    async def test_insufficient_neurons(self) -> None:
        """Less than 4 neurons → empty result."""
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[MagicMock(id="n0")])
        result = await estimate_gromov_delta(storage, sample_size=10)
        assert result.structure_quality == "insufficient_data"
        assert result.sample_count == 0

    async def test_disconnected_graph(self) -> None:
        """Disconnected neurons → unreachable pairs skipped."""
        storage = AsyncMock()
        neurons = [MagicMock(id=f"n{i}") for i in range(5)]
        storage.find_neurons = AsyncMock(return_value=neurons)
        # No edges at all
        storage.get_synapses_for_neurons = AsyncMock(return_value={f"n{i}": [] for i in range(5)})

        result = await estimate_gromov_delta(storage, sample_size=5, seed=42)
        # All pairs unreachable → no valid tuples → delta stays 0
        assert result.delta == 0.0

    async def test_exactly_four_neurons(self) -> None:
        """Minimum viable: exactly 4 neurons forming a path."""
        storage = AsyncMock()
        neurons = [MagicMock(id=f"n{i}") for i in range(4)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        # Path: n0-n1-n2-n3
        synapse_map = {
            "n0": [MagicMock(target_id="n1")],
            "n1": [MagicMock(target_id="n2")],
            "n2": [MagicMock(target_id="n3")],
            "n3": [],
        }
        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)

        result = await estimate_gromov_delta(storage, sample_size=4, seed=42)
        assert result.sample_count == 4
        assert result.tuple_count == 1  # C(4,4) = 1

    async def test_seed_reproducibility(self) -> None:
        """Same seed → same result."""
        storage = AsyncMock()
        neurons = [MagicMock(id=f"n{i}") for i in range(6)]
        storage.find_neurons = AsyncMock(return_value=neurons)
        synapse_map = {
            "n0": [MagicMock(target_id="n1"), MagicMock(target_id="n2")],
            "n1": [MagicMock(target_id="n3")],
            "n2": [MagicMock(target_id="n4")],
            "n3": [MagicMock(target_id="n5")],
            "n4": [],
            "n5": [],
        }
        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)

        r1 = await estimate_gromov_delta(storage, sample_size=6, seed=123)
        r2 = await estimate_gromov_delta(storage, sample_size=6, seed=123)
        assert r1.delta == r2.delta
        assert r1.normalized_delta == r2.normalized_delta


# ── GromovResult ─────────────────────────────────────────────────────


class TestGromovResult:
    """Tests for GromovResult dataclass."""

    def test_empty(self) -> None:
        result = GromovResult.empty()
        assert result.structure_quality == "insufficient_data"
        assert result.sample_count == 0
        assert result.delta == 0.0

    def test_frozen(self) -> None:
        result = GromovResult(
            delta=1.0,
            normalized_delta=0.5,
            structure_quality="mixed",
            sample_count=100,
            tuple_count=500,
            diameter=2,
        )
        with pytest.raises(AttributeError):
            result.delta = 2.0  # type: ignore[misc]

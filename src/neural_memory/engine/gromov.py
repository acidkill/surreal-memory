"""Gromov delta-hyperbolicity estimation for brain structure quality.

Measures how tree-like the brain's neuron graph is. Lower delta = more
hierarchical structure = better retrieval quality. Higher delta = flat/
scattered memories = needs consolidation.

Uses 4-point condition sampling with BFS shortest paths.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage as StorageBackend

logger = logging.getLogger(__name__)

# Sampling limits
_DEFAULT_SAMPLE_SIZE = 200
_MAX_TUPLES = 5000
_BFS_MAX_HOPS = 15


@dataclass(frozen=True)
class GromovResult:
    """Result of Gromov delta-hyperbolicity estimation."""

    delta: float
    normalized_delta: float
    structure_quality: str
    sample_count: int
    tuple_count: int
    diameter: int

    @staticmethod
    def empty() -> GromovResult:
        """Return empty result for brains with insufficient neurons."""
        return GromovResult(
            delta=0.0,
            normalized_delta=0.0,
            structure_quality="insufficient_data",
            sample_count=0,
            tuple_count=0,
            diameter=0,
        )


def classify_structure(normalized_delta: float) -> str:
    """Classify brain structure from normalized delta.

    Args:
        normalized_delta: delta / diameter, in [0, 1].

    Returns:
        Quality label.
    """
    if normalized_delta < 0.1:
        return "tree-like"
    if normalized_delta < 0.3:
        return "hierarchical"
    if normalized_delta < 0.5:
        return "mixed"
    return "flat"


async def estimate_gromov_delta(
    storage: StorageBackend,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
) -> GromovResult:
    """Estimate Gromov delta-hyperbolicity by sampling neuron 4-tuples.

    For each 4-tuple of neurons, computes 6 pairwise BFS distances, then
    the 4-point condition delta. Returns the maximum delta across all
    sampled tuples.

    Args:
        storage: Storage backend to query neurons and synapses.
        sample_size: Number of neurons to sample (reservoir sampling).
        seed: Random seed for reproducibility.

    Returns:
        GromovResult with delta, normalized delta, and structure classification.
    """
    sample_size = min(sample_size, 1000)

    # Sample neurons
    neuron_ids = await _sample_neuron_ids(storage, sample_size, seed)
    if len(neuron_ids) < 4:
        return GromovResult.empty()

    # Build adjacency list for BFS (bidirectional — treat as undirected)
    adjacency = await _build_adjacency(storage, neuron_ids)

    # Compute all-pairs shortest paths for sampled neurons
    dist_cache: dict[tuple[str, str], int] = {}
    diameter = 0

    # Generate 4-tuples (cap at _MAX_TUPLES)
    rng = random.Random(seed)
    all_combos = list(itertools.combinations(neuron_ids, 4))
    if len(all_combos) > _MAX_TUPLES:
        rng.shuffle(all_combos)
        all_combos = all_combos[:_MAX_TUPLES]

    max_delta = 0.0

    for a, b, c, d in all_combos:
        # Get 6 pairwise distances
        d_ab = _get_or_compute_dist(a, b, adjacency, dist_cache)
        d_cd = _get_or_compute_dist(c, d, adjacency, dist_cache)
        d_ac = _get_or_compute_dist(a, c, adjacency, dist_cache)
        d_bd = _get_or_compute_dist(b, d, adjacency, dist_cache)
        d_ad = _get_or_compute_dist(a, d, adjacency, dist_cache)
        d_bc = _get_or_compute_dist(b, c, adjacency, dist_cache)

        # Skip if any pair is unreachable
        if -1 in (d_ab, d_cd, d_ac, d_bd, d_ad, d_bc):
            continue

        # 4-point condition: sort the 3 sums
        s1 = d_ab + d_cd
        s2 = d_ac + d_bd
        s3 = d_ad + d_bc
        sums = sorted([s1, s2, s3])
        delta = (sums[2] - sums[1]) / 2.0

        max_delta = max(max_delta, delta)
        diameter = max(diameter, d_ab, d_cd, d_ac, d_bd, d_ad, d_bc)

    if diameter == 0:
        return GromovResult(
            delta=0.0,
            normalized_delta=0.0,
            structure_quality="tree-like",
            sample_count=len(neuron_ids),
            tuple_count=len(all_combos),
            diameter=0,
        )

    normalized = max_delta / diameter
    return GromovResult(
        delta=max_delta,
        normalized_delta=round(normalized, 4),
        structure_quality=classify_structure(normalized),
        sample_count=len(neuron_ids),
        tuple_count=len(all_combos),
        diameter=diameter,
    )


async def _sample_neuron_ids(
    storage: StorageBackend,
    sample_size: int,
    seed: int | None,
) -> list[str]:
    """Sample neuron IDs using reservoir sampling via find_neurons."""
    # Fetch up to sample_size neurons
    neurons = await storage.find_neurons(limit=sample_size)
    ids = [n.id for n in neurons]

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(ids)

    return ids[:sample_size]


async def _build_adjacency(
    storage: StorageBackend,
    neuron_ids: list[str],
) -> dict[str, set[str]]:
    """Build bidirectional adjacency list from synapses.

    Only includes edges where both endpoints are in the sample set.
    """
    adjacency: dict[str, set[str]] = {nid: set() for nid in neuron_ids}
    id_set = set(neuron_ids)

    # Batch fetch outgoing synapses
    synapses_by_neuron = await storage.get_synapses_for_neurons(neuron_ids, direction="out")
    for source_id, synapses in synapses_by_neuron.items():
        for syn in synapses:
            if syn.target_id in id_set:
                adjacency[source_id].add(syn.target_id)
                adjacency[syn.target_id].add(source_id)  # Undirected

    return adjacency


def _bfs_distance(
    source: str,
    target: str,
    adjacency: dict[str, set[str]],
) -> int:
    """BFS shortest path distance in undirected graph.

    Returns:
        Shortest distance, or -1 if unreachable within _BFS_MAX_HOPS.
    """
    if source == target:
        return 0

    visited = {source}
    queue: deque[tuple[str, int]] = deque([(source, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= _BFS_MAX_HOPS:
            continue

        for neighbor in adjacency.get(current, set()):
            if neighbor == target:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return -1  # Unreachable


def _get_or_compute_dist(
    a: str,
    b: str,
    adjacency: dict[str, set[str]],
    cache: dict[tuple[str, str], int],
) -> int:
    """Get cached distance or compute via BFS."""
    key = (min(a, b), max(a, b))
    if key not in cache:
        cache[key] = _bfs_distance(a, b, adjacency)
    return cache[key]

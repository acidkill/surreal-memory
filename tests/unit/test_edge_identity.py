"""Tests for deterministic synapse ids.

Two things are load-bearing here:

* ``alias_edge_id`` must never change. Live brains already carry rows keyed by
  its digest, so a refactor that "improves" the hash would orphan every one of
  them and resurrect the amplification that module exists to stop. The digest
  is pinned as a literal.
* ``deterministic_edge_id`` must sort the endpoints for bidirectional types,
  so semantic linking cannot express one pair as two edges.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.synapse import BIDIRECTIONAL_TYPES, SynapseType
from surreal_memory.engine.dedup.alias_edges import alias_edge_id
from surreal_memory.engine.edge_identity import deterministic_edge_id

# Synthetic, stable endpoints -- never real neuron ids.
A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"
C = "33333333-3333-4333-8333-333333333333"


class TestAliasEdgeIdIsFrozen:
    def test_digest_is_pinned(self) -> None:
        """Live rows carry this exact digest -- changing it orphans them."""
        assert alias_edge_id(A, B) == "alias-d28381e08e1a0cdf36cd6fb6dd05a0d0"

    def test_alias_stays_directional(self) -> None:
        """'X aliases Y' is not 'Y aliases X' -- the ids must differ."""
        assert alias_edge_id(A, B) != alias_edge_id(B, A)

    def test_shape_is_prefix_and_32_hex(self) -> None:
        edge_id = alias_edge_id(A, B)
        prefix, _, digest = edge_id.partition("-")
        assert prefix == "alias"
        assert len(digest) == 32
        assert all(ch in "0123456789abcdef" for ch in digest)


class TestDeterministicEdgeId:
    def test_matches_alias_edge_id_exactly(self) -> None:
        """The generic derivation must not drift from the frozen one.

        ALIAS is not a bidirectional type, so no sorting intervenes and the two
        functions are required to agree byte-for-byte.
        """
        assert deterministic_edge_id(SynapseType.ALIAS, A, B) == alias_edge_id(A, B)
        assert deterministic_edge_id(SynapseType.ALIAS, B, A) == alias_edge_id(B, A)

    def test_similar_to_digest_is_pinned(self) -> None:
        assert (
            deterministic_edge_id(SynapseType.SIMILAR_TO, A, B)
            == "similar_to-66c3446cfe4f9d868c7456792ab0334e"
        )

    @pytest.mark.parametrize("synapse_type", sorted(BIDIRECTIONAL_TYPES, key=lambda t: t.value))
    def test_bidirectional_types_sort_endpoints(self, synapse_type: SynapseType) -> None:
        """(A, B) and (B, A) are the same fact, so they get one id."""
        assert deterministic_edge_id(synapse_type, A, B) == deterministic_edge_id(
            synapse_type, B, A
        )

    @pytest.mark.parametrize(
        "synapse_type",
        [SynapseType.SUPERSEDES, SynapseType.ALIAS, SynapseType.CAUSED_BY],
    )
    def test_directional_types_keep_their_direction(self, synapse_type: SynapseType) -> None:
        assert synapse_type not in BIDIRECTIONAL_TYPES
        assert deterministic_edge_id(synapse_type, A, B) != deterministic_edge_id(
            synapse_type, B, A
        )

    def test_distinct_pairs_get_distinct_ids(self) -> None:
        ids = {
            deterministic_edge_id(SynapseType.SIMILAR_TO, A, B),
            deterministic_edge_id(SynapseType.SIMILAR_TO, A, C),
            deterministic_edge_id(SynapseType.SIMILAR_TO, B, C),
        }
        assert len(ids) == 3

    def test_type_is_part_of_the_identity(self) -> None:
        """The same pair under two types is two different facts."""
        assert deterministic_edge_id(SynapseType.SIMILAR_TO, A, B) != deterministic_edge_id(
            SynapseType.RELATED_TO, A, B
        )

    def test_is_stable_across_calls(self) -> None:
        first = deterministic_edge_id(SynapseType.SIMILAR_TO, A, B)
        second = deterministic_edge_id(SynapseType.SIMILAR_TO, A, B)
        assert first == second

    @pytest.mark.parametrize("synapse_type", list(SynapseType))
    def test_id_is_always_a_safe_record_name(self, synapse_type: SynapseType) -> None:
        """The id is inlined into a SurrealDB record id, so keep it in [A-Za-z0-9_-]."""
        edge_id = deterministic_edge_id(synapse_type, A, B)
        assert all(ch.isalnum() or ch in "_-" for ch in edge_id)

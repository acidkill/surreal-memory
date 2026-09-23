"""Unit pinning tests for the shared id-unfolding helper and its one load-bearing caller.

``_to_public_id`` is the single spelling of "undo the record-name fold" that
``_row_to_neuron``/``_row_to_synapse``/``_row_to_fiber`` share, and
``maturation._canonicalised`` is the one place in the tree that deliberately folded the
OTHER way to match the (then broken) ``Fiber.id``. The audit of the fiber-id round trip
found exactly one load-bearing dependency on the old form and this is it, so the two
belong to the same change and are pinned together here.
"""

from __future__ import annotations

from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.storage.surrealdb._ids import _to_public_id, _to_surreal_id
from surreal_memory.storage.surrealdb.maturation import _canonicalised

_DASH = "0f3c9a21-1b7e-4c55-9d2a-6e8f10b4c7d3"
_UNDERSCORE = "0f3c9a21_1b7e_4c55_9d2a_6e8f10b4c7d3"


class TestToPublicId:
    def test_undoes_the_fold_for_engine_minted_ids(self) -> None:
        """Round trip for the ids the engine actually mints: uuid4 and hex hashes."""
        assert _to_public_id(_to_surreal_id(_DASH)) == _DASH
        assert _to_public_id(_UNDERSCORE) == _DASH

    def test_strips_the_table_prefix(self) -> None:
        assert _to_public_id(f"fiber:{_UNDERSCORE}") == _DASH
        assert _to_public_id(f"neuron:{_UNDERSCORE}") == _DASH

    def test_is_idempotent_on_an_already_public_id(self) -> None:
        assert _to_public_id(_DASH) == _DASH


class TestCanonicalisedFollowsFiberId:
    def _record(self, fiber_id: str) -> MaturationRecord:
        return MaturationRecord(
            fiber_id=fiber_id, brain_id="brain-under-test", stage=MemoryStage.EPISODIC
        )

    def test_hands_out_the_form_fiber_id_carries(self) -> None:
        """``extract_patterns`` does ``f.id in maturation_map`` — the keys must match.

        Pinned because the direction flipped with the fiber-id fix: while ``Fiber.id``
        came back folded, this folded to underscores to match it; now that ``Fiber.id``
        round-trips as dashes, it must follow. Either half alone silently breaks the join.
        """
        assert _canonicalised(self._record(_UNDERSCORE)).fiber_id == _DASH

    def test_leaves_an_already_canonical_record_untouched(self) -> None:
        record = self._record(_DASH)
        assert _canonicalised(record) is record

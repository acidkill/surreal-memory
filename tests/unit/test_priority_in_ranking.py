"""`priority` reaches recall ranking (it previously did not reach it at all).

A memory can be stored with ``priority: 10`` and the retrieval ranking will not notice:
``priority`` appears nowhere in ``_fiber_score``, the reranker, activation or lifecycle.
The value is persisted on ``typed_memory`` and, for the machine-derived variant, as
``metadata["auto_priority"]`` on the fiber — but scoring never reads either. Marking a
memory as critical is therefore a write into a drawer.

Design of the knob (ARCH, measured against a 2299-fiber production copy):

* multiplier ``m = 1 + priority_weight * (p - 5) / 5`` with ``p`` clamped to [0, 10],
  so the default ``p = 5`` of ``smem remember`` is exactly neutral and the range at
  ``priority_weight = 0.2`` is [0.8, 1.2] — a tie-breaker among near-equals, smaller
  than the recency swing (x1.7 for "yesterday vs a week ago"), not a lever;
* applied inside ``base_score`` (before ``activation_signal``), so the additive bonuses
  further down — tag boost, instruction boost, trigger overlap — keep the absolute
  units upstream calibrated them in. For a per-fiber constant the position among the
  *multiplicative* factors cannot change the ordering, only its interaction with the
  additive ones;
* ``auto_priority`` is NOT a full stand-in for a human's ``priority``: it measures
  novelty at encode time, not importance, and it sits on nearly every fiber. It gets
  its own weight, defaulting to 0.0 (inert), so turning it on is a separate, separately
  measured decision.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

_QUERY = "widget beta protocol handshake"


class _StopEncodeError(Exception):
    """Zatrzymuje `_remember`/`_encode_and_store` zaraz po wywołaniu enkodera.

    Interesuje nas WYŁĄCZNIE to, co handler przekazał do `encode()`; reszta ścieżki
    zapisu (typed_memory, hooki, projekty) jest poza zakresem tego pliku.
    """


@pytest.fixture
async def storage() -> AsyncIterator[InMemoryStorage]:
    s = InMemoryStorage()
    brain = Brain.create(name="priority_ranking_test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    yield s
    await s.close()


async def _anchor(storage: InMemoryStorage) -> str:
    """One shared neuron so ``activation_signal`` is identical for every fiber."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=_QUERY)
    await storage.add_neuron(neuron)
    return neuron.id


async def _add_fiber(
    storage: InMemoryStorage,
    neuron_id: str,
    *,
    salience: float,
    metadata: dict | None = None,
) -> Fiber:
    """Fiber with a fixed recall age, differing only in salience and metadata."""
    fiber = Fiber.create(
        neuron_ids={neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=neuron_id,
        summary=_QUERY,
        metadata=metadata,
    )
    fiber = replace(
        fiber,
        salience=salience,
        conductivity=1.0,
        last_conducted=utcnow() - timedelta(hours=24),
    )
    await storage.add_fiber(fiber)
    return fiber


def _pos(fibers_matched: list[str], fiber_id: str) -> int:
    return fibers_matched.index(fiber_id) if fiber_id in fibers_matched else -1


class TestExplicitPriorityAffectsRanking:
    """The important-but-slightly-less-salient memory must be able to win."""

    async def test_priority_10_overtakes_priority_5_with_a_small_salience_deficit(
        self, storage: InMemoryStorage
    ) -> None:
        """0.50 x 1.2 = 0.60 beats 0.55 x 1.0 — arithmetic chosen so the old code fails.

        The margin is deliberately narrow: a knob that only wins when it is also more
        salient would prove nothing, and one that wins regardless of salience would be
        a lever, not a tie-breaker.
        """
        anchor = await _anchor(storage)
        important = await _add_fiber(storage, anchor, salience=0.50, metadata={"priority": 10})
        routine = await _add_fiber(storage, anchor, salience=0.55, metadata={"priority": 5})

        pipeline = ReflexPipeline(storage, BrainConfig(priority_weight=0.2))
        result = await pipeline.query(_QUERY)

        assert _pos(result.fibers_matched, important.id) >= 0
        assert _pos(result.fibers_matched, routine.id) >= 0
        assert _pos(result.fibers_matched, important.id) < _pos(result.fibers_matched, routine.id)

    async def test_priority_weight_zero_is_a_strict_noop(self, storage: InMemoryStorage) -> None:
        """Distinguishability control: with the weight at 0 salience decides again.

        This is also the shape of the upstream PR's default — a merge that changes
        nobody's ranking until they ask for it.
        """
        anchor = await _anchor(storage)
        important = await _add_fiber(storage, anchor, salience=0.50, metadata={"priority": 10})
        routine = await _add_fiber(storage, anchor, salience=0.55, metadata={"priority": 5})

        pipeline = ReflexPipeline(storage, BrainConfig(priority_weight=0.0))
        result = await pipeline.query(_QUERY)

        assert _pos(result.fibers_matched, routine.id) < _pos(result.fibers_matched, important.id)

    async def test_fiber_without_priority_metadata_is_unchanged(
        self, storage: InMemoryStorage
    ) -> None:
        """Backward compatibility for the 2299 fibers that carry no priority at all."""
        anchor = await _anchor(storage)
        plain_high = await _add_fiber(storage, anchor, salience=0.55)
        plain_low = await _add_fiber(storage, anchor, salience=0.50)

        pipeline = ReflexPipeline(storage, BrainConfig(priority_weight=0.2))
        result = await pipeline.query(_QUERY)

        assert _pos(result.fibers_matched, plain_high.id) < _pos(
            result.fibers_matched, plain_low.id
        )

    async def test_out_of_range_priority_cannot_blow_up_the_multiplier(
        self, storage: InMemoryStorage
    ) -> None:
        """A hand-written ``priority: 99`` must not outrank everything forever.

        Clamping to [0, 10] keeps the multiplier inside [1-pw, 1+pw]; a 0.50-salience
        fiber therefore still loses to a 0.70-salience one (0.60 < 0.70).
        """
        anchor = await _anchor(storage)
        absurd = await _add_fiber(storage, anchor, salience=0.50, metadata={"priority": 99})
        sane = await _add_fiber(storage, anchor, salience=0.70, metadata={"priority": 5})

        pipeline = ReflexPipeline(storage, BrainConfig(priority_weight=0.2))
        result = await pipeline.query(_QUERY)

        assert _pos(result.fibers_matched, sane.id) < _pos(result.fibers_matched, absurd.id)


class TestAutoPriorityIsSeparate:
    """Machine-derived novelty must not silently masquerade as human importance."""

    async def test_auto_priority_is_inert_by_default(self, storage: InMemoryStorage) -> None:
        anchor = await _anchor(storage)
        auto_high = await _add_fiber(storage, anchor, salience=0.50, metadata={"auto_priority": 10})
        plain = await _add_fiber(storage, anchor, salience=0.55)

        pipeline = ReflexPipeline(storage, BrainConfig(priority_weight=0.2))
        result = await pipeline.query(_QUERY)

        assert _pos(result.fibers_matched, plain.id) < _pos(result.fibers_matched, auto_high.id)

    async def test_auto_priority_applies_only_under_its_own_weight(
        self, storage: InMemoryStorage
    ) -> None:
        anchor = await _anchor(storage)
        auto_high = await _add_fiber(storage, anchor, salience=0.50, metadata={"auto_priority": 10})
        plain = await _add_fiber(storage, anchor, salience=0.55)

        pipeline = ReflexPipeline(
            storage, BrainConfig(priority_weight=0.2, auto_priority_weight=0.2)
        )
        result = await pipeline.query(_QUERY)

        assert _pos(result.fibers_matched, auto_high.id) < _pos(result.fibers_matched, plain.id)

    async def test_explicit_priority_wins_over_auto_priority_on_the_same_fiber(
        self, storage: InMemoryStorage
    ) -> None:
        """When both keys are present the human's number decides.

        ``priority: 0`` with ``auto_priority: 10`` must be scored DOWN, not up.
        """
        anchor = await _anchor(storage)
        demoted = await _add_fiber(
            storage, anchor, salience=0.55, metadata={"priority": 0, "auto_priority": 10}
        )
        plain = await _add_fiber(storage, anchor, salience=0.50)

        pipeline = ReflexPipeline(
            storage, BrainConfig(priority_weight=0.2, auto_priority_weight=0.2)
        )
        result = await pipeline.query(_QUERY)

        # 0.55 x 0.8 = 0.44 < 0.50 x 1.0
        assert _pos(result.fibers_matched, plain.id) < _pos(result.fibers_matched, demoted.id)


class TestPriorityConfigSurface:
    def test_fork_defaults(self) -> None:
        cfg = BrainConfig()
        assert cfg.priority_weight == 0.2
        assert cfg.auto_priority_weight == 0.0

    def test_both_knobs_migrate_onto_a_stored_brain(self) -> None:
        """They must travel through ``extras`` — the explicit keys do not migrate."""
        from surreal_memory.unified_config import BrainSettings

        settings = BrainSettings.from_dict({"priority_weight": 0.3, "auto_priority_weight": 0.1})
        overrides = settings.runtime_overrides()
        assert overrides["priority_weight"] == 0.3
        assert overrides["auto_priority_weight"] == 0.1


class TestRememberStoresExplicitPriorityOnTheFiber:
    """The write side of the same defect, checked at the MCP boundary.

    ``priority`` was persisted only on ``typed_memory``; the fiber that scoring
    actually reads never saw it. Ranking cannot honour a number it is not given.
    Mirrors ``test_geo_mcp.py::test_valid_location_reaches_encoder_metadata`` — the
    encoder is the seam, the handler under test is real.
    """

    async def _captured_encode_metadata(self, args: dict) -> dict:
        from typing import Any
        from unittest.mock import AsyncMock, MagicMock, patch

        from surreal_memory.mcp.server import MCPServer
        from surreal_memory.unified_config import (
            ResponseConfig,
            ToolTierConfig,
            WriteGateConfig,
        )

        with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
            cfg = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/priority-test.db"),
                tool_tier=ToolTierConfig(tier="full"),
                response=ResponseConfig(),
            )
            cfg.write_gate = WriteGateConfig()
            cfg.encryption.enabled = False
            cfg.safety.auto_redact_min_severity = 3
            mock_get_config.return_value = cfg
            server = MCPServer()

        storage = AsyncMock()
        storage.get_brain = AsyncMock(return_value=MagicMock(id="test-brain", config=MagicMock()))
        storage._current_brain_id = "test-brain"
        storage.brain_id = "test-brain"

        captured: dict[str, Any] = {}

        async def _capture_encode(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise _StopEncodeError

        with (
            patch.object(server, "get_storage", return_value=storage),
            patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
            patch.object(server, "_fire_eternal_trigger"),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
            patch.object(server, "_passive_capture", new_callable=AsyncMock),
            patch("surreal_memory.mcp.remember_handler.MemoryEncoder") as mock_encoder_cls,
        ):
            mock_encoder_cls.return_value.encode = _capture_encode
            with pytest.raises(_StopEncodeError):
                await server._remember(args)

        return captured["metadata"]

    async def test_explicit_priority_reaches_the_fiber_metadata(self) -> None:
        metadata = await self._captured_encode_metadata(
            {"content": "a rule worth remembering", "priority": 10}
        )
        assert metadata["priority"] == 10

    async def test_auto_derived_priority_is_not_written_as_explicit(self) -> None:
        """No ``priority`` argument → no ``priority`` key on the fiber.

        Otherwise every automatically scored memory would look hand-marked, and the
        knob would stop meaning "a human said this matters".
        """
        metadata = await self._captured_encode_metadata({"content": "a routine note"})
        assert "priority" not in metadata


class TestCliRememberStoresExplicitPriorityOnTheFiber:
    """The CLI write path, which is the main one in this fleet.

    `smem remember --priority` persisted the value on `typed_memory` and called the
    encoder without metadata, so the fiber — the thing scoring reads — never carried
    it. Fixing only the MCP handler would have left the everyday path unfixed.
    """

    async def _encode_via_cli_path(self, priority: int | None):
        from unittest.mock import AsyncMock, MagicMock, patch

        from surreal_memory.cli.commands.memory import _encode_and_store
        from surreal_memory.core.memory_types import MemoryType, Priority

        captured: dict = {}

        class _Encoder:
            def __init__(self, *a, **kw):
                pass

            async def encode(self, **kwargs):
                captured.update(kwargs)
                raise _StopEncodeError

        storage = AsyncMock()
        storage.disable_auto_save = MagicMock()
        with (
            patch("surreal_memory.cli.commands.memory.MemoryEncoder", _Encoder),
            patch("surreal_memory.cli.commands.memory.build_dedup_pipeline", return_value=None),
            pytest.raises(_StopEncodeError),
        ):
            await _encode_and_store(
                storage,
                MagicMock(),
                "a rule worth remembering",
                tags=None,
                mem_type=MemoryType.FACT,
                mem_priority=Priority.from_int(priority)
                if priority is not None
                else Priority.NORMAL,
                expiry_days=None,
                project_id=None,
                priority_was_explicit=priority is not None,
            )
        return captured.get("metadata")

    async def test_explicit_priority_reaches_the_fiber_metadata(self) -> None:
        metadata = await self._encode_via_cli_path(10)
        assert metadata is not None and metadata["priority"] == 10

    async def test_no_priority_argument_writes_no_priority_key(self) -> None:
        metadata = await self._encode_via_cli_path(None)
        assert metadata is None or "priority" not in metadata

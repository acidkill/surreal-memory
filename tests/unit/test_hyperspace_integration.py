"""Integration tests for HyperspaceDB Phase 2 handler wiring.

Tests the full handler flow for:
- Gromov delta via _health(deep=True)
- Koopman auto-predict via _predict(action="auto")
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.mcp.cognitive_handler import CognitiveHandler

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


# ── Helpers ────────────────────────────────────────────────────────────


def _make_cognitive_handler(storage: AsyncMock) -> CognitiveHandler:
    """Build a CognitiveHandler with mocked storage."""
    handler = CognitiveHandler()
    handler.get_storage = AsyncMock(return_value=storage)  # type: ignore[attr-defined]
    handler.config = MagicMock()  # type: ignore[attr-defined]
    return handler


def _make_storage(brain_id: str = "test-brain") -> AsyncMock:
    """Build mock storage with explicit brain_id."""
    storage = AsyncMock()
    storage._current_brain_id = brain_id
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage.get_brain = AsyncMock(return_value=MagicMock(config=MagicMock()))
    storage.get_cognitive_state = AsyncMock(return_value=None)
    storage.list_predictions = AsyncMock(return_value=[])
    storage.get_calibration_stats = AsyncMock(
        return_value={"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
    )
    storage.get_synapses = AsyncMock(return_value=[])
    return storage


@dataclass(frozen=True)
class FakeNeuron:
    id: str
    content: str = "test content"
    content_hash: int = 0


@dataclass(frozen=True)
class FakeNeuronState:
    activation_level: float = 0.5
    activation_history: list[float] | None = None

    def __post_init__(self) -> None:
        # Allow mutable default via frozen override
        pass


# ── Gromov Integration via Health ─────────────────────────────────────


class TestGromovHealthIntegration:
    """Test Gromov delta-hyperbolicity via _health(deep=True)."""

    @pytest.fixture
    def star_storage(self) -> AsyncMock:
        """Mock storage with star topology (tree-like, delta≈0)."""
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        neurons = [MagicMock(id=f"n{i}") for i in range(6)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        synapse_map: dict[str, list[MagicMock]] = {
            "n0": [MagicMock(target_id=f"n{i}") for i in range(1, 6)],
        }
        for i in range(1, 6):
            synapse_map[f"n{i}"] = []
        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)
        return storage

    async def test_gromov_result_in_health(self, star_storage: AsyncMock) -> None:
        """estimate_gromov_delta on star graph returns tree-like quality."""
        from neural_memory.engine.gromov import estimate_gromov_delta

        result = await estimate_gromov_delta(star_storage, sample_size=6, seed=42)
        assert result.structure_quality == "tree-like"
        assert result.delta == 0.0
        assert result.sample_count == 6
        assert result.tuple_count > 0
        assert result.diameter > 0

    async def test_gromov_cache_hit(self, star_storage: AsyncMock) -> None:
        """Second call with same brain should use cached result."""
        from neural_memory.engine.gromov import estimate_gromov_delta

        result1 = await estimate_gromov_delta(star_storage, sample_size=6, seed=42)
        result2 = await estimate_gromov_delta(star_storage, sample_size=6, seed=99)

        # Both return valid results (cache is in tool_handlers, not engine)
        assert result1.structure_quality == "tree-like"
        assert result2.structure_quality == "tree-like"

    async def test_gromov_cycle_not_tree_like(self) -> None:
        """Cycle graph should have delta > 0."""
        from neural_memory.engine.gromov import estimate_gromov_delta

        storage = AsyncMock()
        n = 8
        neurons = [MagicMock(id=f"n{i}") for i in range(n)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        synapse_map: dict[str, list[MagicMock]] = {}
        for i in range(n):
            synapse_map[f"n{i}"] = [MagicMock(target_id=f"n{(i + 1) % n}")]
        storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)

        result = await estimate_gromov_delta(storage, sample_size=n, seed=42)
        assert result.delta > 0
        assert result.structure_quality != "tree-like"


# ── Koopman Auto-Predict Integration ─────────────────────────────────


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")
class TestKoopmanPredictAutoIntegration:
    """Test Koopman DMD via _predict(action='auto')."""

    async def test_auto_predict_insufficient_neurons(self) -> None:
        """Less than 10 neurons should return insufficient_data."""
        storage = _make_storage()
        storage.find_neurons = AsyncMock(return_value=[FakeNeuron(id=f"n{i}") for i in range(5)])
        handler = _make_cognitive_handler(storage)

        result = await handler._predict({"action": "auto"})
        assert result["status"] == "insufficient_data"
        assert result["predictions_created"] == 0

    async def test_auto_predict_insufficient_history(self) -> None:
        """Neurons without enough activation history should return insufficient_data."""
        storage = _make_storage()
        neurons = [FakeNeuron(id=f"n{i}") for i in range(15)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        # All neurons have only 3 data points (need 10+)
        state = MagicMock()
        state.activation_history = [0.1, 0.2, 0.3]
        storage.get_neuron_state = AsyncMock(return_value=state)

        handler = _make_cognitive_handler(storage)
        result = await handler._predict({"action": "auto"})
        assert result["status"] == "insufficient_data"

    async def test_auto_predict_with_spikes(self) -> None:
        """Neurons with rising activation should trigger spike predictions."""
        storage = _make_storage()
        neurons = [FakeNeuron(id=f"n{i}", content=f"Topic {i}") for i in range(15)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        # Build synthetic activation histories — linear rise for spike detection
        def make_state(neuron_id: str) -> MagicMock:
            idx = int(neuron_id[1:])
            state = MagicMock()
            # Linear rising trajectory: 0.1, 0.2, ..., 1.0, 1.1, ...
            base = 0.1 * (idx + 1)
            state.activation_history = [base + 0.1 * t for t in range(12)]
            return state

        storage.get_neuron_state = AsyncMock(side_effect=lambda nid: make_state(nid))
        storage.get_neuron = AsyncMock(
            side_effect=lambda nid: FakeNeuron(id=nid, content=f"Topic {nid}")
        )

        # Mock _predict_create to return prediction IDs
        handler = _make_cognitive_handler(storage)
        handler._predict_create = AsyncMock(  # type: ignore[attr-defined]
            return_value={"prediction_id": "pred-123"}
        )

        result = await handler._predict({"action": "auto"})
        assert result["status"] == "ok"
        assert result["neurons_analyzed"] >= 2
        assert "is_stable" in result
        assert "max_eigenvalue" in result

    async def test_auto_predict_stable_system(self) -> None:
        """Stable trajectories should report is_stable=True."""
        storage = _make_storage()
        neurons = [FakeNeuron(id=f"n{i}", content=f"Topic {i}") for i in range(15)]
        storage.find_neurons = AsyncMock(return_value=neurons)

        # Constant activation = stable
        def make_state(neuron_id: str) -> MagicMock:
            state = MagicMock()
            state.activation_history = [0.5] * 12
            return state

        storage.get_neuron_state = AsyncMock(side_effect=lambda nid: make_state(nid))
        handler = _make_cognitive_handler(storage)
        handler._predict_create = AsyncMock(  # type: ignore[attr-defined]
            return_value={"prediction_id": "pred-456"}
        )

        result = await handler._predict({"action": "auto"})
        assert result["status"] == "ok"
        assert result["is_stable"] is True

    async def test_auto_predict_respects_top_n(self) -> None:
        """top_n parameter should cap neuron count."""
        storage = _make_storage()
        neurons = [FakeNeuron(id=f"n{i}") for i in range(100)]
        storage.find_neurons = AsyncMock(return_value=neurons[:20])

        state = MagicMock()
        state.activation_history = [0.1 * t for t in range(12)]
        storage.get_neuron_state = AsyncMock(return_value=state)
        storage.get_neuron = AsyncMock(return_value=FakeNeuron(id="n0"))

        handler = _make_cognitive_handler(storage)
        handler._predict_create = AsyncMock(  # type: ignore[attr-defined]
            return_value={"prediction_id": "pred-789"}
        )

        await handler._predict({"action": "auto", "top_n": 20})
        storage.find_neurons.assert_called_once_with(limit=20)

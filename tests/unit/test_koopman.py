"""Tests for Koopman trajectory prediction via DMD."""

from __future__ import annotations

import pytest

try:
    import importlib.util

    _HAS_NUMPY = importlib.util.find_spec("numpy") is not None
except (ImportError, ModuleNotFoundError):
    _HAS_NUMPY = False

from neural_memory.engine.koopman import (
    KoopmanResult,
    TrajPrediction,
    koopman_extrapolate,
    predict_activation_trajectory,
)

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


# ── DMD Extrapolation ────────────────────────────────────────────────


class TestKoopmanExtrapolate:
    """Tests for low-level DMD extrapolation."""

    def test_linear_trajectory(self) -> None:
        """Linear growth should be perfectly predicted."""
        # 15 time steps, 3 features, each grows linearly
        t = 15
        trajectory = [[float(i), float(i * 2), float(i * 0.5)] for i in range(t)]

        result = koopman_extrapolate(trajectory, steps_ahead=3)

        assert len(result.predicted) == 3
        assert result.truncated_rank > 0

        # Last observed: [14, 28, 7]
        # Next predicted should be close to [15, 30, 7.5]
        pred = result.predicted[0]
        assert abs(pred[0] - 15.0) < 1.0
        assert abs(pred[1] - 30.0) < 2.0
        assert abs(pred[2] - 7.5) < 1.0

    def test_constant_trajectory(self) -> None:
        """Constant values should predict same constant."""
        t = 15
        trajectory = [[1.0, 2.0, 3.0]] * t

        result = koopman_extrapolate(trajectory, steps_ahead=2)

        if result.predicted:
            for pred in result.predicted:
                assert abs(pred[0] - 1.0) < 0.5
                assert abs(pred[1] - 2.0) < 0.5

    def test_insufficient_data(self) -> None:
        """Less than MIN_TRAJECTORY_LENGTH → empty result."""
        trajectory = [[1.0, 2.0]] * 5  # Only 5 steps

        result = koopman_extrapolate(trajectory, steps_ahead=3)

        assert result.predicted == []
        assert result.truncated_rank == 0

    def test_steps_capped(self) -> None:
        """Steps ahead capped at MAX_STEPS_AHEAD."""
        t = 15
        trajectory = [[float(i)] for i in range(t)]

        result = koopman_extrapolate(trajectory, steps_ahead=100)

        assert len(result.predicted) <= 10

    def test_stability_detection(self) -> None:
        """Exponentially growing trajectory → unstable (eigenvalue > 1)."""
        t = 15
        trajectory = [[2.0**i] for i in range(t)]

        result = koopman_extrapolate(trajectory, steps_ahead=1)

        assert result.max_eigenvalue > 1.0
        assert result.is_stable is False

    def test_stable_system(self) -> None:
        """Decaying trajectory → stable (eigenvalue < 1)."""
        t = 15
        trajectory = [[10.0 * (0.9**i)] for i in range(t)]

        result = koopman_extrapolate(trajectory, steps_ahead=1)

        assert result.max_eigenvalue <= 1.01  # Small tolerance
        assert result.is_stable is True

    def test_zero_trajectory(self) -> None:
        """All zeros → graceful handling."""
        t = 15
        trajectory = [[0.0, 0.0]] * t

        result = koopman_extrapolate(trajectory, steps_ahead=2)

        # Should return empty or zero predictions
        assert result.truncated_rank == 0 or all(
            abs(v) < 1e-10 for pred in result.predicted for v in pred
        )

    def test_oscillating_trajectory(self) -> None:
        """Oscillating signal should capture periodicity."""
        import math

        t = 20
        trajectory = [[math.sin(i * 0.5), math.cos(i * 0.5)] for i in range(t)]

        result = koopman_extrapolate(trajectory, steps_ahead=2)

        assert len(result.predicted) == 2
        assert result.truncated_rank > 0


# ── Trajectory Prediction ────────────────────────────────────────────


class TestPredictActivationTrajectory:
    """Tests for the high-level prediction API."""

    def test_basic_prediction(self) -> None:
        """Basic prediction with sufficient data."""
        history = {
            "n1": [float(i) * 0.1 for i in range(15)],
            "n2": [float(i) * 0.05 for i in range(15)],
        }

        result = predict_activation_trajectory(history, steps_ahead=3)

        assert len(result.predicted) == 3
        assert result.neuron_ids == ["n1", "n2"]
        assert result.steps_ahead == 3

    def test_spike_detection(self) -> None:
        """Rapidly growing neuron should be flagged as spike."""
        # n1 grows exponentially, n2 is flat
        history = {
            "n1": [float(2**i) for i in range(15)],
            "n2": [1.0] * 15,
        }

        result = predict_activation_trajectory(history, steps_ahead=3, spike_threshold=2.0)

        assert "n1" in result.spike_neurons

    def test_empty_history(self) -> None:
        result = predict_activation_trajectory({}, steps_ahead=3)
        assert result == TrajPrediction.empty()

    def test_insufficient_data(self) -> None:
        history = {"n1": [1.0, 2.0, 3.0]}  # Only 3 points
        result = predict_activation_trajectory(history, steps_ahead=3)
        assert result.predicted == []

    def test_uneven_lengths_truncated(self) -> None:
        """Different trajectory lengths → truncated to shortest."""
        history = {
            "n1": [float(i) for i in range(15)],
            "n2": [float(i) for i in range(20)],
        }

        result = predict_activation_trajectory(history, steps_ahead=2)

        assert len(result.predicted) == 2

    def test_sorted_neuron_ids(self) -> None:
        """Neuron IDs are sorted for deterministic ordering."""
        history = {
            "z_neuron": [1.0] * 15,
            "a_neuron": [2.0] * 15,
            "m_neuron": [3.0] * 15,
        }

        result = predict_activation_trajectory(history, steps_ahead=1)

        assert result.neuron_ids == ["a_neuron", "m_neuron", "z_neuron"]

    def test_stability_propagated(self) -> None:
        """Stability info from DMD propagates to TrajPrediction."""
        # Exponential growth → unstable
        history = {"n1": [float(2**i) for i in range(15)]}
        result = predict_activation_trajectory(history, steps_ahead=1)
        assert result.is_stable is False
        assert result.max_eigenvalue > 1.0


# ── Dataclass Tests ──────────────────────────────────────────────────


class TestTrajPrediction:
    def test_empty(self) -> None:
        result = TrajPrediction.empty()
        assert result.predicted == []
        assert result.is_stable is True

    def test_unavailable(self) -> None:
        result = TrajPrediction.unavailable("test reason")
        assert result.predicted == []

    def test_frozen(self) -> None:
        result = TrajPrediction.empty()
        with pytest.raises(AttributeError):
            result.is_stable = False  # type: ignore[misc]


class TestKoopmanResult:
    def test_frozen(self) -> None:
        result = KoopmanResult(predicted=[], max_eigenvalue=0.5, is_stable=True, truncated_rank=2)
        with pytest.raises(AttributeError):
            result.max_eigenvalue = 1.0  # type: ignore[misc]

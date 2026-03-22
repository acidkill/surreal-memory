"""Koopman operator trajectory prediction via Dynamic Mode Decomposition.

Linearizes nonlinear activation dynamics to predict future neuron
activation levels. Detects emerging topics and stability of knowledge.

Requires numpy (optional dependency). Gracefully returns empty results
when numpy is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Minimum data points required per trajectory
MIN_TRAJECTORY_LENGTH = 10
# Default energy threshold for SVD truncation
DEFAULT_ENERGY_THRESHOLD = 0.95
# Maximum extrapolation steps
MAX_STEPS_AHEAD = 10


@dataclass(frozen=True)
class TrajPrediction:
    """Predicted activation trajectory for a set of neurons."""

    predicted: list[list[float]]
    neuron_ids: list[str]
    steps_ahead: int
    max_eigenvalue: float
    is_stable: bool
    spike_neurons: list[str]

    @staticmethod
    def empty() -> TrajPrediction:
        return TrajPrediction(
            predicted=[],
            neuron_ids=[],
            steps_ahead=0,
            max_eigenvalue=0.0,
            is_stable=True,
            spike_neurons=[],
        )

    @staticmethod
    def unavailable(reason: str = "numpy not installed") -> TrajPrediction:
        return TrajPrediction(
            predicted=[],
            neuron_ids=[],
            steps_ahead=0,
            max_eigenvalue=0.0,
            is_stable=True,
            spike_neurons=[],
        )


def koopman_extrapolate(
    trajectory: list[list[float]],
    steps_ahead: int = 3,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> KoopmanResult:
    """Linearize nonlinear dynamics via DMD approximation.

    Args:
        trajectory: T time steps x N features. Each row is a snapshot.
        steps_ahead: Number of future steps to predict.
        energy_threshold: SVD truncation energy threshold (0-1).

    Returns:
        KoopmanResult with predictions and stability info.
    """
    if not _HAS_NUMPY:
        return KoopmanResult(
            predicted=[],
            max_eigenvalue=0.0,
            is_stable=True,
            truncated_rank=0,
        )

    steps_ahead = min(steps_ahead, MAX_STEPS_AHEAD)

    arr = np.array(trajectory, dtype=np.float64)
    t_steps, n_features = arr.shape

    if t_steps < MIN_TRAJECTORY_LENGTH:
        return KoopmanResult(
            predicted=[],
            max_eigenvalue=0.0,
            is_stable=True,
            truncated_rank=0,
        )

    # Build data matrices: X = [x_0, ..., x_{T-2}], Y = [x_1, ..., x_{T-1}]
    x_mat = arr[:-1].T  # N x (T-1)
    y_mat = arr[1:].T  # N x (T-1)

    # SVD of X
    u, s, vt = np.linalg.svd(x_mat, full_matrices=False)

    # Truncate to rank r (energy threshold)
    total_energy = np.sum(s**2)
    if total_energy == 0:
        return KoopmanResult(
            predicted=[],
            max_eigenvalue=0.0,
            is_stable=True,
            truncated_rank=0,
        )

    cumulative = np.cumsum(s**2) / total_energy
    rank = int(np.searchsorted(cumulative, energy_threshold) + 1)
    rank = min(rank, len(s))

    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]

    # Koopman approximation: A_tilde = U_r^T @ Y @ V_r @ S_r^{-1}
    s_inv = np.diag(1.0 / s_r)
    a_tilde = u_r.T @ y_mat @ vt_r.T @ s_inv

    # Eigenvalues for stability analysis
    eigenvalues = np.linalg.eigvals(a_tilde)
    max_eig = float(np.max(np.abs(eigenvalues)))

    # Extrapolate from last state
    last_state = u_r.T @ arr[-1]  # Project to reduced space

    predicted: list[list[float]] = []
    current = last_state.copy()
    for _ in range(steps_ahead):
        current = a_tilde @ current
        # Project back to full space
        full_state = u_r @ current
        predicted.append(full_state.tolist())

    return KoopmanResult(
        predicted=predicted,
        max_eigenvalue=max_eig,
        is_stable=max_eig <= 1.0,
        truncated_rank=rank,
    )


@dataclass(frozen=True)
class KoopmanResult:
    """Result from Koopman DMD extrapolation."""

    predicted: list[list[float]]
    max_eigenvalue: float
    is_stable: bool
    truncated_rank: int


def predict_activation_trajectory(
    activation_history: dict[str, list[float]],
    steps_ahead: int = 3,
    spike_threshold: float = 2.0,
) -> TrajPrediction:
    """Predict future activation levels for a set of neurons.

    Args:
        activation_history: Mapping neuron_id → list of activation values
            over time. All lists must be the same length (≥ MIN_TRAJECTORY_LENGTH).
        steps_ahead: Number of future steps to predict.
        spike_threshold: Multiplier over current activation to flag as spike.

    Returns:
        TrajPrediction with per-neuron predictions and spike alerts.
    """
    if not _HAS_NUMPY:
        return TrajPrediction.unavailable()

    if not activation_history:
        return TrajPrediction.empty()

    neuron_ids = sorted(activation_history.keys())
    trajectories = [activation_history[nid] for nid in neuron_ids]

    # Validate: all same length, sufficient data
    lengths = {len(t) for t in trajectories}
    if len(lengths) != 1:
        # Truncate to shortest
        min_len = min(lengths)
        trajectories = [t[:min_len] for t in trajectories]

    t_steps = len(trajectories[0])
    if t_steps < MIN_TRAJECTORY_LENGTH:
        return TrajPrediction.empty()

    # Build trajectory matrix: T x N (time steps x neurons)
    trajectory_matrix = [
        [trajectories[n][t] for n in range(len(neuron_ids))] for t in range(t_steps)
    ]

    result = koopman_extrapolate(trajectory_matrix, steps_ahead)

    if not result.predicted:
        return TrajPrediction.empty()

    # Detect spikes: neurons whose predicted activation > spike_threshold x current
    current_activations = trajectory_matrix[-1]
    spike_neurons: list[str] = []

    for step_pred in result.predicted:
        for i, (pred_val, curr_val) in enumerate(zip(step_pred, current_activations, strict=False)):
            if curr_val > 0 and pred_val > spike_threshold * curr_val:
                if neuron_ids[i] not in spike_neurons:
                    spike_neurons.append(neuron_ids[i])

    return TrajPrediction(
        predicted=result.predicted,
        neuron_ids=neuron_ids,
        steps_ahead=steps_ahead,
        max_eigenvalue=result.max_eigenvalue,
        is_stable=result.is_stable,
        spike_neurons=spike_neurons,
    )

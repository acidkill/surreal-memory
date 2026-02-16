"""Formal Hebbian learning rule with saturation and novelty adaptation.

Implements: Δw = η_eff * pre * post * (w_max - w)

Where η_eff = η * (1 + novelty_boost * e^(-novelty_decay * freq))

This provides:
- Natural saturation: weights near ceiling barely change
- Novelty boost: new synapses learn fast, familiar ones stabilize
- Competitive normalization: total outgoing weight per neuron capped
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.synapse import Synapse


@dataclass(frozen=True)
class LearningConfig:
    """Configuration for the Hebbian learning rule.

    Attributes:
        learning_rate: Base learning rate η
        weight_max: Maximum synapse weight ceiling
        novelty_boost_max: Maximum multiplier for novel synapses
        novelty_decay_rate: How fast novelty decays with reinforcement count
        weight_normalization_budget: Max total outgoing weight per neuron
    """

    learning_rate: float = 0.05
    weight_max: float = 1.0
    novelty_boost_max: float = 3.0
    novelty_decay_rate: float = 0.06
    weight_normalization_budget: float = 5.0


@dataclass(frozen=True)
class WeightUpdate:
    """Result of a learning rule computation.

    Attributes:
        new_weight: The updated synapse weight
        delta: The actual change applied
        effective_rate: The effective learning rate used (with novelty)
        saturated: Whether the weight was near ceiling
    """

    new_weight: float
    delta: float
    effective_rate: float
    saturated: bool


def compute_effective_rate(
    base_rate: float,
    reinforced_count: int,
    novelty_boost_max: float = 3.0,
    novelty_decay_rate: float = 0.06,
) -> float:
    """Compute novelty-adjusted learning rate.

    New synapses (freq=0) get η_eff ≈ (1 + novelty_boost) * η.
    Frequently reinforced synapses converge toward base η.

    Args:
        base_rate: Base learning rate η
        reinforced_count: Number of times synapse was reinforced
        novelty_boost_max: Maximum novelty multiplier
        novelty_decay_rate: Exponential decay rate of novelty

    Returns:
        Effective learning rate with novelty adjustment
    """
    novelty_factor = 1.0 + novelty_boost_max * math.exp(-novelty_decay_rate * reinforced_count)
    return base_rate * novelty_factor


def hebbian_update(
    current_weight: float,
    pre_activation: float,
    post_activation: float,
    reinforced_count: int,
    config: LearningConfig | None = None,
) -> WeightUpdate:
    """Compute Hebbian weight update with saturation and novelty.

    Formula: Δw = η_eff * pre * post * (w_max - w)

    The (w_max - w) term provides natural saturation — weights near
    the ceiling barely change, preventing runaway strengthening.

    Args:
        current_weight: Current synapse weight
        pre_activation: Pre-synaptic neuron activation level [0, 1]
        post_activation: Post-synaptic neuron activation level [0, 1]
        reinforced_count: How many times this synapse was reinforced
        config: Learning configuration (uses defaults if None)

    Returns:
        WeightUpdate with new weight and diagnostics
    """
    if config is None:
        config = LearningConfig()

    # Clamp to valid range on entry
    current_weight = max(0.0, min(config.weight_max, current_weight))

    # Zero pre or post activation → no learning (biological constraint)
    if pre_activation <= 0.0 or post_activation <= 0.0:
        return WeightUpdate(
            new_weight=current_weight,
            delta=0.0,
            effective_rate=0.0,
            saturated=False,
        )

    effective_rate = compute_effective_rate(
        base_rate=config.learning_rate,
        reinforced_count=reinforced_count,
        novelty_boost_max=config.novelty_boost_max,
        novelty_decay_rate=config.novelty_decay_rate,
    )

    # Saturation term: (w_max - w) prevents runaway growth
    headroom = config.weight_max - current_weight
    saturated = headroom < 0.05

    # Hebbian delta: η_eff * pre * post * (w_max - w)
    delta = effective_rate * pre_activation * post_activation * headroom

    # Clamp to valid range
    new_weight = max(0.0, min(config.weight_max, current_weight + delta))

    return WeightUpdate(
        new_weight=new_weight,
        delta=new_weight - current_weight,
        effective_rate=effective_rate,
        saturated=saturated,
    )


def anti_hebbian_update(
    current_weight: float,
    strength: float,
    config: LearningConfig | None = None,
) -> WeightUpdate:
    """Compute anti-Hebbian weight reduction (for conflict resolution).

    Reduces weight proportional to conflict strength, with a floor at 0.

    Args:
        current_weight: Current synapse weight
        strength: Conflict strength [0, 1]
        config: Learning configuration (uses defaults if None)

    Returns:
        WeightUpdate with reduced weight
    """
    if config is None:
        config = LearningConfig()

    delta = -config.learning_rate * strength * current_weight
    new_weight = max(0.0, current_weight + delta)

    return WeightUpdate(
        new_weight=new_weight,
        delta=new_weight - current_weight,
        effective_rate=config.learning_rate,
        saturated=False,
    )


def normalize_outgoing_weights(
    synapses: list[Synapse],
    source_id: str,
    budget: float = 5.0,
) -> list[Synapse]:
    """Normalize outgoing synapse weights to stay within budget.

    If total outgoing weight from a neuron exceeds the budget,
    all outgoing synapses are scaled proportionally to fit.

    This implements competitive normalization — strengthening one
    pathway weakens others relative to the budget.

    Args:
        synapses: List of synapses (may include non-outgoing ones)
        source_id: The neuron whose outgoing weights to normalize
        budget: Maximum total outgoing weight

    Returns:
        New list with normalized weights (unchanged if within budget)
    """
    outgoing = [s for s in synapses if s.source_id == source_id]
    non_outgoing = [s for s in synapses if s.source_id != source_id]

    if not outgoing:
        return list(synapses)

    total = sum(s.weight for s in outgoing)

    if total <= budget:
        return list(synapses)

    scale = budget / total

    from neural_memory.core.synapse import Synapse as SynapseClass

    normalized_outgoing = [
        SynapseClass(
            id=s.id,
            source_id=s.source_id,
            target_id=s.target_id,
            type=s.type,
            weight=s.weight * scale,
            direction=s.direction,
            metadata=s.metadata,
            reinforced_count=s.reinforced_count,
            last_activated=s.last_activated,
            created_at=s.created_at,
        )
        for s in outgoing
    ]

    return non_outgoing + normalized_outgoing

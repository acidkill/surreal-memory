"""Shared conservative eligibility for selection, resume and conditional deletion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from surreal_memory.core.synapse import Synapse


@dataclass(frozen=True)
class PrunePolicy:
    reference_time: datetime
    weight_threshold: float
    min_inactive_days: float
    dream_decay_multiplier: float

    def effective_weight(self, synapse: Synapse) -> float:
        decayed = synapse.time_decay(reference_time=self.reference_time)
        if synapse.reinforced_count < 2:
            if synapse.metadata.get("_inferred", False):
                decayed = decayed.decay(factor=0.5)
            if synapse.metadata.get("_dream", False):
                decayed = decayed.decay(factor=1.0 / self.dream_decay_multiplier)
            if synapse.metadata.get("_semantic_discovery", False):
                decayed = decayed.decay(factor=0.5)
        return decayed.weight

    def eligible(self, synapse: Synapse) -> bool:
        """Structural protections are supplied separately from the current storage snapshot."""
        if synapse.weight >= 0.1 or synapse.created_at > self.reference_time:
            return False
        if self.effective_weight(synapse) >= self.weight_threshold:
            return False
        since = synapse.last_activated or synapse.created_at
        grace = (
            self.min_inactive_days
            if synapse.last_activated
            else max(1.0, self.min_inactive_days / 7)
        )
        return (self.reference_time - since).total_seconds() / 86400 >= grace

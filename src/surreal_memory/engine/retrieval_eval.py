"""Write-denying storage boundary for full-pipeline retrieval evaluation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

_READ_METHODS = frozenset(
    {
        "find_fibers",
        "find_fibers_batch",
        "find_fibers_by_embedding",
        "find_neurons",
        "find_neurons_by_embedding",
        "find_neurons_ranked",
        "get_brain",
        "get_co_activation_counts",
        "get_depth_priors_batch",
        "get_graph_density",
        "get_keyword_df_batch",
        "get_neighbors",
        "get_neuron",
        "get_neurons_batch",
        "get_neuron_state",
        "get_neuron_states_batch",
        "get_source",
        "get_synapses",
        "get_synapses_for_neurons",
        "get_total_fiber_count",
        "get_typed_memories_batch",
        "search_fiber_summaries",
    }
)


class ReadOnlyStorage:
    """Allow only audited retrieval reads; copy results to isolate mutable fixtures.

    Use ``ReflexPipeline(ReadOnlyStorage(storage), config, read_only=True)``.
    ``read_counts`` records calls; ``write_attempts`` must remain empty even when
    retrieval catches a non-critical exception. Raw SQL and connections are denied.
    """

    def __init__(self, storage: Any) -> None:
        self._storage = storage
        self.read_counts: Counter[str] = Counter()
        self.write_attempts: list[str] = []

    @property
    def brain_id(self) -> str:
        return str(self._storage.brain_id)

    @property
    def _brain_id(self) -> str:
        return str(getattr(self._storage, "_brain_id", self.brain_id))

    def __getattr__(self, name: str) -> Any:
        if name not in _READ_METHODS:
            self.write_attempts.append(name)
            raise RuntimeError(f"Read-only evaluation denied storage operation: {name}")
        method = getattr(self._storage, name)

        async def read(*args: Any, **kwargs: Any) -> Any:
            self.read_counts[name] += 1
            return deepcopy(await method(*args, **kwargs))

        return read

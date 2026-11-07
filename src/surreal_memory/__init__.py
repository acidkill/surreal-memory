"""Surreal-Memory public API, lazily loaded to keep command hooks lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "3.12.2"

if TYPE_CHECKING:
    from surreal_memory.core.brain import Brain, BrainConfig
    from surreal_memory.core.brain_mode import (
        BrainMode,
        BrainModeConfig,
        SharedConfig,
        SyncStrategy,
    )
    from surreal_memory.core.fiber import Fiber
    from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
    from surreal_memory.core.synapse import Direction, Synapse, SynapseType
    from surreal_memory.engine.brain_transplant import TransplantFilter, TransplantResult
    from surreal_memory.engine.brain_versioning import BrainVersion, VersionDiff, VersioningEngine
    from surreal_memory.engine.encoder import EncodingResult, MemoryEncoder
    from surreal_memory.engine.reflex_activation import CoActivation, ReflexActivation
    from surreal_memory.engine.retrieval import DepthLevel, ReflexPipeline, RetrievalResult


_EXPORT_MODULES = {
    "Brain": "surreal_memory.core.brain",
    "BrainConfig": "surreal_memory.core.brain",
    "BrainMode": "surreal_memory.core.brain_mode",
    "BrainModeConfig": "surreal_memory.core.brain_mode",
    "SharedConfig": "surreal_memory.core.brain_mode",
    "SyncStrategy": "surreal_memory.core.brain_mode",
    "Fiber": "surreal_memory.core.fiber",
    "Neuron": "surreal_memory.core.neuron",
    "NeuronState": "surreal_memory.core.neuron",
    "NeuronType": "surreal_memory.core.neuron",
    "Direction": "surreal_memory.core.synapse",
    "Synapse": "surreal_memory.core.synapse",
    "SynapseType": "surreal_memory.core.synapse",
    "TransplantFilter": "surreal_memory.engine.brain_transplant",
    "TransplantResult": "surreal_memory.engine.brain_transplant",
    "BrainVersion": "surreal_memory.engine.brain_versioning",
    "VersionDiff": "surreal_memory.engine.brain_versioning",
    "VersioningEngine": "surreal_memory.engine.brain_versioning",
    "EncodingResult": "surreal_memory.engine.encoder",
    "MemoryEncoder": "surreal_memory.engine.encoder",
    "CoActivation": "surreal_memory.engine.reflex_activation",
    "ReflexActivation": "surreal_memory.engine.reflex_activation",
    "DepthLevel": "surreal_memory.engine.retrieval",
    "ReflexPipeline": "surreal_memory.engine.retrieval",
    "RetrievalResult": "surreal_memory.engine.retrieval",
}


def __getattr__(name: str) -> Any:
    """Load the same public symbols on demand, without booting encoding for hooks."""
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep exported API discovery stable for interactive clients."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__",
    "Brain",
    "BrainConfig",
    "BrainMode",
    "BrainModeConfig",
    "BrainVersion",
    "CoActivation",
    "DepthLevel",
    "Direction",
    "EncodingResult",
    "Fiber",
    "MemoryEncoder",
    "Neuron",
    "NeuronState",
    "NeuronType",
    "ReflexActivation",
    "ReflexPipeline",
    "RetrievalResult",
    "SharedConfig",
    "Synapse",
    "SynapseType",
    "SyncStrategy",
    "TransplantFilter",
    "TransplantResult",
    "VersionDiff",
    "VersioningEngine",
]

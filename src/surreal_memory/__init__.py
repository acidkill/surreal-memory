"""Surreal-Memory - Reflex-based memory system for AI agents."""

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

__version__ = "2.17.0"

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

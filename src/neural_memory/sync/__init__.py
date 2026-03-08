"""Real-time and incremental synchronization for NeuralMemory.

Heavy modules (SyncClient, SyncEngine) are imported lazily to avoid
pulling in aiohttp and the full storage stack at startup — which would
add ~300ms to every MCP server cold start.
"""

from neural_memory.sync.device import DeviceInfo, get_device_id, get_device_info, get_device_name
from neural_memory.sync.protocol import (
    ConflictStrategy,
    SyncChange,
    SyncConflict,
    SyncRequest,
    SyncResponse,
    SyncStatus,
)

__all__ = [
    "SyncClient",
    "SyncClientState",
    "DeviceInfo",
    "get_device_id",
    "get_device_info",
    "get_device_name",
    "ConflictStrategy",
    "SyncChange",
    "SyncConflict",
    "SyncRequest",
    "SyncResponse",
    "SyncStatus",
    "SyncEngine",
]


def __getattr__(name: str) -> type:
    """Lazy-load heavy sync modules on first access."""
    if name in ("SyncClient", "SyncClientState"):
        from neural_memory.sync.client import SyncClient, SyncClientState

        globals()["SyncClient"] = SyncClient
        globals()["SyncClientState"] = SyncClientState
        return SyncClient if name == "SyncClient" else SyncClientState
    if name == "SyncEngine":
        from neural_memory.sync.sync_engine import SyncEngine

        globals()["SyncEngine"] = SyncEngine
        return SyncEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

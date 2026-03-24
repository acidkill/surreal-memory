"""Hub endpoints for multi-device incremental sync."""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.server.dependencies import get_storage, require_local_request
from neural_memory.storage.base import NeuralStorage
from neural_memory.sync.protocol import (
    ConflictStrategy,
    MerkleSyncRequest,
    SyncChange,
    SyncRequest,
    SyncResponse,
)
from neural_memory.sync.sync_engine import SyncEngine
from neural_memory.unified_config import get_config
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

_BRAIN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
_DEVICE_ID_PATTERN = re.compile(r"^[a-fA-F0-9]+$")


router = APIRouter(
    prefix="/hub",
    tags=["hub"],
    dependencies=[Depends(require_local_request)],
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class RegisterDeviceRequest(BaseModel):
    device_id: str = Field(..., max_length=32)
    brain_id: str = Field(..., max_length=128)
    device_name: str = Field("", max_length=256)


class SyncChangeItem(BaseModel):
    sequence: int = 0
    entity_type: str = Field(..., max_length=32)
    entity_id: str = Field(..., max_length=128)
    operation: str = Field(..., max_length=16)
    device_id: str = Field("", max_length=32)
    changed_at: str = Field("", max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class HubSyncRequest(BaseModel):
    device_id: str = Field(..., max_length=32)
    brain_id: str = Field(..., max_length=128)
    last_sequence: int = Field(0, ge=0)
    changes: list[SyncChangeItem] = Field(default_factory=list, max_length=1000)
    strategy: str = Field("prefer_recent", max_length=32)


class HubMerkleSyncRequest(BaseModel):
    device_id: str = Field(..., max_length=32)
    brain_id: str = Field(..., max_length=128)
    root_hash: str = Field(..., max_length=128)
    buckets: dict[str, dict[str, str]] = Field(default_factory=dict)
    strategy: str = Field("prefer_recent", max_length=32)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_brain_id(brain_id: str) -> None:
    """Raise HTTPException if brain_id is invalid."""
    if not _BRAIN_ID_PATTERN.match(brain_id):
        raise HTTPException(status_code=422, detail="Invalid brain_id format")


def _validate_device_id(device_id: str) -> None:
    """Raise HTTPException if device_id is invalid (must be hex, max 32 chars)."""
    if not _DEVICE_ID_PATTERN.match(device_id):
        raise HTTPException(
            status_code=422, detail="Invalid device_id format: must be hex characters only"
        )


def _validate_strategy(strategy: str) -> ConflictStrategy:
    """Parse and validate strategy string. Returns ConflictStrategy or raises HTTPException."""
    try:
        return ConflictStrategy(strategy)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid conflict strategy",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register", tags=["hub"], summary="Register a device for a brain")
async def register_device(
    body: RegisterDeviceRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Register a device for a brain.

    Validates the brain_id and device_id formats, sets the brain context,
    then calls register_device on storage.
    """
    _validate_brain_id(body.brain_id)
    _validate_device_id(body.device_id)

    try:
        # Auto-create brain if needed
        existing_brain = await storage.get_brain(body.brain_id)
        if existing_brain is None:
            now = utcnow()
            brain = Brain(
                id=body.brain_id,
                name=body.brain_id,
                config=BrainConfig(),
                created_at=now,
                updated_at=now,
            )
            await storage.save_brain(brain)
            logger.info("Hub auto-created brain %s for device registration", body.brain_id)

        storage.set_brain(body.brain_id)
        device = await storage.register_device(body.device_id, body.device_name)
    except Exception:
        logger.error(
            "Failed to register device %s for brain %s",
            body.device_id,
            body.brain_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to register device")

    return {
        "device_id": device.device_id,
        "device_name": device.device_name,
        "brain_id": body.brain_id,
        "registered_at": device.registered_at.isoformat(),
        "last_sync_sequence": device.last_sync_sequence,
    }


@router.post("/sync", tags=["hub"], summary="Push/pull incremental changes")
async def hub_sync(
    body: HubSyncRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Push device changes to the hub and pull changes not yet seen.

    The requesting device sends its local changes together with the last hub
    sequence it has observed. The hub applies the incoming changes, resolves
    conflicts, and returns the set of changes the device has not seen yet.
    """
    _validate_brain_id(body.brain_id)
    _validate_device_id(body.device_id)
    conflict_strategy = _validate_strategy(body.strategy)

    # Cap the changes list defensively (Pydantic max_length already enforces
    # this at the model level, but we guard again for safety).
    capped_changes = body.changes[:1000]

    sync_changes = [
        SyncChange(
            sequence=item.sequence,
            entity_type=item.entity_type,
            entity_id=item.entity_id,
            operation=item.operation,
            device_id=item.device_id,
            changed_at=item.changed_at or utcnow().isoformat(),
            payload=item.payload,
        )
        for item in capped_changes
    ]

    sync_request = SyncRequest(
        device_id=body.device_id,
        brain_id=body.brain_id,
        last_sequence=body.last_sequence,
        changes=sync_changes,
        strategy=conflict_strategy,
    )

    try:
        # Auto-create brain if it doesn't exist on the hub yet.
        existing_brain = await storage.get_brain(body.brain_id)
        if existing_brain is None:
            now = utcnow()
            brain = Brain(
                id=body.brain_id,
                name=body.brain_id,
                config=BrainConfig(),
                created_at=now,
                updated_at=now,
            )
            await storage.save_brain(brain)
            logger.info("Hub auto-created brain %s for incoming sync", body.brain_id)

        storage.set_brain(body.brain_id)
        sync_engine = SyncEngine(storage, device_id="hub", strategy=conflict_strategy)
        response: SyncResponse = await sync_engine.handle_hub_sync(sync_request)
    except Exception:
        logger.error(
            "Hub sync failed for device %s brain %s",
            body.device_id,
            body.brain_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Sync failed")

    return {
        "hub_sequence": response.hub_sequence,
        "status": response.status.value,
        "message": response.message,
        "changes": [
            {
                "sequence": c.sequence,
                "entity_type": c.entity_type,
                "entity_id": c.entity_id,
                "operation": c.operation,
                "device_id": c.device_id,
                "changed_at": c.changed_at,
                "payload": c.payload,
            }
            for c in response.changes
        ],
        "conflicts": [
            {
                "entity_type": cf.entity_type,
                "entity_id": cf.entity_id,
                "local_device": cf.local_device,
                "remote_device": cf.remote_device,
                "resolution": cf.resolution,
                "details": cf.details,
            }
            for cf in response.conflicts
        ],
    }


@router.get(
    "/status/{brain_id}",
    tags=["hub"],
    summary="Get sync status for a brain",
)
async def hub_status(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Return change-log statistics and the number of registered devices for a brain."""
    _validate_brain_id(brain_id)

    try:
        storage.set_brain(brain_id)
        stats = await storage.get_change_log_stats()
        devices_list = await storage.list_devices()
    except Exception:
        logger.error("Failed to get hub status for brain %s", brain_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve status")

    return {
        "brain_id": brain_id,
        "device_count": len(devices_list),
        "change_log": stats,
    }


@router.get(
    "/devices/{brain_id}",
    tags=["hub"],
    summary="List registered devices for a brain",
)
async def list_devices(
    brain_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Return all devices registered against a brain."""
    _validate_brain_id(brain_id)

    try:
        storage.set_brain(brain_id)
        devices_list = await storage.list_devices()
    except Exception:
        logger.error("Failed to list devices for brain %s", brain_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve devices")

    return {
        "brain_id": brain_id,
        "devices": [
            {
                "device_id": d.device_id,
                "device_name": d.device_name,
                "registered_at": d.registered_at.isoformat(),
                "last_sync_sequence": d.last_sync_sequence,
            }
            for d in devices_list
        ],
    }


@router.post("/sync/merkle", tags=["hub"], summary="Merkle delta sync (Pro)")
async def hub_merkle_sync(
    body: HubMerkleSyncRequest,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Single-round Merkle sync: compare bucket hashes and return diffs.

    Pro-only. Device sends all bucket hashes (~49KB), hub compares in one
    pass, returns only changed entities grouped by prefix.
    """
    _validate_brain_id(body.brain_id)
    _validate_device_id(body.device_id)
    conflict_strategy = _validate_strategy(body.strategy)

    config = get_config()
    if not config.is_pro():
        raise HTTPException(
            status_code=403,
            detail="Merkle sync requires a Pro license",
        )

    try:
        existing_brain = await storage.get_brain(body.brain_id)
        if existing_brain is None:
            now = utcnow()
            brain = Brain(
                id=body.brain_id,
                name=body.brain_id,
                config=BrainConfig(),
                created_at=now,
                updated_at=now,
            )
            await storage.save_brain(brain)

        storage.set_brain(body.brain_id)
        sync_engine = SyncEngine(storage, device_id="hub", strategy=conflict_strategy)

        request = MerkleSyncRequest(
            device_id=body.device_id,
            brain_id=body.brain_id,
            root_hash=body.root_hash,
            buckets=body.buckets,
            strategy=conflict_strategy,
        )
        response = await sync_engine.handle_merkle_sync(request, is_pro=True)
    except HTTPException:
        raise
    except Exception:
        logger.error(
            "Merkle sync failed for device %s brain %s",
            body.device_id,
            body.brain_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Merkle sync failed")

    return {
        "status": response.status,
        "hub_root_hash": response.hub_root_hash,
        "changed_prefixes": response.changed_prefixes,
        "diffs": [
            {
                "entity_type": d.entity_type,
                "prefix": d.prefix,
                "entity_ids": d.entity_ids,
                "entities": d.entities,
            }
            for d in response.diffs
        ],
        "hub_sequence": response.hub_sequence,
        "message": response.message,
    }

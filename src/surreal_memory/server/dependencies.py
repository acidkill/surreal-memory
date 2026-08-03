"""Shared dependencies for API routes."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from surreal_memory.core.brain import Brain
from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


@lru_cache(maxsize=1)
def _parse_trusted_networks(
    networks: tuple[str, ...],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse and cache CIDR network strings into ip_network objects."""
    parsed: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for net in networks:
        if not net:
            continue
        try:
            parsed.append(ipaddress.ip_network(net, strict=False))
        except ValueError:
            logger.warning("Invalid trusted network CIDR: %s (skipped)", net)
    return tuple(parsed)


def is_trusted_host(host: str) -> bool:
    """Check if a host is trusted (localhost or in configured trusted networks).

    Args:
        host: Client IP address or hostname.

    Returns:
        True if the host is localhost or within a trusted network CIDR.
    """
    if host in _LOCALHOST_HOSTS:
        return True

    from surreal_memory.utils.config import get_config

    config = get_config()
    if not config.trusted_networks:
        return False

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False

    parsed = _parse_trusted_networks(tuple(config.trusted_networks))
    return any(addr in net for net in parsed)


async def require_local_request(request: Request) -> None:
    """Reject requests from untrusted sources.

    Allows localhost and any IP within SURREAL_MEMORY_TRUSTED_NETWORKS CIDRs.
    """
    if request.client is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_trusted_host(request.client.host):
        raise HTTPException(status_code=403, detail="Forbidden")


async def get_storage(
    x_brain_id: Annotated[str | None, Header(alias="X-Brain-ID")] = None,
) -> NeuralStorage:
    """Dependency to get storage instance for the requested brain.

    When X-Brain-ID header is provided, resolves a storage instance
    connected to that brain's DB file. When omitted, returns the
    default storage.

    This is overridden by the application at startup.
    """
    raise NotImplementedError("Storage not configured")


async def get_brain(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain_id: Annotated[str | None, Header(alias="X-Brain-ID")] = None,
) -> Brain:
    """Dependency to get and validate brain from header.

    When X-Brain-ID header is omitted, falls back to the active brain
    from config (current_brain).  This makes the header optional for
    simple REST clients while still allowing explicit brain selection.

    The ``get_storage`` dependency already resolves the correct
    brain-specific storage instance based on the same header, so
    ``storage`` here is connected to the right DB file.
    """
    if brain_id is None:
        from surreal_memory.unified_config import get_config

        brain_id = get_config().current_brain

    brain = await storage.get_brain(brain_id)
    if brain is None:
        # Fallback: brain_id might be a name, not a UUID
        brain = await storage.find_brain_by_name(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    # Scope by the brain *name*, never by ``brain.id``. Rows carry brain_id as a
    # plain string equal to the brain name (see unified_config._get_surrealdb_storage
    # and _get_sqlite_storage, which both set_brain(name) for exactly this reason),
    # while a brain created by an older version has a random uuid4 primary key.
    # This dependency resolves before any route body runs, so binding the scope to
    # brain.id made every route read and write a UUID scope that holds no rows —
    # which is also why the `storage.brain_id or brain.name` fix from #97 never
    # took effect server-side: storage.brain_id was already the UUID.
    storage.set_brain(brain.name)
    return brain


@asynccontextmanager
async def storage_for_scope(storage: NeuralStorage, scope: str) -> AsyncIterator[NeuralStorage]:
    """Yield a storage whose implicitly-bound brain IS ``scope``, without mutating ``storage``.

    Several handlers filter on whatever brain the *shared, process-wide*
    storage instance is bound to rather than taking an explicit brain_id.
    Calling ``storage.set_brain(scope)`` on that shared instance to answer one
    request works until something else reads or mutates the same instance
    concurrently -- a request for a different brain, or a background
    maintenance loop reading ``storage.brain_id`` -- and inherits whichever
    brain last won the race.

    The common case (the shared instance is already bound to ``scope``) costs
    nothing and reuses it. Otherwise an isolated storage is opened on the
    scope and closed afterward: only SurrealDB hands out a private instance to
    close; the other backends return the shared one, which must not be closed
    out from under concurrent callers.
    """
    if storage.brain_id == scope:
        yield storage
        return

    from surreal_memory.unified_config import create_isolated_storage, get_config

    scoped = await create_isolated_storage(scope)
    try:
        yield scoped
    finally:
        if get_config().storage_backend == "surrealdb":
            await scoped.close()

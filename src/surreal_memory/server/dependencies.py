"""Shared dependencies for API routes."""

from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache
from typing import Annotated
from urllib.parse import urlsplit

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


def _header_host(value: str | None) -> str | None:
    """Extract the lowercased hostname from an Origin/Referer header value.

    Returns None when the value is empty or has no parseable host — including
    the opaque ``Origin: null`` that browsers send for sandboxed iframes,
    ``file://`` pages, and some cross-origin redirects.
    """
    if not value:
        return None
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


async def require_local_request(request: Request) -> None:
    """Reject requests from untrusted sources.

    Two independent checks are applied and both must pass:

    * **Client IP** — allows localhost and any IP within
      ``SURREAL_MEMORY_TRUSTED_NETWORKS`` CIDRs.
    * **Origin / Referer** — defense-in-depth against the "local dev server +
      web attacker" CSRF class: a browser tab open to any website is also
      "local" from the server's point of view, so the client IP alone is not
      enough. Browsers always send an ``Origin`` header on cross-origin
      fetch/XHR; if one is present (falling back to ``Referer``) and its host
      is not trusted, the request is rejected — *regardless* of the configured
      CORS policy, which ``SURREAL_MEMORY_CORS_ORIGINS=*`` would otherwise
      disable. Same-origin requests and non-browser clients omit the header
      (or send a trusted host) and pass.
    """
    if request.client is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_trusted_host(request.client.host):
        raise HTTPException(status_code=403, detail="Forbidden")

    # Prefer Origin (sent on every cross-origin browser request); fall back to
    # Referer. Decide on the first header that is present: a trusted host passes,
    # anything else (untrusted host or an unparseable/``null`` origin) is rejected.
    for header_name in ("origin", "referer"):
        value = request.headers.get(header_name)
        if not value:
            continue
        host = _header_host(value)
        if host is None or not is_trusted_host(host):
            raise HTTPException(status_code=403, detail="Forbidden")
        break


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

    # Set brain context using the actual brain ID
    storage.set_brain(brain.id)
    return brain

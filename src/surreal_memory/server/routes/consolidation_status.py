"""Sanitized dashboard observation and internal-only consolidation metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from surreal_memory.server.consolidation_status import (
    ConsolidationStatus,
    observe_consolidation,
    render_metrics,
)
from surreal_memory.server.dependencies import require_local_request

router = APIRouter(tags=["consolidation"], dependencies=[Depends(require_local_request)])


@router.get("/api/dashboard/consolidation-status", response_model=ConsolidationStatus)
async def consolidation_status(request: Request, response: Response) -> ConsolidationStatus:
    """Read sanitized saved work status under the existing dashboard access policy."""
    response.headers["Cache-Control"] = "no-store"
    return await observe_consolidation(getattr(request.app.state, "storage", None))


@router.get("/metrics/consolidation", response_class=Response)
async def consolidation_metrics(request: Request) -> Response:
    """Expose bounded metrics to direct internal scrapes; proxy requests are forbidden."""
    if any(
        key in request.headers
        for key in ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")
    ):
        raise HTTPException(status_code=403, detail="Forbidden")
    status = await observe_consolidation(getattr(request.app.state, "storage", None))
    return Response(
        render_metrics(status),
        media_type="text/plain; version=0.0.4",
        headers={"Cache-Control": "no-store"},
    )

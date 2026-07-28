"""Reasoning-training dashboard API routes.

Exposes the reasoning-training pipeline to the dashboard, kept in its own router
(dashboard_api.py is already ~1800 lines). Local-only via require_local_request.

Endpoints (prefix ``/api/dashboard/reasoning``):
- ``GET  /status``            config + per-model trace/pattern stats + coverage + mining-job state
- ``PUT  /config``            partial config update (toggles, mining_models, injection_map, limits)
- ``POST /mine``              trigger a one-off mining run in the background (409 if already running)
- ``GET  /patterns``          list learned pattern fibers (filter by model/category, paginated)
- ``GET  /patterns/{id}``     one pattern's full detail
- ``DELETE /patterns/{id}``   delete one pattern
- ``DELETE /patterns``        delete all patterns for a model
- ``DELETE /traces``          privacy wipe: delete all staged traces for a model
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace as dc_replace
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from surreal_memory.core.brain import Brain
from surreal_memory.engine.reasoning_miner import _MODELS_WITHOUT_THINKING, normalize_model
from surreal_memory.engine.reasoning_progress import (
    PHASE_DISTILLING,
    PHASE_DONE,
    PHASE_IDLE,
    PHASE_INGESTING,
    PHASE_SCANNING,
    MiningProgress,
)
from surreal_memory.server.dependencies import get_brain, get_storage, require_local_request
from surreal_memory.server.models import ErrorResponse
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dashboard/reasoning",
    tags=["reasoning"],
    dependencies=[Depends(require_local_request)],
)

# Pattern fibers are idempotent by _reasoning_signature (bounded population);
# matches the distiller / injection fetch ceiling.
_PATTERN_FETCH_LIMIT = 20_000
_MAX_PAGE = 100
# Model names / injection globs: letters, digits, dot, underscore, dash, glob chars.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._*?-]{1,128}$")

# Background mining jobs, keyed by brain_id so one brain's run neither blocks nor
# leaks status into another's. Assumes a single uvicorn worker (the dashboard is
# launched without workers=N — cli/commands/tools.py); a multi-worker deployment
# would need shared state (e.g. a DB row) instead of these process globals.
_mining_tasks: dict[str, asyncio.Task[None]] = {}
_mining_states: dict[str, dict[str, Any]] = {}


def _brain_scope(brain: Brain) -> str:
    """Return the key every reasoning row is scoped by: the brain NAME.

    Never ``brain.id``. That attribute carries the brain record's UUID primary
    key, while every reasoning row -- traces, pattern fibers, their neurons and
    synapses -- is written under the brain *name*. Passing the UUID into
    ``create_isolated_storage()`` binds the whole mining job to a scope nothing
    else reads, so the patterns are mined into a brain recall never sees.

    Measured on the live brain before this was fixed: 196 pattern fibers,
    92 neurons, 84 synapses and 6070 reasoning traces stranded under the UUID
    while 10905 traces sat under the name.

    Deliberately does NOT consult ``storage.brain_id``: that reports whatever the
    shared singleton is currently bound to, which is the same class of ambient
    coupling this function exists to remove. ``brain`` comes from the request's
    own dependency, so its name is authoritative here.

    One helper, used by every endpoint in this router, so the derivations cannot
    drift apart again -- which is how the status endpoint ended up correct while
    mining and the privacy wipe stayed wrong.
    """
    return brain.name


def _idle_mining_state() -> dict[str, Any]:
    return {
        "running": False,
        "started_at": None,
        "finished_at": None,
        "phase": PHASE_IDLE,
        "files_total": 0,
        "files_scanned": 0,
        "traces_found": 0,
        "traces_ingested": 0,
        "traces_processed": 0,
        "patterns_learned": 0,
        "current_model": None,
        "models_done": 0,
        "models_total": 0,
        "dry_run": False,
        "error": None,
    }


# ── Response / request models ─────────────────────────────────────────────────


class ModelTraceStats(BaseModel):
    """Per-model trace + pattern counts and coverage."""

    model: str
    trace_count: int = 0
    unprocessed: int = 0
    pattern_count: int = 0
    has_thinking_text: bool = True
    last_trace_at: str | None = None
    coverage_percent: float = 0.0


class CategoryCoverage(BaseModel):
    """Coverage of one category for one source model."""

    category: str
    pattern_count: int
    covered: bool


class MiningJobState(BaseModel):
    """State of the (single) background mining job."""

    running: bool
    started_at: str | None = None
    finished_at: str | None = None
    phase: str = PHASE_IDLE
    files_total: int = 0
    files_scanned: int = 0
    traces_found: int = 0
    traces_ingested: int = 0
    traces_processed: int = 0
    patterns_learned: int = 0
    current_model: str | None = None
    models_done: int = 0
    models_total: int = 0
    dry_run: bool = False
    error: str | None = None


class ReasoningStatusResponse(BaseModel):
    """Full reasoning-training status for the dashboard."""

    config: dict[str, Any]
    detected_models: list[str]
    per_model: list[ModelTraceStats]
    coverage_by_model: dict[str, list[CategoryCoverage]]
    total_traces: int
    unprocessed_traces: int
    total_patterns: int
    mining: MiningJobState


class ReasoningConfigUpdate(BaseModel):
    """Partial reasoning-training config update (all fields optional)."""

    mining_enabled: bool | None = None
    injection_enabled: bool | None = None
    mining_models: list[str] | None = None
    injection_map: dict[str, str] | None = None
    categories: list[str] | None = None
    min_trace_chars: int | None = Field(None, ge=0, le=1_000_000)
    max_trace_chars: int | None = Field(None, ge=1, le=10_000_000)
    scan_lookback_days: int | None = Field(None, ge=0, le=100_000)
    retention_days: int | None = Field(None, ge=1, le=100_000)
    max_traces_total: int | None = Field(None, ge=1, le=10_000_000)
    min_cluster_support: int | None = Field(None, ge=1, le=100_000)
    min_confidence: float | None = Field(None, ge=0.0, le=1.0)
    min_patterns_per_category: int | None = Field(None, ge=1, le=100_000)
    injection_max_patterns: int | None = Field(None, ge=1, le=1000)
    injection_max_chars: int | None = Field(None, ge=1, le=1_000_000)
    # Per-model distillation targets (model -> 0..100). Validated in update_config.
    pattern_targets: dict[str, int] | None = None


class MineRequest(BaseModel):
    """Trigger a one-off mining run."""

    backfill: bool = Field(False, description="Scan the full history (scan_lookback_days=0)")
    dry_run: bool = Field(
        False, description="Don't write — report as a no-op (parity with consolidation)"
    )
    models: list[str] | None = Field(None, description="Restrict mining to these source models")


class MineResponse(BaseModel):
    """Response to a mining trigger."""

    status: str
    mining: MiningJobState


class PatternSummary(BaseModel):
    """One learned reasoning pattern (list view)."""

    id: str
    source_model: str
    category: str
    title: str
    confidence: float
    frequency: int
    signature: str


class PatternDetail(PatternSummary):
    """One learned reasoning pattern with its full strategy/description."""

    strategy: str
    description: str
    summary: str


class PatternsListResponse(BaseModel):
    """Paginated learned-pattern list."""

    patterns: list[PatternSummary]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    """Result of a delete operation."""

    deleted: int


# ── Helpers ───────────────────────────────────────────────────────────────────


def _has_thinking(model: str) -> bool:
    """A model has minable thinking text unless it's on the denylist (prefix match)."""
    norm = normalize_model(model)
    return not any(norm.startswith(prefix) for prefix in _MODELS_WITHOUT_THINKING)


def _validate_model_name(name: str, *, field: str) -> str:
    """Validate a model name / injection glob; 422 on a bad shape."""
    cleaned = name.strip()
    if not cleaned or not _MODEL_NAME_RE.match(cleaned):
        raise HTTPException(status_code=422, detail=f"Invalid {field} name: {name!r}")
    return cleaned


def _pattern_meta(fiber: Any) -> dict[str, Any]:
    return fiber.metadata if isinstance(fiber.metadata, dict) else {}


def _to_summary(fiber: Any) -> PatternSummary:
    md = _pattern_meta(fiber)
    return PatternSummary(
        id=str(fiber.id),
        source_model=str(md.get("_source_model", "")),
        category=str(md.get("_reasoning_category", "")),
        title=str(md.get("_reasoning_title", "")),
        confidence=float(md.get("_reasoning_confidence", 0.0) or 0.0),
        frequency=int(md.get("_reasoning_frequency", 0) or 0),
        signature=str(md.get("_reasoning_signature", "")),
    )


async def _fetch_pattern_fibers(storage: NeuralStorage) -> list[Any]:
    """All learned pattern fibers for the current brain (bounded population)."""
    return await storage.find_fibers(metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT)


# ── GET /status ───────────────────────────────────────────────────────────────


@router.get("/status", response_model=ReasoningStatusResponse, summary="Reasoning-training status")
async def get_status(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain: Annotated[Brain, Depends(get_brain)],
) -> ReasoningStatusResponse:
    """Config + per-model trace/pattern stats + per-model category coverage."""
    from surreal_memory.unified_config import get_config

    config = get_config()
    rt = config.reasoning_training
    brain_id = _brain_scope(brain)

    stats = await storage.get_reasoning_stats(brain_id)
    by_model_traces: dict[str, Any] = stats.get("by_model", {})
    fibers = await _fetch_pattern_fibers(storage)

    # Per-model pattern counts and per-category coverage from one fiber fetch.
    pattern_counts: dict[str, int] = {}
    cat_counts: dict[str, dict[str, int]] = {}
    for f in fibers:
        md = _pattern_meta(f)
        source = str(md.get("_source_model", ""))
        if not source:
            continue
        pattern_counts[source] = pattern_counts.get(source, 0) + 1
        if float(md.get("_reasoning_confidence", 0.0) or 0.0) < rt.min_confidence:
            continue
        cat = str(md.get("_reasoning_category", ""))
        if cat in rt.categories:
            model_cats = cat_counts.setdefault(source, {})
            model_cats[cat] = model_cats.get(cat, 0) + 1

    # detected_models = union of DISTINCT trace models, config mining_models,
    # pattern source models, and injection sources.
    detected = sorted(
        set(by_model_traces)
        | set(rt.mining_models)
        | set(pattern_counts)
        | {s for _, s in rt.injection_map}
    )

    per_model: list[ModelTraceStats] = []
    coverage_by_model: dict[str, list[CategoryCoverage]] = {}
    for model in detected:
        tstats = by_model_traces.get(model, {})
        counts = cat_counts.get(model, {})
        n_covered = sum(
            1 for c in rt.categories if counts.get(c, 0) >= rt.min_patterns_per_category
        )
        cov_pct = round(n_covered / len(rt.categories) * 100.0, 1) if rt.categories else 0.0
        trace_count = int(tstats.get("trace_count", 0) or 0)
        per_model.append(
            ModelTraceStats(
                model=model,
                trace_count=trace_count,
                unprocessed=int(tstats.get("unprocessed", 0) or 0),
                pattern_count=pattern_counts.get(model, 0),
                has_thinking_text=_has_thinking(model) or trace_count > 0,
                last_trace_at=tstats.get("last_trace_at") or None,
                coverage_percent=cov_pct,
            )
        )
        coverage_by_model[model] = [
            CategoryCoverage(
                category=c,
                pattern_count=counts.get(c, 0),
                covered=counts.get(c, 0) >= rt.min_patterns_per_category,
            )
            for c in rt.categories
        ]

    return ReasoningStatusResponse(
        config=rt.to_dict(),
        detected_models=detected,
        per_model=per_model,
        coverage_by_model=coverage_by_model,
        total_traces=int(stats.get("total", 0) or 0),
        unprocessed_traces=int(stats.get("unprocessed", 0) or 0),
        total_patterns=len(fibers),
        mining=MiningJobState(**_mining_states.get(brain_id, _idle_mining_state())),
    )


# ── PUT /config ───────────────────────────────────────────────────────────────


@router.put(
    "/config",
    responses={422: {"model": ErrorResponse}},
    summary="Update reasoning-training config",
)
async def update_config(body: ReasoningConfigUpdate) -> dict[str, Any]:
    """Partial config update. 422 on a model without thinking text or a bad name."""
    from surreal_memory.unified_config import get_config, set_config

    config = get_config()
    rt = config.reasoning_training
    changes: dict[str, Any] = {}

    if body.mining_enabled is not None:
        changes["mining_enabled"] = body.mining_enabled
    if body.injection_enabled is not None:
        changes["injection_enabled"] = body.injection_enabled

    if body.mining_models is not None:
        models: list[str] = []
        for m in body.mining_models:
            name = _validate_model_name(m, field="mining_models")
            # A source model must actually produce thinking text to be minable.
            if not _has_thinking(name):
                raise HTTPException(
                    status_code=422,
                    detail=f"Model {name!r} has no thinking text and cannot be mined",
                )
            models.append(name)
        changes["mining_models"] = tuple(dict.fromkeys(models))

    if body.injection_map is not None:
        pairs: list[tuple[str, str]] = []
        for target, source in body.injection_map.items():
            t = _validate_model_name(target, field="injection_map target")
            s = _validate_model_name(source, field="injection_map source")
            # The source is matched literally (==) against a pattern's _source_model
            # (reasoning_injection), so a glob would silently never match — reject it.
            if "*" in s or "?" in s:
                raise HTTPException(
                    status_code=422,
                    detail=f"Injection source {source!r} must be a literal model, not a glob",
                )
            # The source (whose patterns get injected) must have thinking text;
            # the target (recipient) may be a thinking-less model like opus.
            if not _has_thinking(s):
                raise HTTPException(
                    status_code=422,
                    detail=f"Injection source {s!r} has no thinking text",
                )
            pairs.append((t, s))
        changes["injection_map"] = tuple(pairs)

    if body.categories is not None:
        cats: list[str] = []
        for c in body.categories:
            name = c.strip()
            if not name:
                continue
            if not _MODEL_NAME_RE.match(name):
                raise HTTPException(status_code=422, detail=f"Invalid category name: {c!r}")
            cats.append(name[:64])
        deduped = tuple(dict.fromkeys(cats))
        if not deduped:
            # 422 rather than a silent no-op (categories drive coverage math).
            raise HTTPException(status_code=422, detail="categories cannot be empty")
        changes["categories"] = deduped

    if body.pattern_targets is not None:
        targets: dict[str, int] = {}
        for model, count in body.pattern_targets.items():
            name = _validate_model_name(model, field="pattern_targets")
            # A target model must actually produce thinking text to be distillable.
            if not _has_thinking(name):
                raise HTTPException(
                    status_code=422,
                    detail=f"Model {name!r} has no thinking text and cannot be a pattern target",
                )
            try:
                targets[name] = max(0, min(int(count), 100))
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=422,
                    detail=f"pattern_targets[{name!r}] must be an integer 0-100",
                ) from None
        changes["pattern_targets"] = targets

    for field_name in (
        "min_trace_chars",
        "max_trace_chars",
        "scan_lookback_days",
        "retention_days",
        "max_traces_total",
        "min_cluster_support",
        "min_confidence",
        "min_patterns_per_category",
        "injection_max_patterns",
        "injection_max_chars",
    ):
        value = getattr(body, field_name)
        if value is not None:
            changes[field_name] = value

    new_rt = dc_replace(rt, **changes)
    updated = dc_replace(config, reasoning_training=new_rt)
    updated.save()
    set_config(updated)
    return {"status": "updated", "config": new_rt.to_dict()}


# ── POST /mine ────────────────────────────────────────────────────────────────


async def _run_mining(
    brain_id: str, backfill: bool, dry_run: bool, models: list[str] | None
) -> None:
    """Background mining job: ingest reasoning traces then distill patterns.

    Uses direct ingest+distill (not ConsolidationEngine) so the request's
    backfill / models overrides are honored — the engine's reasoning handlers
    reload config from disk and would ignore them. Runs on an ISOLATED storage
    (create_isolated_storage) so a concurrently-served request's set_brain() on
    the shared SurrealDB singleton can't redirect this job's graph writes into
    the wrong brain.
    """
    from surreal_memory.engine.reasoning_distiller import distill_reasoning_patterns
    from surreal_memory.engine.reasoning_miner import ingest_reasoning_traces
    from surreal_memory.unified_config import create_isolated_storage, get_config

    state = _mining_states.setdefault(brain_id, _idle_mining_state())
    try:
        if dry_run:
            # Parity with consolidation dry_run: no scanning, no writes.
            state.update(phase=PHASE_DONE, traces_ingested=0, patterns_learned=0)
            return

        config = get_config()
        rt = config.reasoning_training
        overrides: dict[str, Any] = {}
        if backfill:
            overrides["scan_lookback_days"] = 0
        if models:
            overrides["mining_models"] = tuple(models)
        run_config = dc_replace(config, reasoning_training=dc_replace(rt, **overrides))

        def _on_progress(p: MiningProgress) -> None:
            # Merge per phase so distill snapshots (which don't know file counts)
            # don't zero out the ingest phase's file/trace counters.
            updates: dict[str, Any] = {"phase": p.phase}
            if p.phase in (PHASE_SCANNING, PHASE_INGESTING):
                updates.update(
                    files_total=p.files_total,
                    files_scanned=p.files_scanned,
                    traces_found=p.traces_found,
                    traces_ingested=p.traces_ingested,
                )
            elif p.phase == PHASE_DISTILLING:
                updates.update(
                    traces_processed=p.traces_processed,
                    patterns_learned=p.patterns_learned,
                    current_model=p.current_model,
                    models_done=p.models_done,
                    models_total=p.models_total,
                )
            state.update(updates)

        storage = await create_isolated_storage(brain_id)
        # We own the isolated SurrealDB instance and must close it; SQLite / in-memory
        # return the shared instance, which must NOT be closed out from under others.
        owns_storage = config.storage_backend == "surrealdb"
        try:
            logger.info("reasoning mining started for brain %s (backfill=%s)", brain_id, backfill)
            state["phase"] = PHASE_SCANNING
            ingest = await ingest_reasoning_traces(
                storage, brain_id, run_config, backfill=backfill, progress=_on_progress
            )
            logger.info(
                "reasoning ingest done for brain %s: %d/%d files, %d traces (%d new)",
                brain_id,
                ingest.files_scanned,
                ingest.files_total,
                ingest.traces_scanned,
                ingest.traces_ingested,
            )
            state.update(
                phase=PHASE_DISTILLING,
                files_total=ingest.files_total,
                files_scanned=ingest.files_scanned,
                traces_found=ingest.traces_scanned,
                traces_ingested=ingest.traces_ingested,
            )
            distill = await distill_reasoning_patterns(
                storage, brain_id, run_config, drain=True, progress=_on_progress
            )
            state.update(
                phase=PHASE_DONE,
                traces_ingested=ingest.traces_ingested,
                traces_processed=distill.traces_processed,
                patterns_learned=distill.patterns_learned,
                models_done=distill.models_seen,
                models_total=distill.models_seen,
            )
            logger.info(
                "reasoning mining done for brain %s: %d patterns from %d traces processed",
                brain_id,
                distill.patterns_learned,
                distill.traces_processed,
            )
        finally:
            if owns_storage:
                await storage.close()
    except Exception:  # never crash; surface a generic error via job state
        logger.exception("reasoning mining job failed for brain %s", brain_id)
        state["error"] = "mining failed; see server logs"
    finally:
        state["running"] = False
        state["finished_at"] = utcnow().isoformat()


@router.post(
    "/mine",
    response_model=MineResponse,
    status_code=202,
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Trigger a one-off mining run",
)
async def trigger_mining(
    body: MineRequest,
    brain: Annotated[Brain, Depends(get_brain)],
) -> MineResponse:
    """Start a background mining run. 400 if mining disabled, 409 if already running.

    Job state and the concurrency guard are per-brain, so a run on one brain never
    blocks or leaks status into another.
    """
    from surreal_memory.unified_config import get_config

    config = get_config()
    if not config.reasoning_training.mining_enabled:
        # Privacy: never scan ~/.claude transcripts unless mining is enabled.
        raise HTTPException(status_code=400, detail="Mining is disabled; enable it first")

    # Scope on the NAME. Keying the job on brain.id also made the status endpoint
    # -- which reads by name -- miss a running job and report idle throughout.
    scope = _brain_scope(brain)

    existing = _mining_tasks.get(scope)
    if existing is not None and not existing.done():
        raise HTTPException(status_code=409, detail="A mining run is already in progress")

    models = None
    if body.models:
        models = [_validate_model_name(m, field="models") for m in body.models]

    state = _idle_mining_state()
    state.update(running=True, started_at=utcnow().isoformat(), dry_run=body.dry_run)
    _mining_states[scope] = state
    _mining_tasks[scope] = asyncio.create_task(
        _run_mining(scope, body.backfill, body.dry_run, models)
    )
    return MineResponse(status="started", mining=MiningJobState(**state))


# ── GET /patterns ─────────────────────────────────────────────────────────────


@router.get("/patterns", response_model=PatternsListResponse, summary="List learned patterns")
async def list_patterns(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain: Annotated[Brain, Depends(get_brain)],
    model: str | None = Query(None, description="Filter by source model"),
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=_MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> PatternsListResponse:
    """List learned reasoning patterns (filter by source model / category)."""
    fibers = await _fetch_pattern_fibers(storage)
    summaries = [_to_summary(f) for f in fibers]
    if model:
        summaries = [s for s in summaries if s.source_model == model]
    if category:
        summaries = [s for s in summaries if s.category == category]
    summaries.sort(key=lambda s: s.confidence * s.frequency, reverse=True)
    total = len(summaries)
    page = summaries[offset : offset + limit]
    return PatternsListResponse(patterns=page, total=total, limit=limit, offset=offset)


@router.get(
    "/patterns/{pattern_id}",
    response_model=PatternDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Get one pattern's detail",
)
async def get_pattern(
    pattern_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain: Annotated[Brain, Depends(get_brain)],
) -> PatternDetail:
    """Return one learned pattern with its full strategy/description."""
    fiber = await storage.get_fiber(pattern_id)
    if fiber is None or not _pattern_meta(fiber).get("_reasoning_pattern"):
        raise HTTPException(status_code=404, detail="Pattern not found")
    md = _pattern_meta(fiber)
    base = _to_summary(fiber)
    return PatternDetail(
        **base.model_dump(),
        strategy=str(md.get("_reasoning_strategy", "")),
        description=str(md.get("_reasoning_description", "")),
        summary=str(fiber.summary or ""),
    )


@router.delete(
    "/patterns/{pattern_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete one learned pattern",
)
async def delete_pattern(
    pattern_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain: Annotated[Brain, Depends(get_brain)],
) -> DeleteResponse:
    """Delete one learned pattern fiber.

    The pattern's private title-neuron is currently left as a harmless graph orphan
    (follow-up ticket); the shared reasoning_category neuron is kept by design.
    """
    fiber = await storage.get_fiber(pattern_id)
    if fiber is None or not _pattern_meta(fiber).get("_reasoning_pattern"):
        raise HTTPException(status_code=404, detail="Pattern not found")
    deleted = await storage.delete_fiber(pattern_id)
    return DeleteResponse(deleted=1 if deleted else 0)


@router.delete(
    "/patterns", response_model=DeleteResponse, summary="Delete all patterns for a model"
)
async def delete_patterns_by_model(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain: Annotated[Brain, Depends(get_brain)],
    model: str = Query(..., description="Delete all patterns whose source model matches"),
) -> DeleteResponse:
    """Delete every learned pattern for a source model.

    Only the pattern fibers are removed; the shared reasoning_category concept-neuron
    is intentionally kept (other models' patterns reference it), and the pattern's
    private title-neuron is currently left as a harmless graph orphan (follow-up
    ticket). Raw mined thinking lives in traces — DELETE /traces is the full wipe.
    """
    if not model.strip():
        return DeleteResponse(deleted=0)  # never a blanket wipe
    fibers = await _fetch_pattern_fibers(storage)
    victims = [f for f in fibers if _pattern_meta(f).get("_source_model") == model]
    deleted = 0
    for f in victims:
        if await storage.delete_fiber(str(f.id)):
            deleted += 1
    return DeleteResponse(deleted=deleted)


# ── DELETE /traces (privacy wipe) ─────────────────────────────────────────────


@router.delete("/traces", response_model=DeleteResponse, summary="Wipe staged traces for a model")
async def delete_traces_by_model(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    brain: Annotated[Brain, Depends(get_brain)],
    model: str = Query(..., description="Delete all staged reasoning traces for this model"),
) -> DeleteResponse:
    """Privacy wipe: delete all staged reasoning traces for a model.

    Scoped by name: wiping under brain.id deleted from a scope the ingest path
    does not write to, so the user was told traces were removed while the ones
    under the brain name survived.
    """
    deleted = await storage.delete_reasoning_traces_by_model(_brain_scope(brain), model)
    return DeleteResponse(deleted=deleted)

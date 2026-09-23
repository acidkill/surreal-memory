"""Durable, brain-scoped state for resumable consolidation runs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from surreal_memory import __version__
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow

logger = logging.getLogger(__name__)

PROGRESS_FORMAT_VERSION = 1
CONSOLIDATION_ENGINE_VERSION = f"{__version__}:checkpoint-v1"
LEASE_SECONDS = 180
INCOMPLETE_STATUSES = {"queued", "running", "paused", "failed"}


class ConsolidationProgressError(RuntimeError):
    """Base error for durable consolidation coordination."""


class ConsolidationPausedError(ConsolidationProgressError):
    """The strategy reached its budget at a committed work-unit boundary."""


class ConsolidationLeaseBusyError(ConsolidationProgressError):
    """Another worker currently owns this brain's consolidation lease."""


class ConsolidationResumeMismatchError(ConsolidationProgressError):
    """An unfinished run cannot safely resume with the requested parameters."""


class ConsolidationLeaseLostError(ConsolidationProgressError):
    """The worker lost its lease and must stop before starting another work unit."""


def options_fingerprint(options: Mapping[str, Any]) -> str:
    """Return a stable digest for all behavior-affecting run parameters."""
    encoded = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _parse_reference_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_naive_utc(value)
    if isinstance(value, str):
        return ensure_naive_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise ConsolidationResumeMismatchError("unfinished consolidation has no valid reference_time")


def supports_persistent_progress(storage: Any) -> bool:
    """Whether this backend implements the shared SurrealDB progress contract."""
    required = (
        "get_consolidation_progress",
        "start_consolidation_progress",
        "claim_consolidation_progress",
        "save_consolidation_progress",
        "acquire_consolidation_lease",
        "renew_consolidation_lease",
        "release_consolidation_lease",
    )
    return bool(getattr(storage, "current_brain_id", None)) and supports_storage_methods(
        storage, required
    )


def supports_storage_methods(storage: Any, method_names: Sequence[str]) -> bool:
    """Return true only when the concrete backend overrides the optional API.

    Static class inspection avoids treating dynamically-created mock attributes
    or the base class's NotImplemented stubs as backend capabilities.
    """
    storage_type = type(storage)
    for name in method_names:
        implementation = inspect.getattr_static(storage_type, name, None)
        base_implementation = inspect.getattr_static(NeuralStorage, name, None)
        if not callable(implementation) or implementation is base_implementation:
            return False
    return True


class ConsolidationProgressSession:
    """Own a distributed lease and persist committed strategy checkpoints."""

    def __init__(
        self,
        storage: Any,
        brain_id: str,
        owner_token: str,
        state: dict[str, Any],
        *,
        lease_seconds: int = LEASE_SECONDS,
    ) -> None:
        self.storage = storage
        self.brain_id = brain_id
        self.owner_token = owner_token
        self.state = state
        self.lease_seconds = lease_seconds
        self.resumed = False
        self._lease_lost = False
        self._closed = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(
        cls,
        storage: Any,
        requested_strategies: Sequence[str],
        fingerprint: str,
        reference_time: datetime,
        *,
        explicit_reference_time: datetime | None = None,
    ) -> ConsolidationProgressSession | None:
        """Acquire the brain lease, then resume or initialize its active record."""
        if not supports_persistent_progress(storage):
            return None

        brain_id = str(storage.current_brain_id)
        owner_token = str(uuid4())
        acquired = await storage.acquire_consolidation_lease(
            brain_id, owner_token, lease_seconds=LEASE_SECONDS
        )
        if not acquired:
            raise ConsolidationLeaseBusyError(
                f"brain {brain_id!r} already has an active consolidation worker"
            )

        try:
            requested = sorted(set(requested_strategies))
            existing = await storage.get_consolidation_progress(brain_id)
            if existing and str(existing.get("status")) in INCOMPLETE_STATUSES:
                if existing.get("engine_version") != CONSOLIDATION_ENGINE_VERSION:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation uses a different code version; "
                        "its checkpoint was left untouched"
                    )
                if existing.get("format_version") != PROGRESS_FORMAT_VERSION:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation uses an incompatible progress format; "
                        "its checkpoint was left untouched"
                    )
                if existing.get("schema_version") != SCHEMA_VERSION:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation uses a different database schema version; "
                        "its checkpoint was left untouched"
                    )
                if existing.get("options_fingerprint") != fingerprint:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation has different strategy/configuration "
                        "parameters; its checkpoint was left untouched"
                    )
                if sorted(existing.get("requested_strategies") or []) != requested:
                    raise ConsolidationResumeMismatchError(
                        "requested strategies differ from the unfinished run; "
                        "its checkpoint was left untouched"
                    )
                previous_reference_time = _parse_reference_time(existing.get("reference_time"))
                if (
                    explicit_reference_time is not None
                    and ensure_naive_utc(explicit_reference_time) != previous_reference_time
                ):
                    raise ConsolidationResumeMismatchError(
                        "explicit reference_time differs from the unfinished run; "
                        "its checkpoint was left untouched"
                    )
                claimed = await storage.claim_consolidation_progress(brain_id, owner_token)
                if claimed is None:
                    raise ConsolidationLeaseBusyError(
                        "unfinished consolidation changed while its lease was acquired"
                    )
                session = cls(storage, brain_id, owner_token, dict(claimed))
                session.resumed = True
            else:
                state: dict[str, Any] = {
                    "brain_id": brain_id,
                    "run_id": str(uuid4()),
                    "schema_version": SCHEMA_VERSION,
                    "format_version": PROGRESS_FORMAT_VERSION,
                    "engine_version": CONSOLIDATION_ENGINE_VERSION,
                    "requested_strategies": requested,
                    "completed_strategies": [],
                    "options_fingerprint": fingerprint,
                    "reference_time": ensure_naive_utc(reference_time),
                    "status": "running",
                    "current_strategy": None,
                    "phase": "starting",
                    "cursor": None,
                    "strategy_states": {},
                    "counters": {},
                    "owner_token": owner_token,
                    "started_at": utcnow(),
                    "updated_at": utcnow(),
                    "last_error": None,
                }
                record = await storage.start_consolidation_progress(state)
                session = cls(storage, brain_id, owner_token, dict(record))
            session._heartbeat_task = asyncio.create_task(
                session._renew_lease(), name=f"consolidation-lease:{brain_id}"
            )
            return session
        except BaseException:
            await storage.release_consolidation_lease(brain_id, owner_token)
            raise

    @property
    def lease_lost(self) -> bool:
        """Whether the background heartbeat has fenced this worker out."""
        return self._lease_lost

    @property
    def reference_time(self) -> datetime:
        return _parse_reference_time(self.state.get("reference_time"))

    @property
    def completed_strategies(self) -> set[str]:
        return set(self.state.get("completed_strategies") or [])

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        states = self.state.get("strategy_states") or {}
        current = states.get(strategy, {})
        return dict(current) if isinstance(current, Mapping) else {}

    @property
    def last_checkpoint(self) -> str:
        strategy = self.state.get("current_strategy") or "none"
        per_strategy = self.strategy_state(str(strategy))
        phase = per_strategy.get("phase") or self.state.get("phase") or "unknown"
        cursor = per_strategy.get("cursor") or self.state.get("cursor") or "start"
        return f"{strategy} phase={phase} cursor={cursor}"

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: Sequence[str] | None = None,
        counters: Mapping[str, int | float] | None = None,
    ) -> None:
        """Persist the latest committed unit boundary for one strategy."""
        if self._lease_lost:
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} was lost; stopping before more work"
            )
        self.state["status"] = "running"
        self.state["current_strategy"] = strategy
        self.state["phase"] = phase
        self.state["cursor"] = cursor
        states = dict(self.state.get("strategy_states") or {})
        strategy_state = dict(states.get(strategy) or {})
        strategy_state.update(phase=phase, cursor=cursor, updated_at=utcnow())
        if pending is not None:
            strategy_state["pending"] = list(pending)
        elif "pending" in strategy_state:
            strategy_state.pop("pending")
        if counters is not None:
            strategy_state["counters"] = dict(counters)
            all_counters = dict(self.state.get("counters") or {})
            all_counters[strategy] = dict(counters)
            self.state["counters"] = all_counters
        states[strategy] = strategy_state
        self.state["strategy_states"] = states
        await self._save()

    async def complete_strategy(self, strategy: str) -> None:
        completed = list(self.state.get("completed_strategies") or [])
        if strategy not in completed:
            completed.append(strategy)
        self.state["completed_strategies"] = completed
        self.state["current_strategy"] = None
        self.state["phase"] = "strategy_completed"
        self.state["cursor"] = None
        states = dict(self.state.get("strategy_states") or {})
        current = dict(states.get(strategy) or {})
        current.update(phase="completed", cursor=None, updated_at=utcnow())
        current.pop("pending", None)
        states[strategy] = current
        self.state["strategy_states"] = states
        await self._save()

    async def pause(self) -> None:
        self.state["status"] = "paused"
        await self._save()

    async def fail(self, error: str) -> None:
        self.state["status"] = "failed"
        self.state["last_error"] = error[:2000]
        await self._save()

    async def complete(self) -> None:
        self.state["status"] = "completed"
        self.state["current_strategy"] = None
        self.state["phase"] = "completed"
        self.state["cursor"] = None
        self.state["last_error"] = None
        await self._save()

    async def _save(self) -> None:
        if self._lease_lost:
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} was lost; checkpoint was not saved"
            )
        saved = await self.storage.save_consolidation_progress(
            self.brain_id, self.owner_token, self.state
        )
        if saved is None:
            self._lease_lost = True
            raise ConsolidationLeaseLostError(
                f"lease fence for brain {self.brain_id!r} rejected the checkpoint"
            )
        self.state = dict(saved)

    async def _renew_lease(self) -> None:
        interval = max(10, self.lease_seconds // 3)
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                if not await self.storage.renew_consolidation_lease(
                    self.brain_id,
                    self.owner_token,
                    lease_seconds=self.lease_seconds,
                ):
                    self._lease_lost = True
                    logger.error(
                        "Consolidation lease lost for brain %s; the active strategy "
                        "will stop at its next committed checkpoint",
                        self.brain_id,
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._lease_lost = True
            logger.exception("Consolidation lease renewal failed for brain %s", self.brain_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        try:
            await self.storage.release_consolidation_lease(self.brain_id, self.owner_token)
        except Exception:
            logger.exception("Failed to release consolidation lease for brain %s", self.brain_id)

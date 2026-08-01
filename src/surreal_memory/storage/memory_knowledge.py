"""In-memory knowledge storage mixin — sources, alerts, cognitive state, knowledge gaps."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any
from uuid import uuid4

from surreal_memory.core.alert import Alert, AlertStatus
from surreal_memory.core.source import Source, SourceStatus
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.utils.timeutils import utcnow

_ALERT_DEDUP_COOLDOWN = timedelta(hours=6)

_MAX_SOURCE_LIMIT = 1_000
_MAX_ALERT_LIMIT = 200
_MAX_LIST_LIMIT = 200
_TOPIC_MAX_LEN = 500

_VISIBLE_ALERT_STATUSES = (AlertStatus.ACTIVE, AlertStatus.SEEN, AlertStatus.ACKNOWLEDGED)
_PENDING_ALERT_STATUSES = (AlertStatus.ACTIVE, AlertStatus.SEEN)

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2}
_UNRANKED_SEVERITY = 3

_COGNITIVE_LIST_KEYS = (
    "neuron_id",
    "confidence",
    "evidence_for_count",
    "evidence_against_count",
    "status",
    "predicted_at",
    "resolved_at",
    "last_evidence_at",
    "created_at",
)


def _clamp_confidence(value: float) -> float:
    """Clamp a confidence score into the persisted [0.01, 0.99] range."""
    return max(0.01, min(0.99, value))


def _clamp_priority(value: float) -> float:
    """Clamp a knowledge-gap priority into the persisted [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def _project_cognitive(state: dict[str, Any]) -> dict[str, Any]:
    """Project a stored cognitive state onto the list/prediction column subset."""
    return {key: state[key] for key in _COGNITIVE_LIST_KEYS}


def _copy_gap(gap: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy of a stored knowledge gap."""
    copied = dict(gap)
    copied["related_neuron_ids"] = list(gap["related_neuron_ids"])
    return copied


class InMemoryKnowledgeMixin:
    """Mixin providing source registry, alert, cognitive state, and knowledge-gap CRUD."""

    # Declared in InMemoryStorage.__init__
    _sources: dict[str, dict[str, Source]]
    _alerts: dict[str, dict[str, Alert]]
    _cognitive_states: dict[str, dict[str, dict[str, Any]]]
    _knowledge_gaps: dict[str, dict[str, dict[str, Any]]]
    _synapses: dict[str, dict[str, Synapse]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ========== Source Registry ==========

    async def add_source(self, source: Source) -> str:
        """Insert a source record. Returns the source ID."""
        # Keyed by source.brain_id, not the active brain — mirrors the SQLite INSERT.
        if source.brain_id not in self._sources:
            self._sources[source.brain_id] = {}
        self._sources[source.brain_id][source.id] = source
        return source.id

    async def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID within the current brain."""
        brain_id = self._get_brain_id()
        return self._sources.get(brain_id, {}).get(source_id)

    async def find_source_by_name(self, name: str) -> Source | None:
        """Find a source by exact name within the current brain."""
        brain_id = self._get_brain_id()
        for source in self._sources.get(brain_id, {}).values():
            if source.name == name:
                return source
        return None

    async def list_sources(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Source]:
        """List sources for the current brain, with optional filters."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, _MAX_SOURCE_LIMIT)

        results: list[Source] = []
        for source in self._sources.get(brain_id, {}).values():
            if source_type is not None and source.source_type.value != source_type:
                continue
            if status is not None and source.status.value != status:
                continue
            results.append(source)

        results.sort(key=lambda s: s.created_at, reverse=True)
        return results[:safe_limit]

    async def update_source(
        self,
        source_id: str,
        status: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        trust: float | None = None,
    ) -> bool:
        """Update a source. Returns True if the row was modified."""
        new_status: SourceStatus | None = None
        if status is not None:
            try:
                new_status = SourceStatus(status)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {status!r}. Must be one of {[s.value for s in SourceStatus]}"
                )

        brain_id = self._get_brain_id()
        sources = self._sources.get(brain_id, {})
        source = sources.get(source_id)
        if source is None:
            return False

        changes: dict[str, Any] = {"updated_at": utcnow()}
        if new_status is not None:
            changes["status"] = new_status
        if version is not None:
            changes["version"] = version
        if metadata is not None:
            changes["metadata"] = dict(metadata)
        if trust is not None:
            changes["trust"] = trust

        sources[source_id] = replace(source, **changes)
        return True

    async def delete_source(self, source_id: str) -> bool:
        """Delete a source. Returns True if deleted."""
        brain_id = self._get_brain_id()
        sources = self._sources.get(brain_id, {})
        if source_id in sources:
            del sources[source_id]
            return True
        return False

    async def count_neurons_for_source(self, source_id: str) -> int:
        """Count neurons linked to a source via SOURCE_OF synapses."""
        brain_id = self._get_brain_id()
        targets = {
            synapse.target_id
            for synapse in self._synapses.get(brain_id, {}).values()
            if synapse.source_id == source_id and synapse.type == SynapseType.SOURCE_OF
        }
        return len(targets)

    # ========== Alerts ==========

    async def record_alert(self, alert: Alert) -> str:
        """Insert a new alert, respecting the dedup cooldown.

        Returns the alert ID if inserted, empty string if suppressed.
        """
        brain_id = self._get_brain_id()
        if brain_id not in self._alerts:
            self._alerts[brain_id] = {}
        alerts = self._alerts[brain_id]

        cutoff = utcnow() - _ALERT_DEDUP_COOLDOWN
        for existing in alerts.values():
            if existing.alert_type != alert.alert_type:
                continue
            if existing.status not in _PENDING_ALERT_STATUSES:
                continue
            if existing.created_at > cutoff:
                return ""

        # Lifecycle timestamps are not persisted on insert — mirrors the SQLite INSERT columns.
        alerts[alert.id] = replace(
            alert,
            brain_id=brain_id,
            seen_at=None,
            acknowledged_at=None,
            resolved_at=None,
            metadata=dict(alert.metadata),
        )
        return alert.id

    async def get_active_alerts(self, limit: int = 50) -> list[Alert]:
        """Get active/seen/acknowledged alerts (not resolved)."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, _MAX_ALERT_LIMIT)

        visible = [
            alert
            for alert in self._alerts.get(brain_id, {}).values()
            if alert.status in _VISIBLE_ALERT_STATUSES
        ]
        visible.sort(key=lambda a: a.created_at, reverse=True)
        visible.sort(key=lambda a: _SEVERITY_RANK.get(a.severity, _UNRANKED_SEVERITY))
        return visible[:safe_limit]

    async def count_pending_alerts(self) -> int:
        """Count active + seen alerts (not acknowledged or resolved)."""
        brain_id = self._get_brain_id()
        return sum(
            1
            for alert in self._alerts.get(brain_id, {}).values()
            if alert.status in _PENDING_ALERT_STATUSES
        )

    async def mark_alert_acknowledged(self, alert_id: str) -> bool:
        """Mark a single alert as acknowledged. Returns True if updated."""
        brain_id = self._get_brain_id()
        alerts = self._alerts.get(brain_id, {})
        alert = alerts.get(alert_id)
        if alert is None or alert.status not in _PENDING_ALERT_STATUSES:
            return False

        alerts[alert_id] = replace(alert, status=AlertStatus.ACKNOWLEDGED, acknowledged_at=utcnow())
        return True

    async def mark_alerts_seen(self, alert_ids: list[str]) -> int:
        """Mark alerts as seen. Returns count of updated rows."""
        if not alert_ids:
            return 0

        brain_id = self._get_brain_id()
        alerts = self._alerts.get(brain_id, {})
        now = utcnow()

        updated = 0
        for alert_id in alert_ids:
            alert = alerts.get(alert_id)
            if alert is None or alert.status != AlertStatus.ACTIVE:
                continue
            alerts[alert_id] = replace(alert, status=AlertStatus.SEEN, seen_at=now)
            updated += 1
        return updated

    async def resolve_alerts_by_type(self, alert_types: list[str]) -> int:
        """Resolve all active/seen alerts of given types. Returns count."""
        if not alert_types:
            return 0

        brain_id = self._get_brain_id()
        alerts = self._alerts.get(brain_id, {})
        wanted = set(alert_types)
        now = utcnow()

        updated = 0
        for alert_id, alert in list(alerts.items()):
            if alert.alert_type.value not in wanted:
                continue
            if alert.status not in _PENDING_ALERT_STATUSES:
                continue
            alerts[alert_id] = replace(alert, status=AlertStatus.RESOLVED, resolved_at=now)
            updated += 1
        return updated

    # ========== Cognitive State ==========

    async def upsert_cognitive_state(
        self,
        neuron_id: str,
        *,
        confidence: float = 0.5,
        evidence_for_count: int = 0,
        evidence_against_count: int = 0,
        status: str = "active",
        predicted_at: str | None = None,
        resolved_at: str | None = None,
        schema_version: int = 1,
        parent_schema_id: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Insert or update a cognitive state record."""
        brain_id = self._get_brain_id()
        if brain_id not in self._cognitive_states:
            self._cognitive_states[brain_id] = {}
        states = self._cognitive_states[brain_id]
        existing = states.get(neuron_id)

        states[neuron_id] = {
            "neuron_id": neuron_id,
            "confidence": _clamp_confidence(confidence),
            "evidence_for_count": evidence_for_count,
            "evidence_against_count": evidence_against_count,
            "status": status,
            "predicted_at": predicted_at,
            "resolved_at": resolved_at,
            "schema_version": schema_version,
            "parent_schema_id": parent_schema_id,
            "last_evidence_at": last_evidence_at,
            "created_at": existing["created_at"] if existing else utcnow().isoformat(),
        }

    async def get_cognitive_state(self, neuron_id: str) -> dict[str, Any] | None:
        """Get cognitive state for a neuron."""
        brain_id = self._get_brain_id()
        state = self._cognitive_states.get(brain_id, {}).get(neuron_id)
        return dict(state) if state is not None else None

    async def list_cognitive_states(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List cognitive states, optionally filtered by status."""
        brain_id = self._get_brain_id()
        capped_limit = min(limit, _MAX_LIST_LIMIT)

        matching = [
            state
            for state in self._cognitive_states.get(brain_id, {}).values()
            if not status or state["status"] == status
        ]
        matching.sort(key=lambda s: s["confidence"], reverse=True)
        return [_project_cognitive(s) for s in matching[:capped_limit]]

    async def update_cognitive_evidence(
        self,
        neuron_id: str,
        *,
        confidence: float,
        evidence_for_count: int,
        evidence_against_count: int,
        status: str,
        resolved_at: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Update only evidence-related fields of a cognitive state.

        Unlike upsert_cognitive_state, this preserves predicted_at,
        schema_version, parent_schema_id, and created_at unchanged.
        """
        brain_id = self._get_brain_id()
        state = self._cognitive_states.get(brain_id, {}).get(neuron_id)
        if state is None:
            return

        state["confidence"] = _clamp_confidence(confidence)
        state["evidence_for_count"] = evidence_for_count
        state["evidence_against_count"] = evidence_against_count
        state["status"] = status
        state["resolved_at"] = resolved_at
        state["last_evidence_at"] = last_evidence_at

    async def list_predictions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List predictions (cognitive states with predicted_at set)."""
        brain_id = self._get_brain_id()
        capped_limit = min(limit, _MAX_LIST_LIMIT)

        matching = [
            state
            for state in self._cognitive_states.get(brain_id, {}).values()
            if state["predicted_at"] is not None and (not status or state["status"] == status)
        ]
        matching.sort(key=lambda s: str(s["predicted_at"]))
        return [_project_cognitive(s) for s in matching[:capped_limit]]

    async def get_calibration_stats(self) -> dict[str, int]:
        """Get prediction calibration statistics."""
        brain_id = self._get_brain_id()
        stats = {"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}

        for state in self._cognitive_states.get(brain_id, {}).values():
            if state["predicted_at"] is None:
                continue
            state_status = state["status"]
            if state_status == "confirmed":
                stats["correct_count"] += 1
                stats["total_resolved"] += 1
            elif state_status == "refuted":
                stats["wrong_count"] += 1
                stats["total_resolved"] += 1
            elif state_status == "pending":
                stats["pending_count"] += 1

        return stats

    # ========== Knowledge Gaps ==========

    async def add_knowledge_gap(
        self,
        *,
        topic: str,
        detection_source: str,
        priority: float = 0.5,
        related_neuron_ids: list[str] | None = None,
    ) -> str:
        """Create a new knowledge gap record. Returns the generated gap ID."""
        brain_id = self._get_brain_id()
        if brain_id not in self._knowledge_gaps:
            self._knowledge_gaps[brain_id] = {}
        gap_id = str(uuid4())

        self._knowledge_gaps[brain_id][gap_id] = {
            "id": gap_id,
            "topic": topic[:_TOPIC_MAX_LEN],
            "detected_at": utcnow().isoformat(),
            "detection_source": detection_source,
            "related_neuron_ids": list(related_neuron_ids or []),
            "resolved_at": None,
            "resolved_by_neuron_id": None,
            "priority": _clamp_priority(priority),
        }
        return gap_id

    async def get_knowledge_gap(self, gap_id: str) -> dict[str, Any] | None:
        """Get a single knowledge gap by ID."""
        brain_id = self._get_brain_id()
        gap = self._knowledge_gaps.get(brain_id, {}).get(gap_id)
        return _copy_gap(gap) if gap is not None else None

    async def list_knowledge_gaps(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List knowledge gaps sorted by priority descending."""
        brain_id = self._get_brain_id()
        capped_limit = min(limit, _MAX_LIST_LIMIT)

        matching = [
            gap
            for gap in self._knowledge_gaps.get(brain_id, {}).values()
            if include_resolved or gap["resolved_at"] is None
        ]
        matching.sort(key=lambda g: g["priority"], reverse=True)
        return [_copy_gap(g) for g in matching[:capped_limit]]

    async def resolve_knowledge_gap(
        self,
        gap_id: str,
        *,
        resolved_by_neuron_id: str | None = None,
    ) -> bool:
        """Mark a knowledge gap as resolved. Returns True if it was open and got resolved."""
        brain_id = self._get_brain_id()
        gap = self._knowledge_gaps.get(brain_id, {}).get(gap_id)
        if gap is None or gap["resolved_at"] is not None:
            return False

        gap["resolved_at"] = utcnow().isoformat()
        gap["resolved_by_neuron_id"] = resolved_by_neuron_id
        return True

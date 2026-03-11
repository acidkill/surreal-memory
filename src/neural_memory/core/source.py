"""Source data structures — tracking memory origins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow


class SourceType(StrEnum):
    """Types of memory sources."""

    LAW = "law"
    CONTRACT = "contract"
    LEDGER = "ledger"
    DOCUMENT = "document"
    API = "api"
    MANUAL = "manual"
    WEBSITE = "website"
    BOOK = "book"
    RESEARCH = "research"


class SourceStatus(StrEnum):
    """Lifecycle status of a source."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"
    DRAFT = "draft"


@dataclass(frozen=True)
class Source:
    """A registered source that memories can originate from.

    Sources are first-class entities that answer "where did this come from?"
    for any neuron linked via a SOURCE_OF synapse.
    """

    id: str
    brain_id: str
    name: str
    source_type: SourceType = SourceType.DOCUMENT
    version: str = ""
    effective_date: datetime | None = None
    expires_at: datetime | None = None
    status: SourceStatus = SourceStatus.ACTIVE
    file_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        brain_id: str,
        name: str,
        source_type: SourceType | str = SourceType.DOCUMENT,
        version: str = "",
        effective_date: datetime | None = None,
        expires_at: datetime | None = None,
        status: SourceStatus | str = SourceStatus.ACTIVE,
        file_hash: str = "",
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
    ) -> Source:
        """Factory method — preferred over direct __init__."""
        return cls(
            id=source_id or str(uuid4()),
            brain_id=brain_id,
            name=name,
            source_type=SourceType(source_type),
            version=version,
            effective_date=effective_date,
            expires_at=expires_at,
            status=SourceStatus(status),
            file_hash=file_hash,
            metadata=metadata or {},
            created_at=utcnow(),
            updated_at=utcnow(),
        )

    def with_status(self, status: SourceStatus | str) -> Source:
        """Return new Source with updated status."""
        from dataclasses import replace

        return replace(self, status=SourceStatus(status), updated_at=utcnow())

    def with_version(self, version: str) -> Source:
        """Return new Source with updated version."""
        from dataclasses import replace

        return replace(self, version=version, updated_at=utcnow())

    @property
    def is_active(self) -> bool:
        """Whether this source is currently active."""
        return self.status == SourceStatus.ACTIVE

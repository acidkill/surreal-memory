"""Lease-published, bounded snapshots of semantic-source validation evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

_IDS_PER_PAGE = 4096
_REFS_PER_PAGE = 512
_MAX_PAGE_BYTES = 1_000_000
ValidationCheckpoint = Callable[[dict[str, Any], dict[str, int]], Awaitable[None]]


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


class SemanticSourceValidation:
    """Keep source proof separate from source/apply cursors and their counters.

    Immutable pages are written first. Only the caller's lease-fenced checkpoint
    publishes their root; abandoned writes cannot advance the next invocation.
    """

    def __init__(
        self,
        storage: Any,
        token: str,
        *,
        reference_time: datetime,
        run_id: str,
        owner_token: str,
        reference: Mapping[str, Any] | None = None,
        budget_check: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.storage = storage
        self.token = token
        if reference_time.tzinfo is not None:
            reference_time = reference_time.astimezone(UTC).replace(tzinfo=None)
        self.reference_time = reference_time.isoformat(timespec="microseconds")
        self.run_id = run_id
        self.owner_token = owner_token
        self.reference = dict(reference) if reference is not None else None
        self.session = storage._create_semantic_source_fence_session(
            token, created_before=reference_time
        )
        self._restored = False
        self.budget_check = budget_check
        self._saved_pages: set[str] = set()

    def _scope(self) -> dict[str, Any]:
        return {
            "brain_id": self.storage.current_brain_id,
            "run_id": self.run_id,
            "owner_token": self.owner_token,
            "source_token": self.token,
            "reference_time": self.reference_time,
        }

    async def restore(self) -> None:
        """Restore only published, hash-checked pages in this frozen run's scope."""
        if self._restored:
            return
        if self.reference is not None:
            ref = self.reference
            if (
                not isinstance(ref.get("state_id"), str)
                or type(ref.get("revision")) is not int
                or ref["revision"] < 1
                or ref.get("source_token") != self.token
            ):
                raise RuntimeError("invalid semantic source validation reference")
            root = await self.storage.load_semantic_discovery_state(
                ref["state_id"], ref["revision"]
            )
            if not isinstance(root, Mapping) or len(_encoded(root)) > _MAX_PAGE_BYTES:
                raise RuntimeError("semantic source validation snapshot unavailable")
            if ref.get("sha256") != _digest(root):
                raise RuntimeError("semantic source validation snapshot hash mismatch")
            scope = self._scope()
            if (
                root.get("kind") != "semantic_source_validation"
                or root.get("version") != 1
                or any(root.get(key) != scope[key] for key in scope if key != "owner_token")
            ):
                raise RuntimeError("semantic source validation scope changed")
            watermark = root.get("watermark")
            database_rows = root.get("scanned_database_rows")
            source_rows = root.get("scanned_source_rows")
            pages = root.get("classifier_pages")
            if (
                type(watermark) is not int
                or watermark < getattr(self.session, "token_versionstamp", 0)
                or type(database_rows) is not int
                or database_rows < 0
                or not isinstance(source_rows, Mapping)
                or set(source_rows) != {"neuron", "synapse"}
                or any(type(value) is not int or value < 0 for value in source_rows.values())
                or not isinstance(pages, list)
                or not isinstance(root.get("validation_token"), str)
                or not root["validation_token"]
                or (
                    root.get("target_token") is not None
                    and not isinstance(root["target_token"], str)
                )
            ):
                raise RuntimeError("invalid semantic source validation cursor")
            ids: dict[str, set[str]] = {"neuron": set(), "synapse": set()}
            seen_pages: set[tuple[str, int]] = set()
            pending: list[tuple[Any, tuple[Mapping[str, Any], ...]]] = [
                (page_ref, ()) for page_ref in reversed(pages)
            ]
            while pending:
                page_ref, ancestors = pending.pop()
                if not isinstance(page_ref, Mapping):
                    raise RuntimeError("invalid semantic source validation page reference")
                page_key = (page_ref.get("state_id"), page_ref.get("revision"))
                if (
                    not isinstance(page_key[0], str)
                    or type(page_key[1]) is not int
                    or page_key in seen_pages
                    or page_ref.get("source_token") != self.token
                    or len(ancestors) > 63
                ):
                    raise RuntimeError("invalid semantic source validation page reference")
                seen_pages.add((str(page_key[0]), int(page_key[1])))
                page = await self.storage._load_semantic_discovery_state_page(
                    *page_key, parent_ref=ref, ancestor_refs=ancestors
                )
                if not isinstance(page, Mapping) or page_ref.get("sha256") != _digest(page):
                    raise RuntimeError("semantic source validation page hash mismatch")
                if (
                    len(_encoded(page)) > _MAX_PAGE_BYTES
                    or page.get("kind")
                    not in {"semantic_source_validation_ids", "semantic_source_validation_index"}
                    or page.get("version") != 1
                    or any(page.get(key) != scope[key] for key in scope if key != "owner_token")
                ):
                    raise RuntimeError("invalid semantic source validation classifier page")
                self._saved_pages.add(str(page_ref["sha256"]))
                if page["kind"] == "semantic_source_validation_index":
                    children = page.get("classifier_pages")
                    if not isinstance(children, list) or not children:
                        raise RuntimeError("invalid semantic source validation classifier index")
                    pending.extend((child, (*ancestors, page_ref)) for child in reversed(children))
                    continue
                if not isinstance(page.get("records"), list):
                    raise RuntimeError("invalid semantic source validation classifier page")
                for record in page["records"]:
                    if (
                        not isinstance(record, list)
                        or len(record) != 2
                        or record[0] not in ids
                        or not isinstance(record[1], str)
                        or not record[1].startswith(record[0] + ":")
                        or record[1] in ids[record[0]]
                    ):
                        raise RuntimeError("invalid semantic source validation classifier record")
                    ids[record[0]].add(record[1])
            # No partially loaded page or malformed root can alter the session.
            self.session.watermark = watermark
            self.session.scanned_database_rows = database_rows
            self.session.scanned_source_rows = dict(source_rows)
            self.session.post_reference_ids = ids
            self.session.validation_token = root["validation_token"]
            self.session.target_token = root.get("target_token")
        self._restored = True

    async def snapshot(self) -> dict[str, Any] | None:
        """Write bounded immutable classifier pages, then their complete root."""
        await self.restore()
        if not callable(getattr(self.session, "verify", None)):
            return None  # Non-SurrealDB storage retains its existing fence contract.
        scope = self._scope()
        pages: list[dict[str, Any]] = []

        async def save(payload: dict[str, Any]) -> dict[str, Any]:
            digest = _digest(payload)
            state_id = "validation-" + digest
            if len(_encoded(payload)) > _MAX_PAGE_BYTES:
                raise RuntimeError("semantic source validation snapshot exceeds 1 MB")
            if digest not in self._saved_pages:
                await self.storage.save_semantic_discovery_state(state_id, 1, payload)
                self._saved_pages.add(digest)
            return {
                "state_id": state_id,
                "revision": 1,
                "state_revision": 1,
                "source_token": self.token,
                "sha256": digest,
            }

        records: list[list[str]] = []
        record_bytes = 0
        for table, ids in sorted(self.session.post_reference_ids.items()):
            for record_id in sorted(ids):
                record = [table, record_id]
                size = len(_encoded(record)) + 1
                if records and (len(records) >= _IDS_PER_PAGE or record_bytes + size > 500_000):
                    pages.append(
                        await save(
                            {
                                **scope,
                                "kind": "semantic_source_validation_ids",
                                "version": 1,
                                "records": records,
                            }
                        )
                    )
                    records = []
                    record_bytes = 0
                records.append(record)
                record_bytes += size
        if records:
            pages.append(
                await save(
                    {
                        **scope,
                        "kind": "semantic_source_validation_ids",
                        "version": 1,
                        "records": records,
                    }
                )
            )
        while len(pages) > _REFS_PER_PAGE:
            pages = [
                await save(
                    {
                        **scope,
                        "kind": "semantic_source_validation_index",
                        "version": 1,
                        "classifier_pages": pages[start : start + _REFS_PER_PAGE],
                    }
                )
                for start in range(0, len(pages), _REFS_PER_PAGE)
            ]
        root = {
            **scope,
            "kind": "semantic_source_validation",
            "version": 1,
            "validation_token": self.session.validation_token,
            "watermark": self.session.watermark,
            "scanned_database_rows": self.session.scanned_database_rows,
            "scanned_source_rows": self.session.scanned_source_rows,
            "classifier_pages": pages,
        }
        # SurrealDB omits NONE object fields; hash the same shape it persists.
        if self.session.target_token is not None:
            root["target_token"] = self.session.target_token
        self.reference = await save(root)
        return self.reference

    async def assert_unchanged(self, checkpoint: ValidationCheckpoint) -> None:
        """Pause normally after publishing proof, without permitting source/apply work."""
        await self.restore()
        verify = getattr(self.session, "verify", None)
        if not callable(verify):
            await self.session.assert_unchanged()
            return

        async def within_budget() -> bool:
            if self.budget_check is not None:
                try:
                    await self.budget_check()
                except ConsolidationPausedError:
                    return False
            return True

        status = await verify(**({"budget_check": within_budget} if self.budget_check else {}))
        if status == "verified":
            return
        if status != "pending":
            raise RuntimeError("invalid semantic source validation result")
        reference = await self.snapshot()
        if reference is None:
            raise RuntimeError("semantic source validation progress was not saved")
        counters = {
            "validation_database_events": self.session.scanned_database_rows,
            "validation_neuron_events": self.session.scanned_source_rows["neuron"],
            "validation_synapse_events": self.session.scanned_source_rows["synapse"],
            "validation_cursor": self.session.watermark,
        }
        await checkpoint(reference, counters)
        raise ConsolidationPausedError(
            "semantic_link source validation paused after verified changefeed prefix "
            f"at {self.session.watermark}; continue with --resume"
        )

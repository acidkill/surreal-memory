"""Durable, bounded source census for consolidation deduplication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any

from surreal_memory.core.constants import GRAPH_ONLY_PLACEHOLDER
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation_progress import ConsolidationProgressError
from surreal_memory.storage.errors import is_duplicate_key_error

_PAGE_SIZE = 500


class DedupAnchorCensus:
    """Freeze dedup anchor projections in immutable, run-scoped source pages.

    The durable rows are the snapshot boundary: pair processing never reloads
    mutable neurons, while progress stores only a keyset cursor and counters.
    """

    def __init__(
        self,
        storage: Any,
        *,
        run_id: str,
        brain_id: str,
        strategy: str,
        reference_time: datetime | None,
        checkpoint_state: Mapping[str, Any],
        checkpoint: Callable[..., Awaitable[None]],
        check_budget: Callable[[], Awaitable[None]],
    ) -> None:
        if not all((run_id, brain_id, strategy)):
            raise ValueError("dedup census identity fields must be non-empty")
        if not callable(getattr(storage, "_query", None)):
            raise TypeError("durable dedup census requires storage._query")
        self._storage = storage
        self._run_id = run_id
        self._brain_id = brain_id
        self._strategy = strategy
        self._reference_time = reference_time
        filter_payload = {"created_before": reference_time.isoformat() if reference_time else None}
        self._filter_fingerprint = hashlib.sha256(
            json.dumps(filter_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._state = dict(checkpoint_state)
        self._checkpoint = checkpoint
        self._check_budget = check_budget
        self._anchor_count = 0
        self._neuron_count = 0
        self._marker_index: int | None = None

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _projection(neuron: Neuron) -> dict[str, Any]:
        return {
            "id": str(neuron.id),
            "content_hash": int(neuron.content_hash or 0),
            "graph_only": neuron.content == GRAPH_ONLY_PLACEHOLDER,
        }

    @staticmethod
    def _decode_anchor(row: Mapping[str, Any]) -> Neuron:
        anchor_id = str(row.get("id") or "")
        content_hash = row.get("content_hash")
        if not anchor_id or not isinstance(content_hash, int):
            raise ConsolidationProgressError("dedup census anchor projection is malformed")
        graph_only = row.get("graph_only")
        if not isinstance(graph_only, bool):
            raise ConsolidationProgressError("dedup census tombstone marker is malformed")
        return Neuron(
            id=anchor_id,
            type=NeuronType.CONCEPT,
            content=GRAPH_ONLY_PLACEHOLDER if graph_only else "",
            metadata={"is_anchor": True},
            content_hash=content_hash,
        )

    def _row_id(self, page_index: int) -> str:
        identity = ":".join(
            (
                self._run_id,
                self._brain_id,
                self._strategy,
                self._filter_fingerprint,
                str(page_index),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        result = await self._storage._query(sql, **params)
        return [dict(row) for row in (result or [])]

    async def _read_page(self, page_index: int) -> dict[str, Any] | None:
        rows = await self._query(
            "SELECT * FROM consolidation_dedup_census WHERE run_id = $run_id "
            "AND brain_id = $brain_id AND strategy = $strategy "
            "AND filter_fingerprint = $filter_fingerprint AND page_index = $page_index LIMIT 1",
            run_id=self._run_id,
            brain_id=self._brain_id,
            strategy=self._strategy,
            filter_fingerprint=self._filter_fingerprint,
            page_index=page_index,
        )
        return rows[0] if rows else None

    def _validate_page(self, row: Mapping[str, Any], page_index: int) -> list[Neuron]:
        raw_anchors = row.get("anchors")
        fingerprint = (
            hashlib.sha256(self._json(raw_anchors).encode("utf-8")).hexdigest()
            if isinstance(raw_anchors, list)
            else ""
        )
        if (
            int(row.get("page_index", -1)) != page_index
            or row.get("complete") is True
            or not isinstance(raw_anchors, list)
            or len(raw_anchors) > _PAGE_SIZE
            or int(row.get("neuron_count", -1)) <= 0
            or int(row.get("neuron_count", -1)) > _PAGE_SIZE
            or int(row.get("anchor_count", -1)) != len(raw_anchors)
            or not isinstance(row.get("first_neuron_id"), str)
            or not isinstance(row.get("last_neuron_id"), str)
            or str(row.get("first_neuron_id")) > str(row.get("last_neuron_id"))
            or str(row.get("page_fingerprint")) != fingerprint
            or str(row.get("run_id")) != self._run_id
            or str(row.get("brain_id")) != self._brain_id
            or str(row.get("strategy")) != self._strategy
            or str(row.get("filter_fingerprint")) != self._filter_fingerprint
        ):
            raise ConsolidationProgressError("dedup census staged page is invalid")
        anchors = [self._decode_anchor(item) for item in raw_anchors if isinstance(item, Mapping)]
        if len(anchors) != len(raw_anchors) or anchors != sorted(anchors, key=lambda item: item.id):
            raise ConsolidationProgressError(
                "dedup census staged anchors are malformed or unordered"
            )
        return anchors

    async def _create_immutable(self, row: dict[str, Any], page_index: int) -> None:
        stage_id = self._row_id(page_index)
        try:
            await self._query(
                "CREATE type::record('consolidation_dedup_census', $stage_id) CONTENT $row",
                stage_id=stage_id,
                row=row,
            )
        except Exception as exc:
            if not is_duplicate_key_error(exc):
                raise
            existing_rows = await self._query(
                "SELECT * FROM type::record('consolidation_dedup_census', $stage_id)",
                stage_id=stage_id,
            )
            if len(existing_rows) != 1:
                raise ConsolidationProgressError(
                    "immutable dedup census page exists but could not be verified"
                ) from exc
            existing = dict(existing_rows[0])
            existing.pop("id", None)
            if self._json(existing) != self._json(row):
                raise ConsolidationProgressError(
                    "immutable dedup census page conflicts with the source; refusing stale overwrite"
                ) from exc

    async def _find_marker(self) -> dict[str, Any] | None:
        rows = await self._query(
            "SELECT * FROM consolidation_dedup_census WHERE run_id = $run_id "
            "AND brain_id = $brain_id AND strategy = $strategy "
            "AND filter_fingerprint = $filter_fingerprint AND complete = true "
            "ORDER BY page_index ASC LIMIT 1",
            run_id=self._run_id,
            brain_id=self._brain_id,
            strategy=self._strategy,
            filter_fingerprint=self._filter_fingerprint,
        )
        return rows[0] if rows else None

    async def build(self) -> int:
        """Finish or resume the census and return the frozen anchor count."""
        marker = await self._find_marker()
        raw_cursor = (
            self._state.get("cursor") if self._state.get("phase") == "dedup_census" else None
        )
        checkpoint: dict[str, Any] | None = None
        if raw_cursor is not None:
            try:
                parsed = json.loads(str(raw_cursor))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ConsolidationProgressError("dedup census checkpoint is malformed") from exc
            if (
                not isinstance(parsed, dict)
                or parsed.get("version") != 1
                or parsed.get("run_id") != self._run_id
                or parsed.get("brain_id") != self._brain_id
                or parsed.get("strategy") != self._strategy
                or parsed.get("filter_fingerprint") != self._filter_fingerprint
                or not isinstance(parsed.get("next_page_index"), int)
                or parsed["next_page_index"] < 0
                or not isinstance(parsed.get("complete"), bool)
            ):
                raise ConsolidationProgressError(
                    "dedup census checkpoint does not match its run, brain, strategy, or filter"
                )
            checkpoint = parsed

        if marker is not None:
            self._validate_marker(marker)
            self._marker_index = int(marker["page_index"])
            self._anchor_count = int(marker["anchor_count"])
            self._neuron_count = int(marker["neuron_count"])
            await self._validate_staged_prefix(self._marker_index, marker)
            return self._anchor_count

        if self._state.get("phase") in {"dedup_pairs", "dedup_window_complete"}:
            raise ConsolidationProgressError(
                "dedup pair checkpoint has no completed immutable anchor census"
            )

        page_count = int(checkpoint["next_page_index"]) if checkpoint else 0
        source_cursor = (
            str(checkpoint["last_neuron_id"])
            if checkpoint and checkpoint.get("last_neuron_id")
            else None
        )
        if checkpoint and (
            page_count < 0 or not isinstance(checkpoint.get("last_neuron_id"), (str, type(None)))
        ):
            raise ConsolidationProgressError("dedup census checkpoint source cursor is invalid")

        # Every checkpointed prefix page must be present, ordered, and content
        # verified before scanning can continue after its keyset cursor.
        anchor_count = 0
        neuron_count = 0
        prior_last_id: str | None = None
        for expected_page in range(page_count):
            row = await self._read_page(expected_page)
            if row is None:
                raise ConsolidationProgressError(
                    "dedup census staged page is missing; refusing to skip the source prefix"
                )
            anchors = self._validate_page(row, expected_page)
            first_id = str(row.get("first_neuron_id") or "")
            last_id = str(row.get("last_neuron_id") or "")
            if (
                not first_id
                or not last_id
                or (prior_last_id is not None and first_id <= prior_last_id)
            ):
                raise ConsolidationProgressError(
                    "dedup census source pages overlap or are unordered"
                )
            prior_last_id = last_id
            anchor_count += len(anchors)
            neuron_count += int(row.get("neuron_count", -1))
        if page_count and prior_last_id != source_cursor:
            raise ConsolidationProgressError("dedup census pages do not match saved source cursor")
        if not page_count and source_cursor is not None:
            raise ConsolidationProgressError("dedup census cursor has no staged source pages")

        get_page = getattr(self._storage, "find_neurons_after_id", None)
        if not callable(get_page):
            raise TypeError("dedup census requires find_neurons_after_id")
        if checkpoint is None:
            checkpoint = {
                "version": 1,
                "run_id": self._run_id,
                "brain_id": self._brain_id,
                "strategy": self._strategy,
                "filter_fingerprint": self._filter_fingerprint,
                "last_neuron_id": None,
                "next_page_index": 0,
                "complete": False,
            }

        complete = bool(checkpoint["complete"])
        while not complete:
            await self._check_budget()
            page = await get_page(
                source_cursor,
                limit=_PAGE_SIZE,
                created_before=self._reference_time,
                ephemeral=False,
                include_embedding=False,
            )
            if not page:
                complete = True
                checkpoint.update(
                    complete=True, last_neuron_id=source_cursor, next_page_index=page_count
                )
                await self._checkpoint(
                    "dedup_census",
                    cursor=json.dumps(checkpoint, sort_keys=True, separators=(",", ":")),
                    pending=[],
                    counters={
                        "pages": page_count,
                        "neurons": neuron_count,
                        "anchors": anchor_count,
                    },
                )
                break
            if page != sorted(page, key=lambda neuron: neuron.id) or (
                source_cursor is not None and page[0].id <= source_cursor
            ):
                raise ConsolidationProgressError(
                    "dedup neuron keyset page is unordered or did not advance"
                )

            anchor_rows = [
                self._projection(neuron)
                for neuron in page
                if neuron.metadata.get("is_anchor", False)
            ]
            serialized = json.loads(self._json(anchor_rows))
            staged = {
                "run_id": self._run_id,
                "brain_id": self._brain_id,
                "strategy": self._strategy,
                "filter_fingerprint": self._filter_fingerprint,
                "page_index": page_count,
                "first_neuron_id": str(page[0].id),
                "last_neuron_id": str(page[-1].id),
                "neuron_count": len(page),
                "anchor_count": len(anchor_rows),
                "page_fingerprint": hashlib.sha256(
                    self._json(serialized).encode("utf-8")
                ).hexdigest(),
                "anchors": serialized,
            }
            await self._create_immutable(staged, page_count)
            source_cursor = str(page[-1].id)
            page_count += 1
            neuron_count += len(page)
            anchor_count += len(anchor_rows)
            complete = len(page) < _PAGE_SIZE
            checkpoint.update(
                last_neuron_id=source_cursor,
                next_page_index=page_count,
                complete=complete,
            )
            await self._checkpoint(
                "dedup_census",
                cursor=json.dumps(checkpoint, sort_keys=True, separators=(",", ":")),
                pending=[],
                counters={"pages": page_count, "neurons": neuron_count, "anchors": anchor_count},
            )

        marker_index = page_count
        marker_row = {
            "run_id": self._run_id,
            "brain_id": self._brain_id,
            "strategy": self._strategy,
            "filter_fingerprint": self._filter_fingerprint,
            "page_index": marker_index,
            "first_neuron_id": None,
            "last_neuron_id": source_cursor,
            "neuron_count": neuron_count,
            "anchor_count": anchor_count,
            "page_fingerprint": hashlib.sha256(b"[]").hexdigest(),
            "anchors": [],
            "complete": True,
        }
        await self._create_immutable(marker_row, marker_index)
        checkpoint.update(complete=True, last_neuron_id=source_cursor, next_page_index=page_count)
        await self._checkpoint(
            "dedup_census",
            cursor=json.dumps(checkpoint, sort_keys=True, separators=(",", ":")),
            pending=[],
            counters={"pages": page_count, "neurons": neuron_count, "anchors": anchor_count},
        )
        self._marker_index = marker_index
        self._anchor_count = anchor_count
        self._neuron_count = neuron_count
        return anchor_count

    def _validate_marker(self, marker: Mapping[str, Any]) -> None:
        if (
            marker.get("complete") is not True
            or marker.get("anchors") != []
            or int(marker.get("page_index", -1)) < 0
            or int(marker.get("anchor_count", -1)) < 0
            or int(marker.get("neuron_count", -1)) < 0
            or str(marker.get("run_id")) != self._run_id
            or str(marker.get("brain_id")) != self._brain_id
            or str(marker.get("strategy")) != self._strategy
            or str(marker.get("filter_fingerprint")) != self._filter_fingerprint
            or str(marker.get("page_fingerprint")) != hashlib.sha256(b"[]").hexdigest()
        ):
            raise ConsolidationProgressError("dedup census completion marker is invalid")

    async def _validate_staged_prefix(self, marker_index: int, marker: Mapping[str, Any]) -> None:
        anchor_count = 0
        neuron_count = 0
        prior_last_id: str | None = None
        for page_index in range(marker_index):
            row = await self._read_page(page_index)
            if row is None:
                raise ConsolidationProgressError(
                    "dedup census completion marker has a missing page"
                )
            anchors = self._validate_page(row, page_index)
            first_id = str(row.get("first_neuron_id") or "")
            last_id = str(row.get("last_neuron_id") or "")
            if (
                not first_id
                or not last_id
                or (prior_last_id is not None and first_id <= prior_last_id)
            ):
                raise ConsolidationProgressError("dedup census pages overlap or are unordered")
            prior_last_id = last_id
            anchor_count += len(anchors)
            neuron_count += int(row.get("neuron_count", -1))
        if (
            anchor_count != int(marker.get("anchor_count", -1))
            or neuron_count != int(marker.get("neuron_count", -1))
            or prior_last_id != marker.get("last_neuron_id")
        ):
            if marker_index == 0 and marker.get("last_neuron_id") is None and neuron_count == 0:
                return
            raise ConsolidationProgressError(
                "dedup census staged pages do not match completion marker"
            )

    async def iter_anchors(self) -> AsyncIterator[Neuron]:
        """Yield the verified frozen anchors with page-sized resident memory."""
        if self._marker_index is None:
            raise RuntimeError("dedup census must be built before iterating anchors")
        for page_index in range(self._marker_index):
            await self._check_budget()
            row = await self._read_page(page_index)
            if row is None:
                raise ConsolidationProgressError("dedup census page disappeared during replay")
            for anchor in self._validate_page(row, page_index):
                yield anchor

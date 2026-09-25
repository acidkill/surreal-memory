"""Fail-closed mutation fence backed by SurrealDB changefeeds.

The fence is deliberately database-wide: any neuron or synapse mutation
invalidates tokens for every brain in the database. This is conservative when
a database contains multiple brains, but avoids relying on changefeed payload
shapes for deletes and cannot miss a relevant write.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

_CHANGEFEED_RETENTION = timedelta(days=7)
_TOKEN_MAX_AGE = timedelta(days=6, hours=15, minutes=36)
_TOKEN_VERSION = 1
_BARRIER_CHANGEFEED_PAGE_SIZE = 128
_BARRIER_CHANGEFEED_MAX_ROWS = 100_000
_SOURCE_CHANGEFEED_PAGE_SIZE = 128
_SOURCE_CHANGEFEED_MAX_ROWS = 100_000
_SOURCE_TABLES = ("neuron", "synapse")


class SemanticSourceFenceError(RuntimeError):
    """Base error for semantic source revision fences."""


class SemanticSourceChangedError(SemanticSourceFenceError):
    """A neuron or synapse mutation occurred after the captured token."""


class SemanticSourceFenceExpiredError(SemanticSourceFenceError):
    """The token is too old or its changefeed history is no longer available."""


class SemanticSourceFenceUnavailableError(SemanticSourceFenceError):
    """The source changefeed could not be checked; callers must fail closed."""


class SurrealDBSemanticSourceRevisionMixin:
    """Capture and verify high-water tokens for neuron/synapse changefeeds."""

    @staticmethod
    def _is_expected_table_reassertion(table: str, record: Mapping[str, Any]) -> bool:
        """Recognize only the unchanged table DDL emitted by schema initialization.

        SurrealDB 3.2 records an idempotent ``ALTER TABLE ... CHANGEFEED 7d``
        as a ``define_table`` mutation. Reject any changed table semantics rather
        than treating all metadata events as harmless source writes.
        """
        metadata_id = record.get("id")
        if not isinstance(metadata_id, int) or isinstance(metadata_id, bool):
            return False
        if set(record) != {"id", "name", "changefeed", "drop", "kind", "permissions", "schemafull"}:
            return False
        feed = record["changefeed"]
        if not isinstance(feed, Mapping) or set(feed) != {"expiry", "original"}:
            return False
        expiry = feed["expiry"]
        if isinstance(expiry, str):
            valid_expiry = expiry == "1w"
        else:
            # The Python SDK decodes the same SurrealDB value as a Duration.
            # Keep its import lazy so this module also works without the extra.
            from surrealdb.data.types.duration import Duration

            valid_expiry = type(expiry) is Duration and expiry.nanoseconds == 604_800_000_000_000

        kind = record["kind"]
        if table == "neuron":
            valid_kind = kind == {"kind": "ANY"}
        elif table == "synapse" and isinstance(kind, Mapping):
            endpoints = (kind.get("in"), kind.get("out"))
            valid_kind = (
                set(kind) == {"kind", "in", "out", "enforced"}
                and all(
                    isinstance(values, list)
                    and len(values) == 1
                    and SurrealDBSemanticSourceRevisionMixin._is_neuron_table(values[0])
                    for values in endpoints
                )
                and kind.get("kind") == "RELATION"
                and kind.get("enforced") is False
            )
        else:
            valid_kind = False
        return (
            record["name"] == table
            and valid_expiry
            and feed["original"] is False
            and record["drop"] is False
            and valid_kind
            and record["permissions"]
            == {"create": False, "delete": False, "select": False, "update": False}
            and record["schemafull"] is (table == "synapse")
        )

    @staticmethod
    def _is_neuron_table(value: Any) -> bool:
        if isinstance(value, str):
            return value == "neuron"
        from surrealdb.data.types.table import Table

        return type(value) is Table and value.table_name == "neuron"

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _query_response(self, sql: str, **params: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed

    async def _server_now(self) -> datetime:
        try:
            value = await self._query_response("RETURN time::now()")
            if isinstance(value, list):
                if not value:
                    raise ValueError("empty server response")
                value = value[0]
            return self._parse_datetime(value)
        except Exception as exc:
            raise SemanticSourceFenceUnavailableError(
                "SurrealDB returned no valid server timestamp"
            ) from exc

    @staticmethod
    def _contains_marker(change: Any, marker_id: str) -> bool:
        if isinstance(change, dict):
            record_id = change.get("id")
            if record_id is not None and str(record_id).endswith(f":{marker_id}"):
                return True
            return any(
                SurrealDBSemanticSourceRevisionMixin._contains_marker(value, marker_id)
                for value in change.values()
            )
        if isinstance(change, (list, tuple)):
            return any(
                SurrealDBSemanticSourceRevisionMixin._contains_marker(value, marker_id)
                for value in change
            )
        return False

    async def _find_marker_versionstamp(
        self, marker_id: str, captured_at: datetime, *, recent_first: bool = True
    ) -> int:
        """Find a recent barrier event using bounded inclusive raw-stamp pagination.

        Start at the server timestamp observed before creating the marker. A
        raw ``SINCE 0`` scan can stop at a gap in retained changefeed history
        even when recent marker events are available. Some backends instead
        omit recent events for a datetime cursor; retry those with ``SINCE 0``.
        Subsequent pages use the raw versionstamp because SurrealDB's SINCE
        cursor is inclusive. Each page skips the already-seen cursor event
        and must advance to a larger raw
        versionstamp. A hard total-row ceiling converts unexpectedly large or
        malformed feeds into a fail-closed error rather than a missed marker.
        """
        if _BARRIER_CHANGEFEED_PAGE_SIZE < 1 or _BARRIER_CHANGEFEED_MAX_ROWS < 1:
            raise SemanticSourceFenceUnavailableError("invalid barrier feed pagination limits")

        cursor = 0
        first_since = (
            f'd"{(captured_at - timedelta(seconds=1)).isoformat(timespec="microseconds")}Z"'
            if recent_first
            else "0"
        )
        scanned = 0
        while scanned < _BARRIER_CHANGEFEED_MAX_ROWS:
            limit = min(
                _BARRIER_CHANGEFEED_PAGE_SIZE,
                _BARRIER_CHANGEFEED_MAX_ROWS - scanned,
            )
            try:
                since = first_since if scanned == 0 else str(cursor)
                events = await self._query(
                    f"SHOW CHANGES FOR TABLE semantic_source_barrier SINCE {since} LIMIT {limit}"
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(word in message for word in ("expired", "retention", "changefeed")):
                    raise SemanticSourceFenceExpiredError(
                        "semantic source barrier changefeed history is unavailable"
                    ) from exc
                raise SemanticSourceFenceUnavailableError(
                    "could not inspect semantic source barrier changefeed"
                ) from exc

            if not events:
                if recent_first:
                    return await self._find_marker_versionstamp(
                        marker_id, captured_at, recent_first=False
                    )
                raise SemanticSourceFenceUnavailableError(
                    "semantic source barrier was not present in its changefeed"
                )

            advanced = False
            for event in events:
                raw_stamp = event.get("versionstamp")
                if not isinstance(raw_stamp, int) or isinstance(raw_stamp, bool) or raw_stamp < 0:
                    raise SemanticSourceFenceUnavailableError(
                        "semantic source barrier feed returned an invalid versionstamp"
                    )
                if raw_stamp < cursor:
                    raise SemanticSourceFenceUnavailableError(
                        "semantic source barrier feed moved backwards"
                    )
                if raw_stamp == cursor:
                    continue
                cursor = raw_stamp
                scanned += 1
                advanced = True
                if self._contains_marker(event.get("changes"), marker_id):
                    return raw_stamp
                if scanned >= _BARRIER_CHANGEFEED_MAX_ROWS:
                    break

            if scanned >= _BARRIER_CHANGEFEED_MAX_ROWS:
                break
            if not advanced:
                raise SemanticSourceFenceUnavailableError(
                    "semantic source barrier pagination did not advance"
                )
            if len(events) < limit:
                break

        if recent_first:
            return await self._find_marker_versionstamp(marker_id, captured_at, recent_first=False)
        raise SemanticSourceFenceUnavailableError(
            "semantic source barrier was not found within the bounded changefeed scan "
            f"({scanned} rows)"
        )

    async def capture_semantic_source_token(self) -> str:
        """Capture a DB-global versionstamp barrier without a contended counter.

        Each capture creates a uniquely keyed short-lived marker row. Its full
        raw changefeed versionstamp is a DB-global commit barrier and is passed
        unchanged to SINCE, whose boundary is inclusive.
        """
        brain_id = self._get_brain_id()
        if not brain_id:
            raise ValueError("current brain id must not be empty")
        captured_at = await self._server_now()
        created_at_readonly = await self._source_created_at_is_readonly()
        marker_id = uuid4().hex
        created = False
        try:
            await self._query(
                "DELETE semantic_source_barrier WHERE created_at < $stale_before",
                stale_before=captured_at - timedelta(minutes=5),
            )
            await self._query(
                "CREATE type::record('semantic_source_barrier', $token_id) "
                "CONTENT $content RETURN AFTER",
                token_id=marker_id,
                content={"brain_id": brain_id, "created_at": captured_at},
            )
            created = True
            marker_versionstamp = await self._find_marker_versionstamp(marker_id, captured_at)
            return json.dumps(
                {
                    "version": _TOKEN_VERSION,
                    "brain_id": brain_id,
                    "versionstamp": marker_versionstamp,
                    "captured_at": captured_at.isoformat(timespec="microseconds") + "Z",
                    # Legacy tokens omit this capability. Only tokens captured
                    # after verifying both source fields carry READONLY may
                    # classify a full-row update by created_at alone.
                    "created_at_readonly": created_at_readonly,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        except SemanticSourceFenceError:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if any(word in message for word in ("expired", "retention", "changefeed")):
                raise SemanticSourceFenceExpiredError(
                    "semantic source barrier changefeed history is unavailable"
                ) from exc
            raise SemanticSourceFenceUnavailableError(
                "could not capture semantic source barrier"
            ) from exc
        finally:
            if created:
                try:
                    await self._query(
                        "DELETE type::record('semantic_source_barrier', $token_id) "
                        "WHERE brain_id = $brain_id",
                        token_id=marker_id,
                        brain_id=brain_id,
                    )
                except Exception as exc:
                    raise SemanticSourceFenceUnavailableError(
                        "could not clean up semantic source barrier"
                    ) from exc

    async def _source_created_at_is_readonly(self) -> bool:
        """Return whether both source tables expose READONLY created_at fields."""
        for table in _SOURCE_TABLES:
            try:
                info = await self._query_response(f"INFO FOR TABLE {table}")
            except Exception as exc:
                raise SemanticSourceFenceUnavailableError(
                    f"could not verify {table} created_at immutability"
                ) from exc
            fields = info.get("fields") if isinstance(info, Mapping) else None
            definition = fields.get("created_at") if isinstance(fields, Mapping) else None
            if not isinstance(definition, str) or "READONLY" not in definition.upper().split():
                return False
        return True

    @staticmethod
    def _decode_token(token: str, brain_id: str) -> tuple[int, datetime, bool]:
        try:
            data = json.loads(token)
            versionstamp = data.get("versionstamp") if isinstance(data, dict) else None
            created_at_readonly = (
                data.get("created_at_readonly", False) if isinstance(data, dict) else None
            )
            if (
                not isinstance(data, dict)
                or data.get("version") != _TOKEN_VERSION
                or data.get("brain_id") != brain_id
                or not isinstance(versionstamp, int)
                or isinstance(versionstamp, bool)
                or versionstamp < 0
                or not isinstance(data.get("captured_at"), str)
                or not isinstance(created_at_readonly, bool)
            ):
                raise ValueError
            captured_at = SurrealDBSemanticSourceRevisionMixin._parse_datetime(data["captured_at"])
            return versionstamp, captured_at, created_at_readonly
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SemanticSourceFenceExpiredError(
                "semantic source token is malformed or belongs to another brain"
            ) from exc

    async def assert_semantic_source_unchanged(
        self, since_token: str, *, created_before: datetime | None = None
    ) -> None:
        """Raise for source changes except provably post-reference-time records.

        Without a frozen cutoff, retain the legacy conservative check. With a
        cutoff, full-record events whose ``created_at`` is after the cutoff are
        irrelevant only when the token attests that both source fields were
        READONLY at capture. Legacy tokens require a visible post-reference
        create before later updates/deletes can be ignored.
        """
        brain_id = self._get_brain_id()
        versionstamp, captured_at, created_at_readonly = self._decode_token(since_token, brain_id)
        server_now = await self._server_now()
        age = server_now - captured_at
        if age < timedelta(0):
            raise SemanticSourceFenceExpiredError("semantic source token is from the future")
        # Leave a 5% safety margin below seven-day source-feed retention; older
        # cursors require a complete source rebuild.
        if age >= _TOKEN_MAX_AGE or age >= _CHANGEFEED_RETENTION:
            raise SemanticSourceFenceExpiredError(
                "semantic source token is older than the safe changefeed window"
            )

        if created_before is not None:
            await self._assert_source_unchanged_before(
                brain_id,
                versionstamp,
                self._parse_datetime(created_before),
                created_at_readonly=created_at_readonly,
            )
            return

        for table in _SOURCE_TABLES:
            try:
                rows = await self._query(
                    f"SHOW CHANGES FOR TABLE {table} SINCE {versionstamp} LIMIT 10"
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(word in message for word in ("expired", "retention", "changefeed")):
                    raise SemanticSourceFenceExpiredError(
                        f"{table} changefeed history is unavailable"
                    ) from exc
                raise SemanticSourceFenceUnavailableError(
                    f"could not inspect {table} changefeed"
                ) from exc
            if rows:
                raise SemanticSourceChangedError(
                    f"{table} changed since the semantic source token was captured"
                )

    async def _assert_source_unchanged_before(
        self,
        brain_id: str,
        versionstamp: int,
        created_before: datetime,
        *,
        created_at_readonly: bool,
    ) -> None:
        """Page source feeds, ignoring only records provably outside the run.

        A post-cutoff ``created_at`` on an update is not proof of a post-cutoff
        creation unless the token was captured after verifying both fields are
        READONLY. Legacy tokens therefore require a visible create event before
        later updates/deletes for that id can be ignored.
        """
        for table in _SOURCE_TABLES:
            cursor = versionstamp
            scanned = 0
            post_reference_ids: set[str] = set()
            while scanned < _SOURCE_CHANGEFEED_MAX_ROWS:
                limit = min(
                    _SOURCE_CHANGEFEED_PAGE_SIZE,
                    _SOURCE_CHANGEFEED_MAX_ROWS - scanned,
                )
                try:
                    events = await self._query(
                        f"SHOW CHANGES FOR TABLE {table} SINCE {cursor} LIMIT {limit}"
                    )
                except Exception as exc:
                    message = str(exc).lower()
                    if any(word in message for word in ("expired", "retention", "changefeed")):
                        raise SemanticSourceFenceExpiredError(
                            f"{table} changefeed history is unavailable"
                        ) from exc
                    raise SemanticSourceFenceUnavailableError(
                        f"could not inspect {table} changefeed"
                    ) from exc

                if not events:
                    break

                page_start = cursor
                skipped_inclusive_boundary = False
                advanced = False
                for event in events:
                    raw_stamp = event.get("versionstamp")
                    if (
                        not isinstance(raw_stamp, int)
                        or isinstance(raw_stamp, bool)
                        or raw_stamp < 0
                        or raw_stamp < cursor
                    ):
                        raise SemanticSourceFenceUnavailableError(
                            f"{table} changefeed returned an invalid versionstamp"
                        )
                    if raw_stamp == page_start and not skipped_inclusive_boundary:
                        skipped_inclusive_boundary = True
                        continue

                    changes = event.get("changes")
                    if not isinstance(changes, (list, tuple)) or not changes:
                        raise SemanticSourceFenceUnavailableError(
                            f"{table} changefeed returned an invalid event"
                        )
                    for change in changes:
                        if not isinstance(change, Mapping) or len(change) != 1:
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an unclassifiable change"
                            )
                        action, record = next(iter(change.items()))
                        if not isinstance(record, Mapping):
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an invalid record"
                            )
                        if action == "define_table":
                            if self._is_expected_table_reassertion(table, record):
                                continue
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an unexpected table definition"
                            )
                        record_id = record.get("id")
                        record_id_text = str(record_id) if record_id is not None else ""
                        if not record_id_text.startswith(f"{table}:"):
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an invalid record id"
                            )
                        if action == "delete":
                            if record_id_text in post_reference_ids:
                                post_reference_ids.remove(record_id_text)
                                continue
                            raise SemanticSourceChangedError(
                                f"{table} changed since the semantic source token was captured"
                            )
                        if action not in {"create", "update"}:
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an unknown change type"
                            )
                        if action == "update" and record_id_text in post_reference_ids:
                            if record.get("brain_id") != brain_id:
                                raise SemanticSourceChangedError(
                                    f"{table} changed since the semantic source token was captured"
                                )
                            continue
                        if action == "update" and not created_at_readonly:
                            raise SemanticSourceChangedError(
                                f"{table} changed since the semantic source token was captured"
                            )
                        if record.get("brain_id") != brain_id:
                            raise SemanticSourceChangedError(
                                f"{table} changed since the semantic source token was captured"
                            )
                        created_at = record.get("created_at")
                        if created_at is None:
                            raise SemanticSourceChangedError(
                                f"{table} changed since the semantic source token was captured"
                            )
                        try:
                            created_at = self._parse_datetime(created_at)
                        except (TypeError, ValueError) as exc:
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an invalid created_at"
                            ) from exc
                        if created_at <= created_before:
                            raise SemanticSourceChangedError(
                                f"{table} changed since the semantic source token was captured"
                            )
                        post_reference_ids.add(record_id_text)

                    cursor = raw_stamp
                    scanned += 1
                    advanced = True
                    if scanned >= _SOURCE_CHANGEFEED_MAX_ROWS:
                        break

                if scanned >= _SOURCE_CHANGEFEED_MAX_ROWS:
                    raise SemanticSourceFenceUnavailableError(
                        f"{table} changefeed exceeded the bounded scan limit"
                    )
                if len(events) < limit:
                    break
                if not advanced:
                    raise SemanticSourceFenceUnavailableError(
                        f"{table} changefeed pagination did not advance"
                    )

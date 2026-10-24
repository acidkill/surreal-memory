"""Fail-closed mutation fence backed by SurrealDB changefeeds.

The fence is deliberately database-wide. With a frozen cutoff, authoritative
reverse diffs can prove that recall changed only statistics, not the semantic
source projection. Unknown changes and deletes remain conservatively fenced.
"""

from __future__ import annotations

import asyncio
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
# SurrealDB applies LIMIT to the database-wide feed before it keeps one table's
# entries, so table feeds are read by paging the database feed (see
# _show_table_changes). These bound that inner scan.
_DATABASE_FEED_PAGE_SIZE = 1_000
_DATABASE_FEED_MAX_ROWS = 2_000_000


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
            and isinstance(feed["original"], bool)
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

    @staticmethod
    def _change_table(change: Any) -> str | None:
        """Table a single feed change belongs to, or None when it cannot be told."""
        if not isinstance(change, Mapping):
            return None
        if set(change) == {"current", "update"}:
            action, record = "update", change["current"]
        elif len(change) == 1:
            action, record = next(iter(change.items()))
        else:
            return None
        if not isinstance(record, Mapping):
            return None
        if action == "define_table":
            name = record.get("name")
            return str(name) if name is not None else None
        record_id = record.get("id")
        return str(record_id).split(":", 1)[0] if record_id is not None else None

    @staticmethod
    def _projection_neutral_update(
        table: str, current: Mapping[str, Any], patches: Any
    ) -> bool | None:
        """Classify authoritative reverse diffs; None means malformed evidence."""
        if not isinstance(patches, list):
            return None
        neutral_fields = (
            {"updated_at", "access_frequency"}
            if table == "neuron"
            else {"weight", "last_activated", "reinforced_count"}
        )
        neutral = True
        for patch in patches:
            if not isinstance(patch, Mapping):
                return None
            operation = patch.get("op")
            expected = {"op", "path"} if operation == "remove" else {"op", "path", "value"}
            if operation not in ("add", "remove", "replace", "change") or set(patch) != expected:
                return None
            if operation == "change" and not isinstance(patch["value"], str):
                return None
            path = patch.get("path")
            if not isinstance(path, str) or not path.startswith("/"):
                return None
            encoded = path[1:].split("/")
            if any("~" in part.replace("~0", "").replace("~1", "") for part in encoded):
                return None
            parts = [part.replace("~1", "/").replace("~0", "~") for part in encoded]
            if parts[0] in neutral_fields and len(parts) == 1:
                continue
            if parts[0] != "metadata":
                neutral = False
                continue
            if table == "synapse":
                continue
            metadata = current.get("metadata", {})
            if not isinstance(metadata, Mapping):
                return None
            if len(parts) > 1:
                neutral = neutral and parts[1] != "_embedding"
                continue
            previous = {} if operation == "remove" else patch["value"]
            if not isinstance(previous, Mapping):
                return None
            try:
                before = json.dumps(previous.get("_embedding"), sort_keys=True, allow_nan=False)
                after = json.dumps(metadata.get("_embedding"), sort_keys=True, allow_nan=False)
            except (TypeError, ValueError):
                return None
            neutral = neutral and before == after
        return neutral

    async def _show_table_changes(
        self, table: str, since: int | str, limit: int
    ) -> list[dict[str, Any]]:
        """``SHOW CHANGES FOR TABLE`` with the LIMIT meaning the callers rely on.

        SurrealDB (3.2) applies LIMIT to the database-wide feed first and only
        then keeps the table's entries, so on a busy database a short or empty
        page does not mean the table's history is exhausted - the next table
        entry can sit behind LIMIT entries of other tables. Page the database
        feed instead and keep the table's changes here: a result shorter than
        ``limit`` now really means the end of the feed. Changes that cannot be
        attributed to a table are kept, so the callers' validation still sees
        (and rejects) them.
        """
        kept: list[dict[str, Any]] = []
        cursor: int | str = since
        last_stamp: int | None = None
        scanned = 0
        while len(kept) < limit:
            events = await self._query(
                f"SHOW CHANGES FOR DATABASE SINCE {cursor} LIMIT {_DATABASE_FEED_PAGE_SIZE}"
            )
            if not events:
                break
            for event in events:
                stamp = event.get("versionstamp")
                if last_stamp is not None and isinstance(stamp, int) and stamp <= last_stamp:
                    continue  # SINCE is inclusive: the previous page's last entry repeats
                changes = event.get("changes")
                if not isinstance(changes, (list, tuple)):
                    kept.append(event)
                    continue
                own = [c for c in changes if self._change_table(c) in (table, None)]
                if own:
                    kept.append({**event, "changes": own})
            scanned += len(events)
            if len(events) < _DATABASE_FEED_PAGE_SIZE:
                break
            newest = events[-1].get("versionstamp")
            if (
                not isinstance(newest, int)
                or isinstance(newest, bool)
                or (last_stamp is not None and newest <= last_stamp)
            ):
                raise SemanticSourceFenceUnavailableError(
                    "database changefeed pagination did not advance"
                )
            if scanned >= _DATABASE_FEED_MAX_ROWS:
                raise SemanticSourceFenceUnavailableError(
                    "database changefeed exceeded the bounded scan limit"
                )
            last_stamp = newest
            cursor = newest
        return kept[:limit]

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
                events = await self._show_table_changes("semantic_source_barrier", since, limit)
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
        """Raise except for proven neutral updates and post-reference records.

        Without a frozen cutoff, retain the legacy conservative check. With a
        cutoff, full-record events whose ``created_at`` is after the cutoff are
        irrelevant only when the token attests that both source fields were
        READONLY at capture. Legacy tokens require a visible post-reference
        create before later updates/deletes can be ignored.
        Reverse diffs may also prove that old records changed only fields
        outside the semantic fingerprint; plain old-record updates stay fenced.
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
                rows = await self._show_table_changes(table, versionstamp, 10)
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

    def _create_semantic_source_fence_session(
        self, since_token: str, *, created_before: datetime
    ) -> _SemanticSourceFenceSession:
        """Create a process-local incremental fence for one frozen discovery run."""
        brain_id = self._get_brain_id()
        versionstamp, _, _ = self._decode_token(since_token, brain_id)
        return _SemanticSourceFenceSession(
            self,
            since_token,
            brain_id,
            versionstamp,
            self._parse_datetime(created_before),
        )

    async def _assert_source_session_unchanged(self, session: _SemanticSourceFenceSession) -> None:
        brain_id = self._get_brain_id()
        versionstamp, captured_at, created_at_readonly = self._decode_token(session.token, brain_id)
        if (
            brain_id != session.brain_id
            or versionstamp != session.token_versionstamp
            or self._parse_datetime(session.created_before) != session.created_before
        ):
            raise SemanticSourceFenceExpiredError(
                "semantic source fence session no longer matches its captured scope"
            )
        server_now = await self._server_now()
        age = server_now - captured_at
        if age < timedelta(0):
            raise SemanticSourceFenceExpiredError("semantic source token is from the future")
        if age >= _TOKEN_MAX_AGE or age >= _CHANGEFEED_RETENTION:
            raise SemanticSourceFenceExpiredError(
                "semantic source token is older than the safe changefeed window"
            )

        # All mutable state is staged locally. Cancellation or any malformed,
        # unknown, expired, or changed-source event leaves the published cursor
        # and post-reference ID sets untouched.
        cursor = session.watermark
        scanned_database_rows = session.scanned_database_rows
        scanned_source_rows = dict(session.scanned_source_rows)
        post_reference_ids = {table: set(ids) for table, ids in session.post_reference_ids.items()}
        last_stamp = cursor
        while True:
            try:
                events = await self._query(
                    f"SHOW CHANGES FOR DATABASE SINCE {cursor} LIMIT {_DATABASE_FEED_PAGE_SIZE}"
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(word in message for word in ("expired", "retention", "changefeed")):
                    raise SemanticSourceFenceExpiredError(
                        "semantic source database changefeed history is unavailable"
                    ) from exc
                raise SemanticSourceFenceUnavailableError(
                    "could not inspect semantic source database changefeed"
                ) from exc
            if not isinstance(events, list):
                raise SemanticSourceFenceUnavailableError(
                    "database changefeed returned an invalid page"
                )
            if not events:
                break

            page_start = cursor
            advanced = False
            for event in events:
                if not isinstance(event, Mapping):
                    raise SemanticSourceFenceUnavailableError(
                        "database changefeed returned an invalid event"
                    )
                stamp = event.get("versionstamp")
                if (
                    not isinstance(stamp, int)
                    or isinstance(stamp, bool)
                    or stamp < 0
                    or stamp < page_start
                    or (last_stamp is not None and stamp < last_stamp)
                ):
                    raise SemanticSourceFenceUnavailableError(
                        "database changefeed returned an invalid versionstamp"
                    )
                if stamp == page_start:
                    continue
                if stamp == last_stamp:
                    raise SemanticSourceFenceUnavailableError(
                        "database changefeed returned duplicate versionstamps"
                    )
                changes = event.get("changes")
                if not isinstance(changes, (list, tuple)) or not changes:
                    raise SemanticSourceFenceUnavailableError(
                        "database changefeed returned an invalid event"
                    )
                source_tables_in_event: set[str] = set()
                for change in changes:
                    table = self._change_table(change)
                    if table is None:
                        raise SemanticSourceFenceUnavailableError(
                            "database changefeed returned an unclassifiable change"
                        )
                    if table not in _SOURCE_TABLES:
                        continue
                    source_tables_in_event.add(table)
                    self._classify_source_change(
                        table,
                        change,
                        brain_id=brain_id,
                        created_before=session.created_before,
                        created_at_readonly=created_at_readonly,
                        post_reference_ids=post_reference_ids[table],
                    )
                for table in source_tables_in_event:
                    scanned_source_rows[table] += 1
                    if scanned_source_rows[table] >= _SOURCE_CHANGEFEED_MAX_ROWS:
                        raise SemanticSourceFenceUnavailableError(
                            f"{table} changefeed exceeded the bounded scan limit"
                        )
                cursor = stamp
                last_stamp = stamp
                scanned_database_rows += 1
                advanced = True
                if scanned_database_rows > _DATABASE_FEED_MAX_ROWS:
                    raise SemanticSourceFenceUnavailableError(
                        "database changefeed exceeded the bounded scan limit"
                    )

            if len(events) < _DATABASE_FEED_PAGE_SIZE:
                break
            if scanned_database_rows >= _DATABASE_FEED_MAX_ROWS:
                raise SemanticSourceFenceUnavailableError(
                    "database changefeed exceeded the bounded scan limit"
                )
            if not advanced:
                raise SemanticSourceFenceUnavailableError(
                    "database changefeed pagination did not advance"
                )

        session.watermark = cursor
        session.scanned_database_rows = scanned_database_rows
        session.scanned_source_rows = scanned_source_rows
        session.post_reference_ids = post_reference_ids

    def _classify_source_change(
        self,
        table: str,
        change: Any,
        *,
        brain_id: str,
        created_before: datetime,
        created_at_readonly: bool,
        post_reference_ids: set[str],
    ) -> None:
        """Apply the shared fail-closed source-event classifier."""
        if not isinstance(change, Mapping):
            raise SemanticSourceFenceUnavailableError(
                f"{table} changefeed returned an unclassifiable change"
            )
        neutral_update = False
        if set(change) == {"current", "update"}:
            action, record = "update", change["current"]
            if isinstance(record, Mapping):
                evidence = self._projection_neutral_update(table, record, change["update"])
                if evidence is None:
                    raise SemanticSourceFenceUnavailableError(
                        f"{table} changefeed returned an invalid reverse diff"
                    )
                neutral_update = evidence
        elif len(change) == 1:
            action, record = next(iter(change.items()))
        else:
            raise SemanticSourceFenceUnavailableError(
                f"{table} changefeed returned an unclassifiable change"
            )
        if not isinstance(record, Mapping):
            raise SemanticSourceFenceUnavailableError(
                f"{table} changefeed returned an invalid record"
            )
        if action == "define_table":
            if self._is_expected_table_reassertion(table, record):
                return
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
                return
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
            return
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
            if neutral_update:
                return
            raise SemanticSourceChangedError(
                f"{table} changed since the semantic source token was captured"
            )
        post_reference_ids.add(record_id_text)

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
                    events = await self._show_table_changes(table, cursor, limit)
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
                        if not isinstance(change, Mapping):
                            raise SemanticSourceFenceUnavailableError(
                                f"{table} changefeed returned an unclassifiable change"
                            )
                        self._classify_source_change(
                            table,
                            change,
                            brain_id=brain_id,
                            created_before=created_before,
                            created_at_readonly=created_at_readonly,
                            post_reference_ids=post_reference_ids,
                        )

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


class _SemanticSourceFenceSession:
    """Invocation-scoped cursor and classifier state; never persisted."""

    def __init__(
        self,
        storage: SurrealDBSemanticSourceRevisionMixin,
        token: str,
        brain_id: str,
        token_versionstamp: int,
        created_before: datetime,
    ) -> None:
        self.storage = storage
        self.token = token
        self.brain_id = brain_id
        self.token_versionstamp = token_versionstamp
        self.created_before = created_before
        self.watermark = token_versionstamp
        self.scanned_database_rows = 0
        self.scanned_source_rows = dict.fromkeys(_SOURCE_TABLES, 0)
        self.post_reference_ids: dict[str, set[str]] = {table: set() for table in _SOURCE_TABLES}
        self._lock = asyncio.Lock()

    async def assert_unchanged(self) -> None:
        """Validate changes since the last completed scan, serially and atomically."""
        async with self._lock:
            await self.storage._assert_source_session_unchanged(self)

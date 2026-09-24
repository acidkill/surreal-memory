"""Fail-closed mutation fence backed by SurrealDB changefeeds.

The fence is deliberately database-wide: any neuron or synapse mutation
invalidates tokens for every brain in the database. This is conservative when
a database contains multiple brains, but avoids relying on changefeed payload
shapes for deletes and cannot miss a relevant write.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

_CHANGEFEED_RETENTION = timedelta(days=7)
_TOKEN_MAX_AGE = timedelta(days=6, hours=15, minutes=36)
_TOKEN_VERSION = 1
_BARRIER_CHANGEFEED_PAGE_SIZE = 128
_BARRIER_CHANGEFEED_MAX_ROWS = 100_000
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

    async def _find_marker_versionstamp(self, marker_id: str) -> int:
        """Find a barrier event using bounded inclusive raw-stamp pagination.

        SurrealDB's SINCE cursor is inclusive. Each subsequent page therefore
        skips the already-seen cursor event and must advance to a larger raw
        versionstamp. A hard total-row ceiling converts unexpectedly large or
        malformed feeds into a fail-closed error rather than a missed marker.
        """
        if _BARRIER_CHANGEFEED_PAGE_SIZE < 1 or _BARRIER_CHANGEFEED_MAX_ROWS < 1:
            raise SemanticSourceFenceUnavailableError("invalid barrier feed pagination limits")

        cursor = 0
        scanned = 0
        while scanned < _BARRIER_CHANGEFEED_MAX_ROWS:
            limit = min(
                _BARRIER_CHANGEFEED_PAGE_SIZE,
                _BARRIER_CHANGEFEED_MAX_ROWS - scanned,
            )
            try:
                events = await self._query(
                    f"SHOW CHANGES FOR TABLE semantic_source_barrier SINCE {cursor} LIMIT {limit}"
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
            marker_versionstamp = await self._find_marker_versionstamp(marker_id)
            return json.dumps(
                {
                    "version": _TOKEN_VERSION,
                    "brain_id": brain_id,
                    "versionstamp": marker_versionstamp,
                    "captured_at": captured_at.isoformat(timespec="microseconds") + "Z",
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

    @staticmethod
    def _decode_token(token: str, brain_id: str) -> tuple[int, datetime]:
        try:
            data = json.loads(token)
            versionstamp = data.get("versionstamp") if isinstance(data, dict) else None
            if (
                not isinstance(data, dict)
                or data.get("version") != _TOKEN_VERSION
                or data.get("brain_id") != brain_id
                or not isinstance(versionstamp, int)
                or isinstance(versionstamp, bool)
                or versionstamp < 0
                or not isinstance(data.get("captured_at"), str)
            ):
                raise ValueError
            captured_at = SurrealDBSemanticSourceRevisionMixin._parse_datetime(data["captured_at"])
            return versionstamp, captured_at
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SemanticSourceFenceExpiredError(
                "semantic source token is malformed or belongs to another brain"
            ) from exc

    async def assert_semantic_source_unchanged(self, since_token: str) -> None:
        """Raise if either source table changed or changefeed history is uncertain."""
        brain_id = self._get_brain_id()
        versionstamp, captured_at = self._decode_token(since_token, brain_id)
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

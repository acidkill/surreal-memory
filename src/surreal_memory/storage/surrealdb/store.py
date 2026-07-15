"""SurrealDB composite storage backend for Surreal-Memory.

Combines document, graph, and vector search capabilities in a single
SurrealDB instance. Implements the full NeuralStorage interface including
sync engine, change log, Merkle hashes, and typed memories.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from surreal_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.core.synapse import Direction, Synapse, SynapseType
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.surrealdb._ids import _safe_brain_id, _to_surreal_id
from surreal_memory.storage.surrealdb.activity import SurrealDBActivityMixin
from surreal_memory.storage.surrealdb.alerts import SurrealDBAlertsMixin
from surreal_memory.storage.surrealdb.cognitive import SurrealDBCognitiveMixin
from surreal_memory.storage.surrealdb.compression import SurrealDBCompressionMixin
from surreal_memory.storage.surrealdb.depth_priors import SurrealDBDepthPriorsMixin
from surreal_memory.storage.surrealdb.keyword_entity import SurrealDBKeywordEntityMixin
from surreal_memory.storage.surrealdb.maturation import SurrealDBMaturationMixin
from surreal_memory.storage.surrealdb.projects import SurrealDBProjectsMixin
from surreal_memory.storage.surrealdb.retrieval_trace import SurrealDBRetrievalTraceMixin
from surreal_memory.storage.surrealdb.review_schedules import SurrealDBReviewSchedulesMixin
from surreal_memory.storage.surrealdb.schema import ensure_schema
from surreal_memory.storage.surrealdb.sources import SurrealDBSourcesMixin
from surreal_memory.storage.surrealdb.tool_events import SurrealDBToolEventsMixin
from surreal_memory.storage.surrealdb.typed_memory import SurrealDBTypedMemoryMixin
from surreal_memory.storage.surrealdb.versions import SurrealDBVersionsMixin
from surreal_memory.utils.geo import GeoFilter, fiber_within
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _serialize_brain_config(config: Any) -> dict[str, Any]:
    """Serialize a ``BrainConfig`` dataclass to a plain dict for storage.

    Persisting the full config (not just a metadata copy) is what makes the
    per-brain retrieval knobs — including reranking — survive a round-trip.
    """
    try:
        return dataclasses.asdict(config)
    except Exception:
        return {}


def _deserialize_brain_config(raw: Any) -> BrainConfig:
    """Rebuild a ``BrainConfig`` from a stored dict, dropping unknown keys.

    Legacy brains stored their *metadata* in the ``config`` column (the old
    ``save_brain`` bug), so a dict with no BrainConfig fields yields a default
    ``BrainConfig``. Unknown/removed keys are filtered so config schema changes
    never break load.
    """
    if not isinstance(raw, dict) or not raw:
        return BrainConfig()
    valid = {f.name for f in dataclasses.fields(BrainConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid}
    if not filtered:
        return BrainConfig()
    try:
        return BrainConfig(**filtered)
    except Exception:
        logger.warning("Failed to deserialize stored brain config; using defaults", exc_info=True)
        return BrainConfig()


def _is_auth_error(exc: Exception) -> bool:
    """True if an exception is a SurrealDB auth / expired-token failure (HTTP 401).

    The SurrealDB Python SDK surfaces an expired root token as an
    ``aiohttp.ClientResponseError`` with ``status == 401``. Match on the status
    attribute first, then fall back to the message text for wrapped errors.
    """
    if getattr(exc, "status", None) == 401:
        return True
    msg = str(exc).lower()
    return "401" in msg or "unauthorized" in msg


_BRAIN_ID_SAFE = re.compile(r"^[a-zA-Z0-9_.\-]+$")

# Bounded concurrency for per-id batch fetches (get_neurons_batch/get_synapses_batch).
# Pipelines direct record selects over the one shared AsyncSurreal connection;
# measured ~1.7x over sequential on a 67k-neuron brain, with diminishing
# returns above this value (single-connection multiplexing, not true parallel
# sockets).
_BATCH_FETCH_CONCURRENCY = 16


def _brain_literal(brain_id: str) -> str:
    """Return ``brain_id`` as a safe inline SurrealQL string literal.

    SurrealDB 3.2.0's planner only uses the ``brain_id`` index when the value is
    an inline literal — a parameterized ``WHERE brain_id = $bid`` falls back to a
    full table scan. On the 64k-neuron table (each row carrying a 1024-float
    vector) that turned ``count() … GROUP ALL`` from 0.01 s into ~2.5 s, and it
    was the single biggest cost behind the dashboard's stats endpoint. brain_id
    is validated to a strict charset, so inlining it is injection-safe.
    """
    if not _BRAIN_ID_SAFE.match(brain_id):
        raise ValueError(f"unsafe brain_id for inline query: {brain_id!r}")
    return f'"{brain_id}"'


def _is_connection_error(exc: Exception) -> bool:
    """True if an exception signals a dropped/closed transport rather than an
    auth or query error.

    A long-lived WebSocket connection severed by a DB container restart (backup,
    upgrade, reboot) never returns 401, so ``_is_auth_error`` misses it. Without
    reconnecting on this, the cached dead connection fails EVERY subsequent query
    and the whole MCP surface returns -32000 until the process is restarted
    (audit finding S-01).
    """
    # OSError covers ConnectionError/ConnectionResetError (both subclasses) and
    # the aiohttp ClientOSError/ClientConnectorError raised during a restart
    # window. TimeoutError (== asyncio.TimeoutError since 3.11) is ALSO an OSError
    # subclass, but a legitimate slow-query timeout under the default HTTP
    # transport (ClientTimeout(total=30)) is a query outcome, not a dropped
    # transport — excluding it avoids a needless reconnect+retry that doubles
    # latency and masks the SDK's own is_timed_out error class.
    if isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if any(k in name for k in ("connectionclosed", "disconnect", "websocket", "transport")):
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "connection closed",
            "connection reset",
            "connection refused",
            "connection lost",
            "not connected",
            "no connection",
            "websocket",
            "broken pipe",
            "going away",
        )
    )


def _from_surreal_id(surreal_id: str) -> str:
    """Extract the original ID from a SurrealDB record ID like 'neuron:abc_def'."""
    if ":" in surreal_id:
        surreal_id = surreal_id.rsplit(":", 1)[1]
    return surreal_id.replace("_", "-")


def _parse_datetime(val: Any) -> datetime | None:
    """Parse a SurrealDB datetime value to Python datetime (naive for consistency)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        # Convert to naive datetime for consistency across codebase
        if val.tzinfo is not None:
            return val.replace(tzinfo=None)
        return val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            # Convert to naive datetime
            if parsed.tzinfo is not None:
                return parsed.replace(tzinfo=None)
            return parsed
        except (ValueError, AttributeError):
            return None
    return None


def _ensure_naive(dt: datetime) -> datetime:
    """Convert datetime to naive (no timezone) for comparison."""
    if dt is None:
        return datetime.min  # noqa: DTZ901 — naive sentinel is intentional
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _parse_neuron_type(value: Any) -> NeuronType:
    """Parse a stored neuron type tolerantly.

    External writers (e.g. an integration plugin) may store enum NAMES in
    uppercase ('CONCEPT'); the engine stores enum VALUES ('concept'). Unknown
    values fall back to CONCEPT instead of raising ValueError, so one foreign
    row cannot break recall over the whole brain.
    """
    try:
        return NeuronType(value)
    except ValueError:
        try:
            return NeuronType(str(value).lower())
        except ValueError:
            logger.warning("Unknown neuron type %r — falling back to 'concept'", value)
            return NeuronType.CONCEPT


def _row_to_neuron(row: dict[str, Any]) -> Neuron:
    """Convert a SurrealDB neuron record to a Neuron."""
    meta = dict(row.get("metadata") or {})
    # Surface the stored vector back into metadata so callers (e.g. reindex's
    # --missing-only) can tell whether a neuron already has an embedding.
    embedding_vec = row.get("embedding_vec")
    if embedding_vec:
        meta["_embedding"] = list(embedding_vec)
    rid = row["id"]
    neuron_id = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    # Strip table prefix and convert underscores back to dashes
    if ":" in neuron_id:
        neuron_id = neuron_id.split(":", 1)[1]
    neuron_id = neuron_id.replace("_", "-")
    return Neuron(
        id=neuron_id,
        type=_parse_neuron_type(row["type"]),
        content=str(row["content"]),
        metadata=meta,
        content_hash=int(row.get("content_hash", 0)),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        ephemeral=bool(row.get("ephemeral", False)),
    )


def _row_to_neuron_state(row: dict[str, Any]) -> NeuronState:
    """Convert a SurrealDB neuron_state record to NeuronState."""
    nid = row.get("neuron_id", "")
    if hasattr(nid, "record_id"):
        nid = str(nid.record_id)
    else:
        nid = str(nid)
    return NeuronState(
        neuron_id=nid,
        activation_level=float(row.get("activation_level", 0.0)),
        access_frequency=int(row.get("access_frequency", 0)),
        last_activated=_parse_datetime(row.get("last_activated")),
        decay_rate=float(row.get("decay_rate", 0.1)),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        firing_threshold=float(row.get("firing_threshold", 0.3)),
        refractory_until=_parse_datetime(row.get("refractory_until")),
        refractory_period_ms=float(row.get("refractory_period_ms", 500.0)),
        homeostatic_target=float(row.get("homeostatic_target", 0.5)),
    )


def _endpoint_to_id(edge_value: Any, legacy_value: Any = None) -> str:
    """Resolve a synapse endpoint id from the native RELATION ``in``/``out`` field.

    ``in``/``out`` come back as a ``RecordID`` pointing at a neuron (``neuron:abc``).
    Falls back to the legacy ``source_id``/``target_id`` string when the edge field
    is absent (old fixtures / pre-migration rows). Returns the bare id with the
    table prefix stripped and underscores denormalised back to dashes, matching
    the ids the rest of the store speaks.
    """
    value = edge_value if edge_value is not None else legacy_value
    if value is None:
        return ""
    part = getattr(value, "id", None)
    if part is not None:
        text = str(part)  # RecordID -> its identifier part
    else:
        text = str(value)
        if ":" in text:
            text = text.split(":", 1)[1]
    return text.replace("_", "-")


def _row_to_synapse(row: dict[str, Any]) -> Synapse:
    """Convert a SurrealDB synapse RELATION record to Synapse.

    Endpoints live in the native ``in``/``out`` edge fields (each a ``RecordID``
    pointing at a neuron). Falls back to the legacy ``source_id``/``target_id``
    string fields so pre-migration fixtures still map.
    """

    rid = row["id"]
    syn_id = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    if ":" in syn_id:
        syn_id = syn_id.split(":", 1)[1]
    syn_id = syn_id.replace("_", "-")
    source_id = _endpoint_to_id(row.get("in"), row.get("source_id"))
    target_id = _endpoint_to_id(row.get("out"), row.get("target_id"))
    syn = Synapse(
        id=syn_id,
        type=SynapseType(row["type"]),
        source_id=source_id,
        target_id=target_id,
        weight=float(row.get("weight", 1.0)),
        direction=Direction(str(row.get("direction", "forward"))),
        metadata=dict(row.get("metadata") or {}),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        last_activated=_parse_datetime(row.get("last_activated")),
        reinforced_count=int(row.get("reinforced_count", 0)),
    )
    return syn


def _row_to_fiber(row: dict[str, Any]) -> Fiber:
    """Convert a SurrealDB fiber record to Fiber."""
    rid = row["id"]
    fiber_id = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    if ":" in fiber_id:
        fiber_id = fiber_id.split(":", 1)[1]
    return Fiber(
        id=fiber_id,
        neuron_ids=set(row.get("neuron_ids") or []),
        synapse_ids=set(row.get("synapse_ids") or []),
        anchor_neuron_id=str(row.get("anchor_neuron_id", "")),
        pathway=list(row.get("pathway") or []),
        conductivity=float(row.get("conductivity", 1.0)),
        last_conducted=_parse_datetime(row.get("last_conducted")),
        time_start=_parse_datetime(row.get("time_start")),
        time_end=_parse_datetime(row.get("time_end")),
        coherence=float(row.get("coherence", 0.0)),
        salience=float(row.get("salience", 0.0)),
        frequency=int(row.get("frequency", 0)),
        summary=row.get("summary"),
        essence=row.get("essence"),
        auto_tags=set(row.get("auto_tags") or []),
        agent_tags=set(row.get("agent_tags") or []),
        metadata=dict(row.get("metadata") or {}),
        compression_tier=int(row.get("compression_tier", 0)),
        pinned=bool(row.get("pinned", False)),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        last_ghost_shown_at=_parse_datetime(row.get("last_ghost_shown_at")),
    )


class SurrealDBStorage(
    SurrealDBTypedMemoryMixin,
    SurrealDBRetrievalTraceMixin,
    SurrealDBProjectsMixin,
    SurrealDBSourcesMixin,
    SurrealDBAlertsMixin,
    SurrealDBCognitiveMixin,
    SurrealDBReviewSchedulesMixin,
    SurrealDBMaturationMixin,
    SurrealDBVersionsMixin,
    SurrealDBKeywordEntityMixin,
    SurrealDBCompressionMixin,
    SurrealDBActivityMixin,
    SurrealDBDepthPriorsMixin,
    SurrealDBToolEventsMixin,
    NeuralStorage,
):
    """SurrealDB-backed storage for Surreal-Memory.

    Multi-model: documents (neurons), graphs (synapses as native RELATE edges
    linking neurons via in/out), and vector search (HNSW via embedding_vec) all
    in one database.

    Usage:
        storage = SurrealDBStorage(url="http://localhost:8001", ...)
        await storage.initialize()
        storage.set_brain("my-brain")
        await storage.add_neuron(neuron)
    """

    def __init__(
        self,
        url: str = "",
        namespace: str = "",
        database: str = "",
        user: str = "",
        password: str = "",
        embedding_dim: int = 3072,
    ) -> None:
        from surreal_memory.storage.surrealdb.connection import SurrealSettings

        settings = SurrealSettings.from_env()
        self._url = url or settings.url
        self._namespace = namespace or settings.namespace
        self._database = database or settings.database
        self._user = user or settings.user
        self._password = password or settings.password
        self._embedding_dim = embedding_dim
        self._conn: Any = None
        self._current_brain_id: str | None = None
        self._change_seq: int = 0
        # Serializes token re-auth so concurrent queries that all hit an
        # expired-token 401 trigger a single reconnect, not a storm.
        self._reauth_lock = asyncio.Lock()
        # ISO GQL (SurrealDB 3.2+) capability, detected once at initialize().
        # get_path uses a GQL SHORTEST-path fast-path when available and falls
        # back to BFS otherwise. See _get_path_gql for the 3.2.0 scoping caveat.
        self._gql_available = False
        # Learn-once: after this many get_path calls where GQL yielded no usable
        # (endpoint-verified) path, stop attempting GQL for the session so the
        # fast-path never adds a wasted round-trip on servers where it can't help.
        self._gql_path_misses = 0

    async def initialize(self) -> None:
        """Connect to SurrealDB, gate on server version, apply schema + migrations."""
        from surrealdb import AsyncSurreal

        from surreal_memory.storage.surrealdb.connection import (
            AUTH_HINT,
            MIN_SERVER_VERSION,
            StorageAuthError,
            StorageVersionError,
            is_credential_error,
            parse_server_version,
        )
        from surreal_memory.storage.surrealdb.migrations import apply_migrations

        self._conn = AsyncSurreal(self._url)
        try:
            await self._conn.signin({"username": self._user, "password": self._password})
        except Exception as exc:
            if is_credential_error(exc):
                raise StorageAuthError(
                    f"SurrealDB authentication failed for user '{self._user}' at {self._url}.",
                    hint=AUTH_HINT,
                ) from exc
            raise
        await self._conn.use(self._namespace, self._database)

        # Hard version gate (>= 3.2.0): the synapse RELATION schema and the
        # auto-migration below require SurrealDB 3.2.0. A failed/unparsable probe
        # warns and continues; only a CONFIRMED old version hard-fails.
        try:
            raw_version = await self._conn.version()
        except Exception:
            logger.warning("Could not read SurrealDB version; skipping version gate.")
            raw_version = None
        if raw_version is not None:
            parsed = parse_server_version(str(raw_version))
            if parsed is not None and parsed < MIN_SERVER_VERSION:
                min_str = ".".join(str(p) for p in MIN_SERVER_VERSION)
                raise StorageVersionError(
                    f"SurrealDB {raw_version} is too old; surreal-memory requires >= {min_str}.",
                    hint=(
                        "Upgrade the image: docker compose -f docker-compose.surrealdb.yml pull "
                        "&& docker compose -f docker-compose.surrealdb.yml up -d (the "
                        "surrealdb_data volume is preserved — back it up first)."
                    ),
                )

        await ensure_schema(self._conn, self._embedding_dim)
        # Auto-run the synapse->RELATE migration on first connect after upgrade.
        await apply_migrations(self._conn)

        # Detect ISO GQL capability (SurrealDB 3.2+) once, non-fatally (2s budget).
        # A labeled MATCH via eval::gql succeeds only when the server was started
        # with --allow-experimental gql AND --allow-eval-query; otherwise it raises
        # and get_path stays on BFS. The neuron LABEL is required — an unlabeled
        # `MATCH (n)` fails to parse even when GQL is enabled.
        try:
            await asyncio.wait_for(
                self._conn.query('RETURN eval::gql("MATCH (n:neuron) RETURN n LIMIT 1")'),
                timeout=2,
            )
            self._gql_available = True
        except Exception:
            self._gql_available = False
            logger.debug("ISO GQL not available; get_path will use BFS.", exc_info=True)

        logger.info(
            "SurrealDB connected: %s ns=%s db=%s (gql=%s)",
            self._url,
            self._namespace,
            self._database,
            self._gql_available,
        )

    async def close(self) -> None:
        """Close the SurrealDB connection.

        Some SDK transports (notably the HTTP connection) do not implement
        ``close()`` — it raises because there is no persistent socket to tear
        down. Treat any close failure as a no-op so callers don't each need
        their own error handling; always drop the connection reference.
        """
        if self._conn is None:
            return
        try:
            await self._conn.close()
        except Exception:
            logger.debug(
                "Connection transport does not support close(); ignoring",
                exc_info=True,
            )
        finally:
            self._conn = None

    @property
    def brain_id(self) -> str | None:
        return self._current_brain_id

    def set_brain(self, brain_id: str) -> None:
        self._current_brain_id = brain_id

    def _get_brain_id(self) -> str:
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        # Store-layer choke point: the returned id is inlined raw into
        # ``device:{brain_id}_{did}`` record literals (register/get/update/delete
        # device) — fail-closed reject a hostile id so it can never break out.
        return _safe_brain_id(self._current_brain_id)

    def _ensure_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("SurrealDB not initialized. Call initialize() first.")
        return self._conn

    def disable_auto_save(self) -> None:
        pass

    def enable_auto_save(self) -> None:
        pass

    async def batch_save(self) -> None:
        pass

    # ================================================================
    # Query helper
    # ================================================================

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a SurrealQL query and return result rows.

        Retries once after re-authenticating if the cached connection's token
        has expired. SurrealDB issues root tokens with a ~1h TTL and the SDK's
        HTTP connection never refreshes them, so a long-lived server connection
        starts returning 401 after an hour — which silently broke the dashboard
        until the container was restarted. Reconnecting on 401 keeps the cached
        connection alive without a restart.

        Also reconnects on a dropped transport (WebSocket closed by a DB restart):
        that surfaces as a connection error, not a 401, so without this the dead
        cached connection fails every query until the process restarts (S-01).
        """
        try:
            result = await self._ensure_conn().query(sql, params)
        except Exception as exc:
            if not (_is_auth_error(exc) or _is_connection_error(exc)):
                raise
            async with self._reauth_lock:
                await self._reconnect()
            result = await self._ensure_conn().query(sql, params)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0] if isinstance(result[0], list) else result
        return []

    async def _reconnect(self) -> None:
        """Re-establish the SurrealDB connection after a token expiry / 401.

        Re-signin + re-select the namespace/database on a fresh connection so
        the cached singleton keeps working. Schema is already applied, so it is
        not re-run here.
        """
        from surrealdb import AsyncSurreal

        from surreal_memory.storage.surrealdb.connection import (
            AUTH_HINT,
            StorageAuthError,
            is_credential_error,
        )

        conn = AsyncSurreal(self._url)
        try:
            await conn.signin({"username": self._user, "password": self._password})
        except Exception as exc:
            if is_credential_error(exc):
                raise StorageAuthError(
                    f"SurrealDB authentication failed for user '{self._user}' at {self._url}.",
                    hint=AUTH_HINT,
                ) from exc
            raise
        await conn.use(self._namespace, self._database)
        self._conn = conn
        logger.info("SurrealDB reconnected after token expiry: %s", self._url)

    # ================================================================
    # Neuron Operations
    # ================================================================

    async def add_neuron(self, neuron: Neuron) -> str:
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()

        sid = _to_surreal_id(neuron.id)
        meta = dict(neuron.metadata)
        embedding_vec = meta.pop("_embedding", None)

        record_data: dict[str, Any] = {
            "id": sid,
            "brain_id": brain_id,
            "type": neuron.type.value,
            "content": neuron.content,
            "content_hash": neuron.content_hash,
            "metadata": meta,
            "ephemeral": neuron.ephemeral,
            "created_at": neuron.created_at,
            "updated_at": utcnow(),
        }
        if embedding_vec is not None:
            record_data["embedding_vec"] = list(embedding_vec)

        await conn.insert("neuron", record_data)

        # Create initial state
        state_data = {
            "id": f"state_{sid}",
            "neuron_id": neuron.id,
            "brain_id": brain_id,
            "activation_level": 0.0,
            "access_frequency": 0,
            "created_at": neuron.created_at,
        }
        try:
            await conn.insert("neuron_state", state_data)
        except Exception:
            pass

        # Record change
        await self._record_change_internal("neuron", neuron.id, "insert", neuron)

        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(neuron_id)
        try:
            result = await conn.select(f"neuron:{sid}")
            if result and isinstance(result, list) and len(result) > 0:
                return _row_to_neuron(result[0])
        except Exception:
            pass
        return None

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        """Fetch multiple neurons by id, concurrently over the shared connection.

        A single ``id IN [...]`` query was measured *slower* than per-id direct
        selects on this SurrealDB version (3.2.0) — IN-membership against a
        RecordID primary key doesn't use the primary index the way a direct
        ``neuron:{id}`` fetch does, so it falls back to a scan-like path (same
        family of index-selection gap as the brain_id-literal-vs-param gotcha
        elsewhere in this store). Bounded concurrency instead pipelines the
        direct selects over the one shared connection (~1.7x measured on a
        67k-neuron brain) without the IN-query regression.
        """
        if not neuron_ids:
            return {}
        semaphore = asyncio.Semaphore(_BATCH_FETCH_CONCURRENCY)

        async def _fetch_one(nid: str) -> tuple[str, Neuron | None]:
            sid = _to_surreal_id(nid)
            async with semaphore:
                try:
                    result = await self._conn.select(f"neuron:{sid}")
                except Exception:
                    return nid, None
            if result and isinstance(result, list) and len(result) > 0:
                return nid, _row_to_neuron(result[0])
            return nid, None

        pairs = await asyncio.gather(*(_fetch_one(nid) for nid in neuron_ids))
        return {nid: neuron for nid, neuron in pairs if neuron is not None}

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
        include_embedding: bool = True,
    ) -> list[Neuron]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if type is not None:
            conditions.append("type = $ntype")
            params["ntype"] = type.value

        if content_exact is not None:
            conditions.append("content = $content_exact")
            params["content_exact"] = content_exact

        if content_contains is not None:
            # Full-text match (@@) via the BM25 index instead of a CONTAINS
            # substring scan: 0.9ms vs 2.9s on a 65k-neuron brain. The analyzer
            # lowercases, so matching is case-insensitive (a net improvement for
            # keyword/entity lookups); it is token-based rather than arbitrary
            # substring, which is the intended semantics for concept recall.
            conditions.append("content @@ $content_contains")
            params["content_contains"] = content_contains

        if time_range is not None:
            start, end = time_range
            conditions.append("created_at >= $time_start AND created_at <= $time_end")
            params["time_start"] = start.isoformat()
            params["time_end"] = end.isoformat()

        if ephemeral is not None:
            conditions.append("ephemeral = $ephemeral")
            params["ephemeral"] = ephemeral

        where = " AND ".join(conditions)
        # OMIT the 1024-3072-float embedding_vec when the caller doesn't need it
        # (dashboard graph/timeline). It is ~4-8 KB/row, so dragging it over tens of
        # thousands of rows is the single biggest dashboard slowdown after a re-embed.
        projection = "SELECT *" if include_embedding else "SELECT * OMIT embedding_vec"
        rows = await self._query(
            f"{projection} FROM neuron WHERE {where} ORDER BY id LIMIT {int(limit)} START {int(offset)}",
            **params,
        )
        return [_row_to_neuron(r) for r in rows]

    async def find_neurons_by_ids(
        self, neuron_ids: list[str], include_embedding: bool = False
    ) -> list[Neuron]:
        """Fetch specific neurons by id in one query, omitting the embedding by
        default. Used by the graph view, which needs only a few thousand nodes out
        of tens of thousands and never uses the vector."""
        if not neuron_ids:
            return []
        # Convert to SurrealDB record ids and keep only injection-safe names
        # (alphanumeric + underscore), since they are interpolated into FROM.
        safe = [
            sid
            for nid in neuron_ids
            for sid in (_to_surreal_id(nid),)
            if sid and all(c.isalnum() or c == "_" for c in sid)
        ]
        if not safe:
            return []
        projection = "SELECT *" if include_embedding else "SELECT * OMIT embedding_vec"
        out: list[Neuron] = []
        # Chunk to keep the FROM record-id list a sane query size.
        for i in range(0, len(safe), 1000):
            things = ", ".join(f"neuron:{s}" for s in safe[i : i + 1000])
            rows = await self._query(f"{projection} FROM {things}")
            out.extend(_row_to_neuron(r) for r in rows)
        return out

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id", "content CONTAINS $prefix"]
        params: dict[str, Any] = {"brain_id": brain_id, "prefix": prefix}

        if type_filter is not None:
            conditions.append("type = $ntype")
            params["ntype"] = type_filter.value

        where = " AND ".join(conditions)
        rows = await self._query(
            f"SELECT id, content, type FROM neuron WHERE {where} LIMIT {int(min(limit, 20))}",
            **params,
        )

        # Get activation info for ranking
        results = []
        for r in rows:
            nid = str(r["id"]).split(":")[-1]
            state = await self.get_neuron_state(nid)
            results.append(
                {
                    "neuron_id": nid,
                    "content": r["content"],
                    "type": r["type"],
                    "access_frequency": state.access_frequency if state else 0,
                    "activation_level": state.activation_level if state else 0.0,
                    "score": (state.activation_level if state else 0.0)
                    + 0.1 * (state.access_frequency if state else 0),
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def update_neuron(self, neuron: Neuron) -> None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(neuron.id)
        meta = dict(neuron.metadata)
        embedding_vec = meta.pop("_embedding", None)

        update_data: dict[str, Any] = {
            "type": neuron.type.value,
            "content": neuron.content,
            "content_hash": neuron.content_hash,
            "metadata": meta,
            "ephemeral": neuron.ephemeral,
            "updated_at": utcnow(),
        }
        if embedding_vec is not None:
            update_data["embedding_vec"] = list(embedding_vec)

        await conn.merge(f"neuron:{sid}", update_data)
        await self._record_change_internal("neuron", neuron.id, "update", neuron)

    async def update_neuron_embeddings(self, pairs: list[tuple[str, list[float]]]) -> None:
        """Write embedding vectors for many neurons in a single round-trip.

        Inline embedding on write (encoder) would otherwise issue one ``merge`` per
        neuron — tens of round-trips per save. This batches them into one
        multi-statement ``UPDATE`` (param-bound, so injection-safe), setting only
        ``embedding_vec``/``updated_at``. The change-log is intentionally skipped:
        the vector is derived from content that already logged its create, and a
        peer can re-embed locally, so it needn't sync as a separate delta.
        """
        if not pairs:
            return
        stmts: list[str] = []
        params: dict[str, Any] = {}
        for i, (nid, vec) in enumerate(pairs):
            params[f"id{i}"] = _to_surreal_id(nid)
            params[f"v{i}"] = list(vec)
            stmts.append(
                f"UPDATE type::record('neuron', $id{i}) "
                f"SET embedding_vec = $v{i}, updated_at = time::now()"
            )
        await self._query(";\n".join(stmts) + ";", **params)

    async def delete_neuron(self, neuron_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        safe_brain = _safe_brain_id(brain_id)
        sid = _to_surreal_id(neuron_id)

        # Delete connected synapses first (belt-and-braces: SurrealDB also auto-cleans
        # edges when the neuron record is deleted). Endpoints are native in/out now.
        #
        # Measured live: a single "brain_id = $brain_id AND (in = ... OR out = ...)"
        # query cost ~1.2s/call regardless of whether brain_id/the record id are
        # inlined or param-bound — SurrealDB 3.2.0's planner doesn't use either
        # `idx_synapse_in`/`idx_synapse_out` index across an OR of two different
        # fields, so it falls back to a full scan. Splitting into two single-field
        # DELETEs (each hits its own index) measured ~5ms total for both — this was
        # the dominant cost behind prune's non-dry-run timeout on a large brain.
        await self._query(f"DELETE synapse WHERE brain_id = '{safe_brain}' AND in = neuron:{sid}")
        await self._query(f"DELETE synapse WHERE brain_id = '{safe_brain}' AND out = neuron:{sid}")
        # Delete state (record id is state_<sid>, matching the writer in add_neuron)
        await self._query(f"DELETE neuron_state:state_{sid}")

        try:
            await conn.delete(f"neuron:{sid}")
            await self._record_change_internal("neuron", neuron_id, "delete")
            return True
        except Exception:
            return False

    async def delete_neurons_batch(self, neuron_ids: list[str]) -> int:
        """Delete multiple neurons sequentially.

        Unlike ``get_neurons_batch``'s reads, concurrent deletes here raised
        live ``Transaction conflict: Transaction write conflict`` errors —
        SurrealDB's transaction isolation conflicts on concurrent writes to
        the same tables (``synapse``, ``change_log``), it doesn't just slow
        down like concurrent reads do. ``delete_neuron`` is cheap now (~ms,
        see its docstring), so sequential is already well within budget.
        """
        count = 0
        for nid in neuron_ids:
            if await self.delete_neuron(nid):
                count += 1
        return count

    async def has_neuron_by_content_hash(self, content_hash: int) -> bool:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM neuron WHERE brain_id = $brain_id AND content_hash = $ch LIMIT 1",
            brain_id=brain_id,
            ch=content_hash,
        )
        return len(rows) > 0

    # ================================================================
    # Neuron State Operations
    # ================================================================

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(neuron_id)
        try:
            result = await conn.select(f"neuron_state:state_{sid}")
            if result:
                return _row_to_neuron_state(result[0] if isinstance(result, list) else result)
        except Exception:
            pass
        return None

    async def get_neuron_states_batch(self, neuron_ids: list[str]) -> dict[str, NeuronState]:
        """Fetch many neuron states in ONE query per chunk.

        The base fallback loops ``get_neuron_state`` once per id. Consolidation's
        prune scan calls this per 5000-neuron page, so on a ~67k-neuron brain the
        fallback fired ~67k point selects and dominated the strategy runtime
        (~170s measured → 120s budget blown). The ``neuron_state`` row carries a
        plain ``neuron_id``, so a single ``IN $ids`` select returns the whole
        page; ids are param-bound (injection-safe). Chunked so an oversized id
        list can't build a pathologically large statement.
        """
        result: dict[str, NeuronState] = {}
        if not neuron_ids:
            return result
        brain_id = self._get_brain_id()
        chunk = 5000
        for start in range(0, len(neuron_ids), chunk):
            ids = list(neuron_ids[start : start + chunk])
            rows = await self._query(
                "SELECT * FROM neuron_state WHERE brain_id = $brain_id AND neuron_id IN $ids",
                brain_id=brain_id,
                ids=ids,
            )
            for r in rows:
                state = _row_to_neuron_state(r)
                result[state.neuron_id] = state
        return result

    async def update_neuron_state(self, state: NeuronState) -> None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(state.neuron_id)

        update_data: dict[str, Any] = {
            "activation_level": state.activation_level,
            "access_frequency": state.access_frequency,
            "decay_rate": state.decay_rate,
            "firing_threshold": state.firing_threshold,
            "refractory_period_ms": state.refractory_period_ms,
            "homeostatic_target": state.homeostatic_target,
        }
        if state.last_activated:
            update_data["last_activated"] = state.last_activated
        if state.refractory_until:
            update_data["refractory_until"] = state.refractory_until

        await conn.merge(f"neuron_state:state_{sid}", update_data)

    # ================================================================
    # Synapse Operations
    # ================================================================

    async def add_synapse(self, synapse: Synapse) -> str:
        from surrealdb import RecordID

        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        sid = _to_surreal_id(synapse.id)
        ss = _to_surreal_id(synapse.source_id)
        st = _to_surreal_id(synapse.target_id)

        # Native RELATION edge: endpoints are the built-in in/out RecordIDs. INSERT
        # RELATION (not conn.insert, which does not work on a RELATION table) keeps
        # the custom edge id so fiber.synapse_ids / change_log / Merkle stay stable.
        record_data: dict[str, Any] = {
            "id": RecordID("synapse", sid),
            "in": RecordID("neuron", ss),
            "out": RecordID("neuron", st),
            "brain_id": brain_id,
            "type": synapse.type.value,
            "weight": synapse.weight,
            "direction": synapse.direction,
            "metadata": dict(synapse.metadata),
            "created_at": synapse.created_at,
            "reinforced_count": synapse.reinforced_count,
        }
        await conn.query("INSERT RELATION INTO synapse $row", {"row": record_data})

        await self._record_change_internal("synapse", synapse.id, "insert", synapse)
        return synapse.id

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(synapse_id)
        try:
            result = await conn.select(f"synapse:{sid}")
            if result:
                return _row_to_synapse(result[0] if isinstance(result, list) else result)
        except Exception:
            pass
        return None

    async def get_synapses_batch(self, synapse_ids: list[str]) -> dict[str, Synapse]:
        """Fetch multiple synapses by id, concurrently over the shared connection.

        See ``get_neurons_batch`` for why this is bounded-concurrent per-id
        selects rather than a single ``id IN [...]`` query.
        """
        if not synapse_ids:
            return {}
        conn = self._ensure_conn()
        semaphore = asyncio.Semaphore(_BATCH_FETCH_CONCURRENCY)

        async def _fetch_one(syn_id: str) -> tuple[str, Synapse | None]:
            sid = _to_surreal_id(syn_id)
            async with semaphore:
                try:
                    result = await conn.select(f"synapse:{sid}")
                except Exception:
                    return syn_id, None
            if result:
                return syn_id, _row_to_synapse(result[0] if isinstance(result, list) else result)
            return syn_id, None

        pairs = await asyncio.gather(*(_fetch_one(sid) for sid in synapse_ids))
        return {sid: synapse for sid, synapse in pairs if synapse is not None}

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
        limit: int | None = None,
    ) -> list[Synapse]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if source_id is not None:
            conditions.append("in = type::record('neuron', $source_id)")
            params["source_id"] = _to_surreal_id(source_id)
        if target_id is not None:
            conditions.append("out = type::record('neuron', $target_id)")
            params["target_id"] = _to_surreal_id(target_id)
        if type is not None:
            conditions.append("type = $stype")
            params["stype"] = type.value
        if min_weight is not None:
            conditions.append("weight >= $min_weight")
            params["min_weight"] = min_weight

        where = " AND ".join(conditions)
        query_str = f"SELECT * FROM synapse WHERE {where}"
        if limit is not None:
            query_str += f" LIMIT {limit}"
        rows = await self._query(query_str, **params)
        return [_row_to_synapse(r) for r in rows]

    async def update_synapse(self, synapse: Synapse) -> None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(synapse.id)

        update_data: dict[str, Any] = {
            "type": synapse.type.value,
            "weight": synapse.weight,
            "direction": synapse.direction,
            "metadata": dict(synapse.metadata),
            "reinforced_count": synapse.reinforced_count,
        }
        if synapse.last_activated:
            update_data["last_activated"] = synapse.last_activated

        await conn.merge(f"synapse:{sid}", update_data)
        await self._record_change_internal("synapse", synapse.id, "update", synapse)

    async def delete_synapse(self, synapse_id: str) -> bool:
        conn = self._ensure_conn()
        sid = _to_surreal_id(synapse_id)
        try:
            await conn.delete(f"synapse:{sid}")
            await self._record_change_internal("synapse", synapse_id, "delete")
            return True
        except Exception:
            return False

    async def delete_synapses_batch(self, synapse_ids: set[str] | list[str]) -> int:
        """Delete multiple synapses sequentially.

        See ``delete_neurons_batch`` — concurrent writes conflict under
        SurrealDB's transaction isolation, unlike concurrent reads.
        """
        count = 0
        for sid in synapse_ids:
            if await self.delete_synapse(sid):
                count += 1
        return count

    # ================================================================
    # Graph Traversal
    # ================================================================

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        brain_id = self._get_brain_id()
        base_params: dict[str, Any] = {
            "brain_id": brain_id,
            "nid": _to_surreal_id(neuron_id),
        }

        extra: list[str] = []
        if synapse_types:
            type_list = ", ".join(f"'{t.value}'" for t in synapse_types)
            extra.append(f"type IN [{type_list}]")
        if min_weight is not None:
            extra.append("weight >= $min_weight")
            base_params["min_weight"] = min_weight

        # Query each edge direction with its own indexed equality (idx_synapse_in
        # / idx_synapse_out). A combined ``in = .. OR out = ..`` disables the
        # index and full-scans the whole synapse table (~950ms/call vs ~2ms). in
        # is the source endpoint (outgoing edges), out the target (incoming).
        edge_cols: list[str] = []
        if direction in ("out", "both"):
            edge_cols.append("in")
        if direction in ("in", "both"):
            edge_cols.append("out")

        seen: set[str] = set()
        results: list[tuple[Neuron, Synapse]] = []
        for col in edge_cols:
            conditions = [
                "brain_id = $brain_id",
                f"{col} = type::record('neuron', $nid)",
                *extra,
            ]
            where = " AND ".join(conditions)
            # Inline both endpoint neurons via the native edge links (in.*/out.*)
            # so a single query returns the neighbour records — kills the N+1
            # get_neuron call that ran once per edge before the RELATION migration.
            syn_rows = await self._query(
                f"SELECT *, in.* AS in_neuron, out.* AS out_neuron FROM synapse WHERE {where}",
                **base_params,
            )
            for sr in syn_rows:
                syn = _row_to_synapse(sr)
                if syn.id in seen:
                    continue
                seen.add(syn.id)
                # The neighbour is the endpoint that is not the queried neuron.
                # Use the domain helper so a row that (defensively) touches
                # neither end is skipped rather than resolving to the wrong end.
                other_id = syn.other_end(neuron_id)
                if other_id is None:
                    continue
                neighbor_row = (
                    sr.get("out_neuron") if other_id == syn.target_id else sr.get("in_neuron")
                )
                neighbor = _row_to_neuron(neighbor_row) if neighbor_row else None
                if neighbor is None:
                    # Orphan endpoint or inline missing — fall back to direct fetch.
                    neighbor = await self.get_neuron(other_id)
                if neighbor is not None:
                    results.append((neighbor, syn))
        return results

    @property
    def gql_available(self) -> bool:
        """True when the server exposes ISO GQL (detected at initialize())."""
        return self._gql_available

    async def _get_path_gql(
        self,
        source_id: str,
        target_id: str,
        max_hops: int,
        bidirectional: bool,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """GQL SHORTEST-path fast-path over synapse edges (source -> target).

        Returns the path as ``[(Neuron, Synapse), ...]`` or ``None`` when GQL does
        not yield a verified source->target path (so get_path falls back to BFS).

        SurrealDB 3.2.0 caveat: the experimental ISO GQL dialect (eval::gql) cannot
        anchor/filter a MATCH node by its record id — string compares don't match a
        RecordID and casts/functions/param binding are unsupported. We still issue
        the correctly-scoped query (ids are hard-`[A-Za-z0-9_]`-sanitised by
        ``_to_surreal_id``, so a hostile source/target id cannot break out of the
        inlined ``{id:"…"}`` string literal to inject GQL) and VERIFY the returned
        path's endpoints in Python, returning None if they
        don't connect source->target. Today that verification fails whenever id
        scoping is unavailable, so BFS handles the lookup; the fast-path activates
        automatically if a future SurrealDB makes GQL node-id scoping work.
        """
        src = _to_surreal_id(source_id)
        tgt = _to_surreal_id(target_id)
        arrow = "-[:synapse]-" if bidirectional else "-[:synapse]->"
        gql = (
            f'MATCH p = SHORTEST 1 (s:neuron {{id:"{src}"}})'
            f'{arrow}{{1,{max_hops}}}(t:neuron {{id:"{tgt}"}}) RETURN p'
        )
        rows = await self._query("RETURN eval::gql($gql)", gql=gql)
        path_seq = rows[0].get("p") if rows and isinstance(rows[0], dict) else None
        if not path_seq or not isinstance(path_seq, list):
            return None

        # The path alternates node, edge, node, ...; edges sit at odd indices.
        result: list[tuple[Neuron, Synapse]] = []
        for i in range(1, len(path_seq), 2):
            edge_row = path_seq[i]
            node_row = path_seq[i + 1] if i + 1 < len(path_seq) else None
            if not isinstance(edge_row, dict) or not isinstance(node_row, dict):
                return None
            result.append((_row_to_neuron(node_row), _row_to_synapse(edge_row)))

        # Verify the path actually connects source_id -> target_id.
        if not result or result[-1][0].id != target_id:
            return None
        first_edge = result[0][1]
        if source_id not in (first_edge.source_id, first_edge.target_id):
            return None
        return result

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        bidirectional: bool = False,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """Path finding between two neurons via synapses.

        Uses a GQL SHORTEST-path fast-path when the server exposes ISO GQL and it
        yields an endpoint-verified path; otherwise (and on any GQL failure) uses
        the universal BFS fallback.
        """
        if source_id == target_id:
            src = await self.get_neuron(source_id)
            return (
                [(src, Synapse.create(source_id, target_id, SynapseType.RELATED_TO))]
                if src
                else None
            )

        # GQL SHORTEST-path fast-path (endpoint-verified) when available.
        if self._gql_available:
            try:
                gql_path = await self._get_path_gql(source_id, target_id, max_hops, bidirectional)
            except Exception:
                gql_path = None
                logger.debug("GQL path lookup failed; falling back to BFS.", exc_info=True)
            if gql_path is not None:
                self._gql_path_misses = 0
                return gql_path
            # GQL produced no usable path — after repeated misses stop trying so the
            # fast-path never keeps adding a wasted round-trip where it cannot help.
            self._gql_path_misses += 1
            if self._gql_path_misses >= 3:
                self._gql_available = False
                logger.debug("Disabling GQL path fast-path after repeated misses.")

        # BFS (universal fallback).
        visited: set[str] = {source_id}
        queue: list[tuple[str, list[tuple[Neuron, Synapse]]]] = [(source_id, [])]

        while queue and len(queue[0][1]) < max_hops:
            current_id, path = queue.pop(0)
            dir_literal: Literal["out", "in", "both"] = "both" if bidirectional else "out"
            neighbors = await self.get_neighbors(current_id, direction=dir_literal)

            for neighbor, synapse in neighbors:
                if neighbor.id == target_id:
                    return path + [(neighbor, synapse)]
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [(neighbor, synapse)]))

        return None

    # ================================================================
    # Fiber Operations
    # ================================================================

    async def add_fiber(self, fiber: Fiber) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        fid = _to_surreal_id(fiber.id)

        record_data: dict[str, Any] = {
            "id": fid,
            "brain_id": brain_id,
            "neuron_ids": list(fiber.neuron_ids),
            "synapse_ids": list(fiber.synapse_ids),
            "anchor_neuron_id": fiber.anchor_neuron_id,
            "pathway": fiber.pathway,
            "conductivity": fiber.conductivity,
            "coherence": fiber.coherence,
            "salience": fiber.salience,
            "frequency": fiber.frequency,
            "summary": fiber.summary,
            "essence": fiber.essence,
            "auto_tags": list(fiber.auto_tags),
            "agent_tags": list(fiber.agent_tags),
            "metadata": dict(fiber.metadata),
            "compression_tier": fiber.compression_tier,
            "pinned": fiber.pinned,
            "created_at": fiber.created_at,
        }
        if fiber.last_conducted:
            record_data["last_conducted"] = fiber.last_conducted
        if fiber.time_start:
            record_data["time_start"] = fiber.time_start
        if fiber.time_end:
            record_data["time_end"] = fiber.time_end
        if fiber.last_ghost_shown_at:
            record_data["last_ghost_shown_at"] = fiber.last_ghost_shown_at

        await conn.insert("fiber", record_data)
        await self._record_change_internal("fiber", fiber.id, "insert")
        return fiber.id

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        conn = self._ensure_conn()
        fid = _to_surreal_id(fiber_id)
        try:
            result = await conn.select(f"fiber:{fid}")
            if result:
                return _row_to_fiber(result[0] if isinstance(result, list) else result)
        except Exception:
            pass
        return None

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
        near: GeoFilter | None = None,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if contains_neuron:
            conditions.append("$contains_neuron IN neuron_ids")
            params["contains_neuron"] = contains_neuron
        if min_salience is not None:
            conditions.append("salience >= $min_salience")
            params["min_salience"] = min_salience
        if metadata_key is not None:
            # Push the marker-existence filter into SurQL so LIMIT applies AFTER it.
            # As a post-LIMIT Python filter it silently dropped any fiber carrying the
            # key (e.g. a learned `_habit_pattern` workflow) that sat beyond the first
            # `limit` rows on a large brain — `smem habits list` then showed nothing.
            # metadata_key is an internal marker constant, but validate it to a bare
            # identifier so the interpolated field path can never carry an injection.
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", metadata_key):
                raise ValueError(f"invalid metadata_key: {metadata_key!r}")
            conditions.append(f"metadata.{metadata_key} != NONE")
        if near is not None:
            # Server-side pre-filter: drop locationless fibers (exercises FLEXIBLE-field
            # traversal). No spatial index exists, so geo::distance would not improve
            # complexity and its lon/lat order + metre unit vary across 3.x — the exact
            # haversine post-filter below is the version-independent source of truth.
            conditions.append("metadata.location != NONE")
        if tags:
            # Push tag membership into SurQL (AND semantics) so LIMIT applies AFTER tag
            # filtering — else a tagged subset (e.g. a chat session) can fall outside the
            # LIMIT window on a large brain and silently vanish. SurrealDB stores tags as
            # separate `auto_tags`/`agent_tags` arrays (no combined `tags` field); the
            # Python `f.tags` union post-filter below still applies. Tags are stored
            # lowercased, so match the lowercased form (parameterised — no injection).
            for i, tag in enumerate(tags):
                key = f"tag_{i}"
                conditions.append(f"(${key} IN auto_tags OR ${key} IN agent_tags)")
                params[key] = tag.lower()

        where = " AND ".join(conditions)
        # Over-fetch when a Python post-filter (time/metadata_key/near) further narrows.
        fetch_limit = min(int(limit) * 3, 3000) if (near is not None or tags) else int(limit)
        rows = await self._query(f"SELECT * FROM fiber WHERE {where} LIMIT {fetch_limit}", **params)

        fibers = [_row_to_fiber(r) for r in rows]

        # Post-filter for complex conditions
        if tags:
            # Normalize query tags to lowercase so "KB" matches fibers stored as "kb"
            normalized_tags = {t.lower() for t in tags}
            fibers = [f for f in fibers if normalized_tags.issubset(f.tags)]
        if time_overlaps:
            start, end = time_overlaps
            # Normalize to naive UTC for comparison
            if start.tzinfo is not None:
                start = start.replace(tzinfo=None)
            if end.tzinfo is not None:
                end = end.replace(tzinfo=None)
            fibers = [
                f
                for f in fibers
                if f.time_start
                and f.time_end
                and _ensure_naive(f.time_start) <= end
                and _ensure_naive(f.time_end) >= start
            ]
        # metadata_key is now filtered in SurQL (see WHERE above), so no post-filter here.
        # Exact geospatial hard filter (haversine) — version-independent source of truth.
        if near is not None:
            fibers = [f for f in fibers if fiber_within(f, near)]

        return fibers[:limit]

    async def update_fiber(self, fiber: Fiber) -> None:
        conn = self._ensure_conn()
        fid = _to_surreal_id(fiber.id)

        update_data: dict[str, Any] = {
            "neuron_ids": list(fiber.neuron_ids),
            "synapse_ids": list(fiber.synapse_ids),
            "anchor_neuron_id": fiber.anchor_neuron_id,
            "pathway": fiber.pathway,
            "conductivity": fiber.conductivity,
            "coherence": fiber.coherence,
            "salience": fiber.salience,
            "frequency": fiber.frequency,
            "summary": fiber.summary,
            "essence": fiber.essence,
            "auto_tags": list(fiber.auto_tags),
            "agent_tags": list(fiber.agent_tags),
            "metadata": dict(fiber.metadata),
            "compression_tier": fiber.compression_tier,
            "pinned": fiber.pinned,
        }
        if fiber.last_conducted:
            update_data["last_conducted"] = fiber.last_conducted
        if fiber.time_start:
            update_data["time_start"] = fiber.time_start
        if fiber.time_end:
            update_data["time_end"] = fiber.time_end
        if fiber.last_ghost_shown_at:
            update_data["last_ghost_shown_at"] = fiber.last_ghost_shown_at

        await conn.merge(f"fiber:{fid}", update_data)
        await self._record_change_internal("fiber", fiber.id, "update")

    async def delete_fiber(self, fiber_id: str) -> bool:
        conn = self._ensure_conn()
        fid = _to_surreal_id(fiber_id)
        try:
            await conn.delete(f"fiber:{fid}")
            await self._record_change_internal("fiber", fiber_id, "delete")
            return True
        except Exception:
            return False

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
        exclude_expired: bool = False,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        order_dir = "DESC" if descending else "ASC"
        # When requested, drop fibers whose typed_memory is past its expires_at
        # (soft-forgotten) so they leave recall immediately, without waiting for
        # consolidation cleanup (issue #36).
        expired_clause = (
            " AND id NOT IN (SELECT VALUE fiber_id FROM typed_memory "
            "WHERE brain_id = $brain_id AND expires_at != NONE "
            "AND expires_at <= time::now())"
            if exclude_expired
            else ""
        )
        rows = await self._query(
            f"SELECT * FROM fiber WHERE brain_id = $brain_id{expired_clause} "
            f"ORDER BY {order_by} {order_dir} LIMIT {int(limit)}",
            brain_id=brain_id,
        )
        return [_row_to_fiber(r) for r in rows]

    # ================================================================
    # Brain Operations
    # ================================================================

    async def save_brain(self, brain: Brain) -> None:
        conn = self._ensure_conn()
        # brain.id is inlined raw into ``merge("brain:{id}")`` and the fallback
        # ``UPDATE brain:{id} SET ...`` statement below — fail-closed reject a
        # hostile id at the store layer (not just the REST route).
        _safe_brain_id(brain.id)

        record_data: dict[str, Any] = {
            "id": brain.id,  # Use original ID to avoid underscore conversion
            "name": brain.name,
            "config": _serialize_brain_config(brain.config),
            "metadata": dict(brain.metadata),
            "created_at": brain.created_at,
            "updated_at": brain.updated_at,
        }
        try:
            await conn.insert("brain", record_data)
        except Exception:
            # Try to update existing record
            try:
                await conn.merge(f"brain:{brain.id}", record_data)
            except Exception:
                # Query and update if merge also fails
                rows = await self._query(
                    "SELECT * FROM brain WHERE id = $id", id=f"brain:{brain.id}"
                )
                if rows:
                    for field, value in record_data.items():
                        await self._query(
                            f"UPDATE brain:{brain.id} SET {field} = $value", value=value
                        )

    async def get_brain(self, brain_id: str) -> Brain | None:
        self._ensure_conn()
        try:
            # Query all brains and filter manually (string matching in SurrealDB is problematic)
            rows = await self._query("SELECT * FROM brain")
            if rows and len(rows) > 0:
                for r in rows:
                    rid = r["id"]
                    name = str(r.get("name") or "")
                    # Compare record ID string
                    rid_str = str(rid) if not hasattr(rid, "id") else f"brain:{rid.id}"
                    # Brains are addressed by name; match the name field first,
                    # then fall back to a record-id match for id-based lookups.
                    # Without the name match, get_brain(name) never resolves a
                    # brain whose record id is a random UUID, so the bootstrap
                    # re-creates a fresh brain on every start (orphan-row leak).
                    if name == brain_id or brain_id in rid_str or rid_str.endswith(f":{brain_id}"):
                        bid_str = (
                            str(rid.id).replace("_", "-")
                            if hasattr(rid, "id")
                            else str(rid).split(":")[-1].replace("_", "-")
                        )
                        return Brain(
                            id=bid_str,
                            name=name,
                            config=_deserialize_brain_config(r.get("config")),
                            metadata=dict(r.get("metadata") or {}),
                            created_at=_parse_datetime(r.get("created_at")) or utcnow(),
                            updated_at=_parse_datetime(r.get("updated_at")) or utcnow(),
                        )
        except Exception:
            pass
        return None

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} not found")

        raw_neurons = await self.find_neurons(limit=100000)
        raw_synapses = await self.get_synapses()
        raw_fibers = await self.get_fibers(limit=100000)

        neurons: list[dict[str, Any]] = [
            {
                "id": n.id,
                "type": n.type.value,
                "content": n.content,
                "metadata": dict(n.metadata),
                "created_at": n.created_at.isoformat(),
            }
            for n in raw_neurons
        ]
        synapses: list[dict[str, Any]] = [
            {
                "id": s.id,
                "source_id": s.source_id,
                "target_id": s.target_id,
                "type": s.type.value,
                "weight": s.weight,
                "direction": s.direction.value,
                "metadata": dict(s.metadata),
            }
            for s in raw_synapses
        ]
        fibers: list[dict[str, Any]] = [
            {
                "id": f.id,
                "neuron_ids": list(f.neuron_ids),
                "synapse_ids": list(f.synapse_ids),
                "anchor_neuron_id": f.anchor_neuron_id,
                "pathway": f.pathway,
                "conductivity": f.conductivity,
                "salience": f.salience,
            }
            for f in raw_fibers
        ]

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config=dict(brain.metadata),
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        bid = target_brain_id or snapshot.brain_id
        self.set_brain(bid)

        brain = Brain(
            id=bid,
            name=snapshot.brain_name,
            metadata=dict(snapshot.config),
        )
        await self.save_brain(brain)

        for nd in snapshot.neurons:
            try:
                neuron = Neuron(
                    id=str(nd.get("id", "")),
                    type=_parse_neuron_type(nd["type"]),
                    content=str(nd["content"]),
                    metadata=dict(nd.get("metadata") or {}),
                    created_at=_parse_datetime(nd.get("created_at")) or utcnow(),
                )
                await self.add_neuron(neuron)
            except Exception:
                pass

        for sd in snapshot.synapses:
            try:
                synapse = Synapse(
                    id=str(sd.get("id", "")),
                    source_id=str(sd.get("source_id", "")),
                    target_id=str(sd.get("target_id", "")),
                    type=SynapseType(sd["type"]),
                    weight=float(sd.get("weight", 1.0)),
                    direction=Direction(str(sd.get("direction", "forward"))),
                    metadata=dict(sd.get("metadata") or {}),
                    created_at=_parse_datetime(sd.get("created_at")) or utcnow(),
                )
                await self.add_synapse(synapse)
            except Exception:
                pass

        for fd in snapshot.fibers:
            try:
                fiber = Fiber(
                    id=str(fd.get("id", "")),
                    neuron_ids=set(fd.get("neuron_ids") or []),
                    synapse_ids=set(fd.get("synapse_ids") or []),
                    anchor_neuron_id=str(fd.get("anchor_neuron_id", "")),
                    pathway=list(fd.get("pathway") or []),
                    conductivity=float(fd.get("conductivity", 1.0)),
                    salience=float(fd.get("salience", 0.0)),
                )
                await self.add_fiber(fiber)
            except Exception:
                pass

        return bid

    # ================================================================
    # Statistics
    # ================================================================

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        # Inline the (validated) brain_id so the count() aggregates use the
        # brain_id index instead of a full scan — see _brain_literal. The three
        # scans are independent, so run them concurrently.
        lit = _brain_literal(brain_id)
        neuron_rows, synapse_rows, fiber_rows = await asyncio.gather(
            self._query(f"SELECT count() AS c FROM neuron WHERE brain_id = {lit} GROUP ALL"),
            self._query(f"SELECT count() AS c FROM synapse WHERE brain_id = {lit} GROUP ALL"),
            self._query(f"SELECT count() AS c FROM fiber WHERE brain_id = {lit} GROUP ALL"),
        )

        def _count(rows: list[Any]) -> int:
            if rows and len(rows) > 0:
                return int(rows[0].get("c", 0))
            return 0

        return {
            "neuron_count": _count(neuron_rows),
            "synapse_count": _count(synapse_rows),
            "fiber_count": _count(fiber_rows),
        }

    async def get_enhanced_stats(
        self, brain_id: str, include_neuron_types: bool = True
    ) -> dict[str, Any]:
        stats = await self.get_stats(brain_id)

        # Neuron type breakdown. This GROUP BY over the whole neuron table is the
        # single most expensive query on a large brain (~2.6 s for 64k neurons)
        # and DiagnosticsEngine never reads it — so callers that only need the
        # health metrics pass include_neuron_types=False to skip it entirely.
        type_counts: dict[str, int] = {}
        if include_neuron_types:
            type_rows = await self._query(
                "SELECT type, count() AS c FROM neuron WHERE brain_id = $bid GROUP BY type",
                bid=brain_id,
            )
            type_counts = {str(r.get("type", "unknown")): int(r.get("c", 0)) for r in type_rows}

        # Synapse stats by type — required by DiagnosticsEngine for diversity
        # (Shannon entropy over by_type counts) and recall_confidence (avg_weight).
        # Without this block the dashboard reported diversity=0 / "0 of 8 types
        # used" even though the brain uses many synapse types. Brain-explicit
        # ($bid) query, so it is race-free across the shared storage singleton.
        synapse_stats: dict[str, Any] = {
            "avg_weight": 0.0,
            "total_reinforcements": 0,
            "by_type": {},
        }
        syn_rows = await self._query(
            "SELECT type, count() AS cnt, math::mean(weight) AS avg_w, "
            "math::sum(reinforced_count) AS total_r "
            "FROM synapse WHERE brain_id = $bid GROUP BY type",
            bid=brain_id,
        )
        total_weight = 0.0
        total_count = 0
        total_reinforcements = 0
        for row in syn_rows:
            stype = str(row.get("type", "unknown"))
            cnt = int(row.get("cnt", 0))
            avg_w = float(row.get("avg_w") or 0.0)
            total_r = int(row.get("total_r") or 0)
            synapse_stats["by_type"][stype] = {
                "count": cnt,
                "avg_weight": round(avg_w, 4),
                "total_reinforcements": total_r,
            }
            total_weight += avg_w * cnt
            total_count += cnt
            total_reinforcements += total_r
        if total_count > 0:
            synapse_stats["avg_weight"] = round(total_weight / total_count, 4)
        synapse_stats["total_reinforcements"] = total_reinforcements

        return {
            **stats,
            "neuron_types": type_counts,
            "synapse_stats": synapse_stats,
        }

    # ================================================================
    # Clear
    # ================================================================

    async def clear(self, brain_id: str) -> None:
        await self._query("DELETE neuron WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE neuron_state WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE synapse WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE fiber WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE change_log WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE device WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE merkle_hash WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE typed_memory WHERE brain_id = $bid", bid=brain_id)

    # ================================================================
    # Vector Search (for cone queries / semantic search)
    # ================================================================

    async def find_neurons_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        type_filter: NeuronType | None = None,
    ) -> list[tuple[Neuron, float]]:
        """Find neurons by vector similarity using SurrealDB KNN operator."""
        brain_id = self._get_brain_id()
        conditions = [
            "brain_id = $brain_id",
        ]
        params: dict[str, Any] = {
            "brain_id": brain_id,
            "vec": query_embedding,
        }

        if type_filter is not None:
            conditions.append("type = $ntype")
            params["ntype"] = type_filter.value

        where = " AND ".join(conditions)
        # SurrealDB KNN syntax: WHERE embedding_vec <|k, ef|> $vec
        rows = await self._query(
            f"SELECT *, vector::distance::knn() AS score "
            f"FROM neuron WHERE {where} AND embedding_vec <|{int(limit)},100|> $vec",
            **params,
        )

        results: list[tuple[Neuron, float]] = []
        for r in rows:
            raw_score = r.pop("score", None)
            score = float(raw_score) if raw_score is not None else 0.0
            # SurrealDB returns distance (lower = more similar), convert to similarity
            similarity = 1.0 / (1.0 + score) if score >= 0 else 0.0
            results.append((_row_to_neuron(r), similarity))
        return results

    # ================================================================
    # Change Log (for sync)
    # ================================================================

    async def _record_change_internal(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        entity: Any | None = None,
        device_id: str = "",
    ) -> None:
        """Internal helper to record a change."""
        try:
            brain_id = self._get_brain_id()
        except ValueError:
            return

        payload: dict[str, Any] | None = None
        if entity is not None:
            if isinstance(entity, Neuron):
                meta = dict(entity.metadata)
                meta.pop("_embedding", None)
                payload = {
                    "type": entity.type.value,
                    "content": entity.content,
                    "metadata": meta,
                    "content_hash": entity.content_hash,
                    "ephemeral": entity.ephemeral,
                }
            elif isinstance(entity, Synapse):
                payload = {
                    "type": entity.type.value,
                    "source_id": entity.source_id,
                    "target_id": entity.target_id,
                    "weight": entity.weight,
                    "direction": entity.direction,
                }

        conn = self._ensure_conn()
        self._change_seq += 1
        change_id = f"change_{self._change_seq}_{uuid4().hex[:8]}"
        record = {
            "id": change_id,
            "brain_id": brain_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation,
            "device_id": device_id,
            "payload": payload,
            "changed_at": utcnow(),
            "synced": False,
            "sequence": self._change_seq,
        }
        try:
            await conn.insert("change_log", record)
        except Exception:
            pass

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        self._change_seq += 1
        change_id = f"change_{self._change_seq}_{uuid4().hex[:8]}"
        record = {
            "id": change_id,
            "brain_id": brain_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation,
            "device_id": device_id,
            "payload": payload,
            "changed_at": utcnow(),
            "synced": False,
            "sequence": self._change_seq,
        }
        await conn.insert("change_log", record)
        return self._change_seq

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[Any]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM change_log WHERE brain_id = $brain_id AND sequence > $seq "
            "ORDER BY sequence ASC LIMIT $limit",
            brain_id=brain_id,
            seq=sequence,
            limit=limit,
        )
        from surreal_memory.sync.protocol import SyncChange

        changes = []
        for r in rows:
            changes.append(
                SyncChange(
                    sequence=int(r.get("sequence", 0)),
                    entity_type=str(r.get("entity_type", "")),
                    entity_id=str(r.get("entity_id", "")),
                    operation=str(r.get("operation", "")),
                    device_id=str(r.get("device_id", "")),
                    changed_at=str(r.get("changed_at", "")),
                    payload=r.get("payload") or {},
                )
            )
        return changes

    async def get_unsynced_changes(self, limit: int = 1000) -> list[Any]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM change_log WHERE brain_id = $brain_id AND synced = false "
            "ORDER BY sequence ASC LIMIT $limit",
            brain_id=brain_id,
            limit=limit,
        )
        from surreal_memory.sync.protocol import SyncChange

        return [
            SyncChange(
                sequence=int(r.get("sequence", 0)),
                entity_type=str(r.get("entity_type", "")),
                entity_id=str(r.get("entity_id", "")),
                operation=str(r.get("operation", "")),
                device_id=str(r.get("device_id", "")),
                changed_at=str(r.get("changed_at", "")),
                payload=r.get("payload") or {},
            )
            for r in rows
        ]

    async def mark_synced(self, up_to_sequence: int) -> int:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM change_log WHERE brain_id = $brain_id AND sequence <= $seq AND synced = false",
            brain_id=brain_id,
            seq=up_to_sequence,
        )
        count = 0
        for r in rows:
            cid = str(r.get("id", ""))
            try:
                await self._conn.merge(cid, {"synced": True})
                count += 1
            except Exception:
                pass
        return count

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        brain_id = self._get_brain_id()
        await self._query(
            "DELETE FROM change_log WHERE brain_id = $brain_id AND synced = true "
            "AND changed_at < time::ago($days, 'd')",
            brain_id=brain_id,
            days=older_than_days,
        )
        return 0

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        brain_id = self._get_brain_id()

        # Check what's already seeded
        existing = await self._query(
            "SELECT entity_id FROM change_log WHERE brain_id = $brain_id AND operation = 'insert' GROUP BY entity_id",
            brain_id=brain_id,
        )
        existing_ids = {str(r.get("entity_id", "")) for r in existing}

        counts = {"neurons": 0, "synapses": 0, "fibers": 0}

        # Seed neurons
        neurons = await self.find_neurons(limit=100000)
        for n in neurons:
            if n.id not in existing_ids:
                await self.record_change("neuron", n.id, "insert", device_id)
                counts["neurons"] += 1

        # Seed synapses
        synapses = await self.get_synapses()
        for s in synapses:
            if s.id not in existing_ids:
                await self.record_change("synapse", s.id, "insert", device_id)
                counts["synapses"] += 1

        # Seed fibers
        fibers = await self.get_fibers(limit=100000)
        for f in fibers:
            if f.id not in existing_ids:
                await self.record_change("fiber", f.id, "insert", device_id)
                counts["fibers"] += 1

        return counts

    async def get_change_log_stats(self) -> dict[str, Any]:
        brain_id = self._get_brain_id()
        total_rows = await self._query(
            "SELECT count() AS c FROM change_log WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )
        pending_rows = await self._query(
            "SELECT count() AS c FROM change_log WHERE brain_id = $bid AND synced = false GROUP ALL",
            bid=brain_id,
        )
        synced_rows = await self._query(
            "SELECT count() AS c FROM change_log WHERE brain_id = $bid AND synced = true GROUP ALL",
            bid=brain_id,
        )
        last_seq_rows = await self._query(
            "SELECT sequence FROM change_log WHERE brain_id = $bid ORDER BY sequence DESC LIMIT 1",
            bid=brain_id,
        )

        def _cnt(rows: list[Any]) -> int:
            return int(rows[0].get("c", 0)) if rows else 0

        def _max(rows: list[Any]) -> int:
            return int(rows[0].get("sequence", 0)) if rows else 0

        return {
            "total": _cnt(total_rows),
            "pending": _cnt(pending_rows),
            "synced": _cnt(synced_rows),
            "last_sequence": _max(last_seq_rows),
        }

    # ================================================================
    # Device Registry
    # ================================================================

    async def register_device(self, device_id: str, device_name: str = "") -> Any:
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        did = _to_surreal_id(device_id)

        from surreal_memory.sync.device import DeviceInfo

        record = {
            "id": f"{brain_id}_{did}",
            "device_id": device_id,
            "brain_id": brain_id,
            "device_name": device_name,
            "registered_at": utcnow(),
            "last_sync_sequence": 0,
        }
        try:
            await conn.insert("device", record)
        except Exception:
            await conn.merge(f"device:{brain_id}_{did}", record)

        return DeviceInfo(
            device_id=device_id,
            device_name=device_name,
            registered_at=utcnow(),
        )

    async def get_device(self, device_id: str) -> Any | None:
        brain_id = self._get_brain_id()
        did = _to_surreal_id(device_id)
        try:
            result = await self._conn.select(f"device:{brain_id}_{did}")
            if result:
                r = result[0] if isinstance(result, list) else result
                from surreal_memory.sync.device import DeviceInfo

                return DeviceInfo(
                    device_id=str(r.get("device_id", device_id)),
                    device_name=str(r.get("device_name", "")),
                    registered_at=_parse_datetime(r.get("registered_at")) or utcnow(),
                )
        except Exception:
            pass
        return None

    async def list_devices(self) -> list[Any]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM device WHERE brain_id = $brain_id ORDER BY registered_at ASC",
            brain_id=brain_id,
        )
        from surreal_memory.sync.device import DeviceInfo

        return [
            DeviceInfo(
                device_id=str(r.get("device_id", "")),
                device_name=str(r.get("device_name", "")),
                registered_at=_parse_datetime(r.get("registered_at")) or utcnow(),
            )
            for r in rows
        ]

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        brain_id = self._get_brain_id()
        did = _to_surreal_id(device_id)
        try:
            await self._conn.merge(
                f"device:{brain_id}_{did}",
                {
                    "last_sync_at": utcnow(),
                    "last_sync_sequence": last_sync_sequence,
                },
            )
        except Exception:
            pass

    async def remove_device(self, device_id: str) -> bool:
        brain_id = self._get_brain_id()
        did = _to_surreal_id(device_id)
        try:
            await self._conn.delete(f"device:{brain_id}_{did}")
            return True
        except Exception:
            return False

    # ================================================================
    # Merkle Hash Operations
    # ================================================================

    async def compute_merkle_root(self, entity_type: str, *, is_pro: bool = False) -> str | None:
        if not is_pro:
            return None

        brain_id = self._get_brain_id()
        rows = await self._query(
            f"SELECT id, updated_at FROM {entity_type} WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )

        # Build 2-level prefix tree
        buckets: dict[str, list[str]] = {}
        for r in rows:
            eid = str(r.get("id", "")).split(":")[-1]
            if len(eid) >= 2:
                prefix = eid[:2]
                buckets.setdefault(prefix, []).append(eid)

        conn = self._ensure_conn()
        # Build lookup for updated_at by entity ID
        updated_lookup = {}
        for row in rows:
            eid_raw = str(row.get("id", "")).split(":")[-1]
            updated_lookup[eid_raw] = str(row.get("updated_at", ""))

        # Compute and store bucket hashes
        for prefix, ids in sorted(buckets.items()):
            ids_sorted = sorted(ids)
            leaf_hashes = []
            for eid in ids_sorted:
                updated = updated_lookup.get(eid, "")
                leaf_hash = sha256(f"{eid}|{updated}".encode()).hexdigest()
                leaf_hashes.append(leaf_hash)

            bucket_hash = sha256("|".join(sorted(leaf_hashes)).encode()).hexdigest()
            merkle_id = f"{brain_id}_{entity_type}_{prefix}"

            try:
                await conn.insert(
                    "merkle_hash",
                    {
                        "id": _to_surreal_id(merkle_id),
                        "brain_id": brain_id,
                        "entity_type": entity_type,
                        "prefix": prefix,
                        "hash": bucket_hash,
                        "computed_at": utcnow(),
                    },
                )
            except Exception:
                await conn.merge(
                    f"merkle_hash:{_to_surreal_id(merkle_id)}",
                    {"hash": bucket_hash, "computed_at": utcnow()},
                )

        # Compute root from bucket hashes
        all_hashes = sorted(
            [sha256("|".join(sorted(buckets[p])).encode()).hexdigest() for p in sorted(buckets)]
        )
        root = sha256("|".join(all_hashes).encode()).hexdigest()
        return root

    async def get_merkle_tree(self, entity_type: str, *, is_pro: bool = False) -> dict[str, str]:
        if not is_pro:
            return {}
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT prefix, hash FROM merkle_hash WHERE brain_id = $brain_id AND entity_type = $et",
            brain_id=brain_id,
            et=entity_type,
        )
        return {str(r["prefix"]): str(r["hash"]) for r in rows}

    async def invalidate_merkle_prefix(
        self, entity_type: str, entity_id: str, *, is_pro: bool = False
    ) -> None:
        if not is_pro:
            return
        # Just recompute on next sync — no-op here is fine for v1

    async def get_merkle_root(self, *, is_pro: bool = False) -> str | None:
        if not is_pro:
            return None
        roots = []
        for et in ("neuron", "synapse", "fiber"):
            r = await self.compute_merkle_root(et, is_pro=is_pro)
            if r:
                roots.append(r)
        if not roots:
            return None
        return sha256("|".join(sorted(roots)).encode()).hexdigest()

    async def get_bucket_entity_ids(
        self, entity_type: str, prefix: str, *, is_pro: bool = False
    ) -> list[str]:
        if not is_pro:
            return []
        brain_id = self._get_brain_id()
        rows = await self._query(
            f"SELECT id FROM {entity_type} WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        return [
            str(r["id"]).split(":")[-1]
            for r in rows
            if str(r["id"]).split(":")[-1].startswith(prefix)
        ]

    # ================================================================
    # Lifecycle Operations
    # ================================================================

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        sid = _to_surreal_id(neuron_id)
        try:
            await self._conn.merge(f"neuron:{sid}", {"lifecycle_state": lifecycle_state})
        except Exception:
            pass

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        sid = _to_surreal_id(neuron_id)
        try:
            await self._conn.merge(f"neuron:{sid}", {"frozen": frozen})
        except Exception:
            pass

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT lifecycle_state, count() AS c FROM neuron WHERE brain_id = $bid GROUP BY lifecycle_state",
            bid=brain_id,
        )
        return {str(r.get("lifecycle_state", "active")): int(r.get("c", 0)) for r in rows}

    # ================================================================
    # Fiber helpers (Phase 1 completions)
    # ================================================================

    async def find_brain_by_name(self, name: str) -> Brain | None:
        rows = await self._query(
            "SELECT * FROM brain WHERE name = $name LIMIT 1",
            name=name,
        )
        if not rows:
            return None
        r = rows[0]
        rid = r.get("id", "")
        bid = (
            str(rid.id).replace("_", "-")
            if hasattr(rid, "id")
            else str(rid).split(":")[-1].replace("_", "-")
        )
        return Brain(
            id=bid,
            name=str(r.get("name", "")),
            config=_deserialize_brain_config(r.get("config")),
            metadata=dict(r.get("metadata") or {}),
            created_at=_parse_datetime(r.get("created_at")) or utcnow(),
            updated_at=_parse_datetime(r.get("updated_at")) or utcnow(),
        )

    async def list_brain_names(self) -> list[str]:
        """List distinct brain names from the SurrealDB brain table.

        The dashboard's default enumeration globs local sqlite fixture files,
        which do not exist on the SurrealDB backend; this lets it list the
        brains that actually hold data. GROUP BY collapses duplicate rows.
        """
        rows = await self._query("SELECT name FROM brain GROUP BY name")
        return sorted({str(r.get("name")) for r in rows if r.get("name")})

    async def get_all_neuron_states(self) -> list[NeuronState]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM neuron_state WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        return [_row_to_neuron_state(r) for r in rows]

    async def count_activated_neuron_states(self, brain_id: str | None = None) -> int:
        """Count neuron_states with access_frequency > 0 via a DB aggregate.

        Diagnostics used to load every neuron_state (~64k rows) just to count the
        activated ones — a multi-second scan on the dashboard. This does it in a
        single ``count() … GROUP ALL`` (~0.4 s)."""
        bid = brain_id or self._get_brain_id()
        rows = await self._query(
            "SELECT count() AS c FROM neuron_state"
            " WHERE brain_id = $bid AND access_frequency > 0 GROUP ALL",
            bid=bid,
        )
        return int(rows[0].get("c", 0)) if rows else 0

    async def get_connected_neuron_ids(self, brain_id: str | None = None) -> set[str]:
        """Return the set of neuron ids that are an endpoint of any synapse.

        Uses ``GROUP BY in`` / ``GROUP BY out`` on the native RELATE edge (the
        `source_id`/`target_id` fields are computed, so `array::group` on them
        yields nothing — but the real `in`/`out` record links group fine). This
        replaces loading ~185k Synapse objects (~10 s) for the orphan-rate metric
        with two distinct-key scans (~2 s total)."""
        bid = brain_id or self._get_brain_id()

        async def _endpoints(field: str) -> list[Any]:
            return await self._query(
                f"SELECT VALUE {field} FROM synapse WHERE brain_id = $bid GROUP BY {field}",
                bid=bid,
            )

        # The two distinct-endpoint scans are independent, so run them
        # concurrently — on a large brain this ~halves the wall time (measured
        # 2.5 s -> ~1.3 s) versus scanning in then out.
        in_rows, out_rows = await asyncio.gather(_endpoints("in"), _endpoints("out"))
        connected: set[str] = set()
        for rid in (*in_rows, *out_rows):
            connected.add(_from_surreal_id(str(rid)))
        return connected

    async def get_synapse_degrees(self, brain_id: str | None = None) -> dict[str, int]:
        """Per-neuron synapse degree via DB ``GROUP BY`` on the RELATE endpoints.

        Replaces loading every synapse into Python just to count endpoints
        (the dashboard graph's ranking step). Note: grouping must target the
        real ``in``/``out`` record links — the ``source_id``/``target_id``
        fields are computed and do not aggregate."""
        bid = brain_id or self._get_brain_id()

        async def _degree(field: str) -> list[Any]:
            return await self._query(
                f"SELECT {field} AS nid, count() AS deg FROM synapse"
                f" WHERE brain_id = $bid GROUP BY {field}",
                bid=bid,
            )

        # The in/out degree scans are independent — run them concurrently.
        in_rows, out_rows = await asyncio.gather(_degree("in"), _degree("out"))
        degree: dict[str, int] = {}
        for r in (*in_rows, *out_rows):
            nid = _endpoint_to_id(r.get("nid"))
            if nid:
                degree[nid] = degree.get(nid, 0) + int(r.get("deg", 0) or 0)
        return degree

    async def get_edges_for_neurons(self, neuron_ids: list[str]) -> list[Synapse]:
        """Outgoing synapses of the given neurons via the indexed graph traversal.

        ``->synapse`` on a record id uses the RELATE edge index, so fetching the
        edges of a few thousand selected nodes is sub-second — versus ~10 s for a
        full ``SELECT * FROM synapse`` table scan with 185k+ edges."""
        if not neuron_ids:
            return []
        safe = [
            sid
            for nid in neuron_ids
            for sid in (_to_surreal_id(nid),)
            if sid and all(c.isalnum() or c == "_" for c in sid)
        ]
        edges: list[Synapse] = []
        for i in range(0, len(safe), 500):
            things = ", ".join(f"neuron:{s}" for s in safe[i : i + 500])
            rows = await self._query(
                "SELECT id, ->synapse.{id, out, type, weight, direction, created_at} AS edges"
                f" FROM {things}"
            )
            for row in rows:
                src = row.get("id")
                for e in row.get("edges") or []:
                    d = dict(e)
                    d["in"] = src
                    try:
                        edges.append(_row_to_synapse(d))
                    except Exception:
                        continue
        return edges

    async def get_all_synapses(self, include_metadata: bool = True) -> list[Synapse]:
        brain_id = self._get_brain_id()
        # OMIT the metadata blob when the caller (e.g. the dashboard graph) only
        # needs endpoints/type/weight — it roughly halves the transfer for the
        # ~185k-row synapse scan.
        projection = "SELECT *" if include_metadata else "SELECT * OMIT metadata"
        rows = await self._query(
            f"{projection} FROM synapse WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        return [_row_to_synapse(r) for r in rows]

    async def batch_update_ghost_shown(self, fiber_ids: list[str], timestamp: datetime) -> int:
        count = 0
        for fid in fiber_ids:
            sid = _to_surreal_id(fid)
            try:
                await self._conn.merge(f"fiber:{sid}", {"last_ghost_shown_at": timestamp})
                count += 1
            except Exception:
                pass
        return count

    async def update_fiber_metadata(self, fiber_id: str, metadata: dict[str, Any]) -> None:
        sid = _to_surreal_id(fiber_id)
        try:
            await self._conn.merge(f"fiber:{sid}", {"metadata": metadata, "updated_at": utcnow()})
        except Exception:
            pass

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
        tags: set[str] | None = None,
    ) -> list[Fiber]:
        seen: set[str] = set()
        results: list[Fiber] = []
        for nid in neuron_ids:
            fibers = await self.find_fibers(contains_neuron=nid, limit=limit_per_neuron, tags=tags)
            for f in fibers:
                if f.id not in seen:
                    seen.add(f.id)
                    results.append(f)
        return results

    async def get_total_fiber_count(self) -> int:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT count() AS cnt FROM fiber WHERE brain_id = $brain_id GROUP ALL",
            brain_id=brain_id,
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_stale_fiber_count(self, brain_id: str, stale_days: int = 90) -> int:
        from datetime import timedelta

        cutoff = utcnow() - timedelta(days=stale_days)
        rows = await self._query(
            "SELECT count() AS cnt FROM fiber"
            " WHERE brain_id = $brain_id AND (last_conducted < $cutoff OR last_conducted IS NONE)"
            " GROUP ALL",
            brain_id=brain_id,
            cutoff=cutoff,
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_fiber_stage_counts(self, brain_id: str) -> dict[str, int]:
        rows = await self._query(
            "SELECT compression_tier, count() AS c FROM fiber"
            " WHERE brain_id = $brain_id GROUP BY compression_tier",
            brain_id=brain_id,
        )
        return {str(r.get("compression_tier", 0)): int(r.get("c", 0)) for r in rows}

    async def update_neuron_states_batch(self, states: list[NeuronState]) -> None:
        for state in states:
            await self.update_neuron_state(state)

    async def update_synapses_batch(self, synapses: list[Synapse]) -> None:
        for synapse in synapses:
            await self.update_synapse(synapse)

    async def cleanup_ephemeral_neurons(self, max_age_hours: float = 24.0) -> int:
        brain_id = self._get_brain_id()
        from datetime import timedelta

        cutoff = utcnow() - timedelta(hours=max_age_hours)
        rows = await self._query(
            "SELECT id FROM neuron WHERE brain_id = $brain_id AND ephemeral = true AND created_at < $cutoff",
            brain_id=brain_id,
            cutoff=cutoff,
        )
        count = 0
        for r in rows:
            # Re-sanitise the id read back from the DB rather than trusting the
            # write-path invariant (defence in depth): _to_surreal_id strips the
            # table prefix and folds, so ``neuron:{nid}`` stays a safe literal.
            nid = _to_surreal_id(str(r.get("id", "")))
            try:
                await self._conn.delete(f"neuron:{nid}")
                count += 1
            except Exception:
                pass
        return count

    async def update_neuron_ephemeral(self, neuron_id: str, ephemeral: bool) -> None:
        sid = _to_surreal_id(neuron_id)
        try:
            await self._conn.merge(f"neuron:{sid}", {"ephemeral": ephemeral})
        except Exception:
            pass

    async def update_neurons_ephemeral_batch(self, neuron_ids: list[str], ephemeral: bool) -> None:
        for nid in neuron_ids:
            await self.update_neuron_ephemeral(nid, ephemeral)

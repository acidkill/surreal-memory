"""Offline contract tests for exact activation CAS and CLI routing."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from surreal_memory.cli.main import app
from surreal_memory.core.brain import BrainConfig
from surreal_memory.engine.consolidation import ConsolidationConfig
from surreal_memory.engine.consolidation_progress import options_fingerprint
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.surrealdb.activation_migration import (
    ActivationMigrationError,
    SurrealDBActivationMigrationMixin,
)


@dataclass(frozen=True)
class BrainRecordFixture:
    """Structural row fixture; real SDK transport is an integration gate."""

    table_name: str
    id: str


class Store(SurrealDBActivationMigrationMixin):
    def __init__(self) -> None:
        self.config = {**asdict(BrainConfig()), "future_knob": {"preserve": [1, 2]}}
        self.rows = [
            {
                "id": BrainRecordFixture("brain", "exact-uuid"),
                "name": "default",
                "config": self.config,
            }
        ]
        self.scope = "default"
        self.lease = False
        self.release_count = 0
        self.update_count = 0
        self.race = False
        self.lose_response = False
        self.cancel = False
        self.lose_lease = False
        self.typed_marker = "original server types"
        self.typed_race = False

    def _get_brain_id(self) -> str:
        return self.scope

    def config_hash(self) -> str:
        # Unit stand-in for server-native hashing; typed NULL/NONE is tested on DB.
        return sha256(
            json.dumps([self.config, self.typed_marker], sort_keys=True).encode()
        ).hexdigest()

    async def _query(self, sql: str, **params: object) -> list[dict]:  # type: ignore[type-arg]
        assert sql == (
            "SELECT *, crypto::sha256(type::string(config)) AS config_sha256 "
            "FROM brain WHERE name = $name"
        )
        assert params == {"name": "default"}
        return [{**deepcopy(row), "config_sha256": self.config_hash()} for row in self.rows]

    async def acquire_consolidation_lease(
        self, brain_id: str, owner_token: str, *, lease_seconds: int = 120
    ) -> bool:
        if self.lease:
            return False
        self.lease = True
        if self.race:
            self.config["future_knob"] = {"drift": True}
        if self.typed_race:
            self.typed_marker = "different server types"
        return True

    async def release_consolidation_lease(self, brain_id: str, owner_token: str) -> bool:
        self.release_count += 1
        assert owner_token.startswith("activation-")
        if self.lose_lease:
            return False
        self.lease = False
        return True

    async def _run_query(self, sql: str, params: dict) -> object:  # type: ignore[type-arg]
        self.update_count += 1
        assert "SET config.activation_strategy = $to_strategy" in sql
        assert "crypto::sha256(type::string(config)) = $expected_config_sha256" in sql
        assert "expected_config" not in params
        assert "$token" not in sql and "token" not in params
        assert "owner_token = $migration_owner" in sql and "expires_at > time::now()" in sql
        assert params["brain_id"] == "exact-uuid"
        if self.cancel:
            raise asyncio.CancelledError
        if self.config_hash() == params["expected_config_sha256"]:
            self.config["activation_strategy"] = params["to_strategy"]
        if self.lose_response:
            raise ConnectionResetError("synthetic lost response")
        return []


async def migrate(store: Store, **overrides: object) -> dict:  # type: ignore[type-arg]
    kwargs = {
        "brain_id": "exact-uuid",
        "expected_config": deepcopy(store.config),
        "expected_config_sha256": store.config_hash(),
        "from_strategy": "classic",
        "to_strategy": "hybrid",
    }
    kwargs.update(overrides)
    return await store.set_activation_strategy(**kwargs)  # type: ignore[arg-type]


async def test_forward_reverse_preserve_all_config_and_confirmation() -> None:
    store = Store()
    before = deepcopy(store.config)
    receipt = await migrate(store)
    assert receipt["confirmed"] is True and receipt["reconciled"] is False
    assert store.config == {**before, "activation_strategy": "hybrid"}
    await migrate(store, from_strategy="hybrid", to_strategy="classic")
    assert store.config == before
    assert store.release_count == 2 and not store.lease


@pytest.mark.parametrize(
    "case", ["wrong-id", "config-drift", "from-mismatch", "wrong-scope", "wrong-name", "duplicate"]
)
async def test_preflight_failures_do_not_write(case: str) -> None:
    store = Store()
    overrides: dict[str, object] = {}
    if case == "wrong-id":
        overrides["brain_id"] = "prefix"
    elif case == "config-drift":
        overrides["expected_config"] = {"activation_strategy": "classic"}
    elif case == "from-mismatch":
        overrides["from_strategy"], overrides["to_strategy"] = "hybrid", "classic"
    elif case == "wrong-scope":
        store.scope = "other"
    elif case == "wrong-name":
        store.rows[0]["name"] = "other"
    else:
        store.rows *= 2
    with pytest.raises(ActivationMigrationError):
        await migrate(store, **overrides)
    assert store.update_count == store.release_count == 0


async def test_active_lease_does_not_release_other_owner() -> None:
    store = Store()
    store.lease = True
    with pytest.raises(ActivationMigrationError, match="active consolidation lease"):
        await migrate(store)
    assert store.update_count == store.release_count == 0 and store.lease


async def test_full_config_cas_race_releases_lease() -> None:
    store = Store()
    store.race = True
    with pytest.raises(ActivationMigrationError, match="readback failed"):
        await migrate(store)
    assert store.config["activation_strategy"] == "classic"
    assert store.release_count == 1 and not store.lease


async def test_lost_response_is_reconciled_without_replaying_write() -> None:
    store = Store()
    store.lose_response = True
    receipt = await migrate(store)
    assert receipt["reconciled"] is True
    assert store.update_count == 1 and store.release_count == 1


async def test_cancellation_releases_owned_lease() -> None:
    store = Store()
    store.cancel = True
    with pytest.raises(asyncio.CancelledError):
        await migrate(store)
    assert store.release_count == 1 and not store.lease


async def test_lost_ownership_is_not_reported_as_success() -> None:
    store = Store()
    store.lose_lease = True
    with pytest.raises(ActivationMigrationError, match="ownership was lost"):
        await migrate(store)


def test_activation_only_change_preserves_consolidation_options_fingerprint() -> None:
    config = asdict(ConsolidationConfig())
    before = options_fingerprint({"config": config})
    classic = BrainConfig()
    hybrid = classic.with_updates(activation_strategy="hybrid")
    assert hybrid.activation_strategy == "hybrid" and classic.activation_strategy == "classic"
    assert asdict(hybrid) == {**asdict(classic), "activation_strategy": "hybrid"}
    assert options_fingerprint({"config": asdict(ConsolidationConfig())}) == before
    assert hybrid.activation_threshold == 0.2
    assert (hybrid.ppr_damping, hybrid.ppr_iterations, hybrid.ppr_epsilon) == (0.15, 20, 1e-6)


def test_cli_exact_command_reads_snapshot_and_guarded_method() -> None:
    store = Store()

    @asynccontextmanager
    async def session():  # type: ignore[no-untyped-def]
        yield store

    with (
        patch(
            "surreal_memory.storage.surrealdb.activation_migration.activation_migration_storage",
            session,
        ),
        patch("surreal_memory.cli.commands.brain.get_storage") as generic_storage,
    ):
        result = CliRunner().invoke(
            app,
            [
                "brain",
                "set-activation",
                "--brain",
                "default",
                "--from",
                "classic",
                "--to",
                "hybrid",
            ],
        )
    assert result.exit_code == 0, result.output
    assert '"confirmed": true' in result.output
    generic_storage.assert_not_called()


def test_cli_rejects_other_brain_before_connecting() -> None:
    with patch("surreal_memory.cli.commands.brain.get_storage") as get_storage:
        result = CliRunner().invoke(
            app,
            ["brain", "set-activation", "--brain", "other", "--from", "classic", "--to", "hybrid"],
        )
    assert result.exit_code != 0
    get_storage.assert_not_called()


async def test_failed_readback_is_unknown_and_never_replays_update() -> None:
    store = Store()
    original = store.get_activation_snapshot
    count = 0

    async def snapshot(**kwargs: object) -> dict:  # type: ignore[type-arg]
        nonlocal count
        count += 1
        if count == 2:
            raise ConnectionResetError("synthetic failed readback")
        return await original()

    with patch.object(store, "get_activation_snapshot", snapshot):
        with pytest.raises(ActivationMigrationError, match="outcome unknown"):
            await migrate(store)
    assert store.update_count == 1 and store.release_count == 1


async def test_failed_acquisition_reply_releases_only_its_token() -> None:
    store = Store()

    async def lost_reply(*args: object, **kwargs: object) -> bool:
        raise ConnectionResetError("synthetic lost lease reply")

    with patch.object(store, "acquire_consolidation_lease", lost_reply):
        with pytest.raises(ConnectionResetError):
            await migrate(store)
    assert store.update_count == 0 and store.release_count == 1


def test_concrete_store_uses_existing_lease_and_single_attempt_query() -> None:
    from surreal_memory.storage.surrealdb.consolidation_state import (
        SurrealDBConsolidationStateMixin,
    )
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    assert SurrealDBStorage.acquire_consolidation_lease is (
        SurrealDBConsolidationStateMixin.acquire_consolidation_lease
    )
    assert SurrealDBStorage.release_consolidation_lease is (
        SurrealDBConsolidationStateMixin.release_consolidation_lease
    )
    assert SurrealDBStorage.set_activation_strategy is (
        SurrealDBActivationMigrationMixin.set_activation_strategy
    )


async def test_connection_only_session_does_not_initialize_or_create_brain() -> None:
    from surreal_memory.storage.surrealdb.activation_migration import activation_migration_storage

    store = MagicMock()
    store._reconnect = AsyncMock()
    store.close = AsyncMock()
    store._ensure_conn.return_value.version = AsyncMock(return_value="surrealdb-3.2.0")
    with patch("surreal_memory.storage.surrealdb.store.SurrealDBStorage", return_value=store):
        async with activation_migration_storage() as connected:
            assert connected is store
    store._reconnect.assert_awaited_once_with()
    store.set_brain.assert_called_once_with("default")
    store.initialize.assert_not_called()
    store.save_brain.assert_not_called()
    store.close.assert_awaited_once()


@pytest.mark.parametrize("version", ["unknown", "surrealdb-2.9.0"])
async def test_connection_only_session_fails_closed_on_version(version: str) -> None:
    from surreal_memory.storage.surrealdb.activation_migration import activation_migration_storage

    store = MagicMock()
    store._reconnect = AsyncMock()
    store.close = AsyncMock()
    store._ensure_conn.return_value.version = AsyncMock(return_value=version)
    with patch("surreal_memory.storage.surrealdb.store.SurrealDBStorage", return_value=store):
        with pytest.raises(ActivationMigrationError, match="supported SurrealDB"):
            async with activation_migration_storage():
                pytest.fail("must not enter unsupported migration session")
    store.close.assert_awaited_once()
    store.set_brain.assert_not_called()


class PipelineReachedError(Exception):
    """Stop client execution after its real pipeline has loaded stored config."""


def readback_store(store: Store):  # type: ignore[no-untyped-def]
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage.__new__(SurrealDBStorage)
    storage._conn = object()
    storage.set_brain("default")
    storage._query = AsyncMock(side_effect=lambda *args, **kwargs: deepcopy(store.rows))  # type: ignore[method-assign]
    return storage


def assert_hybrid_pipeline(storage, config):  # type: ignore[no-untyped-def]
    with patch(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        return_value=(False, "", ""),
    ):
        pipeline = ReflexPipeline(storage, config)
    assert pipeline._config.activation_strategy == "hybrid"
    assert pipeline._ppr_activator is not None
    assert pipeline._use_reflex is True
    raise PipelineReachedError


def test_new_cli_recall_pipeline_reads_migrated_config() -> None:
    store = Store()
    asyncio.run(migrate(store))
    storage = readback_store(store)
    with (
        patch("surreal_memory.cli.commands.memory.get_config"),
        patch("surreal_memory.cli.commands.memory.get_storage", AsyncMock(return_value=storage)),
        patch(
            "surreal_memory.cli.commands.memory.ReflexPipeline", side_effect=assert_hybrid_pipeline
        ) as factory,
    ):
        result = CliRunner().invoke(app, ["recall", "synthetic query", "--depth", "1"])
    assert isinstance(result.exception, PipelineReachedError), result.output
    factory.assert_called_once()


async def test_new_mcp_recall_pipeline_reads_migrated_config() -> None:
    from surreal_memory.mcp.server import MCPServer

    store = Store()
    await migrate(store)
    storage = readback_store(store)
    with patch("surreal_memory.mcp.server.get_config", return_value=MagicMock()):
        server = MCPServer()
    with (
        patch.object(server, "get_storage", AsyncMock(return_value=storage)),
        patch.object(server, "_get_active_session", AsyncMock(return_value=None)),
        patch("surreal_memory.engine.retrieval.ReflexPipeline") as factory,
    ):
        factory.side_effect = assert_hybrid_pipeline
        with pytest.raises(PipelineReachedError):
            await server._recall({"query": "synthetic query", "depth": 1})
    factory.assert_called_once()


async def test_new_http_query_pipeline_reads_migrated_config() -> None:
    from surreal_memory.server.dependencies import get_brain
    from surreal_memory.server.models import QueryRequest
    from surreal_memory.server.routes.memory import query_memory

    store = Store()
    await migrate(store)
    storage = readback_store(store)
    brain = await get_brain(storage, brain_id="default")
    with patch(
        "surreal_memory.server.routes.memory.ReflexPipeline", side_effect=assert_hybrid_pipeline
    ) as factory:
        with pytest.raises(PipelineReachedError):
            await query_memory(QueryRequest(query="synthetic query"), brain, storage)
    factory.assert_called_once()


@pytest.mark.parametrize("proof", [None, True, "", "a" * 63, "g" * 64, "A" * 64])
async def test_config_hash_requires_canonical_sha256_before_writes(proof: object) -> None:
    store = Store()
    with pytest.raises(ActivationMigrationError, match="SHA-256"):
        await migrate(store, expected_config_sha256=proof)
    assert store.update_count == store.release_count == 0


async def test_server_type_drift_rejected_even_when_python_config_matches() -> None:
    store = Store()
    before = await store.get_activation_snapshot()
    store.typed_marker = "server NULL changed to NONE"
    assert (await store.get_activation_snapshot())["config"] == before["config"]
    with pytest.raises(ActivationMigrationError, match="changed"):
        await migrate(store, expected_config_sha256=before["config_sha256"])
    assert store.update_count == store.release_count == 0


async def test_server_type_drift_after_preflight_is_blocked_by_hash_cas() -> None:
    store = Store()
    store.typed_race = True
    with pytest.raises(ActivationMigrationError, match="readback failed"):
        await migrate(store)
    assert store.config["activation_strategy"] == "classic"
    assert store.update_count == store.release_count == 1

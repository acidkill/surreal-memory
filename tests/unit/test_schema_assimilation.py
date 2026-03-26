"""Tests for schema assimilation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.schema_assimilation import (
    AssimilationAction,
    AssimilationResult,
    _extract_shared_entities,
    assimilate_or_accommodate,
    batch_schema_assimilation,
)


def _make_neuron(content: str, tags: list[str] | None = None, ntype: NeuronType = NeuronType.CONCEPT) -> Neuron:
    return Neuron.create(
        content=content,
        type=ntype,
        metadata={"tags": tags or []},
    )


class TestAssimilationResult:
    def test_frozen(self) -> None:
        r = AssimilationResult(action=AssimilationAction.NO_SCHEMA)
        with pytest.raises(AttributeError):
            r.action = AssimilationAction.SKIPPED  # type: ignore[misc]


class TestExtractSharedEntities:
    def test_finds_capitalized_terms(self) -> None:
        contents = [
            "Django is a framework used by many teams",
            "Django supports advanced queries and indexing",
            "We switched to Django for better performance",
        ]
        shared = _extract_shared_entities(contents)
        assert "Django" in shared

    def test_empty_input(self) -> None:
        assert _extract_shared_entities([]) == []

    def test_no_shared_entities(self) -> None:
        contents = ["hello world", "foo bar"]
        shared = _extract_shared_entities(contents)
        assert shared == []


class TestAssimilateOrAccommodate:
    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = False
        neuron = _make_neuron("test", ["python"])
        storage = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.SKIPPED

    @pytest.mark.asyncio
    async def test_no_tags(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10
        neuron = _make_neuron("test", [])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.NO_SCHEMA

    @pytest.mark.asyncio
    async def test_too_few_domain_memories(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10
        neuron = _make_neuron("Python async patterns", ["python"])

        storage = AsyncMock()
        # No existing schemas
        storage.find_neurons = AsyncMock(side_effect=[
            [],  # schema search
            [_make_neuron(f"fact {i}", ["python"]) for i in range(5)],  # domain search
        ])

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.NO_SCHEMA

    @pytest.mark.asyncio
    async def test_schema_created(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10
        neuron = _make_neuron("Python async patterns", ["python"])

        domain_neurons = [_make_neuron(f"Python fact {i}", ["python"]) for i in range(12)]
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(side_effect=[
            [],  # no existing schemas
            domain_neurons,  # enough domain memories
        ])
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.SCHEMA_CREATED
        assert result.schema_id is not None
        assert result.version == 1
        storage.add_neuron.assert_called_once()

    @pytest.mark.asyncio
    async def test_assimilated(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10

        schema = _make_neuron("Schema: python patterns", ["python"], NeuronType.SCHEMA)
        schema_meta = {**schema.metadata, "schema_version": 1}
        from dataclasses import replace
        schema = replace(schema, metadata=schema_meta)

        neuron = _make_neuron("Python decorators are useful", ["python"])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[schema])
        storage.add_synapse = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.ASSIMILATED
        assert result.schema_id == schema.id
        storage.add_synapse.assert_called_once()

    @pytest.mark.asyncio
    async def test_accommodated(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 10

        schema = _make_neuron("Python is fast for development", ["python"], NeuronType.SCHEMA)
        schema_meta = {**schema.metadata, "schema_version": 1}
        from dataclasses import replace
        schema = replace(schema, metadata=schema_meta)

        # Contradictory memory
        neuron = _make_neuron("Python is not fast for development", ["python"])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[schema])
        storage.add_neuron = AsyncMock()
        storage.add_synapse = AsyncMock()

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.ACCOMMODATED
        assert result.version == 2
        storage.add_neuron.assert_called_once()  # new schema created

    @pytest.mark.asyncio
    async def test_small_brain_skipped(self) -> None:
        """With schema_min_cluster_size=200, small brains skip schema creation."""
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 200

        neuron = _make_neuron("test fact", ["python"])
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(side_effect=[
            [],  # no schemas
            [_make_neuron(f"fact {i}", ["python"]) for i in range(50)],
        ])

        result = await assimilate_or_accommodate(neuron, storage, config)
        assert result.action == AssimilationAction.NO_SCHEMA


class TestBatchSchemaAssimilation:
    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = False
        storage = AsyncMock()

        count = await batch_schema_assimilation(storage, config)
        assert count == 0

    @pytest.mark.asyncio
    async def test_dry_run(self) -> None:
        config = MagicMock()
        config.schema_assimilation_enabled = True
        config.schema_min_cluster_size = 3

        neurons = [_make_neuron(f"fact {i}", ["python"]) for i in range(5)]
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(side_effect=[
            [],  # no existing schemas
            neurons,  # all neurons
        ])

        count = await batch_schema_assimilation(storage, config, dry_run=True)
        assert count >= 1
        storage.add_neuron.assert_not_called()

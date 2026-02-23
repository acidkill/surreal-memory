"""Tests for FalkorDB neuron storage operations."""

from __future__ import annotations

import pytest

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage


class TestNeuronCRUD:
    """Basic neuron create/read/update/delete."""

    @pytest.mark.asyncio
    async def test_add_and_get_neuron(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        neuron = make_neuron(content="hello world")
        result_id = await storage.add_neuron(neuron)

        assert result_id == neuron.id
        retrieved = await storage.get_neuron(neuron.id)
        assert retrieved is not None
        assert retrieved.id == neuron.id
        assert retrieved.content == "hello world"
        assert retrieved.type == NeuronType.CONCEPT

    @pytest.mark.asyncio
    async def test_add_duplicate_raises(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        neuron = make_neuron(content="unique")
        await storage.add_neuron(neuron)

        with pytest.raises(ValueError, match="already exists"):
            await storage.add_neuron(neuron)

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(
        self, storage: FalkorDBStorage
    ) -> None:
        result = await storage.get_neuron("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_neurons_batch(
        self, storage: FalkorDBStorage, sample_neurons: list[Neuron]
    ) -> None:
        for n in sample_neurons:
            await storage.add_neuron(n)

        ids = [n.id for n in sample_neurons[:3]]
        batch = await storage.get_neurons_batch(ids)

        assert len(batch) == 3
        for nid in ids:
            assert nid in batch
            assert batch[nid].id == nid

    @pytest.mark.asyncio
    async def test_update_neuron(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        neuron = make_neuron(content="original")
        await storage.add_neuron(neuron)

        from dataclasses import replace

        updated = replace(neuron, content="modified")
        await storage.update_neuron(updated)

        retrieved = await storage.get_neuron(neuron.id)
        assert retrieved is not None
        assert retrieved.content == "modified"

    @pytest.mark.asyncio
    async def test_delete_neuron(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        neuron = make_neuron(content="to delete")
        await storage.add_neuron(neuron)

        deleted = await storage.delete_neuron(neuron.id)
        assert deleted is True

        result = await storage.get_neuron(neuron.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(
        self, storage: FalkorDBStorage
    ) -> None:
        deleted = await storage.delete_neuron("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_metadata_preserved(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        neuron = make_neuron(
            content="with meta",
            metadata={"key": "value", "nested": {"a": 1}},
        )
        await storage.add_neuron(neuron)

        retrieved = await storage.get_neuron(neuron.id)
        assert retrieved is not None
        assert retrieved.metadata["key"] == "value"
        assert retrieved.metadata["nested"]["a"] == 1


class TestNeuronSearch:
    """Neuron find/search operations."""

    @pytest.mark.asyncio
    async def test_find_by_type(
        self, storage: FalkorDBStorage, sample_neurons: list[Neuron]
    ) -> None:
        for n in sample_neurons:
            await storage.add_neuron(n)

        concepts = await storage.find_neurons(type=NeuronType.CONCEPT)
        assert len(concepts) == 1
        assert concepts[0].type == NeuronType.CONCEPT

    @pytest.mark.asyncio
    async def test_find_by_exact_content(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        n1 = make_neuron(content="exact match target")
        n2 = make_neuron(content="something else")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        results = await storage.find_neurons(content_exact="exact match target")
        assert len(results) == 1
        assert results[0].id == n1.id

    @pytest.mark.asyncio
    async def test_find_with_limit(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        for i in range(10):
            await storage.add_neuron(
                make_neuron(content=f"neuron {i}", type=NeuronType.ENTITY)
            )

        results = await storage.find_neurons(
            type=NeuronType.ENTITY, limit=3
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_find_neurons_exact_batch(
        self, storage: FalkorDBStorage, sample_neurons: list[Neuron]
    ) -> None:
        for n in sample_neurons:
            await storage.add_neuron(n)

        batch = await storage.find_neurons_exact_batch(
            contents=["Python language", "Write code"],
            type=None,
        )
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_suggest_neurons(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        await storage.add_neuron(make_neuron(content="Python programming"))
        await storage.add_neuron(make_neuron(content="Python testing"))
        await storage.add_neuron(make_neuron(content="Java programming"))

        suggestions = await storage.suggest_neurons(prefix="Python", limit=5)
        assert len(suggestions) >= 1
        for s in suggestions:
            assert "neuron_id" in s


class TestNeuronState:
    """Neuron activation state operations."""

    @pytest.mark.asyncio
    async def test_update_and_get_state(
        self, storage: FalkorDBStorage, make_neuron
    ) -> None:
        neuron = make_neuron(content="stateful neuron")
        await storage.add_neuron(neuron)

        state = NeuronState(
            neuron_id=neuron.id,
            activation_level=0.8,
            access_frequency=5,
        )
        await storage.update_neuron_state(state)

        retrieved = await storage.get_neuron_state(neuron.id)
        assert retrieved is not None
        assert retrieved.neuron_id == neuron.id
        assert retrieved.activation_level == pytest.approx(0.8, abs=0.01)
        assert retrieved.access_frequency == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_state_returns_none(
        self, storage: FalkorDBStorage
    ) -> None:
        result = await storage.get_neuron_state("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_states_batch(
        self, storage: FalkorDBStorage, sample_neurons: list[Neuron]
    ) -> None:
        for n in sample_neurons[:3]:
            await storage.add_neuron(n)
            state = NeuronState(neuron_id=n.id, activation_level=0.5)
            await storage.update_neuron_state(state)

        ids = [n.id for n in sample_neurons[:3]]
        batch = await storage.get_neuron_states_batch(ids)
        assert len(batch) == 3

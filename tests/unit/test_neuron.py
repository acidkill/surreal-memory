"""Unit tests for Neuron model."""

from __future__ import annotations

from datetime import datetime

import pytest

from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType


class TestNeuron:
    """Tests for Neuron class."""

    def test_create_neuron(self) -> None:
        """Test creating a neuron with factory method."""
        neuron = Neuron.create(
            type=NeuronType.ENTITY,
            content="Alice",
            metadata={"entity_type": "person"},
        )

        assert neuron.type == NeuronType.ENTITY
        assert neuron.content == "Alice"
        assert neuron.metadata == {"entity_type": "person"}
        assert neuron.id  # Should have generated ID
        assert isinstance(neuron.created_at, datetime)

    def test_create_neuron_with_explicit_id(self) -> None:
        """Test creating a neuron with explicit ID."""
        neuron = Neuron.create(
            type=NeuronType.TIME,
            content="3pm",
            neuron_id="custom-id",
        )

        assert neuron.id == "custom-id"

    def test_neuron_is_frozen(self) -> None:
        """Test that Neuron is immutable."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test")

        with pytest.raises(AttributeError):
            neuron.content = "new content"  # type: ignore

    def test_with_metadata(self) -> None:
        """Test creating new neuron with updated metadata."""
        original = Neuron.create(
            type=NeuronType.ENTITY,
            content="Alice",
            metadata={"role": "user"},
        )

        updated = original.with_metadata(status="active", priority=1)

        # Original unchanged
        assert original.metadata == {"role": "user"}

        # New neuron has merged metadata
        assert updated.metadata == {"role": "user", "status": "active", "priority": 1}
        assert updated.id == original.id
        assert updated.content == original.content

    def test_all_neuron_types(self) -> None:
        """Test all neuron types can be created."""
        for neuron_type in NeuronType:
            neuron = Neuron.create(type=neuron_type, content=f"test_{neuron_type}")
            assert neuron.type == neuron_type

    def test_embedding_text_plain_content_unaffected(self) -> None:
        """A neuron with no signature/docstring embeds exactly as before —
        every non-code memory neuron must be unaffected by this method."""
        neuron = Neuron.create(type=NeuronType.ENTITY, content="Alice")
        assert neuron.embedding_text() == "Alice"

    def test_embedding_text_includes_signature_and_docstring(self) -> None:
        """A code-symbol neuron composes content + signature + docstring —
        content alone (e.g. 'MyClass.my_method') is too sparse to embed well."""
        neuron = Neuron.create(
            type=NeuronType.ACTION,
            content="MyClass.my_method",
            metadata={
                "signature": "def my_method(self, x: int) -> str",
                "docstring": "Convert x to its string form.",
            },
        )
        text = neuron.embedding_text()
        assert "MyClass.my_method" in text
        assert "def my_method(self, x: int) -> str" in text
        assert "Convert x to its string form." in text

    def test_embedding_text_signature_only(self) -> None:
        """A symbol with a signature but no docstring still enriches the text."""
        neuron = Neuron.create(
            type=NeuronType.ACTION,
            content="greet",
            metadata={"signature": "def greet(name: str) -> str"},
        )
        text = neuron.embedding_text()
        assert text == "greet\ndef greet(name: str) -> str"

    def test_embedding_text_does_not_mutate_content(self) -> None:
        """content stays the dedup/identity key regardless of embedding_text()."""
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="Widget",
            metadata={"docstring": "A UI widget."},
        )
        neuron.embedding_text()
        assert neuron.content == "Widget"


class TestNeuronState:
    """Tests for NeuronState class."""

    def test_create_state(self) -> None:
        """Test creating neuron state."""
        state = NeuronState(neuron_id="test-1")

        assert state.neuron_id == "test-1"
        assert state.activation_level == 0.0
        assert state.access_frequency == 0
        assert state.last_activated is None
        assert state.decay_rate == 0.1

    def test_activate(self) -> None:
        """Test activating a neuron state (sigmoid-gated)."""
        state = NeuronState(neuron_id="test-1")
        activated = state.activate(level=0.8)

        # Original unchanged
        assert state.activation_level == 0.0
        assert state.access_frequency == 0

        # New state updated (sigmoid maps 0.8 → ~0.858)
        assert activated.activation_level > 0.8
        assert activated.activation_level < 0.9
        assert activated.access_frequency == 1
        assert activated.last_activated is not None

    def test_activate_clamps_level(self) -> None:
        """Test activation level is clamped before sigmoid."""
        state = NeuronState(neuron_id="test-1")

        # Input > 1.0 is clamped to 1.0, then sigmoid(1.0) ≈ 0.953
        high = state.activate(level=1.5)
        assert high.activation_level == pytest.approx(0.9526, abs=0.01)

        # Input < 0.0 is clamped to 0.0, then sigmoid(0.0) ≈ 0.047
        low = state.activate(level=-0.5)
        assert low.activation_level == pytest.approx(0.0474, abs=0.01)

    def test_decay(self) -> None:
        """Test decay over time."""
        state = NeuronState(neuron_id="test-1", activation_level=1.0, decay_rate=0.1)

        # Decay over 1 day
        decayed = state.decay(time_delta_seconds=86400)

        assert decayed.activation_level < 1.0
        assert decayed.activation_level > 0.0

    def test_is_active(self) -> None:
        """Test is_active property."""
        inactive = NeuronState(neuron_id="test-1", activation_level=0.05)
        active = NeuronState(neuron_id="test-2", activation_level=0.5)

        assert not inactive.is_active
        assert active.is_active

    def test_multiple_activations_increment_frequency(self) -> None:
        """Test that multiple activations increment frequency."""
        from datetime import timedelta

        state = NeuronState(neuron_id="test-1")
        t = datetime(2026, 1, 1)
        state = state.activate(0.5, now=t)
        # Wait past refractory period for subsequent activations
        t2 = t + timedelta(seconds=1)
        state = state.activate(0.7, now=t2)
        t3 = t2 + timedelta(seconds=1)
        state = state.activate(0.9, now=t3)

        assert state.access_frequency == 3

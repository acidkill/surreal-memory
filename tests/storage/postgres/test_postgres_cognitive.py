"""Tests for PostgreSQL cognitive layer (cognitive_state, hot_index, knowledge_gaps)."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_upsert_and_get_cognitive_state(storage: Any, sample_neurons: list[Any]) -> None:
    """Upsert a cognitive state and retrieve it."""
    for n in sample_neurons[:1]:
        await storage.add_neuron(n)
    nid = sample_neurons[0].id

    await storage.upsert_cognitive_state(
        nid,
        confidence=0.75,
        evidence_for_count=3,
        evidence_against_count=1,
        status="active",
    )

    state = await storage.get_cognitive_state(nid)
    assert state is not None
    assert state["neuron_id"] == nid
    assert state["confidence"] == pytest.approx(0.75)
    assert state["evidence_for_count"] == 3
    assert state["status"] == "active"


@pytest.mark.asyncio
async def test_upsert_cognitive_state_overwrites(storage: Any, sample_neurons: list[Any]) -> None:
    """Second upsert overwrites the first."""
    for n in sample_neurons[:1]:
        await storage.add_neuron(n)
    nid = sample_neurons[0].id

    await storage.upsert_cognitive_state(nid, confidence=0.5, status="active")
    await storage.upsert_cognitive_state(nid, confidence=0.9, status="confirmed")

    state = await storage.get_cognitive_state(nid)
    assert state is not None
    assert state["confidence"] == pytest.approx(0.9)
    assert state["status"] == "confirmed"


@pytest.mark.asyncio
async def test_list_cognitive_states(storage: Any, sample_neurons: list[Any]) -> None:
    """List cognitive states with optional status filter."""
    for n in sample_neurons[:3]:
        await storage.add_neuron(n)

    await storage.upsert_cognitive_state(sample_neurons[0].id, confidence=0.8, status="active")
    await storage.upsert_cognitive_state(sample_neurons[1].id, confidence=0.3, status="refuted")
    await storage.upsert_cognitive_state(sample_neurons[2].id, confidence=0.6, status="active")

    all_states = await storage.list_cognitive_states()
    assert len(all_states) == 3

    active_only = await storage.list_cognitive_states(status="active")
    assert len(active_only) == 2


@pytest.mark.asyncio
async def test_update_cognitive_evidence(storage: Any, sample_neurons: list[Any]) -> None:
    """Update only evidence fields without touching predicted_at."""
    for n in sample_neurons[:1]:
        await storage.add_neuron(n)
    nid = sample_neurons[0].id

    await storage.upsert_cognitive_state(
        nid, confidence=0.5, status="pending", predicted_at="2026-04-01T00:00:00"
    )
    await storage.update_cognitive_evidence(
        nid,
        confidence=0.8,
        evidence_for_count=5,
        evidence_against_count=1,
        status="active",
        last_evidence_at="2026-03-17T12:00:00",
    )

    state = await storage.get_cognitive_state(nid)
    assert state is not None
    assert state["confidence"] == pytest.approx(0.8)
    assert state["evidence_for_count"] == 5
    assert state["predicted_at"] == "2026-04-01T00:00:00"  # preserved


@pytest.mark.asyncio
async def test_list_predictions(storage: Any, sample_neurons: list[Any]) -> None:
    """List only cognitive states that are predictions."""
    for n in sample_neurons[:2]:
        await storage.add_neuron(n)

    await storage.upsert_cognitive_state(
        sample_neurons[0].id, confidence=0.6, status="pending", predicted_at="2026-04-01"
    )
    await storage.upsert_cognitive_state(
        sample_neurons[1].id,
        confidence=0.7,
        status="active",  # not a prediction
    )

    preds = await storage.list_predictions()
    assert len(preds) == 1
    assert preds[0]["neuron_id"] == sample_neurons[0].id


@pytest.mark.asyncio
async def test_calibration_stats(storage: Any, sample_neurons: list[Any]) -> None:
    """Get calibration stats for predictions."""
    for n in sample_neurons[:3]:
        await storage.add_neuron(n)

    await storage.upsert_cognitive_state(
        sample_neurons[0].id, confidence=0.9, status="confirmed", predicted_at="2026-03-01"
    )
    await storage.upsert_cognitive_state(
        sample_neurons[1].id, confidence=0.1, status="refuted", predicted_at="2026-03-02"
    )
    await storage.upsert_cognitive_state(
        sample_neurons[2].id, confidence=0.5, status="pending", predicted_at="2026-04-01"
    )

    stats = await storage.get_calibration_stats()
    assert stats["correct_count"] == 1
    assert stats["wrong_count"] == 1
    assert stats["total_resolved"] == 2
    assert stats["pending_count"] == 1


@pytest.mark.asyncio
async def test_hot_index_refresh_and_get(storage: Any) -> None:
    """Refresh hot index and retrieve items."""
    items = [
        {
            "slot": 0,
            "category": "hypothesis",
            "neuron_id": "n1",
            "summary": "Test H1",
            "confidence": 0.8,
            "score": 9.5,
        },
        {
            "slot": 1,
            "category": "prediction",
            "neuron_id": "n2",
            "summary": "Test P1",
            "confidence": 0.6,
            "score": 7.0,
        },
    ]

    count = await storage.refresh_hot_index(items)
    assert count == 2

    hot = await storage.get_hot_index(limit=10)
    assert len(hot) == 2
    assert hot[0]["score"] > hot[1]["score"]  # sorted by score desc


@pytest.mark.asyncio
async def test_knowledge_gap_lifecycle(storage: Any) -> None:
    """Add, list, get, and resolve a knowledge gap."""
    gap_id = await storage.add_knowledge_gap(
        topic="Why does latency spike at 3am?",
        detection_source="recall_miss",
        priority=0.8,
        related_neuron_ids=["n1", "n2"],
    )
    assert gap_id

    gaps = await storage.list_knowledge_gaps()
    assert len(gaps) == 1
    assert gaps[0]["topic"] == "Why does latency spike at 3am?"
    assert gaps[0]["related_neuron_ids"] == ["n1", "n2"]

    gap = await storage.get_knowledge_gap(gap_id)
    assert gap is not None
    assert gap["priority"] == pytest.approx(0.8)

    resolved = await storage.resolve_knowledge_gap(gap_id, resolved_by_neuron_id="n3")
    assert resolved is True

    unresolved = await storage.list_knowledge_gaps(include_resolved=False)
    assert len(unresolved) == 0

    all_gaps = await storage.list_knowledge_gaps(include_resolved=True)
    assert len(all_gaps) == 1
    assert all_gaps[0]["resolved_at"] is not None


@pytest.mark.asyncio
async def test_schema_history(storage: Any, sample_neurons: list[Any]) -> None:
    """Walk version chain via parent_schema_id."""
    for n in sample_neurons[:2]:
        await storage.add_neuron(n)

    # v1
    await storage.upsert_cognitive_state(
        sample_neurons[0].id, confidence=0.5, status="active", schema_version=1
    )
    # v2 supersedes v1
    await storage.upsert_cognitive_state(
        sample_neurons[1].id,
        confidence=0.8,
        status="active",
        schema_version=2,
        parent_schema_id=sample_neurons[0].id,
    )

    history = await storage.get_schema_history(sample_neurons[1].id)
    assert len(history) == 2
    assert history[0]["neuron_id"] == sample_neurons[1].id
    assert history[1]["neuron_id"] == sample_neurons[0].id

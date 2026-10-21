"""Validation for recognizing an already committed compression intent."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.compression import CompressionEngine, CompressionTier
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


def _committed_intent() -> tuple[Fiber, dict[str, object], dict[str, object]]:
    fiber = Fiber(
        id="fiber-test",
        neuron_ids={"anchor", "secondary"},
        synapse_ids=set(),
        anchor_neuron_id="anchor",
        compression_tier=int(CompressionTier.EXTRACTIVE),
    )
    intent: dict[str, object] = {
        "version": 1,
        "run_id": "run-test",
        "target_tier": int(CompressionTier.EXTRACTIVE),
        "original_tier": int(CompressionTier.FULL),
        "anchor_id": "anchor",
        "compressed_content": "prepared summary",
        "original_token_count": 20,
        "compressed_token_count": 5,
        "entities_preserved": 2,
        "backup_created": True,
        "source_hashes": [
            {"id": neuron_id, "hash": hashlib.sha256(neuron_id.encode()).hexdigest()}
            for neuron_id in sorted(fiber.neuron_ids)
        ],
    }
    receipt: dict[str, object] = {
        "run_id": "run-test",
        "target_tier": int(CompressionTier.EXTRACTIVE),
        "tokens_saved": 15,
    }
    return fiber, intent, receipt


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), True),
        (("version",), "1"),
        (("target_tier",), True),
        (("target_tier",), "1"),
        (("target_tier",), -1),
        (("target_tier",), 5),
        (("run_id",), ""),
        (("source_hashes",), {"anchor": "a" * 64}),
        (("source_hashes", 0), {"id": "anchor", "hash": "short"}),
        (("anchor_id",), "different-anchor"),
    ],
)
def test_malformed_pending_intent_cannot_use_receipt_fastpath(
    path: tuple[object, ...], value: object
) -> None:
    fiber, intent, receipt = _committed_intent()
    intent = copy.deepcopy(intent)
    target: object = intent
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    assert not CompressionEngine._intent_has_matching_receipt(fiber, intent, receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "another-run"),
        ("target_tier", True),
        ("target_tier", "1"),
        ("tokens_saved", -1),
        ("tokens_saved", True),
        ("tokens_saved", 14),
    ],
)
def test_malformed_receipt_cannot_use_fastpath(field: str, value: object) -> None:
    fiber, intent, receipt = _committed_intent()
    receipt[field] = value

    assert not CompressionEngine._intent_has_matching_receipt(fiber, intent, receipt)


def test_matching_receipt_requires_completed_tier_and_exact_source_membership() -> None:
    fiber, intent, receipt = _committed_intent()
    assert CompressionEngine._intent_has_matching_receipt(fiber, intent, receipt)

    before_tier = Fiber(
        id=fiber.id,
        neuron_ids=fiber.neuron_ids,
        synapse_ids=fiber.synapse_ids,
        anchor_neuron_id=fiber.anchor_neuron_id,
        compression_tier=int(CompressionTier.FULL),
    )
    assert not CompressionEngine._intent_has_matching_receipt(before_tier, intent, receipt)

    extra_source = copy.deepcopy(intent)
    extra_source["source_hashes"].append(
        {"id": "unrelated", "hash": hashlib.sha256(b"unrelated").hexdigest()}
    )
    assert not CompressionEngine._intent_has_matching_receipt(fiber, extra_source, receipt)


@pytest.mark.asyncio
async def test_mismatched_receipt_does_not_clear_intent_after_neuron_edit() -> None:
    store = InMemoryStorage()
    brain = Brain.create(name="compression-checkpoint-test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    neuron = Neuron.create(type=NeuronType.CONCEPT, content="original source content")
    await store.add_neuron(neuron)
    fiber = Fiber(
        id="fiber-mismatched-receipt",
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        compression_tier=int(CompressionTier.EXTRACTIVE),
        metadata={},
        created_at=utcnow(),
    )
    intent = {
        "version": 1,
        "run_id": "run-mismatched-receipt",
        "target_tier": int(CompressionTier.EXTRACTIVE),
        "original_tier": int(CompressionTier.FULL),
        "anchor_id": neuron.id,
        "compressed_content": "prepared summary",
        "original_token_count": 20,
        "compressed_token_count": 5,
        "entities_preserved": 1,
        "backup_created": True,
        "source_hashes": [
            {"id": neuron.id, "hash": hashlib.sha256(neuron.content.encode()).hexdigest()}
        ],
    }
    fiber = replace(
        fiber,
        metadata={
            "_compression_pending": intent,
            "_compression_receipt": {
                "run_id": "run-mismatched-receipt",
                "target_tier": int(CompressionTier.EXTRACTIVE),
                "tokens_saved": 14,
            },
        },
    )
    await store.add_fiber(fiber)
    changed = replace(neuron, content="a later user edit")
    await store.update_neuron(changed)

    with pytest.raises(RuntimeError, match="changed during compression replay"):
        await CompressionEngine(store).compress_fiber(fiber, CompressionTier.EXTRACTIVE)

    saved = await store.get_fiber(fiber.id)
    assert saved is not None
    assert saved.metadata["_compression_pending"] == intent
    assert (await store.get_neuron(neuron.id)).content == "a later user edit"

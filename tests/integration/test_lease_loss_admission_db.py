"""One-shot admission followed by a real CLI resume on an attested disposable database."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from surreal_memory.engine import consolidation_receipts as receipts
from surreal_memory.engine.consolidation import ConsolidationConfig, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressSession,
    options_fingerprint,
)
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow
from tests._disposable_surrealdb import load_target, subprocess_env
from tests.unit._surrealdb_live import ensure_real_surrealdb_sdk
from tests.unit.test_capacity_stop_admission import (
    synthetic_runtime_versions as synthetic_runtime_versions,
)
from tests.unit.test_cluster_consolidation_hourly import hourly
from tests.unit.test_lease_loss_admission import operator

pytestmark = pytest.mark.integration


async def test_admit_then_real_cli_resumes_same_running_checkpoint(tmp_path, monkeypatch):
    target = load_target()
    if target is None:
        pytest.skip("requires registered disposable SurrealDB target")
    ensure_real_surrealdb_sdk()
    target = target.child()
    env = subprocess_env(target, tmp_path / "cli")

    def cli(*args):
        return subprocess.run(  # noqa: S603 - fixed CLI wrapped by verified disposable target
            [sys.executable, "-m", "tests._disposable_surrealdb", "cli", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

    created = await asyncio.to_thread(
        cli, "remember", "Synthetic admission marker", "--type", "fact", "--json"
    )
    assert created.returncode == 0, (created.stdout, created.stderr)
    storage = SurrealDBStorage(**target.storage_kwargs(), database="tests")
    await storage.initialize()
    storage.set_brain("default")
    try:
        strategies = sorted(
            s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL
        )
        config = ConsolidationConfig(
            prune_weight_threshold=0.05, prune_min_inactive_days=7.0, merge_overlap_threshold=0.5
        )
        fingerprint = options_fingerprint(
            {
                "requested_strategies": strategies,
                "config": asdict(config),
                "dream_decay_multiplier": 10.0,
                "tier_config": None,
            }
        )
        session = await ConsolidationProgressSession.open(
            storage, strategies, fingerprint, utcnow()
        )
        assert session is not None
        try:
            for name in (
                "dedup",
                "learn_habits",
                "process_reasoning_traces",
                "process_tool_events",
            ):
                await session.complete_strategy(name)
            await session.checkpoint(
                "prune",
                "synapse_pending",
                pending=[str(uuid4())],
                counters={
                    "synapses_pruned": 0,
                    "synapse_pages": 1,
                    "neurons_pruned": 0,
                    "neuron_pages": 0,
                    "prune_atomic_journal": 1,
                },
            )
        finally:
            await session.close()
        cp = await storage.get_consolidation_progress("default")
        assert cp and cp["status"] == "running"
        before = copy.deepcopy(cp)
        api, _, safety, args, cm, _, approval = operator(tmp_path)
        previous = json.loads(cm["data"]["receipt.json"])
        previous["before_run_id"] = cp["run_id"]
        raw = receipts.canonical(previous)
        cm["data"]["receipt.json"] = raw
        brain = await storage.get_brain("default")
        assert brain is not None
        approval.update(
            checkpoint_sha256=receipts.checkpoint_hash(cp),
            previous_receipt_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            run_id=cp["run_id"],
            options_fingerprint=cp["options_fingerprint"],
            brain_id=brain.id,
            brain_config_sha256=receipts.checkpoint_hash(asdict(brain.config)),
        )
        Path(args.approval_file).write_text(json.dumps(approval))
        assert await hourly.admit_lease_loss_resume(api, storage, safety, args) == 0
        assert await storage.get_consolidation_progress("default") == before
        admitted = json.loads(cm["data"]["receipt.json"])
        command = receipts.select_command(admitted, before, None)
        assert command == ["smem", "consolidate", "--brain", "default", "--resume"]
        popen = subprocess.Popen
        launched = []

        def launch_real_cli(command, **kwargs):
            launched.append(command)
            return popen(
                [sys.executable, "-m", "tests._disposable_surrealdb", "cli", *command[1:]],
                env=env,
                **kwargs,
            )

        with monkeypatch.context() as context:
            context.setattr(hourly.subprocess, "Popen", launch_real_cli)
            assert await asyncio.wait_for(hourly.run(api, storage, safety), timeout=105) == 0
        assert len(launched) == 1 and launched[0][:5] == command
        terminal = json.loads(cm["data"]["receipt.json"])
        assert terminal["cli_rc"] == 0 and terminal["status"] == "completed"
        final = await storage.get_consolidation_progress("default")
        assert (
            final is not None and final["run_id"] == cp["run_id"] and final["status"] == "completed"
        )
        assert set(final["completed_strategies"]) == set(strategies)
        assert cm["data"][
            "lease_loss_resume_" + receipts.checkpoint_hash(admitted) + ".json"
        ] == receipts.canonical(admitted)
    finally:
        try:
            await storage._query_response("REMOVE DATABASE tests")
        finally:
            await storage.close()

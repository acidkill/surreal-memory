"""Real CLI and MCP remember writes share the verified disposable target guard."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from tests._disposable_surrealdb import TARGET_ENV, load_target, subprocess_env
from tests.unit._surrealdb_live import ensure_real_surrealdb_sdk

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv(TARGET_ENV), reason="requires registered disposable SurrealDB"
    ),
]


async def test_parallel_cli_and_mcp_remember_use_distinct_disposable_namespaces(
    tmp_path: Path,
) -> None:
    target = load_target()
    assert target is not None
    ensure_real_surrealdb_sdk()
    cli_target, mcp_target = target.child(), target.child()
    assert cli_target.namespace != mcp_target.namespace
    cli_env = subprocess_env(cli_target, tmp_path / "cli")
    mcp_env = subprocess_env(mcp_target, tmp_path / "mcp")
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "isolation-test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "smem_remember",
                "arguments": {"content": "RP1 synthetic MCP transport marker", "type": "fact"},
            },
        },
    ]

    def run(kind: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        args = [sys.executable, "-m", "tests._disposable_surrealdb", kind]
        if kind == "cli":
            args += ["remember", "RP1 synthetic CLI transport marker", "--type", "fact", "--json"]
        return subprocess.run(  # noqa: S603 -- fixed acceptance commands, no shell
            args,
            env=env,
            input=""
            if kind == "cli"
            else "".join(json.dumps(message) + "\n" for message in messages),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

    cli, mcp = await asyncio.gather(
        asyncio.to_thread(run, "cli", cli_env), asyncio.to_thread(run, "mcp", mcp_env)
    )
    assert cli.returncode == 0, cli.stderr
    assert mcp.returncode == 0, mcp.stderr
    assert "fiber_id" in json.loads(cli.stdout), cli.stdout
    responses = [json.loads(line) for line in mcp.stdout.splitlines() if line.strip()]
    response = next(item for item in responses if item.get("id") == 2)
    assert "error" not in response, response
    assert not response["result"].get("isError"), response
    assert "fiber_id" in response["result"]["content"][0]["text"], response

    for destination, own_marker, other_marker in (
        (cli_target, "RP1 synthetic CLI transport marker", "RP1 synthetic MCP transport marker"),
        (mcp_target, "RP1 synthetic MCP transport marker", "RP1 synthetic CLI transport marker"),
    ):
        storage = SurrealDBStorage(**destination.storage_kwargs(), database="tests")
        await storage.initialize()
        try:
            rows = await storage._query("SELECT content FROM neuron")
            contents = {row.get("content") for row in rows}
            assert own_marker in contents
            assert other_marker not in contents
        finally:
            await storage.close()

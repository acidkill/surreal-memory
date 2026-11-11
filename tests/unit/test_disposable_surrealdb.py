"""Fail before app networking, including localhost-forwarded production targets."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests import _disposable_surrealdb as isolation


@pytest.fixture
def container() -> dict:
    return {
        "Id": "a" * 64,
        "Config": {
            "Image": "surrealdb/surrealdb:v3.2.4",
            "Cmd": [
                "start",
                "memory",
                "--user",
                "root",
                "--pass",
                "root",
                "--bind",
                "0.0.0.0:8001",
                "--allow-experimental",
                "gql",
                "--allow-eval-query",
            ],
            "Labels": {isolation.LABEL: "b" * 32},
        },
        "State": {"Running": True},
        "Mounts": [],
        "NetworkSettings": {"Ports": {"8001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18001"}]}},
    }


@pytest.fixture
def registered(container: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "target.json"
    path.write_text(json.dumps(isolation._identity(container)))
    monkeypatch.setattr(isolation, "_inspect", Mock(return_value=container))
    return {isolation.TARGET_ENV: str(path)}


@pytest.mark.parametrize(
    "field",
    [
        "SURREALDB_URL",
        "SURREALDB_USER",
        "SURREALDB_PASS",
        "SURREALDB_NS",
        "SURREALDB_DB",
        "SURREALDB_AUTH_LEVEL",
    ],
)
def test_inherited_connection_without_proof_fails_before_inspect(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspect = Mock(side_effect=AssertionError("must not inspect or connect"))
    monkeypatch.setattr(isolation, "_inspect", inspect)
    with pytest.raises(isolation.UnsafeTestTargetError, match="proof"):
        isolation.load_target({field: "synthetic-production-setting"})
    inspect.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        *isolation._URL_KEYS,
        *isolation._USER_KEYS,
        *isolation._PASS_KEYS,
        "SURREALDB_NS",
        "SURREALDB_DB",
        "SURREALDB_AUTH_LEVEL",
    ],
)
def test_inherited_mismatch_fails_before_inspect(field: str, registered: dict[str, str]) -> None:
    with pytest.raises(isolation.UnsafeTestTargetError, match=field):
        isolation.load_target({**registered, field: "synthetic-production-setting"})
    isolation._inspect.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "volume",
        "disk",
        "unlabelled",
        "not_running",
        "public_port",
        "wrong_credentials",
        "wrong_image",
    ],
)
def test_loopback_alone_does_not_prove_disposable_instance(container: dict, change: str) -> None:
    info = copy.deepcopy(container)
    if change == "volume":
        info["Mounts"] = [{"Source": "/synthetic/production"}]
    elif change == "disk":
        info["Config"]["Cmd"][1] = "rocksdb:/data"
    elif change == "unlabelled":
        info["Config"]["Labels"] = {}
    elif change == "not_running":
        info["State"]["Running"] = False
    elif change == "public_port":
        info["NetworkSettings"]["Ports"]["8001/tcp"][0]["HostIp"] = "0.0.0.0"  # noqa: S104 -- rejection fixture
    elif change == "wrong_credentials":
        info["Config"]["Cmd"][5] = "synthetic-production-password"
    elif change == "wrong_image":
        info["Config"]["Image"] = "other/image:latest"
    with pytest.raises(isolation.UnsafeTestTargetError, match="disposable"):
        isolation._identity(info)


def test_replaced_container_identity_is_rejected(
    registered: dict[str, str], container: dict
) -> None:
    container["Id"] = "c" * 64
    with pytest.raises(isolation.UnsafeTestTargetError, match="identity changed"):
        isolation.load_target(registered)


def test_parallel_children_have_distinct_namespaces(registered: dict[str, str]) -> None:
    target = isolation.load_target(registered)
    assert target is not None
    first, second = target.child(), target.child()
    assert first.namespace != second.namespace
    assert first.run_id == second.run_id == target.run_id


def test_subprocess_environment_discards_inherited_config(
    registered: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = isolation.load_target(registered)
    assert target is not None
    monkeypatch.setenv(isolation.TARGET_ENV, registered[isolation.TARGET_ENV])
    for key in (
        "SURREALDB_URL",
        "SURREALDB_PASS",
        "SURREALDB_AUTH_LEVEL",
        "SURREAL_MEMORY_DIR",
        "SURREAL_MEMORY_BRAIN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(key, "synthetic-production-setting")
    env = isolation.subprocess_env(target.child(), tmp_path / "child")
    assert env["SURREALDB_URL"] == target.url
    assert env["SURREALDB_AUTH_LEVEL"] == "root"
    assert env["HOME"] == str(tmp_path / "child")
    assert env["SURREAL_MEMORY_DIR"] == str(tmp_path / "child" / ".surrealmemory")
    assert "SURREAL_MEMORY_BRAIN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "synthetic-production-setting" not in env.values()


def test_process_network_guard_rejects_before_socket_connect() -> None:
    code = """
import socket
from tests._disposable_surrealdb import install_network_guard, UnsafeTestTargetError
install_network_guard(None)
for address in [("127.0.0.1", 8000), ("192.0.2.1", 8000)]:
    with socket.socket() as connection:
        try:
            connection.connect(address)
        except UnsafeTestTargetError:
            continue
        raise AssertionError("unsafe socket connect was allowed")
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(  # noqa: S603 -- fixed self-check, no shell
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_test_listener_allowance_is_removed_on_exit() -> None:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        with isolation.allow_test_listener(listener):
            with socket.create_connection(address, timeout=1):
                pass
        with pytest.raises(isolation.UnsafeTestTargetError):
            socket.create_connection(address, timeout=1)


def test_non_listening_socket_is_not_a_fixture_allowance() -> None:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        with pytest.raises(isolation.UnsafeTestTargetError, match="listening"):
            with isolation.allow_test_listener(listener):
                pass

"""One fail-closed target contract for pytest and CLI/MCP acceptance children.

Register only a labelled, memory-backed Docker instance without persistent mounts. A localhost
URL, test brain name or private HOME alone cannot authorize a database connection.
"""

from __future__ import annotations

import json
import os
import re
import runpy
import socket
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

LABEL = "io.surreal-memory.test-run"
TARGET_ENV = "SMEM_TEST_SURREALDB_TARGET_FILE"
_TASK_ID = uuid.uuid4().hex
_LOCAL_LISTENERS: set[tuple[str, int]] = set()
_URL_KEYS = (
    "SURREALDB_URL",
    "SMEM_TEST_SURREALDB_URL",
    "SURREALDB_TEST_URL",
    "SURREALDB_CHANGE_LOG_COLLAPSE_TEST_URL",
)
_USER_KEYS = ("SURREALDB_USER", "SURREALDB_TEST_USER", "SURREALDB_CHANGE_LOG_COLLAPSE_TEST_USER")
_PASS_KEYS = ("SURREALDB_PASS", "SURREALDB_TEST_PASS", "SURREALDB_CHANGE_LOG_COLLAPSE_TEST_PASS")


class UnsafeTestTargetError(RuntimeError):
    """A test target lacks disposable-instance proof; no app connection is allowed."""


def _inspect(container: str) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603 -- fixed Docker verb, no shell
        ["docker", "inspect", container],  # noqa: S607 -- Docker CLI is required for instance proof
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)[0]


def _identity(info: dict[str, Any]) -> dict[str, str]:
    """Verify actual process, image, credentials and published loopback endpoint."""
    config = info.get("Config", {})
    command = config.get("Cmd") or []
    labels = config.get("Labels") or {}
    run_id = labels.get(LABEL, "")
    image = config.get("Image", "")
    ports = info.get("NetworkSettings", {}).get("Ports", {}).get("8001/tcp") or []
    safe = (
        re.fullmatch(r"[0-9a-f]{32}", run_id)
        and re.fullmatch(r"[0-9a-f]{64}", info.get("Id", ""))
        and re.fullmatch(r"surrealdb/surrealdb:v3\.2\.(0|4)", image)
        and info.get("State", {}).get("Running") is True
        and all(
            mount.get("Type") == "tmpfs" and mount.get("Destination") in {"/data", "/logs"}
            for mount in info.get("Mounts", [])
        )
        and command
        == [
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
        ]
        and len(ports) == 1
        and ports[0].get("HostIp") == "127.0.0.1"
        and str(ports[0].get("HostPort", "")).isdigit()
    )
    if not safe:
        raise UnsafeTestTargetError(
            "requires a labelled disposable SurrealDB container without persistent mounts"
        )
    return {
        "container_id": info["Id"],
        "run_id": run_id,
        "url": "http://127.0.0.1:" + ports[0]["HostPort"],
    }


@dataclass(frozen=True)
class DisposableTarget:
    url: str
    run_id: str
    namespace: str

    def storage_kwargs(self) -> dict[str, str]:
        return {
            "url": self.url,
            "user": "root",
            "password": "root",
            "namespace": self.namespace,
        }

    def environment(self) -> dict[str, str]:
        result = dict.fromkeys(_URL_KEYS, self.url)
        result.update(dict.fromkeys(_USER_KEYS, "root"))
        result.update(dict.fromkeys(_PASS_KEYS, "root"))
        result.update(
            SURREALDB_NS=self.namespace, SURREALDB_DB="tests", SURREALDB_AUTH_LEVEL="root"
        )
        return result

    def child(self) -> DisposableTarget:
        return replace(self, namespace=f"smem_test_{self.run_id}_{uuid.uuid4().hex}")


def load_target(env: Mapping[str, str] | None = None) -> DisposableTarget | None:
    """Reject inherited remote settings before inspecting or connecting to a DB."""
    env = os.environ if env is None else env
    path = env.get(TARGET_ENV)
    connection_keys = (
        *_URL_KEYS,
        *_USER_KEYS,
        *_PASS_KEYS,
        "SURREALDB_NS",
        "SURREALDB_DB",
        "SURREALDB_AUTH_LEVEL",
    )
    if not path:
        if any(env.get(key) for key in connection_keys):
            raise UnsafeTestTargetError(
                "inherited connection settings require disposable-instance proof"
            )
        return None
    try:
        descriptor = json.loads(Path(path).read_text())
        run_id = descriptor["run_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("invalid run id")
        url = descriptor["url"]
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise ValueError("invalid disposable endpoint")
        target = DisposableTarget(url, run_id, f"smem_test_{run_id}_{_TASK_ID}")
        expected = target.environment()
        for key in connection_keys:
            value = env.get(key)
            if value is None:
                continue
            if key == "SURREALDB_NS":
                if re.fullmatch(rf"smem_test_{run_id}_[0-9a-f]{{32}}", value):
                    target = replace(target, namespace=value)
                    continue
            elif value == expected[key]:
                continue
            raise UnsafeTestTargetError(f"inherited {key} disagrees with disposable target")
        if _identity(_inspect(descriptor["container_id"])) != descriptor:
            raise UnsafeTestTargetError("disposable container identity changed")
        return target
    except UnsafeTestTargetError:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise UnsafeTestTargetError("cannot verify disposable container identity") from exc


@contextmanager
def allow_test_listener(listener: socket.socket) -> Iterator[None]:
    """Allow only a fixture-owned listening loopback socket for its lifetime."""
    address = listener.getsockname()
    if (
        address[0] != "127.0.0.1"
        or not address[1]
        or not listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
    ):
        raise UnsafeTestTargetError("requires a fixture-owned listening loopback socket")
    endpoint = address[:2]
    _LOCAL_LISTENERS.add(endpoint)
    try:
        yield
    finally:
        _LOCAL_LISTENERS.remove(endpoint)


def install_network_guard(target: DisposableTarget | None) -> None:
    """Audit before socket connect/DNS; SDK reloads cannot remove this safety net."""
    parsed = urlsplit(target.url) if target else None
    allowed = ("127.0.0.1", parsed.port) if parsed else None

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            address = args[1]
            if not isinstance(address, tuple) or (
                address[:2] != allowed and address[:2] not in _LOCAL_LISTENERS
            ):
                raise UnsafeTestTargetError(
                    "test network connection is outside the disposable target"
                )
        elif (
            event == "socket.getaddrinfo"
            and (args[0], args[1]) != allowed
            and (args[0], args[1]) not in _LOCAL_LISTENERS
        ):
            raise UnsafeTestTargetError("test DNS lookup is outside the disposable target")

    sys.addaudithook(guard)


def subprocess_env(target: DisposableTarget, directory: Path) -> dict[str, str]:
    """Remove inherited app/config/provider settings, then install only this target."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SURREALDB_", "SURREAL_MEMORY_", "SMEM_", "NEURAL_MEMORY_"))
        and key not in {"HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"}
        and not key.endswith(("_API_KEY", "_API_BASE", "_BASE_URL"))
    }
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / ".surrealmemory"
    config.mkdir(exist_ok=True)
    (config / "config.toml").write_text('storage_backend = "surrealdb"\n')
    env.update(target.environment())
    env.update(
        HOME=str(directory),
        XDG_CONFIG_HOME=str(directory / "config"),
        XDG_DATA_HOME=str(directory / "data"),
        XDG_CACHE_HOME=str(directory / "cache"),
        SURREAL_MEMORY_DIR=str(config),
        SURREAL_MEMORY_NO_UPDATE_CHECK="1",
        PYTHONPATH=os.pathsep.join(
            (
                str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1]),
            )
        ),
    )
    env[TARGET_ENV] = os.environ[TARGET_ENV]
    return env


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "register":
        descriptor = _identity(_inspect(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(descriptor))
        return
    if len(sys.argv) < 2 or sys.argv[1] not in {"cli", "mcp"}:
        raise SystemExit("use register CONTAINER MANIFEST, cli ARGS, or mcp")
    target = load_target()
    if target is None:
        raise UnsafeTestTargetError("acceptance child requires a registered disposable container")
    os.environ.update(target.environment())
    install_network_guard(target)
    module = "surreal_memory." + sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()

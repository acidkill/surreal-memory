"""The dead-module guard has to catch what ruff cannot, without false alarms.

Both directions matter. A guard that misses test-only modules is pointless; one
that flags live code gets switched off within a week. The cases below are the
import shapes this codebase actually uses.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dead_modules.py"


@pytest.fixture(scope="module")
def guard():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("check_dead_modules", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_dead_modules"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("check_dead_modules", None)


def _imports(guard, source: str, module_path: str = "surreal_memory/thing.py") -> set[str]:  # type: ignore[no-untyped-def]
    path = guard.SRC / module_path
    return guard._imports_in(path, ast.parse(source))


class TestImportDetection:
    def test_plain_import(self, guard) -> None:  # type: ignore[no-untyped-def]
        assert "surreal_memory.engine.retrieval" in _imports(
            guard, "import surreal_memory.engine.retrieval"
        )

    def test_from_import_of_a_submodule(self, guard) -> None:  # type: ignore[no-untyped-def]
        assert "surreal_memory.storage.base" in _imports(
            guard, "from surreal_memory.storage import base"
        )

    def test_import_nested_in_a_function(self, guard) -> None:  # type: ignore[no-untyped-def]
        # Optional dependencies are loaded this way throughout the codebase, so
        # a guard that only looked at module level would call them all dead.
        source = "def load():\n    from surreal_memory.adapters import langchain\n"

        assert "surreal_memory.adapters.langchain" in _imports(guard, source)

    def test_relative_import(self, guard) -> None:  # type: ignore[no-untyped-def]
        found = _imports(
            guard,
            "from .store import SurrealDBStorage",
            module_path="surreal_memory/storage/surrealdb/typed_memory.py",
        )

        assert "surreal_memory.storage.surrealdb.store" in found

    def test_unrelated_imports_are_ignored(self, guard) -> None:  # type: ignore[no-untyped-def]
        assert _imports(guard, "import json\nfrom pathlib import Path") == set()


class TestStringReferences:
    def test_a_dotted_target_in_a_literal_counts(self, guard) -> None:  # type: ignore[no-untyped-def]
        # How the server is actually started: uvicorn.run("...:create_app").
        tree = ast.parse('uvicorn.run("surreal_memory.server.app:create_app")')

        assert "surreal_memory.server.app.create_app" in guard._string_references({Path("x"): tree})

    def test_an_import_statement_is_not_a_string_reference(self, guard) -> None:  # type: ignore[no-untyped-def]
        # Scanning raw file text instead of literals would match the import line
        # itself, marking every imported module as externally referenced and
        # silently disabling the whole check.
        tree = ast.parse("from surreal_memory.storage.factory import create_storage")

        assert guard._string_references({Path("x"): tree}) == set()


class TestConsoleScripts:
    def test_entry_points_come_from_pyproject(self, guard) -> None:  # type: ignore[no-untyped-def]
        modules = guard._console_script_modules()

        assert "surreal_memory.cli" in modules
        assert "surreal_memory.hooks.pre_compact" in modules


class TestAgainstTheRealTree:
    def test_the_package_has_no_unreachable_modules(self, guard) -> None:  # type: ignore[no-untyped-def]
        dead = guard.find_dead_modules()

        assert dead == [], f"unreachable modules: {dead}"

    def test_router_modules_are_not_false_positives(self, guard) -> None:  # type: ignore[no-untyped-def]
        # Route modules are reached only through their package __init__, which
        # app.py imports. A one-hop rule reports all of them; reachability does
        # not.
        dead = guard.find_dead_modules()

        assert not [name for name in dead if ".server.routes." in name]

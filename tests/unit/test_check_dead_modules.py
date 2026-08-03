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


class TestBrokenImportDetection:
    """A root that names a module the tree no longer has must not vanish silently.

    This is the defect an import of the since-removed ``sqlite_store`` module
    exposed: the graph walk simply never matched it, so the guard reported
    ``No unreachable modules.`` while five files that could not run defined
    part of its reachability set.
    """

    def test_resolves_true_for_a_real_module(self, guard) -> None:  # type: ignore[no-untyped-def]
        assert guard._resolves("surreal_memory.storage.base") is True

    def test_resolves_true_for_a_real_package(self, guard) -> None:  # type: ignore[no-untyped-def]
        assert guard._resolves("surreal_memory.storage") is True

    def test_resolves_false_for_a_module_that_was_removed(self, guard) -> None:  # type: ignore[no-untyped-def]
        assert guard._resolves("surreal_memory.storage.sqlite_store") is False

    def test_resolves_true_for_non_package_names(self, guard) -> None:  # type: ignore[no-untyped-def]
        # _resolves only judges surreal_memory.* names; anything else is out
        # of scope for this check and must not be reported as broken.
        assert guard._resolves("json") is True

    def test_a_root_naming_a_missing_module_is_flagged(self, guard) -> None:  # type: ignore[no-untyped-def]
        modules, _ = guard._scan_imports(
            guard.SRC / "thing.py",
            ast.parse("from surreal_memory.storage.sqlite_store import SQLiteStorage"),
        )

        assert modules == {"surreal_memory.storage.sqlite_store"}
        assert not guard._resolves("surreal_memory.storage.sqlite_store")

    def test_attribute_only_names_are_never_checked(self, guard) -> None:  # type: ignore[no-untyped-def]
        # `from surreal_memory.storage import NeuralStorage` produces the
        # non-module name `surreal_memory.storage.NeuralStorage` -- legitimate,
        # and must not be judged by _resolves at all (only `modules` is).
        modules, attributes = guard._scan_imports(
            guard.SRC / "thing.py",
            ast.parse("from surreal_memory.storage import NeuralStorage"),
        )

        assert modules == {"surreal_memory.storage"}
        assert attributes == {"surreal_memory.storage.NeuralStorage"}
        assert guard._resolves("surreal_memory.storage") is True


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

    def test_the_tree_has_no_broken_imports(self, guard) -> None:  # type: ignore[no-untyped-def]
        report = guard.analyse()

        assert report.broken == {}, f"broken imports: {report.broken}"

    def test_main_exits_nonzero_under_strict_when_broken_and_nothing_dead(
        self, guard, monkeypatch, capsys
    ) -> None:  # type: ignore[no-untyped-def]
        # The regression this guards: a broken import with an otherwise-empty
        # `dead` list used to hit the early `if not dead: return` and exit 0
        # even under --strict.
        monkeypatch.setattr(
            guard,
            "analyse",
            lambda: guard.Report(dead=[], broken={"surreal_memory.storage.sqlite_store": ["x.py"]}),
        )
        monkeypatch.setattr(sys, "argv", ["check_dead_modules.py", "--strict"])

        with pytest.raises(SystemExit) as exc_info:
            guard.main()

        assert exc_info.value.code == 1
        assert "sqlite_store" in capsys.readouterr().out

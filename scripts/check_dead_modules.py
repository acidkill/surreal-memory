#!/usr/bin/env python3
"""Report modules that nothing but their own tests imports.

Usage:
    python scripts/check_dead_modules.py            # report, exit 0
    python scripts/check_dead_modules.py --strict   # exit 1 if anything is dead

Ruff cannot see this class of dead code: a module imported by the test written
for it *is* imported, so F401 stays quiet while nothing in the product reaches
it. Three such modules shipped for months before an audit found them.

Reachability is computed from the points where execution actually starts —
console scripts in ``pyproject.toml``, ``__main__`` modules, dotted paths in
string literals (uvicorn targets, ``importlib`` lookups), and anything an
example, benchmark or script imports — then followed through the import graph.
Lazy imports inside functions count, which matters here because optional
dependencies are loaded that way throughout. ``tests/`` is not a root: a module
that only its own test reaches is exactly what this looks for.

**Scope.** This finds unreachable *modules*. It does not find a module that is
imported but whose symbols nobody calls — ``storage/factory.py`` is imported by
its package ``__init__`` and so counts as reachable, even though
``create_storage`` has no caller. Catching that needs symbol-level analysis.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "surreal_memory"
SRC = ROOT / "src"
PACKAGE_ROOT = SRC / PACKAGE

#: Directories whose imports keep a module alive. `tests/` is deliberately absent.
SCAN_ROOTS = ("src", "examples", "scripts", "benchmarks", "qa")

#: Never reported: package markers and `python -m` entry points.
_EXEMPT_BASENAMES = frozenset({"__init__.py", "__main__.py"})

_DOTTED = re.compile(rf"{PACKAGE}(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _all_modules() -> dict[str, Path]:
    return {
        _module_name(p): p
        for p in sorted(PACKAGE_ROOT.rglob("*.py"))
        if p.name not in _EXEMPT_BASENAMES
    }


def _scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        directory = ROOT / root
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    return files


def _imports_in(path: Path, tree: ast.AST) -> set[str]:
    """Every ``surreal_memory.*`` module path this file imports."""
    found: set[str] = set()

    own_module = _module_name(path) if path.is_relative_to(SRC) else ""
    own_package = own_module.rsplit(".", 1)[0] if "." in own_module else PACKAGE

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = own_package.split(".")
                base = ".".join(parts[: max(len(parts) - node.level + 1, 1)])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if not module.startswith(PACKAGE):
                continue
            found.add(module)
            # `from pkg import submodule` imports a module, not an attribute.
            for alias in node.names:
                found.add(f"{module}.{alias.name}")
    return found


def _string_references(parsed: dict[Path, ast.AST]) -> set[str]:
    """Dotted paths inside string literals — uvicorn targets and friends.

    String *literals* only, never raw file text: an ``import`` line is text
    too, so scanning source verbatim would mark every imported module as
    externally referenced and quietly defeat the whole check.
    """
    found: set[str] = set()
    for tree in parsed.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.update(_DOTTED.findall(node.value.replace(":", ".")))
    return found


def _console_script_modules() -> set[str]:
    try:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    targets = data.get("project", {}).get("scripts", {}).values()
    return {str(target).split(":", 1)[0] for target in targets}


def find_dead_modules() -> list[str]:
    modules = _all_modules()

    parsed: dict[Path, ast.AST] = {}
    for path in _scan_files():
        try:
            parsed[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue

    # Import graph over the package, plus the roots execution actually starts from.
    edges: dict[str, set[str]] = {}
    roots: set[str] = set(_console_script_modules()) | _string_references(parsed)

    for path, tree in parsed.items():
        inside_package = path.is_relative_to(PACKAGE_ROOT)
        importer = _module_name(path) if inside_package else ""
        targets = _imports_in(path, tree)
        if inside_package:
            edges.setdefault(importer, set()).update(targets)
            if path.name in _EXEMPT_BASENAMES:
                # `python -m pkg` and package import both execute these.
                roots.add(importer)
        else:
            # An example, benchmark or script reaching in is an external caller.
            roots |= targets

    reachable: set[str] = set()
    queue = [r for r in roots]
    while queue:
        current = queue.pop()
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(edges.get(current, ()))
        # Importing a module runs its package's __init__, which may pull in more.
        package_init = f"{current.rsplit('.', 1)[0]}.__init__" if "." in current else ""
        if package_init and package_init not in reachable:
            queue.append(package_init)

    return sorted(name for name in modules if name not in reachable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report unreachable modules")
    parser.add_argument("--strict", action="store_true", help="Exit 1 when anything is dead")
    args = parser.parse_args()

    dead = find_dead_modules()
    if not dead:
        print("No unreachable modules.")
        return

    print(f"{len(dead)} module(s) reachable only from tests, if at all:")
    for name in dead:
        print(f"  {name}")
    print()
    print("Delete them, wire them up, or — if something reaches them by a name this")
    print("check cannot see — make that reference visible.")
    sys.exit(1 if args.strict else 0)


if __name__ == "__main__":
    main()

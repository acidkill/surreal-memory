"""Gates that stop a version-skew release from being tagged, and keep the
doc-reference scanner inside the repo.

Both of these exist because of a real failure. Release 3.3.1 bumped
`pyproject.toml` alone; the release workflow compared the tag against
`pyproject.toml` and `__init__.py`, failed, and skipped all eight publish
jobs. CI stayed green throughout, because the only version assertion in the
suite pinned `__version__` against a string that had been left stale too — so
the two wrong values agreed with each other.

The npm coverage matters for a second reason: the npm publish jobs read
`package.json`, so a stale one there does not fail the gate, it republishes an
already-taken version.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    """Import a top-level script from scripts/ that is not an installed module."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pre_ship = _load_script("pre_ship")
sync_refs = _load_script("sync_refs")


NPM_PACKAGES = (
    "integrations/surrealmemory",
    "integrations/surreal-memory-client",
    "vscode-extension",
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _fake_repo(root: Path, version: str = "9.9.9", **overrides: str) -> Path:
    """Build a minimal tree carrying every version string the release reads.

    `overrides` takes the same labels `collect_versions` returns, so a test can
    skew exactly one file and assert the gate notices.
    """

    def v(label: str) -> str:
        return overrides.get(label, version)

    _write(root / "pyproject.toml", f'[project]\nname = "x"\nversion = "{v("pyproject.toml")}"\n')
    _write(
        root / "src/surreal_memory/__init__.py",
        f'__version__ = "{v("src/surreal_memory/__init__.py")}"\n',
    )
    _write(
        root / "tests/unit/test_health_fixes.py",
        f'assert surreal_memory.__version__ == "{v("tests/unit/test_health_fixes.py")}"\n',
    )
    _write(
        root / ".claude-plugin/plugin.json",
        json.dumps({"name": "x", "version": v(".claude-plugin/plugin.json")}),
    )
    _write(
        root / ".claude-plugin/marketplace.json",
        json.dumps(
            {
                "version": v(".claude-plugin/marketplace.json (metadata)"),
                "plugins": [{"version": v(".claude-plugin/marketplace.json (plugins)")}],
            }
        ),
    )
    for pkg in NPM_PACKAGES:
        _write(
            root / pkg / "package.json",
            json.dumps({"name": pkg, "version": v(f"{pkg}/package.json")}),
        )
        _write(
            root / pkg / "package-lock.json",
            json.dumps(
                {
                    "name": pkg,
                    "version": v(f"{pkg}/package-lock.json (root)"),
                    "packages": {
                        "": {"version": v(f'{pkg}/package-lock.json (packages."")')},
                        "node_modules/events": {"version": "3.3.0"},
                    },
                }
            ),
        )
    _write(
        root / "integrations/surrealmemory/openclaw.plugin.json",
        json.dumps(
            {
                "id": "surrealmemory",
                "version": v("integrations/surrealmemory/openclaw.plugin.json"),
            }
        ),
    )
    return root


class TestCollectVersions:
    """`collect_versions` is the single source of truth both gates read."""

    def test_reads_every_versioned_file_and_agrees_when_consistent(self, tmp_path: Path) -> None:
        versions = pre_ship.collect_versions(_fake_repo(tmp_path, "9.9.9"))

        assert set(versions.values()) == {"9.9.9"}
        assert "NOT_FOUND" not in versions.values()

    def test_flags_the_init_skew_that_killed_3_3_1(self, tmp_path: Path) -> None:
        root = _fake_repo(tmp_path, "3.3.1", **{"src/surreal_memory/__init__.py": "3.3.0"})

        versions = pre_ship.collect_versions(root)

        assert versions["src/surreal_memory/__init__.py"] == "3.3.0"
        assert versions["pyproject.toml"] == "3.3.1"

    def test_covers_the_test_pin_that_agreed_with_the_stale_value(self, tmp_path: Path) -> None:
        root = _fake_repo(tmp_path, "3.3.1", **{"tests/unit/test_health_fixes.py": "3.3.0"})

        assert pre_ship.collect_versions(root)["tests/unit/test_health_fixes.py"] == "3.3.0"

    @pytest.mark.parametrize("pkg", NPM_PACKAGES)
    def test_covers_npm_package_json(self, tmp_path: Path, pkg: str) -> None:
        """A stale package.json republishes a taken version instead of failing."""
        label = f"{pkg}/package.json"
        root = _fake_repo(tmp_path, "3.3.1", **{label: "3.3.0"})

        assert pre_ship.collect_versions(root)[label] == "3.3.0"

    @pytest.mark.parametrize("pkg", NPM_PACKAGES)
    def test_covers_both_lockfile_roots(self, tmp_path: Path, pkg: str) -> None:
        label = f'{pkg}/package-lock.json (packages."")'
        root = _fake_repo(tmp_path, "3.3.1", **{label: "3.3.0"})

        versions = pre_ship.collect_versions(root)

        assert versions[label] == "3.3.0"
        assert versions[f"{pkg}/package-lock.json (root)"] == "3.3.1"

    def test_ignores_dependency_versions_inside_lockfiles(self, tmp_path: Path) -> None:
        """`events@3.3.0` in node_modules is an unrelated package, not our version."""
        versions = pre_ship.collect_versions(_fake_repo(tmp_path, "3.3.1"))

        assert "3.3.0" not in versions.values()

    def test_covers_the_openclaw_manifest_that_drifted_since_2_20_1(self, tmp_path: Path) -> None:
        label = "integrations/surrealmemory/openclaw.plugin.json"
        root = _fake_repo(tmp_path, "3.3.1", **{label: "2.20.1"})

        assert pre_ship.collect_versions(root)[label] == "2.20.1"

    def test_missing_file_reports_not_found_rather_than_crashing(self, tmp_path: Path) -> None:
        root = _fake_repo(tmp_path, "3.3.1")
        (root / "vscode-extension/package.json").unlink()

        assert pre_ship.collect_versions(root)["vscode-extension/package.json"] == "NOT_FOUND"


class TestVersionGate:
    """The gate both CI and the release workflow call."""

    def test_passes_when_every_file_agrees(self, tmp_path: Path) -> None:
        ok, problems = pre_ship.verify_versions(_fake_repo(tmp_path, "3.3.1"))

        assert ok is True
        assert problems == []

    def test_fails_and_names_the_skewed_file(self, tmp_path: Path) -> None:
        root = _fake_repo(tmp_path, "3.3.1", **{"src/surreal_memory/__init__.py": "3.3.0"})

        ok, problems = pre_ship.verify_versions(root)

        assert ok is False
        assert any("src/surreal_memory/__init__.py" in p and "3.3.0" in p for p in problems)

    def test_expected_version_catches_a_tag_ahead_of_every_file(self, tmp_path: Path) -> None:
        """v3.3.2 tagged on a tree that still says 3.3.1 everywhere."""
        ok, problems = pre_ship.verify_versions(_fake_repo(tmp_path, "3.3.1"), expected="3.3.2")

        assert ok is False
        assert any("3.3.2" in p for p in problems)

    def test_expected_version_passes_when_the_tag_matches(self, tmp_path: Path) -> None:
        ok, problems = pre_ship.verify_versions(_fake_repo(tmp_path, "3.3.1"), expected="3.3.1")

        assert ok is True
        assert problems == []


class TestSchemaVersionTruth:
    """`derive_truth()` pointed at `storage/sqlite_schema.py`, which #141
    deleted along with the SQLite backend. Every run since has reported
    `schema_version=0` and warned `sqlite_schema.py not found` — silently
    exempting every "schema vN" doc claim from ever being checked."""

    def test_reads_the_surrealdb_schema_module_that_actually_exists(self) -> None:
        from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION

        truth = sync_refs.derive_truth()

        assert truth.schema_version == SCHEMA_VERSION
        assert truth.schema_version > 0
        assert "storage/surrealdb/schema.py not found" not in truth.errors
        assert "sqlite_schema.py not found" not in truth.errors


class TestScannerStaysInsideTheRepo:
    """`sync_refs` walks *.md from the repo root; vendored trees are not ours."""

    @pytest.mark.parametrize(
        "relative",
        [
            ".claude/worktrees/agent-1/CHANGELOG.md",
            ".claude/worktrees/agent-1/.venv/lib/python3.13/site-packages/litellm/README.md",
            "dashboard/node_modules/some-dep/README.md",
            ".venv/lib/python3.13/site-packages/pkg/ARCHITECTURE.md",
            "vscode-extension/node_modules/dep/readme.md",
            # Gitignored session journal: timestamped records of what was true
            # that day, not documentation that drifted.
            ".remember/today-2026-07-11.done.md",
        ],
    )
    def test_skips_vendored_and_worktree_paths_at_any_depth(self, relative: str) -> None:
        assert sync_refs._should_skip(sync_refs.ROOT / relative) is True

    @pytest.mark.parametrize(
        "relative",
        ["README.md", "docs/guides/pro-quickstart.md", "ROADMAP.md"],
    )
    def test_still_scans_real_documentation(self, relative: str) -> None:
        assert sync_refs._should_skip(sync_refs.ROOT / relative) is False

    def test_still_honours_the_historical_skip_list(self) -> None:
        assert sync_refs._should_skip(sync_refs.ROOT / "CHANGELOG.md") is True

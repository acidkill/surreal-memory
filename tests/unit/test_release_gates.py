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


class TestAiosqliteIsOptional:
    """#154.6: `aiosqlite` only powers `smem_train_db`'s SQLite dialect
    (`engine/db_introspector.py`, a lazy, try/except-guarded import) — not the
    SurrealDB backend every install actually runs on. It does not belong in
    the unconditional install."""

    def test_not_in_core_dependencies(self) -> None:
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

        assert not any("aiosqlite" in dep for dep in data["project"]["dependencies"])

    def test_available_as_its_own_extra(self) -> None:
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        extras = data["project"]["optional-dependencies"]

        assert any("aiosqlite" in dep for dep in extras.get("sqlite-import", []))

    def test_dev_extra_still_carries_it(self) -> None:
        """conftest's autouse leak-guard fixture imports it unconditionally —
        a dev install without it fails test collection, not just the tests
        that exercise it."""
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        extras = data["project"]["optional-dependencies"]

        assert any("aiosqlite" in dep for dep in extras["dev"])


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


class TestVerificationProfiles:
    def test_profile_commands_and_failure_propagation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
            calls.append(cmd)
            return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

        monkeypatch.setattr(pre_ship.subprocess, "run", fake_run)
        assert pre_ship.run_profile("full") == 1
        assert calls[-1][3:] == ["tests/", "-m", "not stress", "-n", "auto"]
        assert len(calls) == 4

    def test_unit_profile_is_unit_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(pre_ship.subprocess, "run", fake_run)
        assert pre_ship.run_profile("unit") == 0
        assert calls == [[sys.executable, "-m", "pytest", "tests/unit/", "-m", "not stress"]]

    def test_dashboard_build_uses_temporary_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(pre_ship.subprocess, "run", fake_run)
        assert pre_ship.run_profile("dashboard") == 0
        assert [cmd[4] for cmd in calls[:2]] == ["lint", "test"]
        assert calls[-1][-2] == "--outDir"
        assert not Path(calls[-1][-1]).exists()
        assert "static/dist" not in calls[-1][-1]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("success", ""),
        ("invalid", "did not return exactly one valid SHA-256 digest"),
        ("pushfail", "Image push failed"),
        ("inspectfail", "Image inspection failed"),
    ],
)
def test_production_image_receipt(tmp_path: Path, case: str, message: str) -> None:
    """Exercise the actual publish step without a registry or Docker daemon."""
    import os
    import subprocess
    import textwrap

    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    step = workflow.split("      - name: Publish the tested image and record its digest\n", 1)[1]
    script = step.split("        run: |\n", 1)[1].split(
        "      - name: Upload publication receipt\n", 1
    )[0]
    script = textwrap.dedent(script).replace("${{ runner.temp }}", "$RUNNER_TEMP")
    stub = """docker() {
      case "$1" in
        load|tag) return 0 ;;
        push)
          [[ "$CASE" != pushfail ]] || return 23
          if [[ "$CASE" == invalid ]]; then
            echo 'tag: digest: sha256:bad size: 1'
          else
            echo "tag: digest: sha256:$DIGEST size: 1624"
          fi ;;
        inspect) [[ "$CASE" != inspectfail ]] || return 12; echo sha256:imageid ;;
      esac
    }
    """
    digest = "a" * 64
    env = {
        "PATH": os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "REPOSITORY": "Example/Repo",
        "GITHUB_SHA": "abc",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://example.test",
        "GITHUB_REPOSITORY": "Example/Repo",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "CASE": case,
        "DIGEST": digest,
    }
    result = subprocess.run(  # noqa: S603 — script is read from this repository
        ["/bin/bash", "-c", stub + script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = tmp_path / "production-image-receipt.json"
    if case == "success":
        assert result.returncode == 0, result.stderr
        data = json.loads(receipt.read_text())
        assert data["image_digest"] == f"ghcr.io/example/repo@sha256:{digest}"
        assert data["source_sha"] == "abc"
        assert data["image_id"] == "sha256:imageid"
    else:
        assert result.returncode != 0
        assert message in result.stderr
        assert not receipt.exists()

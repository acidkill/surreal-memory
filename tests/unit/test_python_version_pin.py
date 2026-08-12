"""Gates that keep the development interpreter aligned with the mypy target.

These exist because of a real failure. The repo had no interpreter pin --
`.python-version` was listed in `.gitignore`, inherited from the stock Python
template's pyenv stanza -- so `uv sync` selected whatever interpreter it found
on the machine. `requires-python = ">=3.11"` is only a floor, so a 3.13 venv is
a legal resolution, and on 3.13 the resolver picks distributions published for
3.12+ only.

That is enough to break `make typecheck` on a clean tree. `[tool.mypy]
python_version` is `"3.11"`, and mypy parses *every* file it reads at that
feature version -- including the `.pyi` stubs shipped inside installed
packages. numpy 2.5.x (`Requires-Python: >=3.12`) writes its aliases as PEP-695
`type X = ...` statements, which Python 3.11 cannot parse:

    numpy/__init__.pyi:737: error: Type statement is only supported in
    Python 3.12 and greater  [syntax]

CI never caught it: the Type Check job pins setup-python to 3.11 and installs
`.[dev]`, which has no numpy, so it resolves neither the newer interpreter nor
the newer stubs. It is a local-only landmine, which is exactly the kind CI
cannot be relied on to find.

The invariant that makes the failure impossible: the pinned interpreter, the
`requires-python` floor and mypy's `python_version` must all name the same
Python. When they agree, no installable distribution can require a Python newer
than the one mypy parses for, so no stub can carry syntax mypy will reject.
"""

from __future__ import annotations

import ast
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = REPO_ROOT / ".python-version"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# A PEP-695 type-alias statement, the construct numpy 2.5.x stubs are written
# with. Parsing this is what a pre-3.12 feature version refuses to do.
PEP695_ALIAS = "type Alias = int\n"


def _pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _version_tuple(raw: str) -> tuple[int, int]:
    """Parse a `major.minor[.patch]` string down to its `(major, minor)` pair."""
    major, minor = raw.strip().split(".")[:2]
    return int(major), int(minor)


def _pinned_version() -> tuple[int, int]:
    return _version_tuple(PIN_FILE.read_text(encoding="utf-8").strip())


def _mypy_target() -> tuple[int, int]:
    tool = _pyproject()["tool"]
    assert isinstance(tool, dict)
    mypy_cfg = tool["mypy"]
    assert isinstance(mypy_cfg, dict)
    return _version_tuple(str(mypy_cfg["python_version"]))


def _requires_python_floor() -> tuple[int, int]:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    requires = str(project["requires-python"]).strip()
    assert requires.startswith(">="), f"expected a `>=` floor, got {requires!r}"
    return _version_tuple(requires.removeprefix(">="))


class TestInterpreterPin:
    """The pin has to exist, reach a fresh clone, and name a real version."""

    def test_pin_file_exists(self) -> None:
        assert PIN_FILE.is_file(), (
            ".python-version is missing. Without it `uv sync` picks an arbitrary "
            "interpreter and the type-check environment drifts off the mypy target."
        )

    def test_pin_is_a_bare_version(self) -> None:
        raw = PIN_FILE.read_text(encoding="utf-8").strip()
        assert raw, ".python-version is empty"
        assert "\n" not in raw, f"expected a single version line, got {raw!r}"
        # A bare `major.minor` lets uv/pyenv track the newest patch release,
        # which is what CI's setup-python pin does too.
        assert raw.count(".") == 1, (
            f"pin {raw!r} should be `major.minor` so it tracks patch releases "
            "the same way CI's setup-python pin does"
        )
        assert _version_tuple(raw) >= (3, 11)

    def test_pin_is_not_gitignored(self) -> None:
        """The regression itself: an ignored pin never reaches a fresh clone."""
        if not (REPO_ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        result = subprocess.run(  # noqa: S603
            ["git", "check-ignore", "--quiet", str(PIN_FILE)],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        # `git check-ignore --quiet` exits 0 when the path IS ignored.
        assert result.returncode != 0, (
            ".python-version is matched by a .gitignore rule, so it will not "
            "reach a fresh clone and the interpreter pin is not enforced there."
        )


class TestPinAgreesWithBuildConfig:
    """The three declarations that have to name the same Python."""

    def test_pin_matches_mypy_target(self) -> None:
        assert _pinned_version() == _mypy_target(), (
            "The development interpreter and mypy's python_version must match. "
            "If the venv is newer, the resolver can install stubs written in "
            "syntax mypy refuses to parse at its configured target."
        )

    def test_pin_matches_requires_python_floor(self) -> None:
        assert _pinned_version() == _requires_python_floor(), (
            "Develop on the floor the package claims to support, so the lowest "
            "supported Python is the one that actually gets exercised locally."
        )

    def test_ci_typecheck_job_uses_the_pinned_version(self) -> None:
        yaml = pytest.importorskip("yaml")
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
        steps = workflow["jobs"]["typecheck"]["steps"]
        setup = [s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python")]
        assert setup, "the typecheck job no longer pins an interpreter"
        for step in setup:
            assert _version_tuple(str(step["with"]["python-version"])) == _pinned_version()


class TestStubSyntaxHazard:
    """The mechanism, and the environment it actually has to hold in."""

    def test_mypy_target_cannot_parse_pep695_aliases(self) -> None:
        """Documents why the pin is load-bearing rather than cosmetic.

        If this ever stops raising -- because the mypy target moved to 3.12+ --
        the numpy hazard is gone and this gate can be revisited.
        """
        target = _mypy_target()
        if target >= (3, 12):
            pytest.skip(f"mypy target {target} parses PEP-695 natively")
        with pytest.raises(SyntaxError):
            ast.parse(PEP695_ALIAS, feature_version=target)

    def test_installed_numpy_stubs_parse_at_the_mypy_target(self) -> None:
        """The bug scenario end to end, against whatever is really installed.

        On a venv built off the pin (3.13 + numpy 2.5.1) this fails on
        `numpy/__init__.pyi` with the same syntax error `mypy src/` reports.
        """
        numpy = pytest.importorskip("numpy")
        stubs = sorted(Path(numpy.__file__).parent.rglob("*.pyi"))
        assert stubs, "numpy is installed but ships no stubs -- check the install"

        target = _mypy_target()
        unparseable: list[str] = []
        for stub in stubs:
            try:
                ast.parse(stub.read_text(encoding="utf-8"), feature_version=target)
            except SyntaxError as exc:
                unparseable.append(f"{stub.name}:{exc.lineno}: {exc.msg}")

        assert not unparseable, (
            f"numpy {numpy.__version__} ships stubs that mypy's python_version "
            f"{target[0]}.{target[1]} cannot parse, so `mypy src/` fails inside "
            f"site-packages. The venv is off the .python-version pin: "
            f"{unparseable[:3]}"
        )

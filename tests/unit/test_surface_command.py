"""The `smem surface` command, brain-name validation, and the global path.

`smem doctor` prescribes `smem surface generate`. Until now that produced
`No such command 'surface'`, so the prescribed fix could not be followed.

Two other defects are pinned here. The surface path was resolved from the
*process* CWD, so a server or MCP client wrote wherever it happened to be
running; and the brain name -- which becomes a filename -- was never validated,
which is how a live install ended up with a surface whose brain key was an
entire ``CLIConfig(data_dir=PosixPath(...), ...)`` repr.
"""

from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from surreal_memory.cli.main import app
from surreal_memory.mcp.surface_handler import _surface_brain_name
from surreal_memory.surface.resolver import get_surface_path, validate_brain_name

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI styling from CLI output before matching against it.

    Rich colourises option names *inside* the word -- `--global-path` comes back
    as `-\\x1b[0m\\x1b[1;36m-global\\x1b[0m\\x1b[1;36m-path` when a terminal
    profile enables colour. Asserting on the raw output therefore passed locally
    and failed in CI purely on styling.
    """
    return _ANSI.sub("", text)

# The repr that really landed in a live surface file's `brain:` header.
CLICONFIG_REPR = (
    "CLIConfig(data_dir=PosixPath('/home/user/.surrealmemory'), current_brain='default')"
)


class TestValidateBrainName:
    @pytest.mark.parametrize("name", ["default", "work", "brain-1", "a_b.c", "A" * 128])
    def test_accepts_plain_names(self, name: str) -> None:
        assert validate_brain_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "A" * 129,
            "with space",
            "../escape",
            "nested/name",
            "back\\slash",
            ".",
            "..",
            CLICONFIG_REPR,
        ],
    )
    def test_rejects_anything_that_could_become_a_path(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_brain_name(name)

    def test_the_global_path_refuses_a_bad_name(self) -> None:
        """The name is interpolated into a filename, so it is validated there too."""
        with pytest.raises(ValueError):
            get_surface_path("../../etc/passwd", global_only=True)


class TestGlobalPathIgnoresTheCwd:
    def test_global_only_skips_project_detection(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("SURREAL_MEMORY_DIR", str(home))

        project = tmp_path / "project"
        (project / ".git").mkdir(parents=True)
        monkeypatch.chdir(project)

        # Standing inside a project, the default resolution follows the project...
        project_path = get_surface_path("default", for_write=True)
        # ...but --global-path must not care where the process happens to be.
        global_path = get_surface_path("default", for_write=True, global_only=True)

        assert project_path != global_path
        assert global_path == home / "surfaces" / "default.nm"
        assert str(project) not in str(global_path)

    def test_global_path_is_the_same_from_any_directory(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("SURREAL_MEMORY_DIR", str(home))

        first = tmp_path / "one"
        (first / ".git").mkdir(parents=True)
        second = tmp_path / "two"
        second.mkdir()

        monkeypatch.chdir(first)
        from_project = get_surface_path("default", global_only=True)
        monkeypatch.chdir(second)
        from_elsewhere = get_surface_path("default", global_only=True)

        assert from_project == from_elsewhere


class TestSurfaceCommandExists:
    def test_the_command_doctor_prescribes_is_registered(self) -> None:
        result = runner.invoke(app, ["surface", "--help"])
        assert result.exit_code == 0
        assert "No such command" not in _plain(result.output)

    def test_it_offers_generate_and_show(self) -> None:
        output = _plain(runner.invoke(app, ["surface", "--help"]).output)
        assert "generate" in output
        assert "show" in output

    def test_generate_advertises_the_global_path_flag(self) -> None:
        result = runner.invoke(app, ["surface", "generate", "--help"])
        assert result.exit_code == 0
        assert "--global-path" in _plain(result.output)

    def test_an_invalid_brain_name_is_rejected_before_any_io(self) -> None:
        result = runner.invoke(app, ["surface", "show", "--brain", "../escape"])
        assert result.exit_code == 2
        assert "Invalid brain name" in _plain(result.output)


class TestSurfaceBrainKey:
    """The generator must write the key the reader reads."""

    def test_config_current_brain_wins(self) -> None:
        config = SimpleNamespace(current_brain="work")
        brain = SimpleNamespace(name="something-else")
        assert _surface_brain_name(config, brain) == "work"

    def test_falls_back_to_brain_name_when_config_is_unusable(self) -> None:
        config = SimpleNamespace(current_brain=None)
        brain = SimpleNamespace(name="work")
        assert _surface_brain_name(config, brain) == "work"

    def test_a_config_repr_is_never_used_as_a_brain_key(self) -> None:
        config = SimpleNamespace(current_brain=CLICONFIG_REPR)
        brain = SimpleNamespace(name="work")
        assert _surface_brain_name(config, brain) == "work"

    def test_two_unusable_candidates_fall_back_to_default(self) -> None:
        config = SimpleNamespace(current_brain=CLICONFIG_REPR)
        brain = SimpleNamespace(name="../escape")
        assert _surface_brain_name(config, brain) == "default"

    def test_a_missing_brain_object_is_tolerated(self) -> None:
        config = SimpleNamespace(current_brain="work")
        assert _surface_brain_name(config, None) == "work"

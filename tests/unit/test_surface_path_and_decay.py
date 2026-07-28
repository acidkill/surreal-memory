"""Regression tests for two cwd/bookkeeping defects.

D1: the resolved knowledge-surface path depended on the shell's cwd, because the
global config dir ``~/.surrealmemory`` matched the project marker of the same name
and promoted ``$HOME`` to a "project root".

D2: ``Synapse.decay()`` recorded nothing, so every decay run re-processed (and
re-decayed) the same synapses. Decay must not advance ``last_activated``, which
means "last fired".
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from surreal_memory.core.synapse import LAST_DECAYED_KEY, Synapse, SynapseType
from surreal_memory.surface.resolver import (
    detect_project_root,
    get_surface_path,
    load_surface_text,
    save_surface_text,
)
from surreal_memory.utils.timeutils import utcnow

# ── D1: surface path must not depend on cwd ────────────


class TestProjectRootDetection:
    """detect_project_root must not mistake the home directory for a project."""

    def test_home_with_global_config_dir_is_not_a_project_root(self, tmp_path: Path) -> None:
        """The live bug: ~/.surrealmemory made $HOME look like a project root."""
        home = tmp_path / "home"
        global_dir = home / ".surrealmemory"
        global_dir.mkdir(parents=True)

        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=home),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            patch(
                "surreal_memory.unified_config.get_surrealmemory_dir",
                return_value=global_dir,
            ),
        ):
            assert detect_project_root() is None

    def test_relocated_global_config_dir_is_not_a_project_root(self, tmp_path: Path) -> None:
        """SURREAL_MEMORY_DIR elsewhere must not promote its parent either."""
        home = tmp_path / "home"
        home.mkdir()
        data_dir = tmp_path / "srv" / "data"
        global_dir = data_dir / ".surrealmemory"
        global_dir.mkdir(parents=True)

        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=data_dir),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            patch(
                "surreal_memory.unified_config.get_surrealmemory_dir",
                return_value=global_dir,
            ),
        ):
            assert detect_project_root() is None

    def test_project_owned_nm_dir_is_still_a_project_root(self, tmp_path: Path) -> None:
        """Per-project surfaces stay supported — only the global dir is excluded."""
        home = tmp_path / "home"
        global_dir = home / ".surrealmemory"
        global_dir.mkdir(parents=True)
        project = home / "repos" / "myproject"
        (project / ".surrealmemory").mkdir(parents=True)

        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=project),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            patch(
                "surreal_memory.unified_config.get_surrealmemory_dir",
                return_value=global_dir,
            ),
        ):
            assert detect_project_root() == project

    def test_repo_under_home_is_still_detected(self, tmp_path: Path) -> None:
        """A .git checkout below home is a project; the walk stops at home."""
        home = tmp_path / "home"
        (home / ".surrealmemory").mkdir(parents=True)
        repo = home / "repos" / "checkout"
        (repo / ".git").mkdir(parents=True)
        nested = repo / "src" / "pkg"
        nested.mkdir(parents=True)

        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=nested),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            patch(
                "surreal_memory.unified_config.get_surrealmemory_dir",
                return_value=home / ".surrealmemory",
            ),
        ):
            assert detect_project_root() == repo


class TestSurfacePathConsistency:
    """Generator (write) and doctor (read) must resolve the same file."""

    def test_write_from_home_lands_on_the_global_surface(self, tmp_path: Path) -> None:
        """`surface generate` from ~ must not write ~/.surrealmemory/surface.nm."""
        home = tmp_path / "home"
        global_dir = home / ".surrealmemory"
        global_dir.mkdir(parents=True)

        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=home),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            patch(
                "surreal_memory.unified_config.get_surrealmemory_dir",
                return_value=global_dir,
            ),
        ):
            write_path = get_surface_path("default", for_write=True)

        assert write_path == global_dir / "surfaces" / "default.nm"
        assert write_path != global_dir / "surface.nm"

    def test_generate_from_home_then_read_from_repo_finds_the_file(self, tmp_path: Path) -> None:
        """The false alarm: doctor in a repo said "not generated yet" after a ~ run."""
        home = tmp_path / "home"
        global_dir = home / ".surrealmemory"
        global_dir.mkdir(parents=True)
        repo = home / "repos" / "checkout"
        (repo / ".git").mkdir(parents=True)

        cfg_patch = patch(
            "surreal_memory.unified_config.get_surrealmemory_dir",
            return_value=global_dir,
        )

        # Generate from the home directory...
        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=home),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            cfg_patch,
        ):
            written = save_surface_text("# surface", "default")

        # ...then run the doctor check from a repo checkout.
        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=repo),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            cfg_patch,
        ):
            read_path = get_surface_path("default")
            assert read_path == written
            assert read_path.exists()
            assert load_surface_text("default") == "# surface"

    def test_project_surface_round_trips_within_the_project(self, tmp_path: Path) -> None:
        """Inside a project, write then read resolve to the project-level file."""
        home = tmp_path / "home"
        global_dir = home / ".surrealmemory"
        global_dir.mkdir(parents=True)
        project = home / "repos" / "myproject"
        (project / ".git").mkdir(parents=True)

        with (
            patch("surreal_memory.surface.resolver.Path.cwd", return_value=project),
            patch("surreal_memory.surface.resolver.Path.home", return_value=home),
            patch(
                "surreal_memory.unified_config.get_surrealmemory_dir",
                return_value=global_dir,
            ),
        ):
            written = save_surface_text("# project surface", "default")
            assert written == project / ".surrealmemory" / "surface.nm"
            assert get_surface_path("default") == written
            assert load_surface_text("default") == "# project surface"


# ── D2: decay must bookmark itself, not fake activity ──


def _make_synapse(
    weight: float = 1.0,
    *,
    activated_hours_ago: float | None = None,
    created_hours_ago: float = 100.0,
) -> Synapse:
    """Build a synapse with controlled timestamps."""
    now = utcnow()
    synapse = Synapse.create(
        source_id="n1",
        target_id="n2",
        type=SynapseType.RELATED_TO,
        weight=weight,
    )
    # A synapse cannot have fired before it existed — keep fixtures coherent so the
    # newest-stamp semantics of decay_reference_time are exercised, not fought.
    age_hours = max(created_hours_ago, activated_hours_ago or 0.0)
    return replace(
        synapse,
        created_at=now - timedelta(hours=age_hours),
        last_activated=(
            None if activated_hours_ago is None else now - timedelta(hours=activated_hours_ago)
        ),
    )


class TestDecayBookmark:
    """decay() records when it ran without touching last_activated."""

    def test_decay_does_not_advance_last_activated(self) -> None:
        """last_activated means "last fired" — a decay pass is not a firing."""
        synapse = _make_synapse(activated_hours_ago=48)

        decayed = synapse.decay(factor=0.9)

        assert decayed.last_activated == synapse.last_activated

    def test_decay_leaves_never_activated_synapse_unactivated(self) -> None:
        """A synapse that never fired must not gain a last_activated from decay."""
        synapse = _make_synapse(activated_hours_ago=None)

        assert synapse.decay(factor=0.9).last_activated is None

    def test_decay_records_last_decayed(self) -> None:
        before = utcnow()
        synapse = _make_synapse()

        decayed = synapse.decay(factor=0.9)

        assert decayed.last_decayed is not None
        assert before <= decayed.last_decayed <= utcnow()
        assert decayed.weight == 0.9

    def test_decay_accepts_explicit_timestamp(self) -> None:
        """The decay pass stamps one reference time across the whole run."""
        stamp = utcnow() - timedelta(hours=3)
        synapse = _make_synapse()

        assert synapse.decay(factor=0.5, now=stamp).last_decayed == stamp

    def test_undecayed_synapse_has_no_bookmark(self) -> None:
        assert _make_synapse().last_decayed is None

    def test_decay_does_not_mutate_original_metadata(self) -> None:
        """Frozen dataclass: the decayed copy must not share the source dict."""
        synapse = replace(_make_synapse(), metadata={"_dedup": True})

        decayed = synapse.decay(factor=0.9)

        assert synapse.metadata == {"_dedup": True}
        assert decayed.metadata["_dedup"] is True
        assert LAST_DECAYED_KEY in decayed.metadata

    def test_repeated_decay_refreshes_the_bookmark(self) -> None:
        first_stamp = utcnow() - timedelta(days=2)
        second_stamp = utcnow() - timedelta(days=1)
        synapse = _make_synapse()

        twice = synapse.decay(factor=0.9, now=first_stamp).decay(factor=0.9, now=second_stamp)

        assert twice.last_decayed == second_stamp

    def test_bookmark_accepts_datetime_value_from_storage(self) -> None:
        """SurrealDB decodes object fields into datetimes, not ISO strings."""
        stamp = utcnow() - timedelta(hours=5)
        synapse = replace(_make_synapse(), metadata={LAST_DECAYED_KEY: stamp})

        assert synapse.last_decayed == stamp

    def test_corrupt_bookmark_reads_as_never_decayed(self) -> None:
        """A bad value must degrade, not raise in the middle of a decay run."""
        for bad in ("not-a-date", 12345, None):
            synapse = replace(_make_synapse(), metadata={LAST_DECAYED_KEY: bad})
            assert synapse.last_decayed is None

    def test_time_decay_writes_no_bookmark(self) -> None:
        """time_decay is a read-side estimate; its result is not persisted."""
        synapse = _make_synapse(activated_hours_ago=1440)

        estimated = synapse.time_decay()

        assert estimated.last_decayed is None
        assert estimated.last_activated == synapse.last_activated


class TestDecayReferenceTime:
    """decay_reference_time is the base callers should measure elapsed time from."""

    def test_falls_back_to_created_at_when_never_used(self) -> None:
        synapse = _make_synapse(activated_hours_ago=None, created_hours_ago=100)

        assert synapse.decay_reference_time == synapse.created_at

    def test_prefers_last_activated_over_created_at(self) -> None:
        synapse = _make_synapse(activated_hours_ago=10, created_hours_ago=100)

        assert synapse.decay_reference_time == synapse.last_activated

    def test_prefers_last_decayed_when_it_is_newest(self) -> None:
        stamp = utcnow() - timedelta(hours=1)
        synapse = _make_synapse(activated_hours_ago=10).decay(factor=0.9, now=stamp)

        assert synapse.decay_reference_time == stamp

    def test_reinforcement_after_decay_resets_the_clock(self) -> None:
        """A synapse used since its last decay must not be charged for that gap."""
        decayed = _make_synapse(activated_hours_ago=100).decay(
            factor=0.9, now=utcnow() - timedelta(hours=50)
        )

        reinforced = decayed.reinforce(0.05)

        assert reinforced.decay_reference_time == reinforced.last_activated

    def test_second_pass_over_the_same_window_has_nothing_to_decay(self) -> None:
        """The 57k-rewrite regression: elapsed time must not be re-counted."""
        run_at = utcnow()
        synapse = _make_synapse(activated_hours_ago=240)  # 10 days stale

        def elapsed_days(candidate: Synapse) -> float:
            return (run_at - candidate.decay_reference_time).total_seconds() / 86400

        first_elapsed = elapsed_days(synapse)
        decayed = synapse.decay(factor=0.5, now=run_at)
        second_elapsed = elapsed_days(decayed)

        assert first_elapsed > 9.9
        assert second_elapsed == 0.0

    def test_naive_and_aware_timestamps_compare(self) -> None:
        """Stores hand back both shapes; the property must not raise on mixing."""
        aware = datetime.now(UTC) - timedelta(hours=2)
        synapse = replace(_make_synapse(activated_hours_ago=None), last_activated=aware)

        reference = synapse.decay_reference_time

        assert reference.tzinfo is None
        assert reference == aware.replace(tzinfo=None)

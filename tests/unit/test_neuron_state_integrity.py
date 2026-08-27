"""Guards for the neuron <-> neuron_state pairing.

Two defects, opposite directions, both silent:

* `add_neuron` wrote the neuron and then its `neuron_state` row inside a bare
  `except: pass`. A failed state write returned success, so "the state was
  written" and "the state write failed" were indistinguishable afterwards. A
  neuron with no state row looks permanently un-accessed: it never receives the
  activation boost and is a standing candidate for dead-neuron pruning (#174).
* Nothing ever removed states whose neuron was already gone. Both delete paths
  clean up correctly today, so surviving orphans are historical -- but
  `apply_decay` iterates `get_all_neuron_states()`, so every orphan is work done
  on a neuron that no longer exists, and it inflates any per-pass count taken
  from that loop.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import textwrap

from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "surreal_memory"


class TestStateWriteIsNotSilent:
    """A swallowed write is worse than a failed one: it reports success."""

    @staticmethod
    def _bare_excepts(func) -> list[int]:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            writes_state = any(
                isinstance(c, ast.Constant) and c.value == "neuron_state" for c in ast.walk(node)
            )
            if not writes_state:
                continue
            for handler in node.handlers:
                # `except ...: pass` with no logging and no re-raise
                body_is_pass = all(isinstance(s, ast.Pass) for s in handler.body)
                if body_is_pass:
                    offenders.append(handler.lineno)
        return offenders

    def test_add_neuron_does_not_swallow_the_state_write(self) -> None:
        offenders = self._bare_excepts(SurrealDBStorage.add_neuron)
        assert not offenders, (
            "add_neuron swallows the neuron_state insert -- a caller cannot tell "
            f"a written state from a failed one (handler at +{offenders})"
        )

    def test_batch_path_does_not_swallow_the_state_write(self) -> None:
        """The batch path loses a whole chunk at a time, not one row."""
        offenders = self._bare_excepts(SurrealDBStorage.add_neurons_batch)
        assert not offenders, (
            "add_neurons_batch swallows the neuron_state insert for an entire "
            f"chunk (handler at +{offenders})"
        )


class TestOrphanedStatesAreReclaimable:
    def test_storage_exposes_a_way_to_remove_them(self) -> None:
        assert hasattr(SurrealDBStorage, "prune_orphaned_neuron_states"), (
            "nothing can remove neuron_state rows whose neuron is gone, so they "
            "accumulate and inflate every pass over get_all_neuron_states()"
        )

    def test_some_production_path_calls_it(self) -> None:
        """An unreferenced cleanup is indistinguishable from no cleanup.

        prune_synced_changes shipped with zero call sites for its entire life;
        this asserts the same thing does not happen again here.
        """
        call_sites = [
            path.relative_to(SRC_ROOT).as_posix()
            for path in SRC_ROOT.rglob("*.py")
            if "prune_orphaned_neuron_states" in path.read_text(encoding="utf-8")
            and path.name not in {"base.py", "store.py", "memory_store.py"}
        ]
        assert call_sites, "prune_orphaned_neuron_states is never called"

    def test_the_query_inlines_brain_id(self) -> None:
        """A parameterised brain_id defeats the index -- measured 25x on this DB."""
        source = inspect.getsource(SurrealDBStorage._find_orphaned_neuron_states)
        assert not re.search(r"brain_id\s*=\s*\$\w+", source), (
            "parameterised brain_id falls back to a full scan; inline it with _brain_literal()"
        )


class TestOrphanDetectionMatchesIdSpellings:
    """The two sides spell the same neuron differently.

    A record id is ``neuron:<uuid>`` with ``-`` folded to ``_``; the state's
    ``neuron_id`` keeps the original dashed uuid. Comparing them raw makes every
    live state look orphaned — which, run against a real database, deletes the
    entire table. This test exists because that happened.
    """

    def test_both_sides_are_normalised_before_comparison(self) -> None:
        source = inspect.getsource(SurrealDBStorage._find_orphaned_neuron_states)
        tree = ast.parse(textwrap.dedent(source))
        folds = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_to_surreal_id"
        ]
        assert len(folds) >= 2, (
            "both the live neuron ids and the state's neuron_id must go through "
            "_to_surreal_id before being compared; normalising only one side "
            "marks every live state as an orphan"
        )


class TestPruningIsNotAutomatic:
    """Deleting on the strength of a comparison that can empty the table is not
    something that should run unattended.

    The detection compares neurons against states; normalising only one side
    marks every live state as an orphan. That failure mode is why this is an
    explicitly invoked command and not a consolidation stage.
    """

    def test_consolidation_does_not_call_it(self) -> None:
        source = (SRC_ROOT / "engine" / "consolidation.py").read_text(encoding="utf-8")
        assert "prune_orphaned_neuron_states" not in source, (
            "orphan pruning must not run automatically from consolidation"
        )

    def test_it_is_reachable_as_an_explicit_command(self) -> None:
        source = (SRC_ROOT / "cli" / "commands" / "tools.py").read_text(encoding="utf-8")
        assert "prune_orphaned_neuron_states" in source, "no explicit way to run it"
        assert "count_orphaned_neuron_states" in source, (
            "the command must be able to report without deleting"
        )

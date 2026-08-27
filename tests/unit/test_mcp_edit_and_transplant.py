"""Two MCP surfaces that quietly did less than they promised.

* `smem_edit` refreshes `content_hash` and the embedding when content changes
  (#166), but left `metadata["_structure"]` describing the OLD text. Recall
  reads that field back, so an edited structured memory kept answering with
  fields it no longer contains (#176).
* `TransplantFilter.min_salience` is validated, documented and applied — but no
  MCP caller ever set it, so `smem_transplant` always ran unfiltered with no way
  to ask otherwise (#175).
"""

from __future__ import annotations

import ast
import inspect
import pathlib

from surreal_memory.mcp.tool_schemas import get_tool_schemas

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "surreal_memory"


def _function(source: str, name: str) -> ast.AST:
    """The named function's AST node, sync or async."""
    return next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def _schema(name: str) -> dict:
    return next(s for s in get_tool_schemas() if s["name"] == name)


class TestEditRefreshesStructure:
    """Every field derived from content must be refreshed together."""

    def test_content_refresh_recomputes_structure(self) -> None:
        source = inspect.getsource(
            __import__(
                "surreal_memory.mcp.lifecycle_handler", fromlist=["_content_refreshed"]
            )._content_refreshed
        )
        assert "detect_structure" in source, (
            "_content_refreshed updates content_hash and the embedding but not "
            "metadata['_structure'], which recall reads back — an edited memory "
            "keeps answering with fields the new text no longer has"
        )

    def test_stale_structure_is_removed_not_just_overwritten(self) -> None:
        """Editing structured text into prose must drop the old fields.

        Overwriting only on a hit would leave the previous structure in place
        when the new content has none — the worst case, because recall would
        surface fields that no longer exist anywhere.
        """
        source = (SRC_ROOT / "mcp" / "lifecycle_handler.py").read_text(encoding="utf-8")
        func = _function(source, "_content_refreshed")
        pops = [
            n
            for n in ast.walk(func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "pop"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "_structure"
        ]
        assert pops, "no branch removes a stale _structure when the new content has none"


class TestTransplantExposesMinSalience:
    def test_the_tool_schema_advertises_it(self) -> None:
        props = _schema("smem_transplant")["inputSchema"]["properties"]
        assert "min_salience" in props, (
            "min_salience is validated and applied by the engine but unreachable "
            "through MCP — callers always transplant unfiltered"
        )

    def test_the_schema_states_the_valid_range(self) -> None:
        """An out-of-range value raises from the engine; say so up front."""
        prop = _schema("smem_transplant")["inputSchema"]["properties"]["min_salience"]
        assert prop.get("minimum") == 0.0
        assert prop.get("maximum") == 1.0

    def test_the_handler_passes_it_through(self) -> None:
        source = (SRC_ROOT / "mcp" / "evolution_handler.py").read_text(encoding="utf-8")
        calls = [
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "TransplantFilter"
        ]
        assert calls, "no TransplantFilter construction found"
        passed = {kw.arg for call in calls for kw in call.keywords}
        assert "min_salience" in passed, (
            "the handler builds TransplantFilter without min_salience, so the "
            "schema would advertise an argument that is silently dropped"
        )


class TestEditStructureBehaviour:
    """The structural tests above pin the shape; these pin the outcome."""

    async def test_editing_into_new_structure_replaces_the_old_fields(self) -> None:
        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.mcp.lifecycle_handler import _content_refreshed
        from surreal_memory.storage.memory_store import InMemoryStorage

        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="name: old\nrole: tester",
            metadata={"_structure": {"format": "yaml", "fields": [{"name": "name"}]}},
        )

        updated = await _content_refreshed(
            InMemoryStorage(), neuron, "name: new\nrole: reviewer\nteam: platform"
        )

        fields = {f["name"] for f in updated.metadata["_structure"]["fields"]}
        assert "team" in fields, "structure was not recomputed from the new content"

    async def test_editing_structure_away_drops_it_entirely(self) -> None:
        """Recall reads _structure back; a leftover would surface dead fields."""
        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.mcp.lifecycle_handler import _content_refreshed
        from surreal_memory.storage.memory_store import InMemoryStorage

        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="name: old\nrole: tester",
            metadata={"_structure": {"format": "yaml", "fields": [{"name": "name"}]}},
        )

        updated = await _content_refreshed(
            InMemoryStorage(), neuron, "just some prose with no fields at all"
        )

        assert "_structure" not in updated.metadata, (
            "stale structure survived an edit into unstructured text"
        )

"""K5 (run 013) — codebase indexer fixes.

1. co_occurrence edge cap (codebase_encoder._build_file_result): previously a
   file with >5 symbols got ZERO co-occurrence edges; now every file gets up to
   _MAX_CO_OCCUR_EDGES (20), closest pairs by line proximity first.
2. import-key fix (extraction/codebase._extract_import): the relationship target
   is the imported SYMBOL name, not the dotted module path — so the encoder's
   symbol_id_map lookup now matches and the import edge actually connects.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.codebase_encoder import _MAX_CO_OCCUR_EDGES, CodebaseEncoder
from surreal_memory.extraction.codebase import PythonExtractor


@pytest.fixture
def extractor() -> PythonExtractor:
    return PythonExtractor()


class TestCoOccurEdgeCap:
    @pytest.mark.asyncio
    async def test_large_file_gets_co_occurs_edges(self, tmp_path: Path) -> None:
        """A file with 12 symbols (>5) must still get co-occurrence edges.

        Pre-fix this file got zero — the `<= 5` guard skipped the whole block.
        """
        funcs = "\n".join(f"    def f{i}(self) -> int:\n        return {i}\n" for i in range(12))
        source = f"""\
            class Big:
            {funcs}
"""
        # textwrap.dedent above mangles the leading spaces; build cleanly instead
        body = "".join(f"    def f{i}(self) -> int:\n        return {i}\n\n" for i in range(12))
        source = "class Big:\n\n" + body
        f = tmp_path / "big.py"
        f.write_text(source, encoding="utf-8")

        from unittest.mock import MagicMock

        encoder = CodebaseEncoder(MagicMock(), MagicMock())
        result = encoder._build_file_result(f)
        co_occurs = [s for s in result.synapses_created if s.type == SynapseType.CO_OCCURS]
        assert len(co_occurs) > 0, "large file got zero co-occurrence edges"
        assert len(co_occurs) <= _MAX_CO_OCCUR_EDGES

    @pytest.mark.asyncio
    async def test_small_file_keeps_all_pairs(self, tmp_path: Path) -> None:
        """A 4-symbol file keeps all C(4,2)=6 pairs (well under the budget)."""
        source = textwrap.dedent("""\
            class A:
                def a1(self) -> None: pass
                def a2(self) -> None: pass
            def b() -> None: pass
        """)
        f = tmp_path / "small.py"
        f.write_text(source, encoding="utf-8")

        from unittest.mock import MagicMock

        encoder = CodebaseEncoder(MagicMock(), MagicMock())
        result = encoder._build_file_result(f)
        co_occurs = [s for s in result.synapses_created if s.type == SynapseType.CO_OCCURS]
        # 4 symbol neurons → 6 pairs; all under the 20-edge budget.
        assert len(co_occurs) == 6


class TestImportKeyFix:
    def test_from_import_target_is_symbol_name(
        self, extractor: PythonExtractor, tmp_path: Path
    ) -> None:
        """`from pathlib import Path` → relationship target "Path" (the neuron).

        Previously the target was "pathlib.Path", which never matched the
        encoder's symbol_id_map key ("Path"), so the edge was dropped.
        """
        source = "from pathlib import Path\n"
        f = tmp_path / "imp.py"
        f.write_text(source, encoding="utf-8")
        _, relationships = extractor.extract_file(f)
        import_rels = [r for r in relationships if r.relation == "imports"]
        assert len(import_rels) == 1
        assert import_rels[0].target == "Path"

    def test_import_as_target_is_asname(self, extractor: PythonExtractor, tmp_path: Path) -> None:
        """`import numpy as np` → target "np" (the symbol created), not "numpy"."""
        source = "import numpy as np\n"
        f = tmp_path / "imp.py"
        f.write_text(source, encoding="utf-8")
        _, relationships = extractor.extract_file(f)
        import_rels = [r for r in relationships if r.relation == "imports"]
        assert len(import_rels) == 1
        assert import_rels[0].target == "np"

    @pytest.mark.asyncio
    async def test_import_edge_connects_to_symbol_neuron(self, tmp_path: Path) -> None:
        """End-to-end: the import synapse survives the symbol_id_map resolution.

        The encoder resolves relationships against symbol_id_map (keyed by the
        symbol name). With the target now matching the name, the file→import
        synapse is created instead of silently dropped.
        """
        source = textwrap.dedent("""\
            import os
            from pathlib import Path

            def main() -> None:
                pass
        """)
        f = tmp_path / "imp.py"
        f.write_text(source, encoding="utf-8")

        from unittest.mock import MagicMock

        encoder = CodebaseEncoder(MagicMock(), MagicMock())
        result = encoder._build_file_result(f)
        # "imports" maps to RELATED_TO. At least one file→import edge should
        # resolve (os and Path both have symbol neurons).
        related = [s for s in result.synapses_created if s.type == SynapseType.RELATED_TO]
        assert len(related) >= 2, (
            f"expected ≥2 import (RELATED_TO) edges resolving, got {len(related)}"
        )

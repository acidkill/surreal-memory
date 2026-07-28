"""Write-time dedup must run on every write path, and its alias is not an anchor.

Both halves of this file guard a bug that was measured on the live brain:

* ``build_dedup_pipeline`` was inlined in one handler, so seventeen other write
  paths silently skipped dedup. Four CLI writes of byte-identical content
  produced four separate anchors.
* the alias minted on a dedup hit carried ``is_anchor: True``, so it re-entered
  the census and the candidate pool as a fresh duplicate on the next pass.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass
from typing import Any

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.dedup.factory import build_dedup_pipeline
from surreal_memory.engine.pipeline import PipelineContext
from surreal_memory.engine.pipeline_steps import CreateAnchorStep
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "surreal_memory"

# Bulk ingestion paths that deliberately construct an encoder without dedup.
# Each carries a comment at the call site explaining why; keep the two in sync.
DEDUP_EXEMPT = {
    "engine/doc_trainer.py",  # thousands of chunks; near-identical is expected
    "engine/db_trainer.py",  # schema rows are legitimately similar
    "integration/mapper.py",  # batch import; caller owns identity
}


@dataclass
class _Settings:
    enabled: bool = True
    simhash_threshold: int = 7
    embedding_threshold: float = 0.85
    embedding_ambiguous_low: float = 0.75
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_max_pairs_per_encode: int = 3
    merge_strategy: str = "keep_newer"
    max_candidates: int = 30


@dataclass
class _Config:
    dedup: Any


class TestBuildDedupPipeline:
    def test_returns_a_pipeline_when_enabled(self) -> None:
        pipeline = build_dedup_pipeline(InMemoryStorage(), _Config(dedup=_Settings()))
        assert pipeline is not None

    def test_returns_none_when_disabled(self) -> None:
        """Wiring dedup everywhere must not switch it on for people who said no."""
        pipeline = build_dedup_pipeline(InMemoryStorage(), _Config(dedup=_Settings(enabled=False)))
        assert pipeline is None

    def test_returns_none_when_settings_are_malformed(self) -> None:
        """A broken config must not fail the write -- storing beats deduping."""
        pipeline = build_dedup_pipeline(InMemoryStorage(), _Config(dedup=object()))
        assert pipeline is None

    def test_settings_reach_the_pipeline(self) -> None:
        pipeline = build_dedup_pipeline(
            InMemoryStorage(),
            _Config(dedup=_Settings(simhash_threshold=3, max_candidates=11)),
        )
        assert pipeline is not None
        assert pipeline._config.simhash_threshold == 3
        assert pipeline._config.max_candidates == 11


class TestEveryWritePathIsWired:
    """A source-level invariant, so a *new* write path cannot re-introduce B6.

    Unit-testing each of the eighteen call sites individually would not catch
    the nineteenth. Reading the tree does.
    """

    @staticmethod
    def _encoder_constructions() -> list[tuple[str, int, bool]]:
        found: list[tuple[str, int, bool]] = []
        for path in SRC_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name != "MemoryEncoder":
                    continue
                wired = any(kw.arg == "dedup_pipeline" for kw in node.keywords)
                rel = path.relative_to(SRC_ROOT).as_posix()
                found.append((rel, node.lineno, wired))
        return found

    def test_the_scan_finds_the_call_sites(self) -> None:
        """Guard the guard: an empty scan would make this suite vacuously green."""
        assert len(self._encoder_constructions()) >= 15

    def test_no_unexempted_write_path_skips_dedup(self) -> None:
        unwired = [
            f"{rel}:{line}"
            for rel, line, wired in self._encoder_constructions()
            if not wired and rel not in DEDUP_EXEMPT
        ]
        assert not unwired, (
            "these MemoryEncoder call sites build no dedup pipeline, so "
            f"DedupCheckStep will silently skip them: {unwired}. Either pass "
            "dedup_pipeline=build_dedup_pipeline(storage) or add the file to "
            "DEDUP_EXEMPT with a comment saying why."
        )

    @pytest.mark.parametrize("exempt", sorted(DEDUP_EXEMPT))
    def test_exemptions_are_real_files_that_really_skip_dedup(self, exempt: str) -> None:
        """Stop the allowlist rotting into a list of files that no longer exist."""
        assert (SRC_ROOT / exempt).is_file()
        sites = [c for c in self._encoder_constructions() if c[0] == exempt]
        assert sites, f"{exempt} no longer constructs a MemoryEncoder -- drop the exemption"
        assert any(not wired for _, _, wired in sites)


class TestDedupAliasIsNotAnAnchor:
    @staticmethod
    def _storage() -> InMemoryStorage:
        store = InMemoryStorage()
        brain = Brain.create(name="dedup-wiring", config=BrainConfig(), owner_id="test")
        store._brains[brain.id] = brain
        store.set_brain(brain.id)
        return store

    @staticmethod
    def _ctx(content: str) -> PipelineContext:
        ctx = PipelineContext(
            content=content,
            timestamp=utcnow(),
            metadata={},
            tags=set(),
            language="en",
        )
        ctx.effective_metadata = {}
        return ctx

    async def test_alias_from_a_dedup_hit_is_not_an_anchor(self) -> None:
        storage = self._storage()
        canonical = Neuron.create(
            type=NeuronType.CONCEPT,
            content="the canonical memory",
            metadata={"is_anchor": True},
        )
        await storage.add_neuron(canonical)

        ctx = self._ctx("the canonical memory")
        ctx.effective_metadata["_dedup_reused_anchor"] = canonical

        ctx = await CreateAnchorStep().execute(ctx, storage, BrainConfig())

        assert ctx.anchor_neuron is not None
        assert ctx.anchor_neuron.metadata["is_anchor"] is False
        assert ctx.anchor_neuron.metadata["_dedup_alias_of"] == canonical.id

    async def test_the_fiber_still_resolves_an_anchor(self) -> None:
        """Not being *flagged* an anchor must not stop it *being* the fiber's anchor."""
        storage = self._storage()
        canonical = Neuron.create(
            type=NeuronType.CONCEPT,
            content="the canonical memory",
            metadata={"is_anchor": True},
        )
        await storage.add_neuron(canonical)

        ctx = self._ctx("the canonical memory")
        ctx.effective_metadata["_dedup_reused_anchor"] = canonical

        ctx = await CreateAnchorStep().execute(ctx, storage, BrainConfig())

        assert ctx.anchor_neuron is not None
        assert await storage.get_neuron(ctx.anchor_neuron.id) is not None

    async def test_a_genuinely_new_memory_is_still_an_anchor(self) -> None:
        storage = self._storage()

        ctx = await CreateAnchorStep().execute(
            self._ctx("something nobody has stored before"), storage, BrainConfig()
        )

        assert ctx.anchor_neuron is not None
        assert ctx.anchor_neuron.metadata["is_anchor"] is True
        assert "_dedup_alias_of" not in ctx.anchor_neuron.metadata

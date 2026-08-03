"""Codebase encoder: converts extracted code symbols into neural graph structures.

Maps source code into neurons, synapses, and fibers using the
existing Surreal-Memory types. No external dependencies.

Neuron type mapping:
    File path   → SPATIAL   (location in codebase)
    Function    → ACTION    (executable behavior)
    Class       → CONCEPT   (abstract structure)
    Method      → ACTION    (executable behavior, metadata.parent = class)
    Import      → ENTITY    (named reference)
    Constant    → ENTITY    (named value)

Synapse mapping:
    contains    → CONTAINS  (weight 1.0)
    is_a        → IS_A      (weight 0.9)
    imports     → RELATED_TO (weight 0.7)
    co_occurs   → CO_OCCURS  (weight 0.5)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.encoder import EncodingResult
from surreal_memory.extraction.codebase import CodeSymbolType, get_extractor
from surreal_memory.utils.simhash import is_near_duplicate, simhash

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.storage.base import NeuralStorage

_SYMBOL_TYPE_TO_NEURON: dict[CodeSymbolType, NeuronType] = {
    CodeSymbolType.FUNCTION: NeuronType.ACTION,
    CodeSymbolType.CLASS: NeuronType.CONCEPT,
    CodeSymbolType.METHOD: NeuronType.ACTION,
    CodeSymbolType.IMPORT: NeuronType.ENTITY,
    CodeSymbolType.CONSTANT: NeuronType.ENTITY,
}

_RELATION_TO_SYNAPSE: dict[str, tuple[SynapseType, float]] = {
    "contains": (SynapseType.CONTAINS, 1.0),
    "is_a": (SynapseType.IS_A, 0.9),
    "imports": (SynapseType.RELATED_TO, 0.7),
    "co_occurs": (SynapseType.CO_OCCURS, 0.5),
}

_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
    }
)
_DEFAULT_EXCLUDE: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        "build",
        "dist",
        ".next",
        "vendor",
        # Claude Code tooling directory. Its worktrees/ subdirectory holds
        # full nested checkouts of the same repo (one per agent session), so
        # without this a single `smem index` walks the source tree N+1 times.
        ".claude",
    }
)


class CodebaseEncoder:
    """Encodes source code into the neural memory graph."""

    def __init__(self, storage: NeuralStorage, config: BrainConfig) -> None:
        self._storage = storage
        self._config = config

    def _build_file_result(
        self,
        file_path: Path,
        *,
        mtime: float | None = None,
        content_simhash: int | None = None,
        tags: set[str] | None = None,
    ) -> EncodingResult:
        """Parse a file and build its neurons/synapses/fiber WITHOUT persisting them.

        Pure CPU + a single blocking read (via the extractor) — no storage I/O,
        so many files can be built up-front and flushed together in one batch.
        `mtime`/`content_simhash`, when given, ride along on the file neuron's
        metadata so a later `index_directory` run can tell whether this file
        changed without re-reading every file up front.
        """
        extractor = get_extractor(file_path.suffix)
        symbols, relationships = extractor.extract_file(file_path)

        neurons_created: list[Neuron] = []
        synapses_created: list[Synapse] = []

        # 1. Create file neuron (SPATIAL)
        file_metadata: dict[str, Any] = {
            "indexed": True,
            "symbol_count": len(symbols),
        }
        if mtime is not None:
            file_metadata["mtime"] = mtime
        if content_simhash is not None:
            file_metadata["content_simhash"] = content_simhash

        file_neuron = Neuron.create(
            type=NeuronType.SPATIAL,
            content=str(file_path),
            metadata=file_metadata,
        )
        neurons_created.append(file_neuron)

        # 2. Create symbol neurons
        symbol_id_map: dict[str, str] = {str(file_path): file_neuron.id}

        for sym in symbols:
            neuron_type = _SYMBOL_TYPE_TO_NEURON.get(sym.symbol_type, NeuronType.ENTITY)
            metadata: dict[str, Any] = {
                "symbol_type": sym.symbol_type.value,
                "file_path": sym.file_path,
                "line_start": sym.line_start,
                "line_end": sym.line_end,
                "indexed": True,
            }
            if sym.signature:
                metadata["signature"] = sym.signature
            if sym.docstring:
                metadata["docstring"] = sym.docstring
            if sym.parent:
                metadata["parent"] = sym.parent

            # Build a unique key for this symbol
            sym_key = f"{sym.parent}.{sym.name}" if sym.parent else sym.name

            neuron = Neuron.create(
                type=neuron_type,
                content=sym_key,
                metadata=metadata,
            )
            neurons_created.append(neuron)
            symbol_id_map[sym_key] = neuron.id

        # 3. Create synapses from relationships
        for rel in relationships:
            source_id = symbol_id_map.get(rel.source)
            target_id = symbol_id_map.get(rel.target)

            if not source_id or not target_id:
                continue

            synapse_info = _RELATION_TO_SYNAPSE.get(rel.relation)
            if not synapse_info:
                continue

            synapse_type, weight = synapse_info
            synapse = Synapse.create(
                source_id=source_id,
                target_id=target_id,
                type=synapse_type,
                weight=weight,
            )
            synapses_created.append(synapse)

        # 4. Create co-occurrence synapses (capped to avoid O(n²) explosion)
        symbol_neurons = neurons_created[1:]  # Skip file neuron
        max_co_occurs = 5  # Max files: create all pairs; large files: skip
        if len(symbol_neurons) <= max_co_occurs:
            for i, neuron_a in enumerate(symbol_neurons):
                for neuron_b in symbol_neurons[i + 1 :]:
                    synapse = Synapse.create(
                        source_id=neuron_a.id,
                        target_id=neuron_b.id,
                        type=SynapseType.CO_OCCURS,
                        weight=0.5,
                    )
                    synapses_created.append(synapse)

        # 5. Bundle into fiber
        neuron_ids = {n.id for n in neurons_created}
        synapse_ids = {s.id for s in synapses_created}

        fiber = Fiber.create(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=file_neuron.id,
            summary=f"Code index: {file_path.name}",
            tags=(tags or set()) | {"code_index"},
        )

        return EncodingResult(
            fiber=fiber,
            neurons_created=neurons_created,
            neurons_linked=[],
            synapses_created=synapses_created,
        )

    async def index_file(
        self,
        file_path: Path,
        tags: set[str] | None = None,
    ) -> EncodingResult:
        """Index a single source file into neural graph, persisting immediately.

        For indexing a whole directory, prefer `index_directory`: it batches
        writes across every file into a handful of round-trips instead of one
        set of round-trips per file.

        Args:
            file_path: Path to the source file.
            tags: Optional tags for the fiber.

        Returns:
            EncodingResult with created neurons, synapses, and fiber.
        """
        result = self._build_file_result(file_path, tags=tags)
        await self._storage.add_neurons_batch(result.neurons_created)
        await self._storage.add_synapses_batch(result.synapses_created)
        await self._storage.add_fibers_batch([result.fiber])
        return result

    def _collect_candidate_files(
        self,
        directory: Path,
        exts: set[str],
        excludes: set[str],
    ) -> list[Path]:
        """Walk `directory`, pruning excluded subdirectories in place.

        `os.walk` with the `dirnames[:]` filter below never descends into an
        excluded directory at all (`.git`, `node_modules`, N nested
        `.claude/worktrees` checkouts, ...) — unlike the previous
        `sorted(directory.rglob("*"))`, which materialized and `resolve()`d
        every path in the WHOLE tree before any filter ran (measured: 33s of a
        75s index run on this repo's own ~384k-path tree). `followlinks=True`
        matches `rglob`'s traversal on Python < 3.13 (the oldest version this
        package supports), so the symlink-escape check in `index_directory`
        still has something to guard against.
        """
        candidates: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(directory, followlinks=True):
            dirnames[:] = [d for d in dirnames if d not in excludes]
            base = Path(dirpath)
            for filename in filenames:
                file_path = base / filename
                if file_path.suffix in exts:
                    candidates.append(file_path)
        return candidates

    async def _delete_previous_index(self, old_file_neuron: Neuron) -> None:
        """Remove a changed file's stale neurons/synapses/fiber before rebuilding.

        Sequential deletes, deliberately: SurrealDB raises a hard write
        conflict under concurrent deletes to the same tables (see
        `SurrealDBStorage.delete_neurons_batch`'s docstring) — this only ever
        touches one file's worth of entities per call, so sequential is cheap.
        """
        old_fibers = await self._storage.find_fibers(contains_neuron=old_file_neuron.id, limit=1)
        if not old_fibers:
            return
        old_fiber = old_fibers[0]
        for nid in old_fiber.neuron_ids:
            await self._storage.delete_neuron(nid)
        for sid in old_fiber.synapse_ids:
            await self._storage.delete_synapse(sid)
        await self._storage.delete_fiber(old_fiber.id)

    async def _clear_code_index(self) -> None:
        """Wipe every existing `code_index` fiber and its members (for `force=True`)."""
        while True:
            batch = await self._storage.find_fibers(tags={"code_index"}, limit=2000)
            if not batch:
                return
            for fiber in batch:
                for nid in fiber.neuron_ids:
                    await self._storage.delete_neuron(nid)
                for sid in fiber.synapse_ids:
                    await self._storage.delete_synapse(sid)
                await self._storage.delete_fiber(fiber.id)

    async def index_directory(
        self,
        directory: Path,
        extensions: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        tags: set[str] | None = None,
        *,
        force: bool = False,
    ) -> list[EncodingResult]:
        """Index all matching files in a directory recursively.

        Unchanged files (same mtime, or a touched-but-content-identical file,
        detected by simhash) are skipped entirely: previously every run
        re-indexed and duplicated every neuron/synapse/fiber for every file,
        every time. Writes are batched across the whole directory rather than
        issued per file.

        Args:
            directory: Root directory to scan.
            extensions: File extensions to index. Defaults to common source extensions.
            exclude_patterns: Directory names to skip.
            tags: Optional tags for all created fibers.
            force: Wipe the existing code index first and re-index every
                matching file, ignoring change tracking.

        Returns:
            List of EncodingResult, one per (re-)indexed file.
        """
        exts = extensions if extensions is not None else set(_DEFAULT_EXTENSIONS)
        excludes = exclude_patterns if exclude_patterns is not None else set(_DEFAULT_EXCLUDE)

        if force:
            await self._clear_code_index()

        resolved_base = directory.resolve()
        candidates = sorted(self._collect_candidate_files(directory, exts, excludes))

        existing_by_path: dict[str, Neuron] = {}
        if not force and candidates:
            existing_by_path = await self._storage.find_neurons_exact_batch(
                [str(p) for p in candidates], type=NeuronType.SPATIAL
            )

        all_neurons: list[Neuron] = []
        all_synapses: list[Synapse] = []
        all_fibers: list[Fiber] = []
        results: list[EncodingResult] = []

        for file_path in candidates:
            if not file_path.is_file():
                continue
            # Validate resolved path stays within base directory (symlink escape prevention)
            if not file_path.resolve().is_relative_to(resolved_base):
                continue

            old_neuron = existing_by_path.get(str(file_path))
            mtime = file_path.stat().st_mtime
            content_simhash: int | None = None

            if old_neuron is not None:
                old_mtime = old_neuron.metadata.get("mtime")
                if old_mtime is not None and mtime <= float(old_mtime):
                    continue  # unchanged: same or older mtime than last index

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    content_simhash = simhash(content)
                except OSError:
                    content_simhash = None

                old_simhash = old_neuron.metadata.get("content_simhash")
                if (
                    content_simhash is not None
                    and old_simhash is not None
                    and is_near_duplicate(content_simhash, int(old_simhash))
                ):
                    continue  # mtime touched, content unchanged

                await self._delete_previous_index(old_neuron)

            try:
                result = self._build_file_result(
                    file_path,
                    mtime=mtime,
                    content_simhash=content_simhash,
                    tags=tags,
                )
            except (SyntaxError, UnicodeDecodeError):
                logger.debug("Skipping %s due to parse/decode error", file_path, exc_info=True)
                continue

            results.append(result)
            all_neurons.extend(result.neurons_created)
            all_synapses.extend(result.synapses_created)
            all_fibers.append(result.fiber)

        await self._storage.add_neurons_batch(all_neurons)
        await self._storage.add_synapses_batch(all_synapses)
        await self._storage.add_fibers_batch(all_fibers)

        return results

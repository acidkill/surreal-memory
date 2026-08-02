"""Tests for training file tracking and resume."""

from __future__ import annotations

from pathlib import Path

from surreal_memory.utils.file_hash import compute_file_hash


class TestComputeFileHash:
    """Test file hashing utility."""

    def test_consistent_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Hello World", encoding="utf-8")

        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Content A", encoding="utf-8")
        f2.write_text("Content B", encoding="utf-8")

        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Same content", encoding="utf-8")
        f2.write_text("Same content", encoding="utf-8")

        assert compute_file_hash(f1) == compute_file_hash(f2)


class TestTrainingFilesStorage:
    """Training file CRUD against every backend that stores memories.

    These used to instantiate ``SQLiteStorage`` directly, which is how the gap
    survived: the four methods existed on no other backend, ``DocTrainer``
    probed for them with ``hasattr``, and so on SurrealDB every ``smem train``
    re-encoded the entire corpus and duplicated it.
    """

    async def test_upsert_and_lookup(self, pin_storage) -> None:
        storage = pin_storage

        record_id = await storage.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=10,
            chunks_completed=10,
            status="completed",
            domain_tag="react",
        )
        assert record_id

        record = await storage.get_training_file_by_hash("abc123")
        assert record is not None
        assert record["status"] == "completed"
        assert record["file_path"] == "/docs/test.md"

        assert await storage.get_training_file_by_hash("nonexistent") is None

    async def test_upsert_updates_existing(self, pin_storage) -> None:
        storage = pin_storage

        record_id = await storage.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=5,
            chunks_completed=3,
            status="in_progress",
        )

        # Same hash → update in place, not a second row.
        record_id2 = await storage.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=5,
            chunks_completed=5,
            status="completed",
        )
        assert record_id == record_id2

        record = await storage.get_training_file_by_hash("abc123")
        assert record is not None
        assert record["status"] == "completed"
        assert record["chunks_completed"] == 5

    async def test_update_progress_supports_resume(self, pin_storage) -> None:
        storage = pin_storage
        record_id = await storage.upsert_training_file(
            file_hash="resume-me",
            file_path="/docs/big.md",
            file_size=4096,
            chunks_total=10,
            chunks_completed=0,
            status="in_progress",
        )

        await storage.update_training_file_progress(record_id, chunks_completed=4)
        record = await storage.get_training_file_by_hash("resume-me")
        assert record is not None
        assert record["chunks_completed"] == 4
        assert record["status"] == "in_progress"

        await storage.update_training_file_progress(
            record_id, chunks_completed=10, status="completed"
        )
        record = await storage.get_training_file_by_hash("resume-me")
        assert record is not None
        assert record["chunks_completed"] == 10
        assert record["status"] == "completed"

    async def test_training_stats(self, pin_storage) -> None:
        storage = pin_storage

        await storage.upsert_training_file(
            file_hash="h1",
            file_path="a.md",
            file_size=100,
            chunks_total=5,
            chunks_completed=5,
            status="completed",
        )
        await storage.upsert_training_file(
            file_hash="h2",
            file_path="b.md",
            file_size=200,
            chunks_total=3,
            chunks_completed=1,
            status="in_progress",
        )

        stats = await storage.get_training_stats()
        assert stats["total_files"] == 2
        assert stats["completed"] == 1
        assert stats["in_progress"] == 1
        assert stats["total_chunks"] == 6

    async def test_stats_empty_brain(self, pin_storage) -> None:
        assert await pin_storage.get_training_stats() == {
            "total_files": 0,
            "completed": 0,
            "in_progress": 0,
            "failed": 0,
            "total_chunks": 0,
        }


class TestTrainingDedupEndToEnd:
    """Re-training an unchanged corpus must be a no-op on every backend."""

    async def test_second_run_skips_unchanged_files(self, pin_storage, tmp_path) -> None:
        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine.doc_trainer import DocTrainer, TrainingConfig

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "guide.md").write_text(
            "# Guide\n\n"
            + (
                "The deployment pipeline uses Kubernetes and Helm charts to roll "
                "out services across the staging cluster. " * 6
            ),
            encoding="utf-8",
        )

        trainer = DocTrainer(pin_storage, BrainConfig())
        tc = TrainingConfig(consolidate=False)

        first = await trainer.train_directory(docs, training_config=tc)
        assert first.files_processed == 1
        assert first.chunks_encoded > 0
        fibers_after_first = len(await pin_storage.get_fibers(limit=10_000))

        second = await trainer.train_directory(docs, training_config=tc)
        assert second.files_processed == 0, "unchanged file must be skipped on re-train"
        assert second.chunks_encoded == 0
        assert len(await pin_storage.get_fibers(limit=10_000)) == fibers_after_first

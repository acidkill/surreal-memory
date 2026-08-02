"""Tests for multi-agent brain isolation via env var pinning.

Verifies that SURREAL_MEMORY_BRAIN / SURREAL_MEMORY_BRAIN env vars provide
process-level brain isolation without mutating shared config state.
Covers both MCP server (get_shared_storage) and CLI (_helpers.get_storage).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEnvVarBrainPinning:
    """get_shared_storage() must use env var directly without mutating config."""

    @pytest.mark.asyncio
    async def test_env_var_does_not_mutate_config(self, tmp_path: object) -> None:
        """SURREAL_MEMORY_BRAIN should be used directly, not written to config.current_brain."""
        mock_config = MagicMock()
        mock_config.current_brain = "original-brain"
        mock_config.storage_backend = "memory"
        mock_config.get_brain_db_path.return_value = tmp_path / "env-brain.db"  # type: ignore[operator]

        mock_storage = AsyncMock()
        mock_storage._conn = True  # simulate open connection

        with (
            patch("surreal_memory.unified_config.get_config", return_value=mock_config),
            patch.dict(os.environ, {"SURREAL_MEMORY_BRAIN": "env-brain"}, clear=False),
            patch(
                "surreal_memory.unified_config._get_memory_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_memory,
        ):
            from surreal_memory.unified_config import get_shared_storage

            await get_shared_storage()

            # Config must NOT be mutated
            assert mock_config.current_brain == "original-brain"

            # Storage factory must receive the env var brain name
            mock_get_memory.assert_awaited_once()
            call_args = mock_get_memory.call_args
            assert call_args[0][1] == "env-brain"  # name argument

    @pytest.mark.asyncio
    async def test_no_env_var_reads_from_disk(self, tmp_path: object) -> None:
        """Without env var, should read from config.toml (existing behavior)."""
        mock_config = MagicMock()
        mock_config.current_brain = "config-brain"
        mock_config.storage_backend = "memory"

        mock_storage = AsyncMock()

        with (
            patch("surreal_memory.unified_config.get_config", return_value=mock_config),
            patch.dict(os.environ, {}, clear=False),
            patch(
                "surreal_memory.unified_config._read_current_brain_from_toml",
                return_value="disk-brain",
            ),
            patch(
                "surreal_memory.unified_config._get_memory_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_memory,
        ):
            # Ensure env vars are NOT set
            os.environ.pop("SURREAL_MEMORY_BRAIN", None)

            from surreal_memory.unified_config import get_shared_storage

            await get_shared_storage()

            # Config SHOULD be updated from disk (existing behavior)
            assert mock_config.current_brain == "disk-brain"
            call_args = mock_get_memory.call_args
            assert call_args[0][1] == "disk-brain"

    @pytest.mark.asyncio
    async def test_explicit_brain_name_overrides_everything(self, tmp_path: object) -> None:
        """Explicit brain_name param should override env var and config."""
        mock_config = MagicMock()
        mock_config.current_brain = "config-brain"
        mock_config.storage_backend = "memory"

        mock_storage = AsyncMock()

        with (
            patch("surreal_memory.unified_config.get_config", return_value=mock_config),
            patch.dict(os.environ, {"SURREAL_MEMORY_BRAIN": "env-brain"}, clear=False),
            patch(
                "surreal_memory.unified_config._get_memory_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_memory,
        ):
            from surreal_memory.unified_config import get_shared_storage

            await get_shared_storage(brain_name="explicit-brain")

            # Explicit param wins
            call_args = mock_get_memory.call_args
            assert call_args[0][1] == "explicit-brain"
            # Config untouched
            assert mock_config.current_brain == "config-brain"

    @pytest.mark.asyncio
    async def test_concurrent_env_vars_isolate_brains(self) -> None:
        """Simulate two 'processes' with different env vars — config stays clean."""
        mock_config = MagicMock()
        mock_config.current_brain = "default"
        mock_config.storage_backend = "memory"

        mock_storage = AsyncMock()

        with (
            patch("surreal_memory.unified_config.get_config", return_value=mock_config),
            patch(
                "surreal_memory.unified_config._get_memory_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_memory,
        ):
            from surreal_memory.unified_config import get_shared_storage

            # Agent A
            with patch.dict(os.environ, {"SURREAL_MEMORY_BRAIN": "brain-a"}):
                await get_shared_storage()
                assert mock_get_memory.call_args[0][1] == "brain-a"

            # Agent B
            with patch.dict(os.environ, {"SURREAL_MEMORY_BRAIN": "brain-b"}):
                await get_shared_storage()
                assert mock_get_memory.call_args[0][1] == "brain-b"

            # Config never mutated by either
            assert mock_config.current_brain == "default"


class TestCLIEnvVarBrainPinning:
    """CLI get_storage() must respect SURREAL_MEMORY_BRAIN / SURREAL_MEMORY_BRAIN env vars."""

    @pytest.mark.asyncio
    async def test_cli_respects_smem_brain_env_var(self) -> None:
        """CLI get_storage uses SURREAL_MEMORY_BRAIN when no explicit brain_name given."""
        mock_cli_config = MagicMock()
        mock_cli_config.current_brain = "config-brain"
        mock_cli_config.is_shared_mode = False
        mock_cli_config.use_unified_storage = True

        mock_storage = AsyncMock()

        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_BRAIN": "env-brain"}, clear=False),
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_shared,
        ):
            from surreal_memory.cli._helpers import get_storage

            result = await get_storage(mock_cli_config)

            # Should call get_shared_storage with env var brain, not config brain
            mock_get_shared.assert_awaited_once_with("env-brain")
            assert result is mock_storage

    @pytest.mark.asyncio
    async def test_cli_falls_back_to_config_without_env_var(self) -> None:
        """Without env var, CLI uses config.current_brain (existing behavior)."""
        mock_cli_config = MagicMock()
        mock_cli_config.current_brain = "config-brain"
        mock_cli_config.is_shared_mode = False
        mock_cli_config.use_unified_storage = True

        mock_storage = AsyncMock()

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_shared,
        ):
            os.environ.pop("SURREAL_MEMORY_BRAIN", None)

            from surreal_memory.cli._helpers import get_storage

            await get_storage(mock_cli_config)

            mock_get_shared.assert_awaited_once_with("config-brain")

    @pytest.mark.asyncio
    async def test_cli_explicit_brain_name_overrides_env_var(self) -> None:
        """Explicit brain_name arg overrides env var."""
        mock_cli_config = MagicMock()
        mock_cli_config.current_brain = "config-brain"
        mock_cli_config.is_shared_mode = False
        mock_cli_config.use_unified_storage = True

        mock_storage = AsyncMock()

        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_BRAIN": "env-brain"}, clear=False),
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_shared,
        ):
            from surreal_memory.cli._helpers import get_storage

            await get_storage(mock_cli_config, brain_name="explicit-brain")

            mock_get_shared.assert_awaited_once_with("explicit-brain")

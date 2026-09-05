"""SURREALDB_AUTH_LEVEL selects the sign-in scope, and defaults to root.

SurrealDB accepts a root user only when the sign-in payload carries neither
namespace nor database, and a user defined ON DATABASE only when it carries
both; the intermediate shape is rejected for either. Sign-in sent credentials
alone, unconditionally, so a database-scoped user could not connect at all.
Adding the scope unconditionally would have been worse — DEFAULT_USER is root,
so it would reject every stock installation — hence a setting, defaulting to
the shape that works today.
"""

from __future__ import annotations

import pytest

from surreal_memory.storage.surrealdb.connection import (
    DEFAULT_AUTH_LEVEL,
    SurrealSettings,
    signin_payload,
)


class TestSigninPayload:
    def test_root_sends_credentials_alone(self) -> None:
        assert signin_payload("u", "p", "ns", "db", "root") == {"username": "u", "password": "p"}

    def test_namespace_adds_the_namespace_only(self) -> None:
        assert signin_payload("u", "p", "ns", "db", "namespace") == {
            "username": "u",
            "password": "p",
            "namespace": "ns",
        }

    def test_database_adds_both(self) -> None:
        assert signin_payload("u", "p", "ns", "db", "database") == {
            "username": "u",
            "password": "p",
            "namespace": "ns",
            "database": "db",
        }

    def test_default_is_root(self) -> None:
        """The default must not change how a stock installation signs in."""
        assert DEFAULT_AUTH_LEVEL == "root"
        assert signin_payload("u", "p", "ns", "db") == signin_payload("u", "p", "ns", "db", "root")

    @pytest.mark.parametrize("raw", ["DATABASE", "DataBase", " database "])
    def test_level_is_case_and_space_insensitive(self, raw: str) -> None:
        assert "database" in signin_payload("u", "p", "ns", "db", raw)

    def test_unrecognised_level_falls_back_to_root(self) -> None:
        """An unknown value must not silently change the scope, nor refuse to connect."""
        assert signin_payload("u", "p", "ns", "db", "nonsense") == {
            "username": "u",
            "password": "p",
        }


class TestSettingsFromEnv:
    def test_env_var_selects_the_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SURREALDB_AUTH_LEVEL", "database")
        assert SurrealSettings.from_env().auth_level == "database"

    def test_unset_env_var_means_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SURREALDB_AUTH_LEVEL", raising=False)
        assert SurrealSettings.from_env().auth_level == "root"

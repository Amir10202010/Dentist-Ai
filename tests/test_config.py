"""Configuration that a deployment actually has to survive.

The database URL is the one setting an operator frequently cannot choose:
managed Postgres providers generate it and inject it, so the application has to
accept the shape it is given. These cases are the URLs those providers really
emit, which is why they are spelled out rather than reduced to one parametrised
happy path.
"""

from __future__ import annotations

import pytest

from dentist_ai.core.config import DatabaseSettings


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        # Render, Railway and Fly hand out the bare libpq scheme.
        (
            "postgres://u:pw@host:5432/db",
            "postgresql+asyncpg://u:pw@host:5432/db",
        ),
        (
            "postgresql://u:pw@host:5432/db",
            "postgresql+asyncpg://u:pw@host:5432/db",
        ),
        # Already correct: left exactly as-is.
        (
            "postgresql+asyncpg://u:pw@host:5432/db",
            "postgresql+asyncpg://u:pw@host:5432/db",
        ),
    ],
)
def test_a_managed_providers_url_is_accepted(provided: str, expected: str) -> None:
    assert DatabaseSettings(url=provided).url == expected


def test_sslmode_becomes_the_parameter_asyncpg_understands() -> None:
    """`sslmode` reaches asyncpg.connect() untouched and raises TypeError there.

    Render's external connection strings carry it, so the failure would land on
    the first query rather than at boot.
    """
    settings = DatabaseSettings(url="postgres://u:pw@host:5432/db?sslmode=require")
    assert settings.url == "postgresql+asyncpg://u:pw@host:5432/db?ssl=require"


def test_an_explicit_ssl_is_not_duplicated_by_the_translation() -> None:
    settings = DatabaseSettings(url="postgres://u:pw@host/db?ssl=verify-full&sslmode=require")
    assert settings.url.count("ssl=") == 1
    assert "sslmode" not in settings.url


def test_other_query_parameters_survive() -> None:
    settings = DatabaseSettings(
        url="postgres://u:pw@host/db?sslmode=require&application_name=dentist-ai"
    )
    assert "application_name=dentist-ai" in settings.url


def test_the_default_is_sqlite_and_reports_itself_as_such() -> None:
    assert DatabaseSettings().is_sqlite is True
    assert DatabaseSettings(url="postgres://u:pw@host/db").is_sqlite is False


def test_a_synchronous_driver_is_still_refused() -> None:
    """Rewriting the scheme must not become "accept anything"."""
    with pytest.raises(ValueError, match="async driver"):
        DatabaseSettings(url="postgresql+psycopg2://u:pw@host/db")


def test_an_unsupported_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="async driver"):
        DatabaseSettings(url="mysql+aiomysql://u:pw@host/db")

"""Alembic environment, driven by application settings."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from dentist_ai.core.config import get_settings
from dentist_ai.db.base import Base
from dentist_ai.db.models import (  # noqa: F401 - imported for metadata registration
    AuditEvent,
    Finding,
    Organization,
    Patient,
    Study,
    User,
)
from dentist_ai.db.session import create_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
settings = get_settings()


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # SQLite cannot ALTER most things in place; batch mode rewrites the
        # table so the same migration script works on both backends.
        render_as_batch=settings.database.is_sqlite,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database.url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_engine(settings)
    async with engine.connect() as connection:
        await connection.run_sync(lambda sync_conn: _configure(sync_conn))
        await connection.run_sync(lambda sync_conn: context.run_migrations())
        await connection.commit()
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

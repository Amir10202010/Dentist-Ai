"""Async engine and session lifecycle.

One pooled engine lives for the process lifetime; sessions are handed out by a
dependency that always commits or rolls back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry

from dentist_ai.core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    database = settings.database
    if database.is_sqlite:
        # SQLite has no server-side pool to size, and NullPool avoids
        # cross-task connection sharing in tests.
        engine = create_async_engine(database.url, echo=database.echo, future=True)
        _enable_sqlite_pragmas(engine)
        return engine

    return create_async_engine(
        database.url,
        echo=database.echo,
        pool_size=database.pool_size,
        max_overflow=database.max_overflow,
        pool_recycle=database.pool_recycle_seconds,
        pool_pre_ping=True,
        future=True,
    )


def _enable_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Turn on foreign keys and WAL.

    SQLite ignores ``ON DELETE CASCADE`` unless ``foreign_keys`` is on, which
    would silently leave orphaned findings behind in local development while
    Postgres behaved correctly in production.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection: DBAPIConnection, _: ConnectionPoolEntry) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

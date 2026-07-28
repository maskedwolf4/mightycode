"""Database configuration and async engine/session factory.

The connection string is read from the ``DATABASE_URL`` environment variable.
For Neon, use the pooled ``postgresql+asyncpg://…`` connection string.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def get_database_url() -> str:
    """Return the database URL from environment, converting if needed."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        msg = (
            "DATABASE_URL environment variable is not set. "
            "Set it to a postgresql+asyncpg:// connection string."
        )
        raise RuntimeError(msg)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(url: str | None = None):  # noqa: ANN201
    """Lazily create and cache the async engine."""
    global engine  # noqa: PLW0603
    if engine is None:
        db_url = url or get_database_url()
        engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return engine


def get_session_factory(url: str | None = None) -> async_sessionmaker[AsyncSession]:
    """Lazily create and cache the session factory."""
    global async_session_factory  # noqa: PLW0603
    if async_session_factory is None:
        async_session_factory = async_sessionmaker(
            get_engine(url),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return async_session_factory


def reset_engine() -> None:
    """Reset the cached engine and session factory (useful for testing)."""
    global engine, async_session_factory  # noqa: PLW0603
    engine = None
    async_session_factory = None

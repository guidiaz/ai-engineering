"""SQLAlchemy engine, session factory and per-request session helpers.

Two connection stacks live here, side by side:

* **Synchronous** (``create_engine_from_settings`` / ``SessionLocal`` /
  ``get_session``) — the SQLAlchemy 2.0 sync API. The Session 6 ingestion paths
  are not on the hot user request path (they run as BackgroundTasks or one-shot
  admin operations), so we trade async ergonomics for simplicity there.
* **Asynchronous** (``create_async_engine_from_settings`` /
  ``AsyncSessionLocal`` / ``get_async_session``) — the asyncpg-backed stack,
  added for the Session 8 semantic retriever, which *is* on the hot request
  path and needs non-blocking pgvector queries. Both derive from the single
  ``Settings.DATABASE_URL``; the async variant just swaps the driver.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def create_engine_from_settings() -> Engine:
    """Build the global sync engine from ``Settings.DATABASE_URL`` (singleton)."""
    return create_engine(
        get_settings().DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )


SessionLocal = sessionmaker(
    bind=create_engine_from_settings(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a sync Session and closes it on exit."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# --- Async stack (asyncpg) — added for the Session 8 hot-path retriever -------


def _async_database_url() -> str:
    """Derive the asyncpg URL from the single ``DATABASE_URL`` setting.

    We use ``make_url(...).set(drivername=...)`` rather than string surgery so
    that ``postgresql://``, ``postgresql+psycopg://`` and
    ``postgresql+psycopg2://`` all normalise to ``postgresql+asyncpg://``
    uniformly.

    Caveat: asyncpg does NOT accept libpq-style query parameters such as
    ``sslmode`` / ``options`` that psycopg tolerates. The current
    ``DATABASE_URL`` is clean, but a production URL carrying e.g.
    ``?sslmode=require`` would break the async engine while the sync one keeps
    working — translate those to asyncpg ``connect_args`` here if that day comes.
    """
    return (
        make_url(get_settings().DATABASE_URL)
        .set(drivername="postgresql+asyncpg")
        .render_as_string(hide_password=False)
    )


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    """Build the global async engine (singleton), backed by asyncpg."""
    # pgvector note: the ``pgvector.sqlalchemy.Vector`` column type carries its
    # own bind/result processors that cross the asyncpg boundary (list <-> the
    # DB ``vector`` type) — verified end to end: insert, read-back as list[float]
    # and the ``<=>`` operator all work over asyncpg with no per-connection
    # ``register_vector`` listener. That listener is only needed for RAW asyncpg
    # access; the ORM/Core path used here does not require it.
    return create_async_engine(
        _async_database_url(),
        pool_pre_ping=True,
        future=True,
    )


AsyncSessionLocal = async_sessionmaker(
    bind=create_async_engine_from_settings(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an AsyncSession and closes it on exit."""
    async with AsyncSessionLocal() as session:
        yield session

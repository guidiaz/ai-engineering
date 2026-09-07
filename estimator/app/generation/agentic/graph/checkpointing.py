"""Postgres checkpointing for the estimation graph (Session 13, step 2).

A checkpointer turns the graph from a fire-and-forget call into a **resumable**
one: LangGraph writes the state after every node, keyed by the ``thread_id``
passed in the invocation config. Re-invoking with the same ``thread_id`` replays
the completed nodes instead of re-running them — which for an LLM pipeline means
not paying twice for the same extraction, and getting the *same* decomposition
back even though ``extract_requirements`` is nondeterministic.

It reuses the project's existing Postgres (the pgvector one), so there is no new
service to run.

Two things worth knowing before you touch this:

**The URL needs converting.** ``settings.DATABASE_URL`` is SQLAlchemy's dialect
form (``postgresql+psycopg://``). psycopg — which is what
``langgraph-checkpoint-postgres`` speaks — rejects that prefix, hence
:func:`psycopg_conn_string`.

**The four checkpoint tables are NOT managed by alembic.** ``saver.setup()``
creates and migrates ``checkpoints``, ``checkpoint_writes``, ``checkpoint_blobs``
and ``checkpoint_migrations`` itself; that is how the library is meant to be
used, and its own ``checkpoint_migrations`` table versions them. They live in
the same database as the alembic-managed tables but are deliberately outside
alembic's history — do not add them to a migration, and never run
``alembic revision --autogenerate`` without excluding them, or it will emit a
migration that drops all four.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.config import get_settings

log = structlog.get_logger()

# langgraph's own tables, created by saver.setup(). Listed here so the names are
# discoverable from the code that owns them rather than only from the library.
CHECKPOINT_TABLES = (
    "checkpoints",
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoint_migrations",
)


# The Pydantic models this graph keeps in its state. They get msgpack-encoded
# into the checkpoint rows, and langgraph only decodes types it has been told
# about: today an unlisted type merely warns
# ("Deserializing unregistered type ... will be blocked in a future version"),
# in a later release it fails outright. Declaring them keeps resume working
# across upgrades — and keeps the allowlist narrow instead of `True`, which
# would let a checkpoint row rehydrate arbitrary classes.
_STATE_TYPES: tuple[tuple[str, str], ...] = tuple(
    ("app.generation.agentic.graph.schemas", name)
    for name in (
        "Requirement",
        "Component",
        "BudgetHit",
        "EstimatedComponent",
        "GraphEstimate",
    )
)


def _serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that knows about this graph's state models."""
    return JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)


def psycopg_conn_string(database_url: str | None = None) -> str:
    """Turn a SQLAlchemy database URL into a plain psycopg connection string.

    ``postgresql+psycopg://user:pw@host/db`` → ``postgresql://user:pw@host/db``.
    Any URL that is already in psycopg form is returned unchanged.
    """
    url = database_url if database_url is not None else get_settings().DATABASE_URL
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


@asynccontextmanager
async def postgres_checkpointer(
    database_url: str | None = None,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a Postgres checkpointer, ensuring its tables exist.

    Scoped as a context manager on purpose: the saver owns a connection pool, so
    it must be opened once per process and closed deterministically. Creating one
    per graph invocation would leak connections, and creating one as an
    import-time singleton would leave nothing to close it — there is no FastAPI
    lifespan owning it in this step, because this step has no endpoint.

    Usage::

        async with postgres_checkpointer() as saver:
            estimate = await run_estimation_graph(
                transcript, checkpointer=saver, estimation_id=some_id
            )
    """
    dsn = psycopg_conn_string(database_url)
    async with AsyncPostgresSaver.from_conn_string(dsn, serde=_serializer()) as saver:
        # Idempotent: creates the tables on first use, no-ops afterwards.
        await saver.setup()
        log.info("graph_checkpointer_ready", tables=len(CHECKPOINT_TABLES))
        yield saver

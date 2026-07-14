"""Persistence layer for the Session 6 ingestion subsystem.

Holds the SQLAlchemy engine, the declarative ``Base`` and the row models that
back the pseudonymization mapping table and the ingestion job tracker. Higher
layers consume narrow repositories from ``app.foundation.persistence.repositories``; they
never see SQLAlchemy types directly.
"""

from app.foundation.persistence.database import (
    AsyncSessionLocal,
    SessionLocal,
    create_async_engine_from_settings,
    create_engine_from_settings,
    get_async_session,
    get_session,
)
from app.foundation.persistence.models import (
    Base,
    ChunkRow,
    DocumentRow,
    IngestionJobRow,
    PseudonymMappingRow,
)

__all__ = [
    "AsyncSessionLocal",
    "Base",
    "ChunkRow",
    "DocumentRow",
    "IngestionJobRow",
    "PseudonymMappingRow",
    "SessionLocal",
    "create_async_engine_from_settings",
    "create_engine_from_settings",
    "get_async_session",
    "get_session",
]

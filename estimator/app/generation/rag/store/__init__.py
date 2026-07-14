"""Vector store — persistence of embedded chunks in PostgreSQL + pgvector.

``DocumentIngestor`` writes one budget document and its embedded chunks in a
single transaction (``POST /embeddings/ingest``). The HNSW index and the
semantic retriever that reads from here land alongside this package in
Session 8.
"""

from app.generation.rag.store.ingestor import (
    DocumentAlreadyExists,
    DocumentIngestor,
    IngestOutcome,
)

__all__ = ["DocumentAlreadyExists", "DocumentIngestor", "IngestOutcome"]

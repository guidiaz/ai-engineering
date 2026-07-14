"""Document ingestion into the pgvector store (Session 8).

``DocumentIngestor`` owns the *single transaction* that turns one incoming
budget JSON into one ``documents`` row plus N embedded ``chunks`` rows. The
whole thing commits or nothing does: if the embedder raises mid-way, the
uncommitted ``documents`` insert is rolled back, so we never leave a document
without its chunks.

Sequence (mirrors the exercise contract):

1. Reject if a document already exists for this ``source_path`` (409 upstream).
2. Insert the ``documents`` row and ``flush`` to obtain its generated id.
3. Chunk the budget structurally (CPU-bound, cheap — run inline).
4. Embed every chunk in one batched API call (blocking I/O — run in a thread
   so the event loop is not blocked).
5. ``add_all`` the chunk rows and ``commit``.

The blocking OpenAI call is pushed to a worker thread with
``run_in_threadpool`` because the surrounding session is async: calling the
synchronous embedder inline would stall the event loop for the whole request.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.foundation.persistence.models import ChunkRow, DocumentRow
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.embedding.embedder import EMBEDDING_DIM, OpenAIEmbedder
from app.generation.rag.schemas import IngestRequest

log = structlog.get_logger()

# One structural chunk == one budget component. Stored on every chunk row so
# later steps (the Session 8 retriever / search) can filter by chunk
# granularity. Kept a single obvious constant on purpose — it is part of the
# data contract, and it is the value returned in the /search response.
STRUCTURAL_CHUNK_TYPE = "budget_component"


class DocumentAlreadyExists(Exception):
    """Raised when a document with the same ``source_path`` is already stored.

    Carries the existing ``document_id`` so the router can return it in the
    409 body.
    """

    def __init__(self, document_id: int) -> None:
        self.document_id = document_id
        super().__init__(f"Document already ingested (id={document_id})")


@dataclass(frozen=True)
class IngestOutcome:
    """What the router needs to build the 200 response. No ORM types leak out."""

    document_id: int
    chunks_created: int
    embedding_dimension: int


class DocumentIngestor:
    """Chunk + embed + persist one budget document in a single transaction."""

    def __init__(self, chunker: JSONStructuralChunker, embedder: OpenAIEmbedder) -> None:
        self._chunker = chunker
        self._embedder = embedder

    async def ingest(self, session: AsyncSession, request: IngestRequest) -> IngestOutcome:
        # 1. Duplicate check (check-then-insert; the TOCTOU race is acceptable
        #    here — see README. A UNIQUE constraint would harden it later).
        existing_id = await session.scalar(
            select(DocumentRow.id).where(DocumentRow.source_path == request.source_path)
        )
        if existing_id is not None:
            raise DocumentAlreadyExists(existing_id)

        budget = request.content
        try:
            # 2. Insert the document and flush to get its generated id.
            document = DocumentRow(
                source_path=request.source_path,
                document_type=request.document_type,
                doc_metadata={
                    "budget_id": budget.budget_id,
                    "client_sector": budget.client_metadata.sector,
                    "main_technology": budget.main_technology,
                    "year": budget.year,
                },
            )
            session.add(document)
            await session.flush()

            # 3. Chunk (sync, CPU-bound).
            chunks = self._chunker.chunk([budget])

            # 4. Embed in one batched call, off the event loop.
            embedded = await run_in_threadpool(self._embedder.embed_many, chunks)

            # 5. Persist all chunk rows.
            rows = [
                ChunkRow(
                    document_id=document.id,
                    chunk_type=STRUCTURAL_CHUNK_TYPE,
                    content=chunk.text,
                    embedding=chunk.embedding,
                    chunk_metadata=chunk.metadata,
                )
                for chunk in embedded
            ]
            session.add_all(rows)

            # 6. Commit the whole unit of work.
            await session.commit()
        except Exception:
            # Embedder failure (or anything else) leaves no orphan document.
            await session.rollback()
            raise

        dimension = len(embedded[0].embedding) if embedded else EMBEDDING_DIM
        log.info(
            "document_ingested",
            document_id=document.id,
            chunks_created=len(rows),
            embedding_dimension=dimension,
        )
        return IngestOutcome(
            document_id=document.id,
            chunks_created=len(rows),
            embedding_dimension=dimension,
        )

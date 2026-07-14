"""Semantic retriever (Session 8).

Embeds a natural-language query with the *same* model used at ingestion
(``text-embedding-3-small``) and ranks stored chunks by cosine distance via
pgvector's ``<=>`` operator, exposed through
``ChunkRow.embedding.cosine_distance(...)``.

No ANN index yet: Postgres does a full sequential scan over ``chunks``. For the
programme's sample corpus (tens of documents, hundreds of chunks) that resolves
in a few hundred ms — observing that latency *without* an index is the live
session's starting point. The HNSW index is added there.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.foundation.persistence.models import ChunkRow
from app.generation.rag.embedding.embedder import OpenAIEmbedder

log = structlog.get_logger()


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked search hit. No ORM types leak past this boundary."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict


class SemanticRetriever:
    """Embed a query and return its ``k`` nearest chunks by cosine distance."""

    def __init__(self, embedder: OpenAIEmbedder) -> None:
        self._embedder = embedder

    async def search(self, session: AsyncSession, query: str, k: int) -> list[RetrievedChunk]:
        # Embed the query off the event loop (blocking OpenAI call).
        query_vector = await run_in_threadpool(self._embedder.embed_one, query)

        # cosine_distance maps to pgvector's ``<=>``. We both order by it and
        # select it so the caller gets the score, computed once by Postgres.
        distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.chunk_metadata,
                distance,
            )
            .order_by(distance)
            .limit(k)
        )
        # Time ONLY the DB round-trip. The endpoint's search_time_ms also
        # includes the query-embedding API call; this isolated number is the
        # sequential-scan latency the live session sets out to observe (and to
        # watch collapse once an HNSW index is added).
        scan_start = perf_counter()
        rows = (await session.execute(stmt)).all()
        db_scan_ms = round((perf_counter() - scan_start) * 1000, 1)
        log.info("semantic_search_done", k=k, hits=len(rows), db_scan_ms=db_scan_ms)
        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=float(row.distance),
                metadata=row.chunk_metadata or {},
            )
            for row in rows
        ]

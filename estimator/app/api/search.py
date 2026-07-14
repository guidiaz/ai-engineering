"""HTTP layer for semantic search over the ingested corpus.

Thin router: embed the query, rank chunks by cosine distance, map failures to
status codes. The retrieval logic lives in ``app.generation.rag.retriever``.
"""

from __future__ import annotations

from time import perf_counter

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder
from app.foundation.persistence import get_async_session
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.schemas import SearchRequest, SearchResponse, SearchResultItem

log = structlog.get_logger()

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_async_session),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> SearchResponse:
    """Embed the query and return its ``k`` nearest chunks by cosine distance."""
    if embedder is None:
        log.error("search_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    retriever = SemanticRetriever(embedder)
    started = perf_counter()
    try:
        hits = await retriever.search(session, request.query, request.k)
    except Exception as exc:  # noqa: BLE001 — embedding/query failure becomes a 500.
        log.error(
            "search_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run semantic search.") from exc

    search_time_ms = round((perf_counter() - started) * 1000)
    log.info("search_done", k=request.k, hits=len(hits), search_time_ms=search_time_ms)
    return SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=search_time_ms,
        results=[
            SearchResultItem(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                chunk_type=hit.chunk_type,
                content=hit.content,
                distance=hit.distance,
                metadata=hit.metadata,
            )
            for hit in hits
        ],
    )

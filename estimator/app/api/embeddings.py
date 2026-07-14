"""HTTP layer for the embedding pipeline.

Thin router: it orchestrates chunker -> embedder -> response assembly and maps
failures to status codes. No business logic lives here.
"""

from __future__ import annotations

from time import perf_counter

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.foundation.persistence import get_async_session
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import IngestRequest, IngestResponse
from app.generation.rag.store import DocumentAlreadyExists, DocumentIngestor

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={409: {"description": "A document with this source_path is already ingested."}},
)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_async_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
):
    """Chunk, embed and persist one budget in a single transaction.

    Returns identifiers and metrics only — the chunks and their vectors are
    stored, not echoed back. A repeated ``source_path`` yields a 409.
    """
    if embedder is None:
        # No OPENAI_API_KEY configured. Generic message to the client, detail logged.
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    log.info(
        "embeddings_ingest_received",
        source_path=request.source_path,
        document_type=request.document_type,
    )
    ingestor = DocumentIngestor(chunker, embedder)
    started = perf_counter()
    try:
        outcome = await ingestor.ingest(session, request)
    except DocumentAlreadyExists as exc:
        log.info(
            "embeddings_ingest_conflict",
            source_path=request.source_path,
            document_id=exc.document_id,
        )
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": exc.document_id},
        )
    except Exception as exc:  # noqa: BLE001 — embedding/persistence failure becomes a 500.
        log.error(
            "embeddings_ingest_failed",
            reason="ingest_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to ingest document.") from exc

    ingestion_time_ms = round((perf_counter() - started) * 1000)
    log.info(
        "embeddings_ingest_done",
        document_id=outcome.document_id,
        chunks_created=outcome.chunks_created,
        ingestion_time_ms=ingestion_time_ms,
    )
    return IngestResponse(
        document_id=outcome.document_id,
        chunks_created=outcome.chunks_created,
        embedding_dimension=outcome.embedding_dimension,
        ingestion_time_ms=ingestion_time_ms,
    )


@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> CompareResponse:
    """Run several chunking strategies over the same budgets and compare them.

    Returns per-strategy corpus stats and, if queries are given, the top-k
    chunks each strategy retrieves. Nothing is persisted (Session 8 territory).
    """
    if embedder is None:
        log.error("embeddings_compare_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {exc.args[0]}") from exc
    except RuntimeError as exc:
        # A strategy needs an API key that is not configured.
        log.error("embeddings_compare_failed", reason="missing_api_key", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    comparator = ChunkingComparator(chunkers, embedder)
    log.info(
        "embeddings_compare_received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(request.budgets, request.queries, request.top_k)
    except Exception as exc:  # noqa: BLE001 — any chunker/embedding failure becomes a 500.
        log.error(
            "embeddings_compare_failed",
            reason="comparison_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run chunking comparison.") from exc

    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)

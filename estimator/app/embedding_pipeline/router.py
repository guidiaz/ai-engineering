"""POST /embeddings/ingest — chunk historical budgets and embed them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "budgets_sample.json"
_INGEST_OPENAPI_EXAMPLE: dict = {
    "budgets": json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))[:1],
}

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_embeddings(
    request: Annotated[
        IngestRequest,
        Body(
            openapi_examples={
                "sample_budget": {
                    "summary": "Single budget from budgets_sample.json",
                    "value": _INGEST_OPENAPI_EXAMPLE,
                }
            }
        ),
    ],
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    """Chunk each budget component, embed the fragments, and return vectors + stats."""
    log.info("ingest_request_received", budgets=len(request.budgets))

    chunks = chunker.chunk(request.budgets)

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:
        log.error(
            "embedding_ingest_failed",
            error=str(exc)[:400],
            error_type=type(exc).__name__,
            chunks=len(chunks),
        )
        raise HTTPException(status_code=500, detail="Embedding service failed") from exc

    response = IngestResponse(
        chunks=embedded,
        stats=IngestStats(
            total_budgets=len(request.budgets),
            total_chunks=len(embedded),
            total_tokens=embedder.total_tokens,
            estimated_cost_usd=embedder.estimated_cost_usd,
        ),
    )
    log.info(
        "ingest_completed",
        total_budgets=response.stats.total_budgets,
        total_chunks=response.stats.total_chunks,
        total_tokens=response.stats.total_tokens,
        estimated_cost_usd=response.stats.estimated_cost_usd,
    )
    return response

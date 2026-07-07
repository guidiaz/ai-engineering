"""OpenAI embeddings client for the historical-budget pipeline."""

from __future__ import annotations

import time

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

_EMBEDDING_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100
_RATE_LIMIT_BACKOFF_SECONDS = (1, 2, 4)

# OpenAI pricing for text-embedding-3-small (input tokens, USD). Update when pricing changes.
_PRICE_USD_PER_MILLION_INPUT_TOKENS = 0.02


def estimate_cost_usd(total_tokens: int) -> float:
    """Return estimated embedding cost from a pre-counted token total."""
    return total_tokens * _PRICE_USD_PER_MILLION_INPUT_TOKENS / 1_000_000


class OpenAIEmbedder:
    """Batch embedder backed by OpenAI ``embeddings.create``."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str = _EMBEDDING_MODEL,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0

    def embed_one(self, text: str) -> list[float]:
        return self._create_embeddings([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            self.total_tokens = 0
            self.estimated_cost_usd = 0.0
            return []

        embedded: list[EmbeddedChunk] = []
        total_tokens = 0

        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            batch_tokens = sum(chunk.token_count for chunk in batch)

            started = time.perf_counter()
            vectors = self._create_embeddings([chunk.text for chunk in batch])
            latency_ms = (time.perf_counter() - started) * 1000

            total_tokens += batch_tokens
            log.info(
                "embedding_batch_processed",
                chunks=len(batch),
                total_tokens=batch_tokens,
                latency_ms=round(latency_ms, 2),
                model=self._model,
                batch_size=self._batch_size,
            )

            embedded.extend(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    token_count=chunk.token_count,
                    embedding=vector,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            )

        self.total_tokens = total_tokens
        self.estimated_cost_usd = estimate_cost_usd(total_tokens)
        return embedded

    def _create_embeddings(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(len(_RATE_LIMIT_BACKOFF_SECONDS) + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                return [item.embedding for item in ordered]
            except RateLimitError:
                if attempt == len(_RATE_LIMIT_BACKOFF_SECONDS):
                    raise
                wait_seconds = _RATE_LIMIT_BACKOFF_SECONDS[attempt]
                log.warning(
                    "embedding_rate_limited",
                    attempt=attempt + 1,
                    wait_seconds=wait_seconds,
                )
                time.sleep(wait_seconds)

        raise RuntimeError("embedding retry loop exited unexpectedly")

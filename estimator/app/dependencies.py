"""FastAPI dependency factories for shared singletons."""

from __future__ import annotations

from functools import lru_cache

import redis
import structlog
from openai import OpenAI

from app.cache.semantic import EstimationSemanticCache
from app.config import get_settings
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.services.cache import EstimationCache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper
from app.sessions.store import SessionStore

log = structlog.get_logger()


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
    )


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Lazy OpenAI client used by ``check_input`` (Moderation API) and the
    semantic cache (Embeddings API)."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    settings = get_settings()
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAIEmbedder(client, model=settings.EMBEDDING_MODEL)


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Build the semantic cache, swallowing setup errors so the rest of the
    pipeline keeps working if Redis Stack / RediSearch is not available
    (e.g. running on vanilla redis:7-alpine)."""
    settings = get_settings()
    openai_client = get_openai_client()
    if openai_client is None:
        log.warning("semantic_cache_disabled", reason="no_openai_key")
        return None

    # We use redisvl's OpenAITextVectorizer; it lazy-loads the OpenAI client.
    try:
        from redisvl.utils.vectorize import OpenAITextVectorizer

        vectorizer = OpenAITextVectorizer(
            model=settings.EMBEDDING_MODEL,
            api_config={"api_key": settings.OPENAI_API_KEY},
        )
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
            log_only=settings.SEMANTIC_CACHE_LOG_ONLY,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "semantic_cache_disabled",
            reason="setup_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None


@lru_cache
def get_estimation_service() -> EstimationService:
    settings = get_settings()
    return EstimationService(
        llm_wrapper=get_llm_wrapper(),
        exact_cache=get_cache(),
        semantic_cache=get_semantic_cache(),
        openai_client=get_openai_client(),
        metadata_extractor_model=settings.METADATA_EXTRACTOR_MODEL,
        compression_model=settings.COMPRESSION_MODEL,
        anchor_detection_mode=settings.ANCHOR_DETECTION_MODE,
        conversational_prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION,
        critic_model=settings.CRITIC_MODEL,
        boss_max_iterations=settings.BOSS_MAX_ITERATIONS,
    )


@lru_cache
def get_session_store() -> SessionStore:
    """In-memory store of conversational sessions. Singleton per worker.

    State lives in process memory: docker compose restart clears it, and
    workers > 1 each get their own copy. Documented limitation for the
    Session 5 exercise; persistence is module-3 territory.
    """
    settings = get_settings()
    return SessionStore(max_turns=settings.MAX_CONVERSATION_TURNS)

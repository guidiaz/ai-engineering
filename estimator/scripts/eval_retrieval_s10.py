#!/usr/bin/env python3
"""Measure retrieval quality across the four Session 10 configurations.

Runs the golden set (``evals/golden_retrieval.json``) through the same
``retrieve()`` pipeline the API uses, for each of:

    A  Vector  / no rerank   (the Session 9 baseline)
    B  Hybrid  / no rerank
    C  Vector  / rerank
    D  Hybrid  / rerank

and reports precision@5 (mean over the queries) and query latency (mean over
measured runs) per configuration, plus a per-query precision breakdown.

Method notes:
* Queries are embedded ONCE upfront; embedding latency is an OpenAI round-trip
  shared by every config, so it is excluded from the timings (we measure the
  retrieval/rerank cost the techniques actually add).
* A permissive distance threshold is used so the top-5 is never truncated by the
  relevance floor — we are comparing RANKING quality, not the soft-fail gate.
* A retrieved chunk is relevant iff its parent ``budget_id`` is in the query's
  annotated ``relevant_budget_ids``. precision@5 = relevant_in_top5 / 5.

Usage (host, stack up + corpus ingested + OPENAI_API_KEY in .env)::

    uv run python scripts/eval_retrieval_s10.py

Reranking config runs download the cross-encoder weights on first use; verify
first with ``python -m app.generation.rag.retrieval.verify_reranker``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.s08_common import Stopwatch, require_embedder  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.dependencies import get_reranker  # noqa: E402
from app.generation.rag.retrieval.pipeline import retrieve  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# (id, search label, rerank label, search_mode, rerank). Labels are Spanish:
# the comparative table is the exercise deliverable ("tabla comparativa, en español").
CONFIGS = [
    ("A", "Vectorial", "No", "vector", False),
    ("B", "Híbrida", "No", "hybrid", False),
    ("C", "Vectorial", "Sí", "vector", True),
    ("D", "Híbrida", "Sí", "hybrid", True),
]

# No effective relevance floor: we want a full top-k to grade ranking quality.
NO_FLOOR_THRESHOLD = 2.0
# Latency sampling per (query, config): one discarded warm-up + N measured runs.
MEASURED_RUNS = 3


def precision_at_k(chunks, relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results whose budget is genuinely relevant."""
    top = chunks[:k]
    hits = sum(1 for chunk in top if chunk.budget_id in relevant_ids)
    return hits / k


async def _run_once(query_embedding, query_text, search_mode, rerank, settings, chunk_types, k):
    return await retrieve(
        query_embedding=query_embedding,
        query_text=query_text,
        search_mode=search_mode,
        rerank=rerank,
        top_k=k,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=k,
        distance_threshold=NO_FLOOR_THRESHOLD,
        rrf_k=settings.RRF_K,
        chunk_types=chunk_types,
    )


async def main() -> int:
    settings = get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    chunk_types = golden.get("chunk_types")
    k = int(golden.get("k", 5))

    embedder = require_embedder()
    print(f"Embedding {len(queries)} golden queries (excluded from timings)...")
    embeddings = {q["id"]: embedder.embed_one(q["query"]) for q in queries}

    # Warm the reranker once so its (one-time) model load does not skew the first
    # timed reranking run.
    print("Warming up the cross-encoder (first load downloads weights)...")
    get_reranker().load()

    # results[config_id] = {"precisions": [...], "latencies_ms": [...]}
    results = {cfg[0]: {"precisions": [], "latencies_ms": [], "per_query": {}} for cfg in CONFIGS}
    empty_warning = False

    for cfg_id, _s, _r, search_mode, rerank in CONFIGS:
        for q in queries:
            relevant = set(q["relevant_budget_ids"])
            emb = embeddings[q["id"]]

            # Warm-up (discarded) then measured runs.
            await _run_once(emb, q["query"], search_mode, rerank, settings, chunk_types, k)
            samples = []
            last = None
            for _ in range(MEASURED_RUNS):
                with Stopwatch() as sw:
                    last = await _run_once(
                        emb, q["query"], search_mode, rerank, settings, chunk_types, k
                    )
                samples.append(sw.elapsed_ms)

            if not last.chunks:
                empty_warning = True
            precision = precision_at_k(last.chunks, relevant, k)
            results[cfg_id]["precisions"].append(precision)
            results[cfg_id]["latencies_ms"].extend(samples)
            results[cfg_id]["per_query"][q["id"]] = precision

    _print_report(results, queries, k)
    if empty_warning:
        print(
            "\nWARNING: some configurations returned 0 chunks. Is the base corpus "
            "ingested? Run `uv run python scripts/query_examples.py` to ingest it.",
            file=sys.stderr,
        )
    return 0


def _print_report(results: dict, queries: list, k: int) -> None:
    print(f"\n## Evaluación de recuperación — precisión@{k} y latencia\n")
    print(f"| Configuración | Búsqueda | Reranking | Precisión@{k} | Latencia (ms) |")
    print("| --- | --- | --- | --- | --- |")
    for cfg_id, search_label, rerank_label, _m, _rr in CONFIGS:
        bucket = results[cfg_id]
        mean_p = statistics.fmean(bucket["precisions"])
        mean_l = statistics.fmean(bucket["latencies_ms"])
        print(f"| {cfg_id} | {search_label} | {rerank_label} | {mean_p:.2f} | {mean_l:.1f} |")

    print(f"\n### Precisión@{k} por consulta\n")
    header = "| Consulta | " + " | ".join(cfg[0] for cfg in CONFIGS) + " |"
    print(header)
    print("| --- | " + " | ".join("---" for _ in CONFIGS) + " |")
    for q in queries:
        row = [q["id"]] + [f"{results[cfg[0]]['per_query'][q['id']]:.2f}" for cfg in CONFIGS]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

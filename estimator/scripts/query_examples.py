#!/usr/bin/env python3
"""Semantic-search sanity check — five representative queries against /search.

Replaces the Session 7 ``compare.py`` (which measured cosine similarity between
two loose texts). Instead of embedding pairs by hand, this script *calls the
running ``POST /search`` endpoint* with five queries chosen to probe the corpus
from different angles, and prints the top-k hits for each.

The five angles (see the exercise brief):

1. Known direct component — should have a near-perfect match. Sanity check.
2. Semantic paraphrase — same idea, different vocabulary. Do embeddings capture
   meaning or just words?
3. Different domain — should NOT be in the corpus; expect large distances.
4. Ambiguous query — short and generic; watch the ranking with no clear winner.
5. Very specific query — precise technical vocabulary; does it discriminate?

Preconditions: the estimator API is running and the corpus has been ingested
(``POST /embeddings/ingest``). This script only reads.

Usage::

    # outside the container (from estimator/, with the API on :8000):
    uv run python scripts/query_examples.py

    # against another host / a different k:
    uv run python scripts/query_examples.py --base-url http://localhost:8000 --k 5

    # inside the container network:
    docker compose exec estimator python scripts/query_examples.py \\
        --base-url http://estimator:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402

# (angle label, query text). The example texts are the ones from the brief.
QUERIES: list[tuple[str, str]] = [
    (
        "1. Known direct component (sanity check)",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "2. Semantic paraphrase (different vocabulary)",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "3. Different domain (should be irrelevant / far)",
        "mobile application for restaurant reservations",
    ),
    (
        "4. Ambiguous query (no dominant match)",
        "integration with external system",
    ),
    (
        "5. Very specific query (precise technical vocabulary)",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]

CONTENT_PREVIEW_CHARS = 120


def _preview(text: str, width: int = CONTENT_PREVIEW_CHARS) -> str:
    """First ``width`` chars of ``content`` on a single terminal line."""
    flat = " ".join(text.split())
    return flat[:width] + ("..." if len(flat) > width else "")


def _print_query_block(index: int, total: int, label: str, query: str, body: dict) -> None:
    results = body.get("results", [])
    print("\n" + "-" * 100)
    print(f"[{index}/{total}] {label}")
    print(f'      query: "{query}"')
    print(f"      k={body.get('k')} | search_time_ms={body.get('search_time_ms')} | hits={len(results)}")
    if not results:
        print("      (no results - is the corpus ingested?)")
        return
    print(f"      {'#':>2}  {'chunk_id':>8}  {'distance':>8}  {'chunk_type':<17}  content")
    for rank, hit in enumerate(results, start=1):
        print(
            f"      {rank:>2}  "
            f"{hit['chunk_id']:>8}  "
            f"{hit['distance']:>8.4f}  "
            f"{hit['chunk_type']:<17}  "
            f"{_preview(hit['content'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run five example queries against /search.")
    parser.add_argument(
        "--base-url",
        default=get_settings().ESTIMATOR_API_BASE_URL,
        help="Base URL of the estimator API (default: ESTIMATOR_API_BASE_URL).",
    )
    parser.add_argument("--k", type=int, default=5, help="Results per query (default: 5).")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/search"
    print(f"POST {url}  |  {len(QUERIES)} queries  |  k={args.k}")

    try:
        with httpx.Client(timeout=60) as client:
            for i, (label, query) in enumerate(QUERIES, start=1):
                response = client.post(url, json={"query": query, "k": args.k})
                if response.status_code != 200:
                    print(f"\n[{i}/{len(QUERIES)}] {label}")
                    print(f"      ERROR {response.status_code}: {response.text[:300]}")
                    continue
                _print_query_block(i, len(QUERIES), label, query, response.json())
    except httpx.ConnectError:
        print(
            f"\nERROR: could not reach {url}. Is the API running?\n"
            "  uv run uvicorn app.main:app --reload   (or: docker compose up)",
            file=sys.stderr,
        )
        return 1

    print("\n" + "-" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

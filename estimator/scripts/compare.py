#!/usr/bin/env python
"""Compara dos textos por similitud coseno de sus embeddings.

Reutiliza el ``OpenAIEmbedder`` del pipeline de embeddings (misma clase que usa
``POST /embeddings/ingest``) y la configuración de la app (``get_settings``), de
modo que el modelo de embeddings y la API key salen del mismo sitio que el resto
del servicio.

Uso:

    python scripts/compare.py \
      --text-a "OAuth 2.0 authentication backend for fintech" \
      --text-b "JWT-based authorization service for banking app"

Se puede ejecutar dentro del contenedor (``docker compose exec estimator ...``)
o fuera con ``uv run python scripts/compare.py ...`` teniendo el .env cargado.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# El script vive en ``<proyecto>/scripts/``. Añadimos la raíz del proyecto al
# path para poder importar el paquete ``app`` sin depender del cwd (funciona
# igual dentro del contenedor, con WORKDIR=/app, que fuera con uv run).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dependencies import get_embedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno con biblioteca estándar: producto escalar / producto de normas."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embeda dos textos y devuelve su similitud coseno.",
    )
    parser.add_argument("--text-a", required=True, help="Primer texto a comparar.")
    parser.add_argument("--text-b", required=True, help="Segundo texto a comparar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        embedder = get_embedder()
    except RuntimeError as exc:
        # get_embedder() lanza RuntimeError si falta OPENAI_API_KEY.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        embedding_a = embedder.embed_one(args.text_a)
        embedding_b = embedder.embed_one(args.text_b)
    except Exception as exc:  # p.ej. RateLimitError / cuota agotada
        print(f"Error al generar embeddings: {exc}", file=sys.stderr)
        return 1

    similarity = cosine_similarity(embedding_a, embedding_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

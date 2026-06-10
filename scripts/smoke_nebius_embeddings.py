"""Live smoke for the Nebius embeddings preset.

Requires NEBIUS_API_KEY. Run with:
  set -a; source .env; set +a
  uv run python scripts/smoke_nebius_embeddings.py

This is not part of deterministic eval; pytest covers the OpenAI-compatible client with fakes.
"""
import math

from genacademy_rag.config import Settings
from genacademy_rag.core.providers import OpenAICompatEmbedder


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def main():
    s = Settings.from_env()
    if not s.nebius_api_key:
        raise SystemExit("NEBIUS_API_KEY not set; export it (env-only secret) and rerun")

    embedder = OpenAICompatEmbedder(
        base_url=s.nebius_base_url,
        api_key=s.nebius_api_key,
        model=s.nebius_embed_model,
    )
    texts = [
        "retrieval augmented generation grounds answers in retrieved context",
        "RAG systems search a corpus before generating cited answers",
        "ripe bananas and flour make a simple quick bread",
    ]
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise SystemExit(f"expected {len(texts)} vectors, got {len(vectors)}")

    dim = len(vectors[0])
    if dim <= 0 or any(len(vec) != dim for vec in vectors):
        raise SystemExit("embedding dimensions were empty or inconsistent")
    for vec in vectors:
        if not math.isclose(_norm(vec), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise SystemExit("normalized embedding norm was not approximately 1.0")

    related = _dot(vectors[0], vectors[1])
    unrelated = _dot(vectors[0], vectors[2])
    if related <= unrelated:
        raise SystemExit("related embedding pair did not outrank unrelated pair")

    print(f"NEBIUS EMBEDDINGS SMOKE OK  model={s.nebius_embed_model} dim={dim}")


if __name__ == "__main__":
    main()

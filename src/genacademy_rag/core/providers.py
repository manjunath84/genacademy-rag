"""ModelProvider seam. STEmbedder = local sentence-transformers (offline, deterministic).
OpenAICompatProvider = the one generation seam for every preset (Nebius/OpenRouter/OpenAI/Gemma):
the same base_url + key + model verbatim shape the spike validated."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def generate(self, messages: list[dict], *, json_mode: bool = False,
                 max_tokens: int = 512, temperature: float = 0.0) -> str: ...


class STEmbedder:
    """Local all-MiniLM-L6-v2 (384-dim). Load once (cold ~11.6 s); reuse for every request."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), batch_size=32, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenAICompatProvider:
    """Generation via any OpenAI-compatible endpoint. Call shape from spike/gen_probe.py."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        # OpenAI(api_key="") raises OpenAIError("Missing credentials"); a keyless local Gemma
        # server ignores the value, so pass a placeholder when no key is configured.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - not used in P0
        raise NotImplementedError("Phase 0 embeds locally via STEmbedder")

    def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0) -> str:
        kwargs = dict(model=self._model, messages=list(messages),
                      temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content


class CombinedProvider:
    """Bundles local embed + remote generate behind one ModelProvider for the pipeline."""

    def __init__(self, embedder: STEmbedder, generator: OpenAICompatProvider):
        self._embedder = embedder
        self._generator = generator

    def embed(self, texts):
        return self._embedder.embed(texts)

    def generate(self, messages, **kwargs):
        return self._generator.generate(messages, **kwargs)


def build_provider(settings) -> CombinedProvider:
    """Wire the active preset from Settings. Set GENACADEMY_PROVIDER=nebius for Nebius."""
    return CombinedProvider(
        STEmbedder(settings.embed_model),
        OpenAICompatProvider(settings.gen_base_url, settings.gen_api_key, settings.gen_model),
    )

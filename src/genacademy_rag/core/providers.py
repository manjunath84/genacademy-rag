"""ModelProvider seam. STEmbedder = local sentence-transformers (offline, deterministic).
OpenAICompatEmbedder = OpenAI-compatible embeddings (Nebius preset). OpenAICompatProvider =
the one generation seam for every preset (Nebius/OpenRouter/OpenAI/Gemma): the same base_url +
key + model verbatim shape the spike validated."""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

_EMBED_BATCH = 128


@runtime_checkable
class ModelProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def generate(self, messages: list[dict], *, json_mode: bool = False,
                 max_tokens: int = 512, temperature: float = 0.0) -> str: ...


class EmbedderSetupError(RuntimeError):
    """Raised when an embedding preset is selected but cannot be constructed."""


class STEmbedder:
    """Local all-MiniLM-L6-v2 (384-dim). Load once (cold ~11.6 s); reuse for every request."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), batch_size=32, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenAICompatEmbedder:
    """Embeddings via any OpenAI-compatible endpoint, normalized to the local embedder contract."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = _EMBED_BATCH,
        client_cls: type | None = None,
    ):
        if client_cls is None:
            from openai import OpenAI

            client_cls = OpenAI
        self._client = client_cls(base_url=base_url, api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start:start + self._batch_size])
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                encoding_format="float",
            )
            for item in sorted(response.data, key=lambda i: i.index):
                vec = [float(x) for x in item.embedding]
                norm = math.sqrt(sum(x * x for x in vec))
                out.append([x / norm for x in vec] if norm else vec)
        return out


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

    def __init__(self, embedder, generator: OpenAICompatProvider):
        self._embedder = embedder
        self._generator = generator

    def embed(self, texts):
        return self._embedder.embed(texts)

    def generate(self, messages, **kwargs):
        return self._generator.generate(messages, **kwargs)


def build_embedder(settings, *, client_cls: type | None = None):
    """Wire the active embedding preset. Default local path preserves Phase 0/1 behavior."""
    if settings.embeddings == "local":
        return STEmbedder(settings.embed_model)
    if settings.embeddings == "nebius":
        if not settings.nebius_api_key:
            raise EmbedderSetupError(
                "GENACADEMY_EMBEDDINGS=nebius requires NEBIUS_API_KEY in the environment"
            )
        return OpenAICompatEmbedder(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
            model=settings.nebius_embed_model,
            client_cls=client_cls,
        )
    raise ValueError(
        f"unknown embeddings preset {settings.embeddings!r}; expected 'local' or 'nebius'"
    )


def build_provider(settings) -> CombinedProvider:
    """Wire the active preset from Settings. Set GENACADEMY_PROVIDER=nebius for Nebius."""
    return CombinedProvider(
        build_embedder(settings),
        OpenAICompatProvider(settings.gen_base_url, settings.gen_api_key, settings.gen_model),
    )

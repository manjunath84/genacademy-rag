"""Optional cross-encoder reranker seam for Phase 2 retrieval ordering."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from genacademy_rag.core.types import Chunk

PROVISION_RERANK_MODEL_COMMAND = "uv run python scripts/provision_rerank_model.py"


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]: ...


class RerankerSetupError(RuntimeError):
    """Raised when rerank is enabled but model setup fails."""


def _clone_module_tensors(module) -> None:
    """Move loaded tensors off memory-mapped safetensors storage before inference.

    On this macOS CPU runtime, CrossEncoder forward passes SIGBUSed when backed by mmap'd
    safetensors. Cloning keeps frozen inference values identical while using normal in-memory
    storage; configurable models with shared Parameter objects may lose that object sharing.
    """
    import torch

    for child in module.modules():
        for name, param in list(child.named_parameters(recurse=False)):
            child._parameters[name] = torch.nn.Parameter(  # noqa: SLF001
                param.detach().clone(),
                requires_grad=param.requires_grad,
            )
        for name, buffer in list(child.named_buffers(recurse=False)):
            if buffer is not None:
                child._buffers[name] = buffer.detach().clone()  # noqa: SLF001


class SentenceTransformersCrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str | None = None,
        local_files_only: bool = True,
        cache_dir: Path | None = None,
        cross_encoder_cls: type | None = None,
    ):
        if cross_encoder_cls is None:
            from sentence_transformers import CrossEncoder

            cross_encoder_cls = CrossEncoder
        try:
            self._model = cross_encoder_cls(
                model_name,
                device=device,
                local_files_only=local_files_only,
                cache_folder=str(cache_dir) if cache_dir is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            if local_files_only:
                message = (
                    "Could not load rerank model from local files. "
                    f"Provision it with `{PROVISION_RERANK_MODEL_COMMAND}` before running "
                    "rerank-enabled eval/runtime with local_files_only=true."
                )
            else:
                message = "Could not initialize rerank model during explicit provisioning."
            raise RerankerSetupError(
                f"{message} Original error: {type(exc).__name__}: {exc}"
            ) from exc
        model = getattr(self._model, "model", None)
        if model is not None:
            _clone_module_tensors(model)
        self._batch_size = batch_size

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = [float(score) for score in self._model.predict(pairs, batch_size=self._batch_size)]
        return sorted(zip(chunks, scores, strict=True), key=lambda item: item[1], reverse=True)


def build_reranker(settings) -> Reranker | None:
    if not settings.rerank_enabled:
        return None
    return SentenceTransformersCrossEncoderReranker(
        model_name=settings.rerank_model,
        batch_size=settings.rerank_batch_size,
        device=settings.rerank_device,
        local_files_only=settings.rerank_local_files_only,
        cache_dir=settings.rerank_cache_dir,
    )

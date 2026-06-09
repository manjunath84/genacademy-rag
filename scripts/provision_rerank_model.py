"""One-time provisioning for the Phase 2 rerank model.

This is the only project path that intentionally allows network download for the reranker.
Runtime and eval default to local_files_only=true.
"""
from genacademy_rag.config import Settings
from genacademy_rag.core.reranker import SentenceTransformersCrossEncoderReranker


def main():
    settings = Settings.from_env()
    SentenceTransformersCrossEncoderReranker(
        model_name=settings.rerank_model,
        batch_size=settings.rerank_batch_size,
        device=settings.rerank_device,
        local_files_only=False,
        cache_dir=settings.rerank_cache_dir,
    )
    print(f"provisioned rerank model: {settings.rerank_model}")


if __name__ == "__main__":
    main()

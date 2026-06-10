"""Settings from env. Provider presets keep generation pluggable (base_url + key + model)
behind one ModelProvider.generate() seam — no `if provider == ...` in business logic."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

# preset name -> (base_url env var, default base_url, key env var, model env var)
PROVIDER_PRESETS: dict[str, tuple[str, str, str, str]] = {
    "nebius": (
        "NEBIUS_BASE_URL",
        "https://api.studio.nebius.com/v1",
        "NEBIUS_API_KEY",
        "NEBIUS_MODEL",
    ),
    "openrouter": (
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
    ),
    "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_MODEL"),
    "gemma": ("GEMMA_BASE_URL", "http://127.0.0.1:8085/v1", "GEMMA_API_KEY", "GEMMA_MODEL"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
# Materials live at GenAcademy/CuratedRAGMaterials/ — two levels above REPO_ROOT
# (REPO_ROOT=genacademy-rag/ → parent=Week2-RAG_ContextEngineering/ → parent=GenAcademy/)
CURATED_MATERIALS_DIR = REPO_ROOT.parent.parent / "CuratedRAGMaterials"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    provider: str
    gen_base_url: str
    gen_api_key: str
    gen_model: str
    embed_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    chunker: str
    section_chunk_max_chars: int
    section_chunk_overlap: int
    chroma_dir: Path
    sqlite_path: Path
    session_secret: str
    rerank_enabled: bool
    rerank_model: str
    rerank_local_files_only: bool
    rerank_batch_size: int
    rerank_pool: int
    rerank_device: str | None
    rerank_cache_dir: Path | None
    # Vector store preset (Phase 2). Defaults keep the chroma path identical with no env set.
    vectorstore: str = "chroma"
    pinecone_api_key: str = ""
    pinecone_index: str = "genacademy-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    def __post_init__(self):
        if self.session_secret == "dev-only-change-me":
            warnings.warn(
                "GENACADEMY_SESSION_SECRET is the default dev value — set it before deploying",
                stacklevel=2,
            )

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.environ.get("GENACADEMY_PROVIDER", "openrouter")
        if provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"unknown GENACADEMY_PROVIDER={provider!r}; one of {list(PROVIDER_PRESETS)}"
            )
        base_var, base_default, key_var, model_var = PROVIDER_PRESETS[provider]
        return cls(
            provider=provider,
            gen_base_url=os.environ.get(base_var, base_default),
            gen_api_key=os.environ.get(key_var, ""),
            gen_model=os.environ.get(model_var, ""),
            embed_model=os.environ.get("GENACADEMY_EMBED_MODEL", "all-MiniLM-L6-v2"),
            top_k=int(os.environ.get("GENACADEMY_TOP_K", "5")),
            chunk_size=int(os.environ.get("GENACADEMY_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("GENACADEMY_CHUNK_OVERLAP", "150")),
            chunker=os.environ.get("GENACADEMY_CHUNKER", "fixed"),
            section_chunk_max_chars=int(
                os.environ.get("GENACADEMY_SECTION_CHUNK_MAX_CHARS", "1500")
            ),
            section_chunk_overlap=int(
                os.environ.get("GENACADEMY_SECTION_CHUNK_OVERLAP", "150")
            ),
            chroma_dir=Path(
                os.environ.get("GENACADEMY_CHROMA_DIR", str(DATA_DIR / "chroma"))
            ),
            sqlite_path=Path(
                os.environ.get("GENACADEMY_SQLITE", str(DATA_DIR / "genacademy.sqlite"))
            ),
            session_secret=os.environ.get("GENACADEMY_SESSION_SECRET", "dev-only-change-me"),
            rerank_enabled=_env_bool("GENACADEMY_RERANK_ENABLED", False),
            rerank_model=os.environ.get(
                "GENACADEMY_RERANK_MODEL",
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            ),
            rerank_local_files_only=_env_bool("GENACADEMY_RERANK_LOCAL_FILES_ONLY", True),
            rerank_batch_size=int(os.environ.get("GENACADEMY_RERANK_BATCH_SIZE", "32")),
            rerank_pool=int(os.environ.get("GENACADEMY_RERANK_POOL", "0")),
            rerank_device=os.environ.get("GENACADEMY_RERANK_DEVICE") or None,
            rerank_cache_dir=_env_path("GENACADEMY_RERANK_CACHE_DIR"),
            vectorstore=os.environ.get("GENACADEMY_VECTORSTORE", "chroma"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY", ""),
            pinecone_index=os.environ.get("GENACADEMY_PINECONE_INDEX", "genacademy-rag"),
            pinecone_cloud=os.environ.get("GENACADEMY_PINECONE_CLOUD", "aws"),
            pinecone_region=os.environ.get("GENACADEMY_PINECONE_REGION", "us-east-1"),
        )

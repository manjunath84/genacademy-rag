"""Settings from env. Provider presets keep generation pluggable (base_url + key + model)
behind one ModelProvider.generate() seam — no `if provider == ...` in business logic."""
from __future__ import annotations

import os
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
# Spike path fix: production files live one level above Week2-RAG_ContextEngineering.
CURATED_MATERIALS_DIR = REPO_ROOT.parent.parent / "CuratedRAGMaterials"


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
    chroma_dir: Path
    sqlite_path: Path
    session_secret: str

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
            chroma_dir=Path(
                os.environ.get("GENACADEMY_CHROMA_DIR", str(DATA_DIR / "chroma"))
            ),
            sqlite_path=Path(
                os.environ.get("GENACADEMY_SQLITE", str(DATA_DIR / "genacademy.sqlite"))
            ),
            session_secret=os.environ.get("GENACADEMY_SESSION_SECRET", "dev-only-change-me"),
        )

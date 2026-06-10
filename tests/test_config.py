from genacademy_rag.config import PROVIDER_PRESETS, Settings


def test_provider_preset_resolves_base_url_key_and_model(monkeypatch):
    monkeypatch.setenv("GENACADEMY_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
    s = Settings.from_env()
    assert s.provider == "openrouter"
    assert s.gen_base_url == "https://openrouter.ai/api/v1"
    assert s.gen_api_key == "sk-test"
    assert s.gen_model == "meta-llama/llama-3.1-70b-instruct"


def test_known_presets_present():
    assert {"nebius", "openrouter", "openai", "gemma"} <= set(PROVIDER_PRESETS)


def test_rerank_defaults_are_disabled_and_offline(monkeypatch):
    monkeypatch.delenv("GENACADEMY_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("GENACADEMY_RERANK_MODEL", raising=False)
    monkeypatch.delenv("GENACADEMY_RERANK_LOCAL_FILES_ONLY", raising=False)
    monkeypatch.delenv("GENACADEMY_RERANK_BATCH_SIZE", raising=False)
    monkeypatch.delenv("GENACADEMY_RERANK_POOL", raising=False)
    monkeypatch.delenv("GENACADEMY_RERANK_DEVICE", raising=False)
    monkeypatch.delenv("GENACADEMY_RERANK_CACHE_DIR", raising=False)

    s = Settings.from_env()

    assert s.rerank_enabled is False
    assert s.rerank_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert s.rerank_local_files_only is True
    assert s.rerank_batch_size == 32
    assert s.rerank_pool == 0
    assert s.rerank_device is None
    assert s.rerank_cache_dir is None


def test_vectorstore_defaults_to_chroma_with_empty_pinecone_settings(monkeypatch):
    for name in (
        "GENACADEMY_VECTORSTORE",
        "PINECONE_API_KEY",
        "GENACADEMY_PINECONE_INDEX",
        "GENACADEMY_PINECONE_CLOUD",
        "GENACADEMY_PINECONE_REGION",
    ):
        monkeypatch.delenv(name, raising=False)

    s = Settings.from_env()

    assert s.vectorstore == "chroma"
    assert s.pinecone_api_key == ""
    assert s.pinecone_index == "genacademy-rag"
    assert s.pinecone_cloud == "aws"
    assert s.pinecone_region == "us-east-1"


def test_vectorstore_env_settings_parse(monkeypatch):
    monkeypatch.setenv("GENACADEMY_VECTORSTORE", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "pk-test")
    monkeypatch.setenv("GENACADEMY_PINECONE_INDEX", "custom-index")
    monkeypatch.setenv("GENACADEMY_PINECONE_CLOUD", "gcp")
    monkeypatch.setenv("GENACADEMY_PINECONE_REGION", "us-central1")

    s = Settings.from_env()

    assert s.vectorstore == "pinecone"
    assert s.pinecone_api_key == "pk-test"
    assert s.pinecone_index == "custom-index"
    assert s.pinecone_cloud == "gcp"
    assert s.pinecone_region == "us-central1"


def test_rerank_env_settings_parse(monkeypatch, tmp_path):
    monkeypatch.setenv("GENACADEMY_RERANK_ENABLED", "true")
    monkeypatch.setenv("GENACADEMY_RERANK_MODEL", "custom/reranker")
    monkeypatch.setenv("GENACADEMY_RERANK_LOCAL_FILES_ONLY", "false")
    monkeypatch.setenv("GENACADEMY_RERANK_BATCH_SIZE", "8")
    monkeypatch.setenv("GENACADEMY_RERANK_POOL", "12")
    monkeypatch.setenv("GENACADEMY_RERANK_DEVICE", "cpu")
    monkeypatch.setenv("GENACADEMY_RERANK_CACHE_DIR", str(tmp_path / "hf-cache"))

    s = Settings.from_env()

    assert s.rerank_enabled is True
    assert s.rerank_model == "custom/reranker"
    assert s.rerank_local_files_only is False
    assert s.rerank_batch_size == 8
    assert s.rerank_pool == 12
    assert s.rerank_device == "cpu"
    assert s.rerank_cache_dir == tmp_path / "hf-cache"


def test_chunker_defaults_to_fixed(monkeypatch):
    monkeypatch.delenv("GENACADEMY_CHUNKER", raising=False)
    monkeypatch.delenv("GENACADEMY_SECTION_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("GENACADEMY_SECTION_CHUNK_OVERLAP", raising=False)

    s = Settings.from_env()

    assert s.chunker == "fixed"
    assert s.section_chunk_max_chars == 1500
    assert s.section_chunk_overlap == 150


def test_chunker_env_settings_parse(monkeypatch):
    monkeypatch.setenv("GENACADEMY_CHUNKER", "section")
    monkeypatch.setenv("GENACADEMY_SECTION_CHUNK_MAX_CHARS", "1800")
    monkeypatch.setenv("GENACADEMY_SECTION_CHUNK_OVERLAP", "120")

    s = Settings.from_env()

    assert s.chunker == "section"
    assert s.section_chunk_max_chars == 1800
    assert s.section_chunk_overlap == 120

from genacademy_rag.config import DATA_DIR, PROVIDER_PRESETS, Settings


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
        "GENACADEMY_EMBEDDINGS",
        "NEBIUS_EMBED_MODEL",
        "GENACADEMY_EMBED_DIM",
    ):
        monkeypatch.delenv(name, raising=False)

    s = Settings.from_env()

    assert s.vectorstore == "chroma"
    assert s.pinecone_api_key == ""
    assert s.pinecone_index == "genacademy-rag"
    assert s.pinecone_cloud == "aws"
    assert s.pinecone_region == "us-east-1"
    assert s.embeddings == "local"
    assert s.nebius_base_url == "https://api.studio.nebius.com/v1"
    assert s.nebius_api_key == ""
    assert s.nebius_embed_model == "Qwen/Qwen3-Embedding-8B"
    assert s.embed_dim == 384


def test_unknown_vectorstore_rejected_eagerly(monkeypatch):
    monkeypatch.setenv("GENACADEMY_VECTORSTORE", "faiss")
    try:
        Settings.from_env()
    except ValueError as exc:
        assert "GENACADEMY_VECTORSTORE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_unknown_embeddings_rejected_eagerly(monkeypatch):
    monkeypatch.setenv("GENACADEMY_EMBEDDINGS", "cloud")
    try:
        Settings.from_env()
    except ValueError as exc:
        assert "GENACADEMY_EMBEDDINGS" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_env_bool_rejects_unrecognized_values(monkeypatch):
    # Silent coercion to False would flip default-True safety flags permissive.
    monkeypatch.setenv("GENACADEMY_RERANK_LOCAL_FILES_ONLY", "enabled")
    try:
        Settings.from_env()
    except ValueError as exc:
        assert "GENACADEMY_RERANK_LOCAL_FILES_ONLY" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_env_bool_accepts_explicit_falsy_values(monkeypatch):
    monkeypatch.setenv("GENACADEMY_RERANK_LOCAL_FILES_ONLY", "off")
    assert Settings.from_env().rerank_local_files_only is False


def test_vectorstore_env_settings_parse(monkeypatch):
    monkeypatch.setenv("GENACADEMY_VECTORSTORE", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "pk-test")
    monkeypatch.setenv("GENACADEMY_PINECONE_INDEX", "custom-index")
    monkeypatch.setenv("GENACADEMY_PINECONE_CLOUD", "gcp")
    monkeypatch.setenv("GENACADEMY_PINECONE_REGION", "us-central1")
    monkeypatch.setenv("GENACADEMY_EMBEDDINGS", "nebius")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://nebius.test/v1")
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test")
    monkeypatch.setenv("NEBIUS_EMBED_MODEL", "custom/embedder")
    monkeypatch.setenv("GENACADEMY_EMBED_DIM", "4096")

    s = Settings.from_env()

    assert s.vectorstore == "pinecone"
    assert s.pinecone_api_key == "pk-test"
    assert s.pinecone_index == "custom-index"
    assert s.pinecone_cloud == "gcp"
    assert s.pinecone_region == "us-central1"
    assert s.embeddings == "nebius"
    assert s.nebius_base_url == "https://nebius.test/v1"
    assert s.nebius_api_key == "neb-test"
    assert s.nebius_embed_model == "custom/embedder"
    assert s.embed_dim == 4096


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


def test_deploy_data_dir_drives_default_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("GENACADEMY_DATA_DIR", str(tmp_path / "deploy-data"))
    monkeypatch.delenv("GENACADEMY_CHROMA_DIR", raising=False)
    monkeypatch.delenv("GENACADEMY_SQLITE", raising=False)

    s = Settings.from_env()

    assert s.chroma_dir == tmp_path / "deploy-data" / "chroma"
    assert s.sqlite_path == tmp_path / "deploy-data" / "genacademy.sqlite"


def test_blank_deploy_data_dir_uses_default_paths(monkeypatch):
    monkeypatch.setenv("GENACADEMY_DATA_DIR", "")
    monkeypatch.delenv("GENACADEMY_CHROMA_DIR", raising=False)
    monkeypatch.delenv("GENACADEMY_SQLITE", raising=False)

    s = Settings.from_env()

    assert s.chroma_dir == DATA_DIR / "chroma"
    assert s.sqlite_path == DATA_DIR / "genacademy.sqlite"


def test_secure_cookies_default_false_and_env_parse(monkeypatch):
    monkeypatch.delenv("GENACADEMY_SECURE_COOKIES", raising=False)
    assert Settings.from_env().secure_cookies is False

    monkeypatch.setenv("GENACADEMY_SECURE_COOKIES", "true")
    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    assert Settings.from_env().secure_cookies is True


def test_secure_cookies_reject_default_session_secret(monkeypatch):
    monkeypatch.setenv("GENACADEMY_SECURE_COOKIES", "true")
    monkeypatch.delenv("GENACADEMY_SESSION_SECRET", raising=False)

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "GENACADEMY_SESSION_SECRET" in str(exc)
    else:
        raise AssertionError("expected secure deploy to require a non-default session secret")

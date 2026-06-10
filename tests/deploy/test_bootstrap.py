from genacademy_rag.config import Settings


def _settings(tmp_path):
    return Settings(
        provider="openrouter",
        gen_base_url="https://openrouter.ai/api/v1",
        gen_api_key="",
        gen_model="",
        embed_model="all-MiniLM-L6-v2",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        chunker="fixed",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "genacademy.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
    )


def test_bootstrap_skips_when_eval_collection_has_chunks(monkeypatch, tmp_path, capsys):
    import genacademy_rag.deploy.bootstrap as bootstrap

    settings = _settings(tmp_path)
    state = {"ingest_called": False}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["collection"] = collection

        def get_all_chunks(self):
            return [object()]

    monkeypatch.setattr(bootstrap.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(bootstrap, "ChromaStore", _Store)
    monkeypatch.setattr(
        bootstrap,
        "_run_ingest",
        lambda: state.__setitem__("ingest_called", True),
    )

    bootstrap.main([])

    assert state["collection"] == "eval"
    assert state["ingest_called"] is False
    assert "eval collection already seeded" in capsys.readouterr().out


def test_bootstrap_runs_ingest_when_eval_collection_empty(monkeypatch, tmp_path):
    import genacademy_rag.deploy.bootstrap as bootstrap

    settings = _settings(tmp_path)
    state = {"reset": None}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            pass

        def get_all_chunks(self):
            return []

    def fake_ingest(*, reset_collection):
        state["reset"] = reset_collection

    monkeypatch.setattr(bootstrap.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(bootstrap, "ChromaStore", _Store)
    monkeypatch.setattr(bootstrap, "_run_ingest", fake_ingest)

    bootstrap.main([])

    assert state["reset"] is False


def test_bootstrap_force_reseeds_existing_eval_collection(monkeypatch, tmp_path):
    import genacademy_rag.deploy.bootstrap as bootstrap

    settings = _settings(tmp_path)
    state = {"reset": None}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            pass

        def get_all_chunks(self):
            return [object()]

    monkeypatch.setattr(bootstrap.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(bootstrap, "ChromaStore", _Store)
    monkeypatch.setattr(
        bootstrap,
        "_run_ingest",
        lambda *, reset_collection: state.__setitem__("reset", reset_collection),
    )

    bootstrap.main(["--force"])

    assert state["reset"] is True


def test_run_ingest_uses_module_subprocess(monkeypatch):
    import genacademy_rag.deploy.bootstrap as bootstrap

    state = {}
    monkeypatch.setattr(bootstrap.sys, "executable", "/python")
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda cmd, cwd, check: state.update({"cmd": cmd, "cwd": cwd, "check": check}),
    )

    bootstrap._run_ingest(reset_collection=True)

    assert state["cmd"] == [
        "/python",
        "-m",
        "scripts.ingest_eval_corpus",
        "--chunker",
        "fixed",
        "--reset-collection",
    ]
    assert state["cwd"] == bootstrap.REPO_ROOT
    assert state["check"] is True

import sys

import pytest

import scripts.ingest_eval_corpus as ingest_script
from genacademy_rag.config import Settings
from genacademy_rag.core.types import Document


def _settings(tmp_path, *, chunker="fixed"):
    return Settings(
        provider="openrouter",
        gen_base_url="https://openrouter.ai/api/v1",
        gen_api_key="",
        gen_model="",
        embed_model="all-MiniLM-L6-v2",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        chunker=chunker,
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


@pytest.mark.parametrize(
    ("settings_chunker", "argv"),
    [
        ("section", ["ingest_eval_corpus.py"]),
        ("fixed", ["ingest_eval_corpus.py", "--chunker", "section"]),
    ],
)
def test_ingest_eval_refuses_non_fixed_chunker_for_baseline_collection(
    monkeypatch,
    tmp_path,
    settings_chunker,
    argv,
):
    settings = _settings(tmp_path, chunker=settings_chunker)

    monkeypatch.setattr(ingest_script.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        ingest_script,
        "build_provider",
        lambda s: pytest.fail("baseline guard should run before provider construction"),
    )

    with pytest.raises(SystemExit) as exc:
        ingest_script.main()

    assert "refusing to ingest collection='eval' with chunker='section'" in str(exc.value)


def test_ingest_eval_defaults_to_eval_collection_fixed_chunker_and_primary_sqlite(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    state = {}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["persist_dir"] = persist_dir
            state["collection"] = collection

        def upsert(self, chunks, embeddings):
            state["upserted"] = [c.chunk_id for c in chunks]

    class _Datastore:
        def __init__(self, path):
            state["sqlite_path"] = path

        def seed_users(self):
            state["seeded"] = True

        def add_document(self, **kwargs):
            state["doc_id"] = kwargs["doc_id"]

        def add_chunks_meta(self, chunks):
            state["chunks_meta"] = [c.chunk_id for c in chunks]

    class _Provider:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingest_script.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(ingest_script, "build_provider", lambda s: _Provider())
    monkeypatch.setattr(ingest_script, "ChromaStore", _Store)
    monkeypatch.setattr(ingest_script, "SQLiteDatastore", _Datastore)
    monkeypatch.setattr(
        ingest_script,
        "EVAL_CORPUS",
        [
            {
                "owner": "owner",
                "repo": "repo",
                "sha": "abc123",
                "files": [{"path": "README.md", "kind": "markdown"}],
            }
        ],
    )
    monkeypatch.setattr(ingest_script, "fetch_raw", lambda **kwargs: b"# Title\n\nBody\n")
    monkeypatch.setattr(
        ingest_script,
        "load_markdown",
        lambda **kwargs: Document(
            doc_id="repo/README.md@abc123",
            title="README.md",
            source_type="github",
            text="# Title\n\nBody\n",
            repo="repo",
            file_path="README.md",
            commit_hash="abc123",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["ingest_eval_corpus.py"])

    ingest_script.main()

    assert state["collection"] == "eval"
    assert state["sqlite_path"] == settings.sqlite_path
    assert state["seeded"] is True
    assert state["chunks_meta"] == ["repo/README.md@abc123::0"]


def test_ingest_eval_section_collection_uses_isolated_sqlite_by_default(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    state = {}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["collection"] = collection

        def upsert(self, chunks, embeddings):
            state["chunk_texts"] = [c.text for c in chunks]

    class _Datastore:
        def __init__(self, path):
            state["sqlite_path"] = path

        def seed_users(self):
            pass

        def add_document(self, **kwargs):
            state["n_chunks"] = kwargs["n_chunks"]

        def add_chunks_meta(self, chunks):
            state["page_or_section"] = chunks[0].citation.page_or_section

    class _Provider:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingest_script.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(ingest_script, "build_provider", lambda s: _Provider())
    monkeypatch.setattr(ingest_script, "ChromaStore", _Store)
    monkeypatch.setattr(ingest_script, "SQLiteDatastore", _Datastore)
    monkeypatch.setattr(
        ingest_script,
        "reset_chroma_collection",
        lambda persist_dir, collection: state.setdefault("reset", collection),
        raising=False,
    )
    monkeypatch.setattr(
        ingest_script,
        "EVAL_CORPUS",
        [
            {
                "owner": "owner",
                "repo": "repo",
                "sha": "abc123",
                "files": [{"path": "README.md", "kind": "markdown"}],
            }
        ],
    )
    monkeypatch.setattr(
        ingest_script,
        "fetch_raw",
        lambda **kwargs: b"# Title\n\n| A | B |\n| - | - |\n| C | D |\n",
    )
    monkeypatch.setattr(
        ingest_script,
        "load_markdown",
        lambda **kwargs: Document(
            doc_id="repo/README.md@abc123",
            title="README.md",
            source_type="github",
            text="# Title\n\n| A | B |\n| - | - |\n| C | D |\n",
            repo="repo",
            file_path="README.md",
            commit_hash="abc123",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_eval_corpus.py",
            "--collection",
            "eval_section",
            "--chunker",
            "section",
            "--reset-collection",
        ],
    )

    ingest_script.main()

    assert state["collection"] == "eval_section"
    assert state["reset"] == "eval_section"
    assert state["sqlite_path"] == settings.sqlite_path.with_name(
        "genacademy-eval_section.sqlite"
    )
    assert state["page_or_section"] == "section: Title"

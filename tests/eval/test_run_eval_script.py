import sys

import scripts.run_eval as run_eval
from genacademy_rag.config import Settings
from genacademy_rag.core.pipeline import QueryResult
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan


def test_run_eval_keeps_retrieval_local_when_nebius_embeddings_selected(monkeypatch, tmp_path):
    settings = Settings(
        provider="openrouter",
        gen_base_url="https://openrouter.ai/api/v1",
        gen_api_key="sk-gen",
        gen_model="gen-model",
        embed_model="local/model",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        chunker="fixed",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "db.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
        embeddings="nebius",
        nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
        nebius_api_key="",
        nebius_embed_model="Qwen/Qwen3-Embedding-8B",
        embed_dim=4096,
    )
    citation = Citation(
        doc_id="d1",
        title="README.md",
        source_type="github",
        repo="repo",
        file_path="README.md",
        commit_hash="abc123",
        line_start=1,
        line_end=2,
    )
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text="RAG text", citation=citation)
    question = GoldQuestion(
        id="q1",
        question="What is RAG?",
        category="answerable",
        answerable=True,
        gold=[
            GoldSpan(
                repo="repo",
                file_path="README.md",
                commit_hash="abc123",
                line_start=1,
                line_end=2,
            )
        ],
    )
    state = {}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["collection"] = collection

        def get_all_chunks(self):
            return [chunk]

    class _Embedder:
        def __init__(self, model_name):
            state["embed_model"] = model_name

        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _text in texts]

    class _Generator:
        def __init__(self, base_url, api_key, model):
            state["generator"] = (base_url, api_key, model)

        def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0):
            return '{"answerable": true, "confidence": 5}' if json_mode else "answer"

    class _Retriever:
        def __init__(
            self, *, store, provider, all_chunks, top_k, candidate_k, reranker, rerank_pool
        ):
            state["retriever_provider"] = provider
            state["top_k"] = top_k

        def retrieve(self, query):
            return [RetrievedChunk(chunk=chunk, score=0.9)]

    class _QueryPipeline:
        def __init__(self, *, retriever, provider):
            state["query_provider"] = provider

        def answer(self, question):
            return QueryResult(
                answer="grounded answer",
                citations=[citation],
                refused=False,
                confidence=5,
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_eval.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(run_eval, "ChromaStore", _Store)
    monkeypatch.setattr(run_eval, "STEmbedder", _Embedder)
    monkeypatch.setattr(run_eval, "OpenAICompatProvider", _Generator)
    monkeypatch.setattr(run_eval, "HybridRetriever", _Retriever)
    monkeypatch.setattr(run_eval, "QueryPipeline", _QueryPipeline)
    monkeypatch.setattr(run_eval, "build_reranker", lambda s: None)
    monkeypatch.setattr(run_eval, "load_gold_set", lambda path: [question])
    monkeypatch.setattr(run_eval, "render_report", lambda *args, **kwargs: "# report\n")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--no-judge"])

    run_eval.main()

    assert state["collection"] == "eval"
    assert state["embed_model"] == "local/model"
    assert state["generator"] == ("https://openrouter.ai/api/v1", "sk-gen", "gen-model")
    assert isinstance(state["retriever_provider"], _Embedder)
    assert isinstance(state["query_provider"], _Generator)
    assert (tmp_path / "eval" / "REPORT.md").read_text() == "# report\n"

import json
import sys

import scripts.eval_retrieval as eval_script
from genacademy_rag.config import Settings
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan


def test_eval_retrieval_json_out_includes_config_latency_and_rows(monkeypatch, tmp_path):
    settings = Settings(
        provider="openrouter",
        gen_base_url="https://openrouter.ai/api/v1",
        gen_api_key="",
        gen_model="",
        embed_model="all-MiniLM-L6-v2",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "db.sqlite",
        session_secret="test-secret",
        rerank_enabled=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device="cpu",
        rerank_cache_dir=None,
    )
    citation = Citation(
        doc_id="d1",
        title="README.md",
        source_type="github",
        repo="awesome-agentic-ai-resources",
        file_path="README.md",
        commit_hash="abc123",
        line_start=10,
        line_end=20,
    )
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text="RAG text", citation=citation)
    question = GoldQuestion(
        id="q1",
        question="What is RAG?",
        category="answerable",
        answerable=True,
        gold=[
            GoldSpan(
                repo="awesome-agentic-ai-resources",
                file_path="README.md",
                commit_hash="abc123",
                line_start=10,
                line_end=20,
            )
        ],
    )
    state = {}
    fake_reranker = object()

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

    class _Retriever:
        def __init__(
            self,
            *,
            store,
            provider,
            all_chunks,
            top_k,
            candidate_k,
            reranker,
            rerank_pool,
        ):
            state["top_k"] = top_k
            state["candidate_k"] = candidate_k
            state["reranker"] = reranker
            state["rerank_pool"] = rerank_pool

        def retrieve(self, query):
            return [RetrievedChunk(chunk=chunk, score=0.9)]

    out = tmp_path / "retrieval.json"
    monkeypatch.setattr(eval_script.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(eval_script, "ChromaStore", _Store)
    monkeypatch.setattr(eval_script, "STEmbedder", _Embedder)
    monkeypatch.setattr(eval_script, "HybridRetriever", _Retriever)
    monkeypatch.setattr(eval_script, "build_reranker", lambda s: fake_reranker, raising=False)
    monkeypatch.setattr(eval_script, "load_gold_set", lambda path: [question])
    monkeypatch.setattr(sys, "argv", ["eval_retrieval.py", "--json-out", str(out)])

    eval_script.main()

    payload = json.loads(out.read_text())
    assert state["collection"] == "eval"
    assert state["top_k"] == 5
    assert state["candidate_k"] == 20
    assert state["reranker"] is fake_reranker
    assert state["rerank_pool"] == 0
    assert payload["metrics"] == {
        "n_retrieval_questions": 1,
        "recall@k": 1.0,
        "precision@k": 0.2,
        "mrr": 1.0,
    }
    assert payload["config"]["rerank_enabled"] is True
    assert payload["config"]["rerank_device"] == "cpu"
    assert payload["config"]["candidate_k"] == 20
    assert payload["latency"]["retrieval_ms_mean"] >= 0.0
    assert payload["questions"][0]["retrieval_ms"] >= 0.0
    assert payload["questions"][0]["max_cosine"] == 0.9
    assert payload["questions"][0]["cosine_fallback_answerable_at_0_2"] is True
    assert payload["questions"][0]["retrieved"][0]["chunk_id"] == "d1::0"

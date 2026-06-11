from genacademy_rag.core.graph import build_graph
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from tests.conftest import FakeModelProvider

REFUSAL = "I could not find this in the course materials."


class _Retriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query):
        return self._chunks


def _rc(text):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="r", file_path="README.md", commit_hash="abc123", line_start=1, line_end=2)
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text=text, citation=cit)
    return RetrievedChunk(chunk=chunk, score=0.8)


def test_answerable_path_returns_answer_and_citations():
    provider = FakeModelProvider(canned_json='{"answerable": true, "confidence": 5}',
                                 canned_answer="RAG retrieves then generates.")
    retriever = _Retriever([_rc("RAG retrieves then generates.")])
    graph = build_graph(retriever=retriever, provider=provider)
    out = graph.invoke({"question": "what is RAG?"})
    assert out["refused"] is False
    assert out["answer"] == "RAG retrieves then generates."
    assert out["citations"][0].commit_hash == "abc123"


def test_unanswerable_path_refuses_without_calling_answer():
    provider = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')
    graph = build_graph(retriever=_Retriever([_rc("unrelated")]), provider=provider)
    out = graph.invoke({"question": "who won the 2050 world cup?"})
    assert out["refused"] is True
    assert out["answer"] == REFUSAL
    # the answer-generation path must NOT have been taken (no non-json generate call)
    assert all(c["json_mode"] for c in provider.calls)


def test_answer_prompt_keeps_grounding_and_adds_overview_format():
    from genacademy_rag.core.graph import ANSWER_SYSTEM

    assert "You answer ONLY from the provided course context" in ANSWER_SYSTEM
    assert "Never use outside knowledge" in ANSWER_SYSTEM
    assert "overview paragraph" in ANSWER_SYSTEM
    assert "Be concise" not in ANSWER_SYSTEM


def test_answer_node_generates_with_800_max_tokens(fake_provider):
    captured = {}
    real_generate = fake_provider.generate

    def recording_generate(messages, *, json_mode=False, max_tokens=512, temperature=0.0):
        if not json_mode:
            captured["max_tokens"] = max_tokens
        return real_generate(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    fake_provider.generate = recording_generate

    graph = build_graph(
        retriever=_Retriever([_rc("RAG retrieves then generates.")]),
        provider=fake_provider,
    )
    out = graph.invoke({"question": "What is RAG?"})
    assert out["refused"] is False
    assert captured["max_tokens"] == 800

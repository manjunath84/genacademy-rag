from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from tests.conftest import FakeModelProvider


class _Retriever:
    def retrieve(self, q):
        cit = Citation(
            doc_id="d1", title="README.md", source_type="github",
            repo="r", file_path="README.md", commit_hash="abc123",
            line_start=1, line_end=2,
        )
        return [RetrievedChunk(
            chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0,
                        text="RAG retrieves then generates.", citation=cit),
            score=0.8,
        )]


def test_answerable_query_returns_answer_with_citations():
    provider = FakeModelProvider(
        canned_json='{"answerable": true, "confidence": 5}',
        canned_answer="RAG retrieves then generates.",
    )
    qp = QueryPipeline(retriever=_Retriever(), provider=provider)
    result = qp.answer("what is RAG?")
    assert result.refused is False
    assert result.answer == "RAG retrieves then generates."
    assert result.citations[0].file_path == "README.md"


def test_unanswerable_query_refuses():
    provider = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')
    qp = QueryPipeline(retriever=_Retriever(), provider=provider)
    result = qp.answer("who won the 2050 world cup?")
    assert result.refused is True
    assert "could not find" in result.answer.lower()

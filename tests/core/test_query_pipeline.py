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


def test_answer_result_carries_merged_sources(fake_provider):
    cit = Citation(
        doc_id="d1",
        title="README.md",
        source_type="github",
        repo="awesome-agentic-ai-resources",
        file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        line_start=41,
        line_end=57,
    )
    cit2 = Citation(
        doc_id="d1",
        title="README.md",
        source_type="github",
        repo="awesome-agentic-ai-resources",
        file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        line_start=57,
        line_end=70,
    )

    class _MergingRetriever:
        def retrieve(self, q):
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="d1::0",
                        doc_id="d1",
                        ordinal=0,
                        text="chunk one",
                        citation=cit,
                    ),
                    score=0.9,
                ),
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="d1::1",
                        doc_id="d1",
                        ordinal=1,
                        text="chunk two",
                        citation=cit2,
                    ),
                    score=0.8,
                ),
            ]

    qp = QueryPipeline(retriever=_MergingRetriever(), provider=fake_provider)
    result = qp.answer("What did the course say about chunking?")
    assert result.refused is False
    assert len(result.citations) == 2
    assert len(result.sources) == 1
    assert result.sources[0].range_label == "lines 41–70"


def test_refused_result_has_empty_sources():
    refusing = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')

    class _EmptyRetriever:
        def retrieve(self, q):
            return []

    qp = QueryPipeline(retriever=_EmptyRetriever(), provider=refusing)
    result = qp.answer("Who won the 2030 World Cup?")
    assert result.refused is True
    assert result.sources == []

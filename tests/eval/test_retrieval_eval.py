from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan
from genacademy_rag.eval.retrieval_eval import aggregate, chunk_matches_span, score_question


def _rc(line_start, line_end, commit="abc123", text="x"):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="awesome-agentic-ai-resources", file_path="README.md",
                   commit_hash=commit, line_start=line_start, line_end=line_end)
    return RetrievedChunk(chunk=Chunk(chunk_id=f"d1::{line_start}", doc_id="d1",
                                      ordinal=line_start, text=text, citation=cit), score=1.0)


def test_chunk_matches_span_requires_overlap_and_commit():
    span = GoldSpan(repo="awesome-agentic-ai-resources", file_path="README.md",
                    commit_hash="abc123", line_start=10, line_end=20)
    assert chunk_matches_span(_rc(8, 15).chunk, span)            # overlaps 10-15
    assert not chunk_matches_span(_rc(30, 40).chunk, span)       # no overlap
    # commit mismatch -> no leak
    assert not chunk_matches_span(_rc(8, 15, commit="WRONG").chunk, span)


def test_recall_precision_mrr_on_a_hit_at_rank_2():
    q = GoldQuestion(id="q1", question="x", category="answerable", answerable=True,
                     gold=[GoldSpan("awesome-agentic-ai-resources", "README.md", "abc123", 10, 20)])
    retrieved = [_rc(30, 40), _rc(12, 18), _rc(50, 60)]   # gold is at rank 2
    s = score_question(q, retrieved, k=5)
    assert s["recall"] == 1.0
    assert s["precision"] == 1 / 5           # 1 relevant of k=5 slots
    assert s["mrr"] == 1 / 2                 # first relevant at rank 2


def test_unanswerable_question_excluded_from_retrieval_metrics():
    q = GoldQuestion(id="q2", question="x", category="unanswerable", answerable=False, gold=[])
    s = score_question(q, [_rc(1, 2)], k=5)
    assert s["recall"] is None               # not a retrieval-scored question
    agg = aggregate([s])
    assert agg["n_retrieval_questions"] == 0

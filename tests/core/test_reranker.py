import pytest

from genacademy_rag.core.reranker import (
    RerankerSetupError,
    SentenceTransformersCrossEncoderReranker,
    build_reranker,
)
from genacademy_rag.core.types import Chunk, Citation


def _chunk(i: int, text: str) -> Chunk:
    cit = Citation(doc_id="d1", title="t", source_type="github")
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=text, citation=cit)


class _FakeCrossEncoder:
    instances = []
    scores_by_text: dict[str, float] = {}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        type(self).instances.append(self)

    def predict(self, pairs, batch_size):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        return [self.scores_by_text[pair[1]] for pair in pairs]


def test_cross_encoder_wrapper_sorts_scores_descending_and_passes_runtime_settings(tmp_path):
    _FakeCrossEncoder.instances.clear()
    _FakeCrossEncoder.scores_by_text = {"low": -1.0, "high": 9.0, "middle": 2.5}
    chunks = [_chunk(0, "low"), _chunk(1, "high"), _chunk(2, "middle")]

    reranker = SentenceTransformersCrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cpu",
        batch_size=7,
        local_files_only=True,
        cache_dir=tmp_path,
        cross_encoder_cls=_FakeCrossEncoder,
    )

    ranked = reranker.rerank("question", chunks)

    assert [chunk.chunk_id for chunk, _score in ranked] == ["d1::1", "d1::2", "d1::0"]
    assert [score for _chunk, score in ranked] == [9.0, 2.5, -1.0]
    instance = _FakeCrossEncoder.instances[0]
    assert instance.args == ("cross-encoder/ms-marco-MiniLM-L6-v2",)
    assert instance.kwargs == {
        "device": "cpu",
        "local_files_only": True,
        "cache_folder": str(tmp_path),
    }
    assert instance.calls == [
        {
            "pairs": [("question", "low"), ("question", "high"), ("question", "middle")],
            "batch_size": 7,
        }
    ]


def test_cross_encoder_wrapper_preserves_input_order_for_tied_scores():
    _FakeCrossEncoder.instances.clear()
    _FakeCrossEncoder.scores_by_text = {"first": 1.0, "second": 1.0, "third": 1.0}
    chunks = [_chunk(0, "first"), _chunk(1, "second"), _chunk(2, "third")]

    reranker = SentenceTransformersCrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size=32,
        local_files_only=True,
        cross_encoder_cls=_FakeCrossEncoder,
    )

    assert [chunk.chunk_id for chunk, _score in reranker.rerank("q", chunks)] == [
        "d1::0",
        "d1::1",
        "d1::2",
    ]


def test_cross_encoder_wrapper_handles_empty_input_without_predict_call():
    _FakeCrossEncoder.instances.clear()
    _FakeCrossEncoder.scores_by_text = {}
    reranker = SentenceTransformersCrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size=32,
        local_files_only=True,
        cross_encoder_cls=_FakeCrossEncoder,
    )

    assert reranker.rerank("q", []) == []
    assert _FakeCrossEncoder.instances[0].calls == []


def test_cross_encoder_wrapper_clones_loaded_model_tensors():
    torch = pytest.importorskip("torch")
    instances = []

    class _CrossEncoderWithModel:
        def __init__(self, *args, **kwargs):
            self.model = torch.nn.Linear(2, 1)
            self.original_weight = self.model.weight
            instances.append(self)

        def predict(self, pairs, batch_size):
            return [1.0 for _pair in pairs]

    SentenceTransformersCrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size=32,
        local_files_only=True,
        cross_encoder_cls=_CrossEncoderWithModel,
    )

    instance = instances[0]
    assert instance.model.weight.data_ptr() != instance.original_weight.data_ptr()
    assert torch.equal(instance.model.weight, instance.original_weight)


def test_missing_local_model_error_names_provisioning_command():
    class _BrokenCrossEncoder:
        def __init__(self, *args, **kwargs):
            raise OSError("model files not found")

    with pytest.raises(RerankerSetupError, match="scripts/provision_rerank_model.py"):
        SentenceTransformersCrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            batch_size=32,
            local_files_only=True,
            cross_encoder_cls=_BrokenCrossEncoder,
        )


def test_build_reranker_returns_none_when_disabled(monkeypatch):
    from genacademy_rag.config import Settings

    monkeypatch.setenv("GENACADEMY_RERANK_ENABLED", "false")
    settings = Settings.from_env()

    assert build_reranker(settings) is None

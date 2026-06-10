import math

import pytest

import genacademy_rag.core.providers as providers_module
from genacademy_rag.config import Settings
from genacademy_rag.core.providers import (
    EmbedderSetupError,
    OpenAICompatEmbedder,
    OpenAICompatProvider,
    STEmbedder,
    build_embedder,
    build_provider,
)


def test_st_embedder_returns_384_dim(monkeypatch):
    # Avoid loading the real model in a unit test: stub the encoder.
    emb = STEmbedder.__new__(STEmbedder)
    emb._model = type(
        "M", (), {"encode": staticmethod(lambda xs, **k: [[0.0] * 384 for _ in xs])}
    )()
    vecs = emb.embed(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 384


def test_openai_compat_provider_builds_json_request():
    captured = {}

    class FakeChat:
        class completions:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                msg = type("Msg", (), {"content": '{"answerable": true, "confidence": 4}'})()
                choice = type("Ch", (), {"message": msg})()
                return type("R", (), {"choices": [choice]})()

    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p._client = type("C", (), {"chat": FakeChat})()
    p._model = "test-model"
    out = p.generate([{"role": "user", "content": "hi"}], json_mode=True, max_tokens=64)
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.0
    assert "answerable" in out


def test_openai_compat_embedder_empty_input_makes_no_request():
    class FakeClient:
        def __init__(self, **_kwargs):
            self.embeddings = type("Embeddings", (), {"create": self.create})()

        def create(self, **_kwargs):
            raise AssertionError("empty input must not call embeddings.create")

    embedder = OpenAICompatEmbedder(
        base_url="https://nebius.test/v1",
        api_key="neb-test",
        model="embed-model",
        client_cls=FakeClient,
    )

    assert embedder.embed([]) == []


def test_openai_compat_embedder_sorts_by_index_and_l2_normalizes():
    captured = {}

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {
                    "data": [
                        type("Item", (), {"index": 1, "embedding": [0.0, 6.0, 8.0]})(),
                        type("Item", (), {"index": 0, "embedding": [3.0, 4.0, 0.0]})(),
                    ]
                },
            )()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.embeddings = FakeEmbeddings()

    embedder = OpenAICompatEmbedder(
        base_url="https://nebius.test/v1",
        api_key="neb-test",
        model="embed-model",
        client_cls=FakeClient,
    )

    vecs = embedder.embed(["first", "second"])

    assert captured["client_kwargs"] == {
        "base_url": "https://nebius.test/v1",
        "api_key": "neb-test",
    }
    assert captured["model"] == "embed-model"
    assert captured["input"] == ["first", "second"]
    assert captured["encoding_format"] == "float"
    assert vecs == [
        pytest.approx([0.6, 0.8, 0.0]),
        pytest.approx([0.0, 0.6, 0.8]),
    ]
    assert [math.sqrt(sum(x * x for x in vec)) for vec in vecs] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]


def test_openai_compat_embedder_batches_inputs_at_128():
    call_lengths = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            batch = list(kwargs["input"])
            call_lengths.append(len(batch))
            return type(
                "Response",
                (),
                {
                    "data": [
                        type("Item", (), {"index": i, "embedding": [1.0, 0.0]})()
                        for i, _text in enumerate(batch)
                    ]
                },
            )()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.embeddings = FakeEmbeddings()

    embedder = OpenAICompatEmbedder(
        base_url="https://nebius.test/v1",
        api_key="neb-test",
        model="embed-model",
        client_cls=FakeClient,
    )

    vecs = embedder.embed([f"text {i}" for i in range(129)])

    assert call_lengths == [128, 1]
    assert len(vecs) == 129


def test_build_embedder_defaults_to_local_without_loading_model(monkeypatch):
    recorded = {}

    class FakeSTEmbedder:
        def __init__(self, model_name):
            recorded["model_name"] = model_name

    monkeypatch.setattr(providers_module, "STEmbedder", FakeSTEmbedder)
    monkeypatch.delenv("GENACADEMY_EMBEDDINGS", raising=False)
    monkeypatch.setenv("GENACADEMY_EMBED_MODEL", "local/model")

    embedder = build_embedder(Settings.from_env())

    assert isinstance(embedder, FakeSTEmbedder)
    assert recorded == {"model_name": "local/model"}


def test_build_embedder_nebius_without_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("GENACADEMY_EMBEDDINGS", "nebius")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    with pytest.raises(EmbedderSetupError, match="NEBIUS_API_KEY"):
        build_embedder(Settings.from_env())


def test_build_provider_routes_through_build_embedder(monkeypatch):
    recorded = {}
    embedder = object()

    class FakeGenerator:
        def __init__(self, base_url, api_key, model):
            recorded["generator"] = (base_url, api_key, model)

    monkeypatch.setattr(providers_module, "build_embedder", lambda settings: embedder)
    monkeypatch.setattr(providers_module, "OpenAICompatProvider", FakeGenerator)
    monkeypatch.setenv("GENACADEMY_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    provider = build_provider(Settings.from_env())

    assert provider._embedder is embedder
    assert isinstance(provider._generator, FakeGenerator)
    assert recorded["generator"] == ("https://api.openai.com/v1", "sk-test", "gpt-test")

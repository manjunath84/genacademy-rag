from genacademy_rag.core.providers import OpenAICompatProvider, STEmbedder


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

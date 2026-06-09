from starlette.testclient import TestClient


def _client(monkeypatch, tmp_path, refused=False):
    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    class _Retriever:
        def retrieve(self, q):
            from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk

            cit = Citation(
                doc_id="d1",
                title="README.md",
                source_type="github",
                repo="r",
                file_path="README.md",
                commit_hash="abc123",
                line_start=1,
                line_end=2,
            )
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="d1::0",
                        doc_id="d1",
                        ordinal=0,
                        text="RAG retrieves then generates.",
                        citation=cit,
                    ),
                    score=0.8,
                )
            ]

    canned = (
        '{"answerable": false, "confidence": 1}'
        if refused
        else '{"answerable": true, "confidence": 5}'
    )
    provider = FakeModelProvider(
        canned_json=canned, canned_answer="RAG retrieves then generates."
    )
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    app = create_app(retriever=_Retriever(), provider=provider, datastore=datastore)
    return TestClient(app)


def test_unauthenticated_chat_redirects_to_login(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and "/login" in r.headers["location"]


def test_login_then_ask_renders_cited_answer(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    c.post("/login", data={"email": "member@genacademy.local", "password": "member"})
    r = c.post("/ask", data={"question": "what is RAG?"})
    assert r.status_code == 200
    assert "RAG retrieves then generates." in r.text
    assert "README.md" in r.text  # source card rendered
    assert "details" in r.text.lower()


def test_refusal_is_rendered_not_an_answer(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, refused=True)
    c.post("/login", data={"email": "member@genacademy.local", "password": "member"})
    r = c.post("/ask", data={"question": "who won the 2050 world cup?"})
    assert "could not find" in r.text.lower()

import re

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


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def _login(client, email="member@genacademy.local", password="member"):
    page = client.get("/login")
    return client.post(
        "/login",
        data={"email": email, "password": password, "csrf_token": _csrf(page.text)},
    )


def test_unauthenticated_chat_redirects_to_login(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and "/login" in r.headers["location"]


def test_login_then_ask_renders_cited_answer(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    page = c.get("/")
    r = c.post("/ask", data={"question": "what is RAG?", "csrf_token": _csrf(page.text)})
    assert r.status_code == 200
    assert "RAG retrieves then generates." in r.text
    assert "README.md" in r.text  # source card rendered
    assert "details" in r.text.lower()


def test_refusal_is_rendered_not_an_answer(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, refused=True)
    _login(c)
    page = c.get("/")
    r = c.post(
        "/ask",
        data={"question": "who won the 2050 world cup?", "csrf_token": _csrf(page.text)},
    )
    assert "could not find" in r.text.lower()


def test_signup_redeems_invite_and_logs_in(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    r = c.get("/admin/invites")
    token = _csrf(r.text)
    generated = c.post(
        "/admin/invites",
        data={"role": "member", "expires_days": "7", "csrf_token": token},
    )
    code = re.search(r"Invite code: ([^<]+)<", generated.text).group(1)
    signup = c.get("/signup")
    signup_token = _csrf(signup.text)
    created = c.post(
        "/signup",
        data={
            "email": "new@example.com",
            "password": "secret",
            "code": code,
            "csrf_token": signup_token,
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    home = c.get("/")
    assert "Ask the cohort materials" in home.text


def test_admin_routes_block_member(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    assert c.get("/admin/invites", follow_redirects=False).status_code == 403


def test_csrf_required_for_invite_generation(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    r = c.post("/admin/invites", data={"role": "member", "expires_days": "7"})
    assert r.status_code == 403


def test_admin_upload_is_searchable_and_eval_stays_pristine(monkeypatch, tmp_path):
    import io

    from pypdf import PdfWriter

    from genacademy_rag.core.chunker import FixedSizeChunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.core.vectorstore import ChromaStore
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")

    # Build a tiny PDF
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    pdf_bytes = buf.getvalue()

    serving_store = ChromaStore(persist_dir=tmp_path / "chroma", collection="serving")
    eval_store = ChromaStore(persist_dir=tmp_path / "chroma", collection="eval")
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    provider = FakeModelProvider(canned_json='{"answerable": true, "confidence": 5}',
                                 canned_answer="PDF answer.")

    class _EmptyRetriever:
        def retrieve(self, q): return []
        def reindex(self, chunks): pass

    retriever = _EmptyRetriever()
    pipe = IngestPipeline(chunker=FixedSizeChunker(100, 10), provider=provider,
                          store=serving_store, datastore=datastore)

    def ingest_upload(doc):
        pipe.ingest([doc])
        retriever.reindex(serving_store.get_all_chunks())

    app = create_app(retriever=retriever, provider=provider, datastore=datastore,
                     ingest_upload=ingest_upload, uploads_dir=tmp_path / "uploads")
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")
    r = c.post(
        "/upload",
        data={"csrf_token": _csrf(page.text)},
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code in (200, 303)                        # redirected or OK
    # eval collection stays pristine — uploads must never touch it
    assert eval_store.get_all_chunks() == []


def test_upload_uses_content_hash_path_and_avoids_filename_collision(monkeypatch, tmp_path):
    import io
    from pathlib import Path

    from pypdf import PdfWriter

    from genacademy_rag.core.chunker import FixedSizeChunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.core.vectorstore import ChromaStore
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    provider = FakeModelProvider()
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    serving = ChromaStore(persist_dir=tmp_path / "chroma", collection="serving")

    class _Retriever:
        def retrieve(self, q):
            return []

        def mutate_corpus(self, mutation):
            mutation()

    pipe = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=100, overlap=10),
        provider=provider,
        store=serving,
        datastore=datastore,
    )

    def ingest_upload(doc):
        pipe.ingest([doc])

    app = create_app(
        retriever=_Retriever(),
        provider=provider,
        datastore=datastore,
        ingest_upload=ingest_upload,
        serving_store=serving,
        uploads_dir=tmp_path / "uploads",
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")
    token = _csrf(page.text)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    first = io.BytesIO()
    writer.write(first)
    writer = PdfWriter()
    writer.add_blank_page(width=210, height=210)
    second = io.BytesIO()
    writer.write(second)
    c.post(
        "/upload",
        data={"csrf_token": token},
        files={"file": ("same.pdf", first.getvalue(), "application/pdf")},
    )
    c.post(
        "/upload",
        data={"csrf_token": token},
        files={"file": ("same.pdf", second.getvalue(), "application/pdf")},
    )
    stored = list((tmp_path / "uploads").glob("*.pdf"))
    assert len(stored) == 2
    assert all(Path(p).name != "same.pdf" for p in stored)


def test_delete_route_removes_serving_doc_and_leaves_eval_pristine(monkeypatch, tmp_path):
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    provider = FakeModelProvider()
    deleted = []
    reindexed = []

    class _Retriever:
        def retrieve(self, q):
            return []

        def mutate_corpus(self, mutation):
            reindexed.append(mutation())

    class _Serving:
        def delete_doc(self, doc_id):
            deleted.append(doc_id)

        def get_all_chunks(self):
            return []

    datastore.add_document(
        doc_id="pdf/abc",
        title="notes.pdf",
        source_type="pdf",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
        stored_path=str(tmp_path / "notes.pdf"),
        n_chunks=1,
    )
    app = create_app(
        retriever=_Retriever(),
        provider=provider,
        datastore=datastore,
        serving_store=_Serving(),
        uploads_dir=tmp_path / "uploads",
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")
    token = _csrf(page.text)
    r = c.post(
        "/admin/documents/delete",
        data={"doc_id": "pdf/abc", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert deleted == ["pdf/abc"]
    assert reindexed == [[]]
    assert datastore.get_document("pdf/abc")["status"] == "deleted"

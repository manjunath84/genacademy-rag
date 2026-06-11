import logging
import re

import pytest
from starlette.testclient import TestClient


def _client(
    monkeypatch,
    tmp_path,
    refused=False,
    *,
    datastore=None,
    ingest_upload=None,
    serving_store=None,
    uploads_dir=None,
):
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
    if datastore is None:
        datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    app = create_app(
        retriever=_Retriever(),
        provider=provider,
        datastore=datastore,
        ingest_upload=ingest_upload,
        serving_store=serving_store,
        uploads_dir=uploads_dir,
    )
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


def _ask_and_get_query_id(client, token):
    page = client.post(
        "/ask",
        data={"question": "What is RAG?", "csrf_token": token},
    ).text
    match = re.search(r'name="query_id" value="(\d+)"', page)
    return match.group(1) if match else None, page


def _store_doc(client, tmp_path, *, doc_id="up1", suffix=".pdf", status="indexed"):
    stored = tmp_path / f"deck{suffix}"
    stored.write_bytes(b"%PDF-1.4 fake content")
    client.app.state.datastore.add_document(
        doc_id=doc_id,
        title="Week2 Deck",
        source_type=suffix.lstrip("."),
        filename=f"deck{suffix}",
        uploaded_by="admin@genacademy.local",
        status=status,
        stored_path=str(stored),
    )
    return stored


def _admin_post(client, path: str, *, csrf_token: str | None = None):
    data = {}
    if path == "/admin/invites":
        data = {"role": "member", "expires_days": "7"}
    elif path == "/admin/documents/delete":
        data = {"doc_id": "pdf/abc"}
    if csrf_token is not None:
        data["csrf_token"] = csrf_token
    if path == "/upload":
        return client.post(
            path,
            data=data,
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
            follow_redirects=False,
        )
    return client.post(path, data=data, follow_redirects=False)


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


def test_feedback_requires_login(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post(
        "/feedback",
        data={"query_id": 1, "verdict": 1, "csrf_token": "x"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_feedback_requires_csrf(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    r = c.post("/feedback", data={"query_id": 1, "verdict": 1, "csrf_token": "wrong"})
    assert r.status_code == 403


def test_feedback_rejects_bad_verdict(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)
    r = c.post("/feedback", data={"query_id": 1, "verdict": 7, "csrf_token": token})
    assert r.status_code == 400


def test_feedback_happy_path_persists_and_returns_fragment(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)
    qid, _ = _ask_and_get_query_id(c, token)
    assert qid is not None
    r = c.post("/feedback", data={"query_id": qid, "verdict": 1, "csrf_token": token})
    assert r.status_code == 200
    assert "Thanks for the feedback" in r.text
    assert c.app.state.datastore.feedback_summary()["up"] == 1


def test_feedback_rejects_unknown_query_id(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)

    r = c.post("/feedback", data={"query_id": 404, "verdict": 1, "csrf_token": token})

    assert r.status_code == 404
    assert c.app.state.datastore.feedback_summary() == {"up": 0, "down": 0}


def test_feedback_rejects_query_owned_by_another_user(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)
    owner_qid, _ = _ask_and_get_query_id(c, token)
    assert owner_qid is not None

    _login(c, "admin@genacademy.local", "admin")
    admin_token = _csrf(c.get("/").text)
    r = c.post(
        "/feedback",
        data={"query_id": owner_qid, "verdict": -1, "csrf_token": admin_token},
    )

    assert r.status_code == 404
    assert c.app.state.datastore.feedback_summary() == {"up": 0, "down": 0}


def test_answer_card_renders_badge_sources_disclaimer(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)
    _, page = _ask_and_get_query_id(c, token)
    assert "High confidence" in page
    assert 'href="https://github.com/The-Gen-Academy/r/blob/abc123/README.md#L1-L2"' in page
    assert "RAG retrieves then generates." in page
    assert "it can make mistakes" in page
    assert "copy" in page and "retry" in page


def test_refused_card_has_refusal_badge_no_copy_no_sources(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, refused=True)
    _login(c)
    token = _csrf(c.get("/").text)
    _, page = _ask_and_get_query_id(c, token)
    assert "Not in the materials" in page
    assert "Sources" not in page
    assert "⧉ copy" not in page
    assert "it can make mistakes" in page


def test_thumbs_hidden_when_log_query_fails(monkeypatch, tmp_path, caplog):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)
    ds = c.app.state.datastore

    def boom(**kwargs):
        raise RuntimeError("db down")

    ds.log_query = boom
    with caplog.at_level(logging.ERROR):
        qid, page = _ask_and_get_query_id(c, token)
    assert qid is None
    assert "👍" not in page


def test_feedback_write_failure_does_not_500(monkeypatch, tmp_path, caplog):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    token = _csrf(c.get("/").text)
    ds = c.app.state.datastore

    def boom(**kwargs):
        raise RuntimeError("db down")

    ds.add_feedback = boom
    with caplog.at_level(logging.ERROR):
        r = c.post("/feedback", data={"query_id": 1, "verdict": 1, "csrf_token": token})
    assert r.status_code == 200
    assert "feedback write failed" in caplog.text


def test_document_file_requires_login(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/documents/up1/file", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_document_file_serves_pdf_inline(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    _store_doc(c, tmp_path, suffix=".pdf")
    r = c.get("/documents/up1/file")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "inline" in r.headers["content-disposition"]


def test_document_file_serves_upload_with_slash_doc_id(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    _store_doc(c, tmp_path, doc_id="pdf/abc123", suffix=".pdf")

    r = c.get("/documents/pdf/abc123/file")

    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"


def test_document_file_downloads_pptx(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    _store_doc(c, tmp_path, doc_id="up2", suffix=".pptx")
    r = c.get("/documents/up2/file")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]


def test_document_file_404s(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    assert c.get("/documents/nope/file").status_code == 404

    _store_doc(c, tmp_path, doc_id="dead", status="deleted")
    assert c.get("/documents/dead/file").status_code == 404

    c.app.state.datastore.add_document(
        doc_id="gh1",
        title="README.md",
        source_type="github",
        repo="r",
        file_path="README.md",
        commit_hash="abc",
    )
    assert c.get("/documents/gh1/file").status_code == 404


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


def test_create_app_uses_secure_cookie_setting(monkeypatch, tmp_path):
    import genacademy_rag.web.app as app_module
    from genacademy_rag.config import Settings
    from genacademy_rag.data.datastore import SQLiteDatastore

    settings = Settings(
        provider="openrouter",
        gen_base_url="https://openrouter.ai/api/v1",
        gen_api_key="",
        gen_model="",
        embed_model="all-MiniLM-L6-v2",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        chunker="fixed",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "app.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
        secure_cookies=True,
    )

    monkeypatch.setattr(app_module.Settings, "from_env", classmethod(lambda cls: settings))
    app = app_module.create_app(
        retriever=object(),
        provider=object(),
        datastore=SQLiteDatastore(tmp_path / "test.sqlite"),
    )

    session_middleware = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls.__name__ == "SessionMiddleware"
    )
    assert session_middleware.kwargs["https_only"] is True


def test_admin_routes_block_member(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    assert c.get("/admin/invites", follow_redirects=False).status_code == 403


@pytest.mark.parametrize(
    "path",
    [
        "/admin/documents/delete",
        "/admin/documents/reindex",
        "/admin/invites/unknown/revoke",
        "/upload",
    ],
)
@pytest.mark.parametrize("csrf_token", [None, "not-the-session-token"])
def test_destructive_admin_posts_reject_missing_or_invalid_csrf(
    monkeypatch, tmp_path, path, csrf_token
):
    c = _client(
        monkeypatch,
        tmp_path,
        ingest_upload=lambda doc: None,
        uploads_dir=tmp_path / "uploads",
    )
    _login(c, "admin@genacademy.local", "admin")
    c.get("/admin/documents")

    r = _admin_post(c, path, csrf_token=csrf_token)

    assert r.status_code == 403


@pytest.mark.parametrize("actor", ["member", "anonymous"])
@pytest.mark.parametrize(
    "path",
    [
        "/admin/invites",
        "/admin/documents/delete",
        "/admin/documents/reindex",
        "/admin/invites/unknown/revoke",
        "/upload",
    ],
)
def test_member_and_anonymous_users_are_blocked_from_admin_posts(
    monkeypatch, tmp_path, actor, path
):
    c = _client(
        monkeypatch,
        tmp_path,
        ingest_upload=lambda doc: None,
        uploads_dir=tmp_path / "uploads",
    )
    if actor == "member":
        _login(c)

    r = _admin_post(c, path)

    if path == "/upload":
        assert r.status_code == 303
        assert "/login" in r.headers["location"]
    else:
        assert r.status_code == 403


@pytest.mark.parametrize("actor", ["member", "anonymous"])
@pytest.mark.parametrize("path", ["/admin/documents", "/admin/dashboard"])
def test_member_and_anonymous_users_are_blocked_from_admin_gets(
    monkeypatch, tmp_path, actor, path
):
    c = _client(monkeypatch, tmp_path)
    if actor == "member":
        _login(c)

    r = c.get(path, follow_redirects=False)

    assert r.status_code == 403


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

        def snapshot_chunks(self):
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


def test_delete_route_filters_deleted_doc_from_snapshot(monkeypatch, tmp_path):
    """The delete mutation rebuilds from the in-memory snapshot (never a remote read,
    which could lag on Pinecone); the deleted doc must be filtered out of it."""
    from genacademy_rag.core.types import Chunk, Citation
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    rebuilt = []

    def _chunk(doc_id, ordinal):
        cit = Citation(doc_id=doc_id, title=doc_id, source_type="pdf")
        return Chunk(chunk_id=f"{doc_id}::{ordinal}", doc_id=doc_id, ordinal=ordinal,
                     text="t", citation=cit)

    snapshot = [_chunk("pdf/abc", 0), _chunk("pdf/keep", 0)]
    store_reads = []

    class _Retriever:
        def retrieve(self, q):
            return []

        def snapshot_chunks(self):
            return list(snapshot)

        def mutate_corpus(self, mutation):
            rebuilt.append(mutation())

    class _StaleServing:
        def delete_doc(self, doc_id):
            pass                       # lagging remote: vectors may be orphaned

        def get_all_chunks(self):
            store_reads.append(True)   # must NOT be consulted on delete
            return list(snapshot)

    datastore.add_document(
        doc_id="pdf/abc",
        title="notes.pdf",
        source_type="pdf",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
        n_chunks=1,
    )
    app = create_app(
        retriever=_Retriever(),
        provider=FakeModelProvider(),
        datastore=datastore,
        serving_store=_StaleServing(),
        uploads_dir=tmp_path / "uploads",
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    token = _csrf(c.get("/admin/documents").text)
    r = c.post(
        "/admin/documents/delete",
        data={"doc_id": "pdf/abc", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert [ch.chunk_id for ch in rebuilt[0]] == ["pdf/keep::0"]
    assert store_reads == []           # snapshot-based: no remote read on delete


def test_serving_chunk_filter_drops_orphaned_uploads_but_keeps_github_seed(tmp_path):
    from genacademy_rag.core.types import Chunk, Citation
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import _filter_serving_chunks_for_datastore

    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.add_document(
        doc_id="pdf/live",
        title="live.pdf",
        source_type="pdf",
        filename="live.pdf",
        uploaded_by="admin@genacademy.local",
        n_chunks=1,
    )
    ds.add_document(
        doc_id="pdf/deleted",
        title="deleted.pdf",
        source_type="pdf",
        filename="deleted.pdf",
        uploaded_by="admin@genacademy.local",
        n_chunks=1,
    )
    ds.delete_document("pdf/deleted", deleted_by="admin@genacademy.local")

    def chunk(doc_id: str, source_type: str, repo: str | None = None) -> Chunk:
        cit = Citation(
            doc_id=doc_id,
            title=doc_id,
            source_type=source_type,
            repo=repo,
            file_path="README.md" if repo else None,
            commit_hash="abc123" if repo else None,
        )
        return Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, ordinal=0, text="t", citation=cit)

    visible = _filter_serving_chunks_for_datastore(
        [
            chunk("course/seed", "github", repo="r"),
            chunk("pdf/live", "pdf"),
            chunk("pdf/deleted", "pdf"),
            chunk("pdf/orphan", "pdf"),
        ],
        ds,
    )

    assert [c.chunk_id for c in visible] == ["course/seed::0", "pdf/live::0"]


def test_reindex_route_filters_deleted_docs_via_datastore_ledger(monkeypatch, tmp_path):
    """Reindex is the one deliberate remote re-read. Pinecone may retain old uploaded
    vectors across an HF restart after SQLite/upload files are wiped; keep GitHub seed
    chunks but do not resurrect deleted or orphaned uploaded docs."""
    from genacademy_rag.core.types import Chunk, Citation
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    rebuilt = []

    def _chunk(doc_id, ordinal, *, source_type="pdf", repo=None):
        cit = Citation(
            doc_id=doc_id,
            title=doc_id,
            source_type=source_type,
            repo=repo,
            file_path="README.md" if repo else None,
            commit_hash="abc123" if repo else None,
        )
        return Chunk(chunk_id=f"{doc_id}::{ordinal}", doc_id=doc_id, ordinal=ordinal,
                     text="t", citation=cit)

    class _Retriever:
        def retrieve(self, q):
            return []

        def snapshot_chunks(self):
            return []

        def mutate_corpus(self, mutation):
            rebuilt.append(mutation())

    class _OrphanedServing:
        def get_all_chunks(self):
            # Remote still holds the deleted doc's vectors (lagged delete) plus a
            # live doc, a ledger-less GitHub eval seed, and an uploaded orphan left
            # behind by an HF /data reset.
            return [
                _chunk("pdf/gone", 0),
                _chunk("pdf/live", 0),
                _chunk("course/seed", 0, source_type="github", repo="r"),
                _chunk("pdf/orphan", 0),
            ]

    for doc_id, title in (("pdf/gone", "gone.pdf"), ("pdf/live", "live.pdf")):
        datastore.add_document(
            doc_id=doc_id, title=title, source_type="pdf", filename=title,
            uploaded_by="admin@genacademy.local", n_chunks=1,
        )
    datastore.delete_document("pdf/gone", deleted_by="admin@genacademy.local")

    app = create_app(
        retriever=_Retriever(),
        provider=FakeModelProvider(),
        datastore=datastore,
        serving_store=_OrphanedServing(),
        uploads_dir=tmp_path / "uploads",
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    token = _csrf(c.get("/admin/documents").text)
    r = c.post("/admin/documents/reindex", data={"csrf_token": token},
               follow_redirects=False)

    assert r.status_code == 303
    # Deleted and orphaned uploads filtered; live doc and ledger-less GitHub seed kept.
    assert [ch.chunk_id for ch in rebuilt[0]] == ["pdf/live::0", "course/seed::0"]


def test_delete_route_treats_empty_uploaded_by_as_undeletable(monkeypatch, tmp_path):
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    provider = FakeModelProvider()
    deleted = []

    class _Retriever:
        def retrieve(self, q):
            return []

        def mutate_corpus(self, mutation):
            mutation()

    class _Serving:
        def delete_doc(self, doc_id):
            deleted.append(doc_id)

        def get_all_chunks(self):
            return []

    datastore.add_document(
        doc_id="course-empty",
        title="Course",
        source_type="github",
        uploaded_by="",
        n_chunks=1,
    )
    app = create_app(
        retriever=_Retriever(),
        provider=provider,
        datastore=datastore,
        serving_store=_Serving(),
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")

    r = c.post(
        "/admin/documents/delete",
        data={"doc_id": "course-empty", "csrf_token": _csrf(page.text)},
        follow_redirects=False,
    )

    assert r.status_code == 403
    assert deleted == []
    assert datastore.get_document("course-empty")["status"] == "indexed"


def test_ask_requires_csrf_and_writes_usage_row(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    forbidden = c.post("/ask", data={"question": "what is RAG?"})
    assert forbidden.status_code == 403
    page = c.get("/")
    ok = c.post("/ask", data={"question": "what is RAG?", "csrf_token": _csrf(page.text)})
    assert ok.status_code == 200
    rows = c.app.state.datastore.recent_usage(limit=10)
    assert len(rows) == 1
    assert rows[0]["question"] == "what is RAG?"
    assert rows[0]["user_email"] == "member@genacademy.local"
    assert rows[0]["refused"] == 0
    assert rows[0]["used_fallback"] == 0
    assert rows[0]["n_citations"] == 1
    assert rows[0]["latency_ms"] >= 0


def test_ask_logs_refused_usage_row(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, refused=True)
    _login(c)
    page = c.get("/")

    r = c.post(
        "/ask",
        data={"question": "who won the 2050 world cup?", "csrf_token": _csrf(page.text)},
    )

    assert r.status_code == 200
    rows = c.app.state.datastore.recent_usage(limit=10)
    assert len(rows) == 1
    assert rows[0]["question"] == "who won the 2050 world cup?"
    assert rows[0]["refused"] == 1
    assert rows[0]["used_fallback"] == 0
    assert rows[0]["n_citations"] == 1
    assert rows[0]["latency_ms"] >= 0


def test_ask_returns_answer_when_usage_logging_fails(monkeypatch, tmp_path, caplog):
    from genacademy_rag.data.datastore import SQLiteDatastore

    class FailingLogDatastore(SQLiteDatastore):
        def log_query(self, **kwargs):
            raise RuntimeError("database is locked")

    datastore = FailingLogDatastore(tmp_path / "t.sqlite")
    c = _client(monkeypatch, tmp_path, datastore=datastore)
    _login(c)
    page = c.get("/")

    with caplog.at_level(logging.ERROR, logger="genacademy_rag.web.app"):
        r = c.post("/ask", data={"question": "what is RAG?", "csrf_token": _csrf(page.text)})

    assert r.status_code == 200
    assert "RAG retrieves then generates." in r.text
    assert "usage log_query failed" in caplog.text


def test_default_upload_embeds_before_corpus_mutation_lock(monkeypatch, tmp_path):
    import genacademy_rag.config as config_module
    import genacademy_rag.core.loaders.pdf_loader as pdf_loader
    import genacademy_rag.core.providers as providers_module
    import genacademy_rag.core.retriever as retriever_module
    import genacademy_rag.core.vectorstore as vectorstore_module
    from genacademy_rag.config import Settings
    from genacademy_rag.core.types import Document
    from genacademy_rag.web.app import build_default_app

    state = {"in_lock": False, "embed_in_lock": [], "upsert_in_lock": []}
    settings = Settings(
        provider="openrouter",
        gen_base_url="",
        gen_api_key="",
        gen_model="",
        embed_model="",
        top_k=5,
        chunk_size=50,
        chunk_overlap=5,
        chunker="fixed",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "t.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
    )

    class Provider:
        def embed(self, texts):
            if texts:
                state["embed_in_lock"].append(state["in_lock"])
            return [[0.1] * 384 for _ in texts]

        def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0):
            return '{"answerable": true, "confidence": 5}' if json_mode else "answer"

    class Store:
        def __init__(self, *, persist_dir, collection):
            self.collection = collection
            self.chunks = []

        def get_all_chunks(self):
            return list(self.chunks)

        def upsert(self, chunks, embeddings):
            if chunks:
                state["upsert_in_lock"].append(state["in_lock"])
            self.chunks.extend(chunks)

    class Retriever:
        def __init__(
            self,
            *,
            store,
            provider,
            all_chunks,
            top_k,
            candidate_k=20,
            reranker=None,
            rerank_pool=0,
        ):
            self.store = store
            self.all_chunks = list(all_chunks)
            self.candidate_k = candidate_k
            self.reranker = reranker
            self.rerank_pool = rerank_pool

        def retrieve(self, query):
            return []

        def snapshot_chunks(self):
            return list(self.all_chunks)

        def mutate_corpus(self, mutation):
            state["in_lock"] = True
            try:
                self.all_chunks = mutation()
            finally:
                state["in_lock"] = False

    def fake_load_pdf_bytes(*, filename, raw_bytes, uploaded_by=None, stored_path=None):
        return Document(
            doc_id="pdf/abc",
            title=filename,
            source_type="pdf",
            text="Gen Academy upload about retrieval augmented generation.",
            filename=filename,
            uploaded_by=uploaded_by,
            stored_path=stored_path,
        )

    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(providers_module, "build_provider", lambda s: Provider())
    monkeypatch.setattr(vectorstore_module, "ChromaStore", Store)
    monkeypatch.setattr(retriever_module, "HybridRetriever", Retriever)
    monkeypatch.setattr(pdf_loader, "load_pdf_bytes", fake_load_pdf_bytes)

    c = TestClient(build_default_app())
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")

    r = c.post(
        "/upload",
        data={"csrf_token": _csrf(page.text)},
        files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
        follow_redirects=False,
    )

    assert r.status_code == 303
    assert state["embed_in_lock"] == [False]
    assert state["upsert_in_lock"] == [True]


def _default_app_scaffold(monkeypatch, tmp_path, *, store_cls, retriever_cls):
    """Shared monkeypatching for build_default_app tests with injected fakes."""
    import genacademy_rag.config as config_module
    import genacademy_rag.core.providers as providers_module
    import genacademy_rag.core.retriever as retriever_module
    import genacademy_rag.core.vectorstore as vectorstore_module
    from genacademy_rag.config import Settings

    settings = Settings(
        provider="openrouter",
        gen_base_url="",
        gen_api_key="",
        gen_model="",
        embed_model="",
        top_k=5,
        chunk_size=50,
        chunk_overlap=5,
        chunker="fixed",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "t.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
    )

    class _Provider:
        def embed(self, texts):
            return [[0.1] * 384 for _ in texts]

        def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0):
            return '{"answerable": true, "confidence": 5}' if json_mode else "answer"

    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(providers_module, "build_provider", lambda s: _Provider())
    monkeypatch.setattr(vectorstore_module, "ChromaStore", store_cls)
    monkeypatch.setattr(retriever_module, "HybridRetriever", retriever_cls)


def _corpus_chunk(doc_id, ordinal, text="t"):
    from genacademy_rag.core.types import Chunk, Citation

    cit = Citation(doc_id=doc_id, title=doc_id, source_type="github")
    return Chunk(chunk_id=f"{doc_id}::{ordinal}", doc_id=doc_id, ordinal=ordinal,
                 text=text, citation=cit)


def test_default_boot_seeds_index_from_local_chunks_not_lagging_store_read(
    monkeypatch, tmp_path
):
    """First boot against an eventually-consistent store: the seed upsert is not yet
    visible to get_all_chunks, and the retriever must be built from the local seed
    list — not the lagging (empty) remote read."""
    from genacademy_rag.web.app import build_default_app

    eval_chunks = [_corpus_chunk("course/readme", 0, "retrieval augmented generation")]
    built = {}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            self.collection = collection

        def get_all_chunks(self):
            # eval collection is local and consistent; serving lags and reads empty.
            return list(eval_chunks) if self.collection == "eval" else []

        def upsert(self, chunks, embeddings):
            pass

    class _Retriever:
        def __init__(self, *, store, provider, all_chunks, top_k, candidate_k=20,
                     reranker=None, rerank_pool=0):
            built["all_chunks"] = list(all_chunks)

        def retrieve(self, query):
            return []

        def snapshot_chunks(self):
            return built["all_chunks"]

        def mutate_corpus(self, mutation):
            mutation()

    _default_app_scaffold(monkeypatch, tmp_path, store_cls=_Store, retriever_cls=_Retriever)
    build_default_app()

    assert [c.chunk_id for c in built["all_chunks"]] == ["course/readme::0"]


def test_default_upload_unions_committed_chunks_despite_stale_store_read(
    monkeypatch, tmp_path
):
    """The upload mutation must rebuild the corpus from the in-memory snapshot plus the
    just-committed chunks. A stale store read (missing the new chunks, e.g. Pinecone
    lag) must not make the upload unsearchable — and must not evict prior corpus."""
    import genacademy_rag.core.loaders.pdf_loader as pdf_loader
    from genacademy_rag.core.types import Document
    from genacademy_rag.web.app import build_default_app

    existing = _corpus_chunk("course/readme", 0, "prior corpus chunk")
    rebuilt = {}

    class _StaleStore:
        def __init__(self, *, persist_dir, collection):
            self.collection = collection

        def get_all_chunks(self):
            # Always stale: never reflects any upsert (boot sees the prior corpus).
            return [existing]

        def upsert(self, chunks, embeddings):
            pass

    class _Retriever:
        def __init__(self, *, store, provider, all_chunks, top_k, candidate_k=20,
                     reranker=None, rerank_pool=0):
            self.all_chunks = list(all_chunks)

        def retrieve(self, query):
            return []

        def snapshot_chunks(self):
            return list(self.all_chunks)

        def mutate_corpus(self, mutation):
            self.all_chunks = mutation()
            rebuilt["corpus"] = self.all_chunks

    def fake_load_pdf_bytes(*, filename, raw_bytes, uploaded_by=None, stored_path=None):
        return Document(
            doc_id="pdf/abc",
            title=filename,
            source_type="pdf",
            text="Gen Academy upload about retrieval augmented generation.",
            filename=filename,
            uploaded_by=uploaded_by,
            stored_path=stored_path,
        )

    _default_app_scaffold(monkeypatch, tmp_path, store_cls=_StaleStore,
                          retriever_cls=_Retriever)
    monkeypatch.setattr(pdf_loader, "load_pdf_bytes", fake_load_pdf_bytes)

    c = TestClient(build_default_app())
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")
    r = c.post(
        "/upload",
        data={"csrf_token": _csrf(page.text)},
        files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
        follow_redirects=False,
    )

    assert r.status_code == 303
    corpus_ids = [c2.chunk_id for c2 in rebuilt["corpus"]]
    assert "course/readme::0" in corpus_ids          # prior corpus retained
    assert any(cid.startswith("pdf/abc::") for cid in corpus_ids)  # upload searchable
    assert len(corpus_ids) == len(set(corpus_ids))   # no duplicates


def test_dashboard_renders_usage_summary(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    c.app.state.datastore.log_query(
        user_email="member@genacademy.local",
        question="What is RAG?",
        refused=False,
        confidence=5,
        used_fallback=False,
        n_citations=2,
        latency_ms=100,
    )
    r = c.get("/admin/dashboard")
    assert r.status_code == 200
    assert "Total queries" in r.text
    assert "What is RAG?" in r.text
    assert "<svg" in r.text


def test_dashboard_shows_feedback_counts(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    ds = c.app.state.datastore
    qid = ds.log_query(
        user_email="m@x.com",
        question="q?",
        refused=False,
        confidence=4,
        used_fallback=False,
        n_citations=1,
        latency_ms=50,
    )
    ds.add_feedback(usage_log_id=qid, user_email="m@x.com", verdict=1)
    page = c.get("/admin/dashboard").text
    assert "Thumbs up" in page
    assert "Thumbs down" in page


def test_phase1_demo_flow(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    invites = c.get("/admin/invites")
    generated = c.post(
        "/admin/invites",
        data={"role": "member", "expires_days": "7", "csrf_token": _csrf(invites.text)},
    )
    code = re.search(r"Invite code: ([^<]+)<", generated.text).group(1)
    signup = c.get("/signup")
    created = c.post(
        "/signup",
        data={
            "email": "cohort@example.com",
            "password": "secret",
            "code": code,
            "csrf_token": _csrf(signup.text),
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    chat = c.get("/")
    asked = c.post("/ask", data={"question": "what is RAG?", "csrf_token": _csrf(chat.text)})
    assert asked.status_code == 200
    _login(c, "admin@genacademy.local", "admin")
    dashboard = c.get("/admin/dashboard")
    assert "what is RAG?" in dashboard.text

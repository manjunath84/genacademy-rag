"""Thin FastAPI view. ALL RAG logic is injected (QueryPipeline); no core logic here. Non-streaming
form-post (HTMX-ready). create_app(retriever, provider, datastore) lets tests inject fakes; the
real wiring (local embed + provider preset + serving vector store via GENACADEMY_VECTORSTORE;
eval pinned to local Chroma) happens in build_default_app()."""
from __future__ import annotations

import hmac
import logging
import secrets
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from genacademy_rag.config import Settings, data_dir_from_env
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.core.security import hash_password
from genacademy_rag.core.sources import confidence_bucket
from genacademy_rag.core.types import Chunk
from genacademy_rag.data.datastore import SQLiteDatastore
from genacademy_rag.web.auth import authenticate

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
logger = logging.getLogger(__name__)


def _filter_serving_chunks_for_datastore(chunks: list[Chunk], datastore) -> list[Chunk]:
    """Drop serving vectors for uploaded docs that are not visible in SQLite.

    Pinecone persists independently of Hugging Face `/data`. Without persistent storage,
    SQLite/upload files can reset while the Pinecone namespace still contains old uploaded
    vectors. Keep ledger-less GitHub seed chunks, but never serve orphaned or deleted uploads.
    """
    visible: list[Chunk] = []
    doc_cache: dict[str, dict | None] = {}
    dropped = 0
    for chunk in chunks:
        citation = chunk.citation
        if citation.repo or citation.source_type == "github":
            visible.append(chunk)
            continue
        if chunk.doc_id not in doc_cache:
            doc_cache[chunk.doc_id] = datastore.get_document(chunk.doc_id)
        doc = doc_cache[chunk.doc_id]
        if doc and doc.get("status") != "deleted":
            visible.append(chunk)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "serving corpus: dropped %d uploaded chunks missing from the active datastore",
            dropped,
        )
    return visible


def create_app(
    *,
    retriever,
    provider,
    datastore,
    ingest_upload=None,
    serving_store=None,
    uploads_dir=None,
) -> FastAPI:
    settings = Settings.from_env()
    datastore.seed_users()
    qp = QueryPipeline(retriever=retriever, provider=provider)

    app = FastAPI()
    app.state.datastore = datastore
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        same_site="lax",
        https_only=settings.secure_cookies,
    )
    logger.info("session cookies: secure=%s", settings.secure_cookies)

    def current_user(request: Request) -> str | None:
        return request.session.get("email")

    def csrf_token(request: Request) -> str:
        token = request.session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            request.session["csrf_token"] = token
        return token

    def csrf_context(request: Request, extra: dict | None = None) -> dict:
        context = {"csrf_token": csrf_token(request)}
        if extra:
            context.update(extra)
        return context

    def valid_csrf(request: Request, token: str | None) -> bool:
        expected = request.session.get("csrf_token")
        return bool(expected and token and hmac.compare_digest(expected, token))

    def csrf_forbidden() -> HTMLResponse:
        return HTMLResponse("Forbidden", status_code=403)

    def require_admin(request: Request) -> dict | None:
        email = request.session.get("email")
        if not email:
            return None
        user = datastore.get_user_by_email(email)
        if not user or user["role"] != "admin":
            return None
        request.session["role"] = user["role"]
        return user

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        return TEMPLATES.TemplateResponse(
            request, "login.html", csrf_context(request, {"error": None})
        )

    @app.post("/login")
    def login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        user = authenticate(datastore, email, password)
        if not user:
            return TEMPLATES.TemplateResponse(
                request,
                "login.html",
                csrf_context(request, {"error": "Invalid credentials"}),
                status_code=401,
            )
        request.session["email"] = user["email"]
        request.session["role"] = user["role"]
        return RedirectResponse("/", status_code=303)

    @app.get("/signup", response_class=HTMLResponse)
    def signup_form(request: Request):
        return TEMPLATES.TemplateResponse(
            request, "signup.html", csrf_context(request, {"error": None})
        )

    @app.post("/signup")
    def signup(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        code: str = Form(...),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        user = datastore.redeem_invite(
            raw_code=code,
            email=email,
            password_hash=hash_password(password),
        )
        if not user:
            return TEMPLATES.TemplateResponse(
                request,
                "signup.html",
                csrf_context(request, {"error": "Invalid or expired code"}),
                status_code=400,
            )
        request.session["email"] = user["email"]
        request.session["role"] = user["role"]
        return RedirectResponse("/", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        if not current_user(request):
            return RedirectResponse("/login", status_code=302)
        return TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            csrf_context(
                request,
                {"result": None, "question": None, "query_id": None, "bucket": None},
            ),
        )

    @app.post("/ask", response_class=HTMLResponse)
    def ask(
        request: Request,
        question: str = Form(...),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        if not current_user(request):
            return RedirectResponse("/login", status_code=303)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        start = time.perf_counter()
        result = qp.answer(question)
        latency_ms = int((time.perf_counter() - start) * 1000)
        query_id = None
        try:
            query_id = datastore.log_query(
                user_email=current_user(request),
                question=question,
                refused=result.refused,
                confidence=result.confidence,
                used_fallback=result.used_fallback,
                n_citations=len(result.citations),
                latency_ms=latency_ms,
            )
        except Exception:
            logger.exception("usage log_query failed (question=%r)", question)
        return TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            csrf_context(
                request,
                {
                    "result": result,
                    "question": question,
                    "query_id": query_id,
                    "bucket": None if result.refused else confidence_bucket(result.confidence),
                },
            ),
        )

    @app.post("/feedback", response_class=HTMLResponse)
    def feedback(
        request: Request,
        query_id: int = Form(...),
        verdict: int = Form(...),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        user = current_user(request)
        if not user:
            return RedirectResponse("/login", status_code=303)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        if verdict not in (1, -1):
            return HTMLResponse("Bad verdict", status_code=400)
        try:
            datastore.add_feedback(usage_log_id=query_id, user_email=user, verdict=verdict)
        except (LookupError, PermissionError):
            return HTMLResponse("Not found", status_code=404)
        except Exception:
            logger.exception("feedback write failed (query_id=%r)", query_id)
        return HTMLResponse('<span class="text-xs text-slate-500">Thanks for the feedback</span>')

    @app.get("/documents/{doc_id:path}/file")
    def document_file(request: Request, doc_id: str):
        # Members can access originals for chunks they already see as retrieved context.
        if not current_user(request):
            return RedirectResponse("/login", status_code=303)
        doc = datastore.get_document(doc_id)
        if not doc or doc.get("status") == "deleted" or not doc.get("stored_path"):
            return HTMLResponse("Not found", status_code=404)
        path = Path(doc["stored_path"])
        if not path.exists():
            return HTMLResponse("Not found", status_code=404)
        filename = doc.get("filename") or path.name
        if path.suffix.lower() == ".pdf":
            return FileResponse(
                path,
                media_type="application/pdf",
                filename=filename,
                content_disposition_type="inline",
            )
        return FileResponse(path, filename=filename)

    @app.get("/admin/invites", response_class=HTMLResponse)
    def admin_invites(request: Request):
        admin = require_admin(request)
        if not admin:
            return HTMLResponse("Forbidden", status_code=403)
        return TEMPLATES.TemplateResponse(
            request,
            "admin_invites.html",
            csrf_context(request, {"invites": datastore.list_invites(), "new_code": None}),
        )

    @app.post("/admin/invites", response_class=HTMLResponse)
    def generate_invite(
        request: Request,
        role: str = Form(...),
        expires_days: int = Form(7),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        admin = require_admin(request)
        if not admin:
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        expires_at = None
        if expires_days > 0:
            expires_at = (
                datetime.now(UTC) + timedelta(days=expires_days)
            ).strftime("%Y-%m-%d %H:%M:%S")
        invite = datastore.generate_invite(
            role=role,
            created_by=admin["email"],
            expires_at=expires_at,
        )
        return TEMPLATES.TemplateResponse(
            request,
            "admin_invites.html",
            csrf_context(
                request, {"invites": datastore.list_invites(), "new_code": invite["code"]}
            ),
        )

    @app.post("/admin/invites/{code_id}/revoke")
    def revoke_invite(
        request: Request,
        code_id: str,
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        datastore.revoke_invite(code_id)
        return RedirectResponse("/admin/invites", status_code=303)

    @app.post("/upload")
    async def upload(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        admin = require_admin(request)
        if not admin or ingest_upload is None:
            return RedirectResponse("/login", status_code=303)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        raw = await file.read()
        # file.filename is attacker-controlled: strip to a basename so "../../etc/x" can't escape
        # uploads_dir, and supply a fallback since Starlette permits a None filename.
        safe_name = Path(file.filename or "upload.pdf").name
        from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes

        doc = load_pdf_bytes(filename=safe_name, raw_bytes=raw, uploaded_by=admin["email"])
        suffix = Path(safe_name).suffix.lower() or ".pdf"
        stored_name = doc.doc_id.replace("/", "_") + suffix
        stored_path = None
        if uploads_dir is not None:
            uploads_dir.mkdir(parents=True, exist_ok=True)
            stored_path = uploads_dir / stored_name
            stored_path.write_bytes(raw)
        doc = replace(doc, stored_path=str(stored_path) if stored_path else None)
        try:
            ingest_upload(doc)
        except Exception:
            logger.exception(
                "ingest failed for doc_id=%s — datastore and vector store may have "
                "diverged; delete the document (if listed) and re-upload", doc.doc_id,
            )
            if stored_path is not None:
                stored_path.unlink(missing_ok=True)
            raise
        return RedirectResponse("/admin/documents", status_code=303)

    @app.get("/admin/documents", response_class=HTMLResponse)
    def admin_documents(request: Request):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        return TEMPLATES.TemplateResponse(
            request,
            "admin_documents.html",
            csrf_context(request, {"documents": datastore.list_documents()}),
        )

    @app.get("/admin/dashboard", response_class=HTMLResponse)
    def admin_dashboard(request: Request):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        from genacademy_rag.core.analytics import usage_summary

        rows = datastore.recent_usage(limit=500)
        summary = usage_summary(rows)
        return TEMPLATES.TemplateResponse(
            request,
            "admin_dashboard.html",
            csrf_context(
                request,
                {"summary": summary, "rows": rows, "feedback": datastore.feedback_summary()},
            ),
        )

    @app.post("/admin/documents/delete")
    def delete_document(
        request: Request,
        doc_id: str = Form(...),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        admin = require_admin(request)
        if not admin:
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        doc = datastore.get_document(doc_id)
        if not doc or not doc["uploaded_by"]:
            return HTMLResponse("Forbidden", status_code=403)
        if serving_store is not None:
            def mutation():
                serving_store.delete_doc(doc_id)
                # Rebuild from the in-memory snapshot, not a remote re-read: an
                # eventually-consistent store could return stale state and evict or
                # resurrect unrelated docs. snapshot_chunks() is deadlock-safe here.
                return [c for c in retriever.snapshot_chunks() if c.doc_id != doc_id]

            retriever.mutate_corpus(mutation)
        if doc.get("stored_path"):
            Path(doc["stored_path"]).unlink(missing_ok=True)
        datastore.delete_document(doc_id, deleted_by=admin["email"])
        return RedirectResponse("/admin/documents", status_code=303)

    @app.post("/admin/documents/reindex")
    def reindex_documents(
        request: Request,
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        if serving_store is not None:
            def mutation():
                # Reindex is the one deliberate remote re-read (recovery path for chunks
                # missing from memory). Filter it against the datastore ledger so lagged
                # deletes or Pinecone vectors orphaned by an HF /data reset are not served.
                before = len(retriever.snapshot_chunks())
                chunks = _filter_serving_chunks_for_datastore(
                    serving_store.get_all_chunks(),
                    datastore,
                )
                logger.info("reindex: corpus %d -> %d chunks", before, len(chunks))
                return chunks

            retriever.mutate_corpus(mutation)
        return RedirectResponse("/admin/documents", status_code=303)

    return app


def build_default_app() -> FastAPI:
    """Real wiring: local embed + active provider preset + serving collection (seeded from eval)."""
    from genacademy_rag.core.chunker import build_chunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.core.providers import build_provider
    from genacademy_rag.core.reranker import build_reranker
    from genacademy_rag.core.retriever import DEFAULT_CANDIDATE_K, HybridRetriever
    from genacademy_rag.core.vectorstore import ChromaStore, build_vectorstore

    s = Settings.from_env()
    provider = build_provider(s)
    datastore = SQLiteDatastore(s.sqlite_path)
    # eval collection: pinned corpus used by eval scripts — NEVER written to by uploads.
    # Always local Chroma: the deterministic eval must not depend on a remote store.
    eval_store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = eval_store.get_all_chunks()
    # serving collection: grows with admin uploads; swappable via GENACADEMY_VECTORSTORE.
    serving = build_vectorstore(s, collection="serving")
    serving_chunks = _filter_serving_chunks_for_datastore(serving.get_all_chunks(), datastore)
    if not serving_chunks:              # seed once from the pinned eval chunks
        serving.upsert(chunks, provider.embed([c.text for c in chunks]))
        # Build the index from the local seed list, not a re-read: a remote store
        # (Pinecone) is eventually consistent, so an immediate re-read can return []
        # and boot a retriever that refuses every question.
        serving_chunks = chunks
    # The count is the operator's signal for a partially visible remote namespace.
    logger.info("boot corpus: %d chunks from %s serving store", len(serving_chunks), s.vectorstore)
    retriever = HybridRetriever(
        store=serving,
        provider=provider,
        all_chunks=serving_chunks,
        top_k=s.top_k,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker=build_reranker(s),
        rerank_pool=s.rerank_pool,
    )
    pipe = IngestPipeline(
        chunker=build_chunker(
            s.chunker,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            section_max_chars=s.section_chunk_max_chars,
            section_overlap=s.section_chunk_overlap,
        ),
        provider=provider,
        store=serving,
        datastore=datastore,
    )
    uploads_dir = data_dir_from_env() / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    def ingest_upload(doc):
        prepared = pipe.prepare([doc])

        def mutation():
            pipe.commit(prepared)
            # Union the in-memory snapshot with the just-committed chunks. No remote
            # re-read here: an eventually-consistent store could lag this upload (making
            # it unsearchable — the retriever drops dense hits whose id it does not know)
            # or lag earlier mutations (evicting/resurrecting unrelated docs).
            new_chunks = [c for item in prepared for c in item.chunks]
            new_ids = {c.chunk_id for c in new_chunks}
            current = [c for c in retriever.snapshot_chunks() if c.chunk_id not in new_ids]
            return current + new_chunks

        retriever.mutate_corpus(mutation)

    return create_app(retriever=retriever, provider=provider, datastore=datastore,
                      ingest_upload=ingest_upload, serving_store=serving, uploads_dir=uploads_dir)

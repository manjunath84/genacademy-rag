"""Thin FastAPI view. ALL RAG logic is injected (QueryPipeline); no core logic here. Non-streaming
form-post (HTMX-ready). create_app(retriever, provider, datastore) lets tests inject fakes; the
real wiring (local embed + provider preset + ingested Chroma) happens in build_default_app()."""
from __future__ import annotations

import hmac
import secrets
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from genacademy_rag.config import Settings
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.core.security import hash_password
from genacademy_rag.data.datastore import SQLiteDatastore
from genacademy_rag.web.auth import authenticate

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


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
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret, same_site="lax")

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
            request, "chat.html", csrf_context(request, {"result": None, "question": None})
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
        result = qp.answer(question)
        return TEMPLATES.TemplateResponse(
            request, "chat.html", csrf_context(request, {"result": result, "question": question})
        )

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
        if not doc or doc["uploaded_by"] is None:
            return HTMLResponse("Forbidden", status_code=403)
        if serving_store is not None:
            def mutation():
                serving_store.delete_doc(doc_id)
                return serving_store.get_all_chunks()

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
            retriever.mutate_corpus(lambda: serving_store.get_all_chunks())
        return RedirectResponse("/admin/documents", status_code=303)

    return app


def build_default_app() -> FastAPI:
    """Real wiring: local embed + active provider preset + serving collection (seeded from eval)."""
    from genacademy_rag.config import DATA_DIR
    from genacademy_rag.core.chunker import FixedSizeChunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.core.providers import build_provider
    from genacademy_rag.core.retriever import HybridRetriever
    from genacademy_rag.core.vectorstore import ChromaStore

    s = Settings.from_env()
    provider = build_provider(s)
    # eval collection: pinned corpus used by eval scripts — NEVER written to by uploads
    eval_store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = eval_store.get_all_chunks()
    # serving collection: grows with admin uploads
    serving = ChromaStore(persist_dir=s.chroma_dir, collection="serving")
    if not serving.get_all_chunks():    # seed once from the pinned eval chunks
        serving.upsert(chunks, provider.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store=serving, provider=provider,
                                all_chunks=serving.get_all_chunks(), top_k=s.top_k)
    datastore = SQLiteDatastore(s.sqlite_path)
    pipe = IngestPipeline(chunker=FixedSizeChunker(s.chunk_size, s.chunk_overlap),
                          provider=provider, store=serving, datastore=datastore)
    uploads_dir = DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    def ingest_upload(doc):
        def mutation():
            pipe.ingest([doc])
            return serving.get_all_chunks()

        retriever.mutate_corpus(mutation)

    return create_app(retriever=retriever, provider=provider, datastore=datastore,
                      ingest_upload=ingest_upload, serving_store=serving, uploads_dir=uploads_dir)

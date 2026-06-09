"""Thin FastAPI view. ALL RAG logic is injected (QueryPipeline); no core logic here. Non-streaming
form-post (HTMX-ready). create_app(retriever, provider, datastore) lets tests inject fakes; the
real wiring (local embed + provider preset + ingested Chroma) happens in build_default_app()."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from genacademy_rag.config import Settings
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.data.datastore import SQLiteDatastore
from genacademy_rag.web.auth import authenticate

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app(*, retriever, provider, datastore, ingest_upload=None, uploads_dir=None) -> FastAPI:
    settings = Settings.from_env()
    datastore.seed_users()
    qp = QueryPipeline(retriever=retriever, provider=provider)

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

    def current_user(request: Request) -> str | None:
        return request.session.get("email")

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        return TEMPLATES.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    def login(request: Request, email: str = Form(...), password: str = Form(...)):
        user = authenticate(datastore, email, password)
        if not user:
            return TEMPLATES.TemplateResponse(
                request,
                "login.html",
                {"error": "Invalid credentials"},
                status_code=401,
            )
        request.session["email"] = user["email"]
        request.session["role"] = user["role"]
        return RedirectResponse("/", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        if not current_user(request):
            return RedirectResponse("/login", status_code=302)
        return TEMPLATES.TemplateResponse(request, "chat.html", {"result": None, "question": None})

    @app.post("/ask", response_class=HTMLResponse)
    def ask(request: Request, question: str = Form(...)):
        if not current_user(request):
            return RedirectResponse("/login", status_code=303)
        result = qp.answer(question)
        return TEMPLATES.TemplateResponse(
            request, "chat.html", {"result": result, "question": question}
        )

    @app.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...)):  # noqa: B008
        if request.session.get("role") != "admin" or ingest_upload is None:
            return RedirectResponse("/login", status_code=303)
        raw = await file.read()
        if uploads_dir is not None:
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / file.filename).write_bytes(raw)
        from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes
        doc = load_pdf_bytes(filename=file.filename, raw_bytes=raw,
                             uploaded_by=request.session.get("email"))
        ingest_upload(doc)
        return RedirectResponse("/", status_code=303)

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
        pipe.ingest([doc])
        retriever.reindex(serving.get_all_chunks())

    return create_app(retriever=retriever, provider=provider, datastore=datastore,
                      ingest_upload=ingest_upload, uploads_dir=uploads_dir)

"""Thin FastAPI view. ALL RAG logic is injected (QueryPipeline); no core logic here. Non-streaming
form-post (HTMX-ready). create_app(retriever, provider, datastore) lets tests inject fakes; the
real wiring (local embed + provider preset + ingested Chroma) happens in build_default_app()."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from genacademy_rag.config import Settings
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.data.datastore import SQLiteDatastore
from genacademy_rag.web.auth import authenticate

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app(*, retriever, provider, datastore) -> FastAPI:
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

    return app


def build_default_app() -> FastAPI:
    """Real wiring: local embed + active provider preset + the ingested eval Chroma collection."""
    from genacademy_rag.core.providers import build_provider
    from genacademy_rag.core.retriever import HybridRetriever
    from genacademy_rag.core.vectorstore import ChromaStore

    s = Settings.from_env()
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()
    retriever = HybridRetriever(store=store, provider=provider, all_chunks=chunks, top_k=s.top_k)
    datastore = SQLiteDatastore(s.sqlite_path)
    return create_app(retriever=retriever, provider=provider, datastore=datastore)

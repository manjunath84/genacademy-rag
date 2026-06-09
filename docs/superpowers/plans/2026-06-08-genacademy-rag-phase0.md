# GenAcademy RAG — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the graded spine — ingest the commit-pinned GitHub eval corpus → chunk with citation metadata → hybrid retrieve (dense + BM25 + RRF) → LangGraph `grade → {answer + citations | refuse}` → non-streaming chat UI → a runnable 15-question evaluation report — such that a cohort member asks a course question and gets a cited answer or an honest refusal, and the retrieval eval is green by end of Day 2.

**Architecture:** Pure core / thin view. All RAG + data logic lives in `src/genacademy_rag/core` and `…/data` with **no FastAPI/HTMX imports**; the only HTTP/template layer is `…/web`. Every pluggable seam (`ModelProvider`, `VectorStore`, `Retriever`, `Chunker`, `Datastore`, `Loader`) is an interface with **one** Phase-0 implementation; the second implementation is a Phase-2 demo. The external generation API is an **injected dependency** (`ModelProvider`), so the whole graph/pipeline/eval is unit-tested offline against a `FakeModelProvider`; only 1–2 `@pytest.mark.integration` tests hit a live endpoint and are skipped without a key.

**Tech Stack:** Python 3.12 · `uv` (committed lockfile) · `ruff` · `pytest`. LangGraph (one graph: refusal branch) + `langchain-core` primitives. Generation via the OpenAI-compatible SDK pointed at a provider preset (dev: OpenRouter open Llama; mandatory final call: Nebius open model — one config line). Embeddings: local `sentence-transformers` `all-MiniLM-L6-v2` (384-dim). Dense store: raw `chromadb` (precomputed embeddings). Sparse: `rank-bm25`. Fusion: hand-rolled RRF. Relational: `sqlite3` (stdlib). Loaders: `pypdf`, `nbformat`. View: FastAPI + Starlette `SessionMiddleware` + Jinja2 + HTMX (non-streaming form-post).

---

## Scope of this plan

This plan covers **Phase 0 only — the complete gradeable deliverable**. By the spec's own framing (`specs/roadmap.md`, `docs/design.md` §5) Phase 0 alone is a complete, gradeable submission: working cited bot + refusal path + the 15-question eval report.

**Phases 1 and 2 are deliberately not task-broken-down here.** They are SHOULD/stretch and depend on Phase 0 *outcomes* (e.g. Phase 2 cross-encoder rerank is reported as a before/after **eval delta** — its tasks cannot be written before Phase 0 produces baseline eval numbers). Each gets its own plan once Phase 0 is green. The seams they plug into are noted in **§ "Forward pointer (Phases 1–2)"** at the end. Do not build them ahead of a finished Phase 0 (`AGENTS.md` §5).

## Source of truth & guardrails (read before coding)

This plan is written from `docs/design.md` + `docs/spike-findings.md` (the authoritative, reviewed spec). **`docs/architecture-decisions.md` is reasoning-only; do not implement from it** — it has drifted and caused regressions. Where they disagree, design.md/spike-findings.md win.

Non-negotiables threaded through every task (`AGENTS.md` §3, `specs/tech-stack.md`):
- **Pure core / thin view** — a `from fastapi import …` inside `core/` is a review reject.
- **Citations captured at ingest, never reconstructed** — GitHub chunks carry `{repo, file_path, line_start/end, commit_hash}`; file chunks carry `{doc_id, title, page/section, char_span}`. The `commit_hash` is one unbroken chain: fetch → chunk metadata → retrieved chunk → eval scorer.
- **Refusal path is load-bearing** — low confidence ⇒ refuse, never answer from priors. It is graded.
- **Pluggability = interface + config preset, not scattered `if provider == …`.**
- **Eval corpus is commit-pinned; `Mastering-Agentic-AI-Week2` is NEVER fetched or read** (the sample solution — reading it is disqualifying). The ingest allowlist hardcodes only the two permitted repos+SHAs, and a test asserts Week-2 is absent.
- **Pin exact LangChain/LangGraph versions** in `pyproject.toml`. **Secrets via env only.**

## Spike facts baked in (from `docs/spike-findings.md`, run 2026-06-08)

- Eval corpus pinned SHAs (gold set anchors here):
  - `awesome-agentic-ai-resources` @ `5dfb8691180dc4956107e86839998ba3a2ebd94f` → `README.md` (28 KB, 114 table rows).
  - `Mastering-Agentic-AI-Week1` @ `3aa31dfede8c76422be91f2ecdbc59eddc690b1d` → `Langchain Basics/Langchain_Fundamentals.ipynb` (26 KB, 25 cells), `Langchain Basics/README.md`, `Langchain Basics/langchain_prompts.py`.
  - Both public under `The-Gen-Academy` org. Eval corpus is tiny (~54 KB total) → only **two substantive docs** (README + Week-1 notebook), so genuine multi-document gold questions must span those two.
- Embedding: `all-MiniLM-L6-v2`, **dim = 384**, cold load ~11.6 s (once at startup), single-query embed ~12 ms, batch ingest ~377 chunks/s. **Note:** this model's `max_seq_length` is 256 tokens — chunk size is set to respect it (Task 4).
- Generation JSON-mode **works on an open Llama model** (`meta-llama/llama-3.1-70b-instruct` via OpenRouter): single call ~3.75 s, warm ~0.48 s, 10/10 no throttling. → grader/judge use the **JSON-mode primary path**, not the cosine fallback. OpenAI `gpt-4o-mini` also validated (second preset).
- End-to-end: embed 12 ms + generate ~4 s ≪ 8 s ceiling.
- Guidebook PDF (`Mastering-Agentic-AI-Getting-Started-Guidebook.pdf`): 20.2 MB but 15 pages; `pypdf` extracts cleanly (printable ratio 0.994, 0 empty pages) — **no OCR**. Production corpus only, not eval-gating.
- **Path fix:** production files live at `../../CuratedRAGMaterials/` from `genacademy-rag/` (i.e. `GenAcademy/CuratedRAGMaterials/`), **not** `../CuratedRAGMaterials/` as some design prose says. Production loaders (Task 17) use the corrected path.
- Provider presets behind one OpenAI-compatible `ModelProvider.generate()` seam (base_url + key + model): **Nebius** (mandatory final call; open model, credit lands today/Mon), **OpenRouter** (dev stand-in, validated), **OpenAI** `gpt-4o-mini` (second preset = the graded "model swap" demo), local Gemma (offline fallback). The second working provider **is** the swap demo.

## Day mapping (the hard timing rule)

`AGENTS.md` §2.5 / `roadmap.md`: **eval runnable + scores table by end of Day 2; UI polish only if eval green.**
- **Day 1:** Tasks 0–6 (scaffold → ingest → retrieve) **in parallel with** Task 12 gold-set annotation (~6 h reading, the #1 risk — start immediately).
- **Day 2:** Tasks 7–11, 13 → **retrieval eval green** (the protected artifact, `scripts/eval_retrieval.py`). Task 14 faithfulness if time.
- **Day 3 (only if eval green):** Task 15 — finalize the full eval report (refusal correctness + faithfulness + the hand-filled failure-analysis table). **Report done before any UI.**
- **Day 4:** Task 16 (chat UI), then Task 17 (SHOULD: PDF + upload) if time. *(Separate days, not a "Day 3–4" bundle, so the eval-before-UI rule is structural — never a same-day scope race.)*

---

## File structure

```
genacademy-rag/
├── pyproject.toml                         # deps (exact-pinned), ruff + pytest config
├── uv.lock                                # committed
├── .env.example                           # provider presets, paths (mirror spike/.env.example)
├── scripts/
│   └── ingest_eval_corpus.py              # commit-pinned eval ingest entry point (Task 10)
├── src/genacademy_rag/
│   ├── __init__.py
│   ├── config.py                          # Settings from env: provider preset, paths, k, chunk size (Task 0)
│   ├── core/                              # PURE — no fastapi/template imports
│   │   ├── __init__.py
│   │   ├── types.py                       # Document, Chunk, Citation, RetrievedChunk, GraphState (Task 1)
│   │   ├── providers.py                   # ModelProvider protocol + STEmbedder + OpenAICompatProvider (Task 2)
│   │   ├── chunker.py                     # Chunker protocol + FixedSizeChunker (Task 3)
│   │   ├── loaders/
│   │   │   ├── __init__.py                # Loader protocol + EVAL_CORPUS allowlist (Task 4)
│   │   │   ├── github_fetcher.py          # fetch raw blobs at pinned SHA (Task 4)
│   │   │   ├── markdown_loader.py         # (Task 4)
│   │   │   ├── jupyter_loader.py          # (Task 4)
│   │   │   └── pdf_loader.py              # production, SHOULD (Task 17)
│   │   ├── vectorstore.py                 # VectorStore protocol + ChromaStore (Task 5)
│   │   ├── retriever.py                   # Retriever protocol + HybridRetriever + rrf_fuse (Task 6)
│   │   ├── grader.py                      # grade_answerability: JSON-mode + cosine fallback (Task 7)
│   │   ├── graph.py                       # build_graph: retrieve→grade→{answer|refuse} (Task 8)
│   │   └── pipeline.py                    # IngestPipeline + QueryPipeline orchestration (Tasks 10–11)
│   ├── data/
│   │   ├── __init__.py
│   │   └── datastore.py                   # Datastore protocol + SQLiteDatastore (Task 9)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── gold_schema.py                 # GoldQuestion dataclass + loader/validator (Task 12)
│   │   ├── gold/gold_set.yaml             # 15 annotated questions (Task 12)
│   │   ├── retrieval_eval.py              # recall@k, precision@k, MRR (Task 13) — PROTECTED
│   │   ├── faithfulness_eval.py           # LLM-judge + citation-grounding fallback (Task 14) — CUTTABLE
│   │   └── report.py                      # scores table + failure table → markdown (Task 15)
│   └── web/                               # THIN view — only HTTP/templates
│       ├── __init__.py
│       ├── app.py                         # FastAPI app + routes (Task 16)
│       ├── auth.py                        # session login, 2 seeded users (Task 16)
│       └── templates/                     # login.html, chat.html (Task 16)
└── tests/
    ├── conftest.py                        # FakeModelProvider, tmp fixtures (Task 1)
    ├── core/  …                           # one test module per core unit
    ├── eval/  …
    └── integration/test_live_provider.py  # @pytest.mark.integration (Task 2)
```

---

## Task 0: Project scaffold, dependencies, config

**Files:**
- Create: `pyproject.toml`, `.env.example`, `src/genacademy_rag/__init__.py`, `src/genacademy_rag/config.py`
- Create: `tests/__init__.py`, `tests/test_config.py`

- [ ] **Step 1: Initialise the uv project and add exact-pinned deps**

Run (from repo root):
```bash
uv init --package --name genacademy-rag --python 3.12 .
uv add langgraph langchain-core "chromadb" "rank-bm25" "sentence-transformers" \
       "openai" "pypdf" "nbformat" "fastapi" "uvicorn[standard]" "jinja2" \
       "python-multipart" "itsdangerous" "pyyaml" "requests"
uv add --dev pytest ruff
```
Then **convert the resolved versions to exact pins**: open `pyproject.toml`, and for **every** dependency replace the `>=X` specifier `uv` wrote with `==<resolved version>`. Read the resolved versions from `uv.lock` (or `uv pip list`). This satisfies the hard rule "pin **exact** LangChain/LangGraph versions" (`specs/tech-stack.md` §6) — and we pin everything for a reproducible deploy. Expect `langgraph` and `langchain-core` in the 1.0.x line (current as of 2026-06).

- [ ] **Step 2: Add ruff + pytest config to `pyproject.toml`**

Append:
```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: hits a live external API; skipped without a key"]
addopts = "-m 'not integration'"
```
This makes `pytest` skip integration tests by default (run them explicitly with `pytest -m integration`).

- [ ] **Step 3: Write `.env.example`** (mirror `spike/.env.example`; the app reads the same vars)

```bash
# Copy to .env (gitignored). Generation providers are OpenAI-compatible (base_url + key + model)
# behind ModelProvider.generate(). Pick the active one with GENACADEMY_PROVIDER.
GENACADEMY_PROVIDER=openrouter        # openrouter | openai | nebius | gemma  (dev default = openrouter)

NEBIUS_API_KEY=
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1
NEBIUS_MODEL=                          # set when credit lands (the MANDATORY final call)

OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct

OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

GEMMA_BASE_URL=http://127.0.0.1:8085/v1
GEMMA_MODEL=gemma-3-12b-it

GENACADEMY_SESSION_SECRET=dev-only-change-me
```

- [ ] **Step 4: Write the failing test** `tests/test_config.py`

```python
import os

from genacademy_rag.config import Settings, PROVIDER_PRESETS


def test_provider_preset_resolves_base_url_key_and_model(monkeypatch):
    monkeypatch.setenv("GENACADEMY_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
    s = Settings.from_env()
    assert s.provider == "openrouter"
    assert s.gen_base_url == "https://openrouter.ai/api/v1"
    assert s.gen_api_key == "sk-test"
    assert s.gen_model == "meta-llama/llama-3.1-70b-instruct"


def test_known_presets_present():
    assert {"nebius", "openrouter", "openai", "gemma"} <= set(PROVIDER_PRESETS)
```

- [ ] **Step 5: Run it, verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.config`.

- [ ] **Step 6: Implement `src/genacademy_rag/config.py`**

```python
"""Settings from env. Provider presets keep generation pluggable (base_url + key + model)
behind one ModelProvider.generate() seam — no `if provider == ...` in business logic."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# preset name -> (base_url env var, default base_url, key env var, model env var)
PROVIDER_PRESETS: dict[str, tuple[str, str, str, str]] = {
    "nebius": ("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1", "NEBIUS_API_KEY", "NEBIUS_MODEL"),
    "openrouter": ("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"),
    "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_MODEL"),
    "gemma": ("GEMMA_BASE_URL", "http://127.0.0.1:8085/v1", "GEMMA_API_KEY", "GEMMA_MODEL"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
# Spike path fix: production files live one level above Week2-RAG_ContextEngineering.
CURATED_MATERIALS_DIR = REPO_ROOT.parent.parent / "CuratedRAGMaterials"


@dataclass(frozen=True)
class Settings:
    provider: str
    gen_base_url: str
    gen_api_key: str
    gen_model: str
    embed_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    chroma_dir: Path
    sqlite_path: Path
    session_secret: str

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.environ.get("GENACADEMY_PROVIDER", "openrouter")
        if provider not in PROVIDER_PRESETS:
            raise ValueError(f"unknown GENACADEMY_PROVIDER={provider!r}; one of {list(PROVIDER_PRESETS)}")
        base_var, base_default, key_var, model_var = PROVIDER_PRESETS[provider]
        return cls(
            provider=provider,
            gen_base_url=os.environ.get(base_var, base_default),
            gen_api_key=os.environ.get(key_var, ""),
            gen_model=os.environ.get(model_var, ""),
            embed_model=os.environ.get("GENACADEMY_EMBED_MODEL", "all-MiniLM-L6-v2"),
            top_k=int(os.environ.get("GENACADEMY_TOP_K", "5")),
            chunk_size=int(os.environ.get("GENACADEMY_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("GENACADEMY_CHUNK_OVERLAP", "150")),
            chroma_dir=Path(os.environ.get("GENACADEMY_CHROMA_DIR", str(DATA_DIR / "chroma"))),
            sqlite_path=Path(os.environ.get("GENACADEMY_SQLITE", str(DATA_DIR / "genacademy.sqlite"))),
            session_secret=os.environ.get("GENACADEMY_SESSION_SECRET", "dev-only-change-me"),
        )
```

- [ ] **Step 7: Run tests, verify pass + lint clean**

Run: `uv run pytest tests/test_config.py -v && uv run ruff check src tests`
Expected: 2 passed; ruff reports no errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .env.example src/genacademy_rag tests/test_config.py tests/__init__.py
git commit -m "chore: scaffold package, pin exact deps, env-driven provider presets"
```

---

## Task 1: Core types + the FakeModelProvider test seam

**Files:**
- Create: `src/genacademy_rag/core/__init__.py`, `src/genacademy_rag/core/types.py`
- Create: `tests/conftest.py`, `tests/core/__init__.py`, `tests/core/test_types.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_types.py`

```python
from genacademy_rag.core.types import Citation, Chunk, Document, RetrievedChunk


def test_github_citation_round_trips_through_chunk():
    cit = Citation(
        doc_id="d1", title="README.md", source_type="github",
        repo="awesome-agentic-ai-resources", file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        line_start=10, line_end=18,
    )
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text="hello", citation=cit)
    assert chunk.citation.commit_hash.startswith("5dfb869")
    assert chunk.citation.line_end == 18


def test_citation_to_flat_metadata_omits_none():
    cit = Citation(doc_id="d1", title="g.pdf", source_type="pdf",
                   page_or_section="p3", char_start=0, char_end=99)
    flat = cit.to_metadata()
    assert flat["source_type"] == "pdf"
    assert flat["page_or_section"] == "p3"
    assert "repo" not in flat  # None values dropped (chromadb metadata cannot be None)


def test_document_carries_commit_hash():
    doc = Document(doc_id="d1", title="README.md", source_type="github", text="x",
                   repo="r", file_path="README.md", commit_hash="abc123")
    assert doc.commit_hash == "abc123"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.types`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/types.py`**

```python
"""Pure data types threaded end-to-end. The `commit_hash` chain (fetch → chunk → retrieved
→ eval scorer) lives on Citation so the eval scorer can verify gold provenance."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from typing_extensions import TypedDict


@dataclass(frozen=True)
class Citation:
    doc_id: str
    title: str
    source_type: str  # 'github' | 'pdf' | 'docx' | ...
    # GitHub provenance
    repo: Optional[str] = None
    file_path: Optional[str] = None
    commit_hash: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    # File provenance
    page_or_section: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    def to_metadata(self) -> dict:
        """Flatten for chromadb metadata (str/int/float/bool only; None not allowed)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_metadata(cls, meta: dict) -> "Citation":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in meta.items() if k in fields})


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}::{ordinal}"
    doc_id: str
    ordinal: int
    text: str
    citation: Citation


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source_type: str
    text: str
    repo: Optional[str] = None
    file_path: Optional[str] = None
    commit_hash: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class GraphState(TypedDict, total=False):
    question: str
    retrieved: list  # list[RetrievedChunk]
    answerable: bool
    confidence: int
    answer: str
    citations: list  # list[Citation]
    refused: bool
```

- [ ] **Step 4: Implement the shared `FakeModelProvider` in `tests/conftest.py`**

This is the seam that makes every downstream TDD step deterministic and offline.

```python
"""Shared test fixtures. FakeModelProvider replaces the live OpenAI-compatible API so the
graph/pipeline/eval are tested deterministically without network or keys."""
import hashlib

import pytest


class FakeModelProvider:
    """Deterministic embed (hash-seeded 384-d vector) + scriptable generate.

    - embed(): stable per text, so retrieval order is reproducible.
    - generate(): returns canned_json when json_mode else canned_answer.
    """

    def __init__(self, canned_json: str = '{"answerable": true, "confidence": 5}',
                 canned_answer: str = "A grounded answer."):
        self.canned_json = canned_json
        self.canned_answer = canned_answer
        self.calls: list[dict] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(384)]
            out.append(vec)
        return out

    def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return self.canned_json if json_mode else self.canned_answer


@pytest.fixture
def fake_provider():
    return FakeModelProvider()
```

- [ ] **Step 5: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_types.py -v && uv run ruff check src tests`
Expected: 3 passed; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/genacademy_rag/core tests/conftest.py tests/core
git commit -m "feat(core): typed Document/Chunk/Citation + FakeModelProvider test seam"
```

---

## Task 2: ModelProvider — local embeddings + OpenAI-compatible generation

**Files:**
- Create: `src/genacademy_rag/core/providers.py`
- Create: `tests/core/test_providers.py`, `tests/integration/__init__.py`, `tests/integration/test_live_provider.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_providers.py`

```python
from genacademy_rag.core.providers import STEmbedder, OpenAICompatProvider


def test_st_embedder_returns_384_dim(monkeypatch):
    # Avoid loading the real model in a unit test: stub the encoder.
    emb = STEmbedder.__new__(STEmbedder)
    emb._model = type("M", (), {"encode": staticmethod(lambda xs, **k: [[0.0] * 384 for _ in xs])})()
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
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.providers`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/providers.py`**

```python
"""ModelProvider seam. STEmbedder = local sentence-transformers (offline, deterministic).
OpenAICompatProvider = the one generation seam for every preset (Nebius/OpenRouter/OpenAI/Gemma):
the same base_url + key + model verbatim shape the spike validated."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def generate(self, messages: list[dict], *, json_mode: bool = False,
                 max_tokens: int = 512, temperature: float = 0.0) -> str: ...


class STEmbedder:
    """Local all-MiniLM-L6-v2 (384-dim). Load once (cold ~11.6 s); reuse for every request."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), batch_size=32, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenAICompatProvider:
    """Generation via any OpenAI-compatible endpoint. Verbatim call shape from spike/gen_probe.py."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        # OpenAI(api_key="") raises OpenAIError("Missing credentials"); a keyless local Gemma
        # server ignores the value, so pass a placeholder when no key is configured.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - not used in P0
        raise NotImplementedError("Phase 0 embeds locally via STEmbedder")

    def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0) -> str:
        kwargs = dict(model=self._model, messages=list(messages),
                      temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content


class CombinedProvider:
    """Bundles local embed + remote generate behind one ModelProvider for the pipeline."""

    def __init__(self, embedder: STEmbedder, generator: OpenAICompatProvider):
        self._embedder = embedder
        self._generator = generator

    def embed(self, texts):
        return self._embedder.embed(texts)

    def generate(self, messages, **kwargs):
        return self._generator.generate(messages, **kwargs)


def build_provider(settings) -> CombinedProvider:
    """Wire the active preset from Settings. The mandatory Nebius call = set GENACADEMY_PROVIDER=nebius."""
    return CombinedProvider(
        STEmbedder(settings.embed_model),
        OpenAICompatProvider(settings.gen_base_url, settings.gen_api_key, settings.gen_model),
    )
```

- [ ] **Step 4: Write the integration test** `tests/integration/test_live_provider.py`

```python
import os

import pytest

from genacademy_rag.config import Settings
from genacademy_rag.core.providers import OpenAICompatProvider


@pytest.mark.integration
def test_live_json_mode_returns_parseable_object():
    s = Settings.from_env()
    if not s.gen_api_key:
        pytest.skip("no generation key set")
    p = OpenAICompatProvider(s.gen_base_url, s.gen_api_key, s.gen_model)
    out = p.generate(
        [{"role": "system", "content": "Reply ONLY with JSON."},
         {"role": "user", "content": 'Return {"answerable": true, "confidence": 5}.'}],
        json_mode=True, max_tokens=64,
    )
    import json
    parsed = json.loads(out)
    assert "answerable" in parsed
```

- [ ] **Step 5: Run unit tests (pass) + confirm integration is skipped by default**

Run: `uv run pytest tests/core/test_providers.py -v && uv run pytest tests/integration -v`
Expected: unit tests 2 passed; integration shows `1 deselected` (default `-m 'not integration'`). With a key: `uv run pytest -m integration -v` → 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/genacademy_rag/core/providers.py tests/core/test_providers.py tests/integration
git commit -m "feat(core): ModelProvider seam (local embed + OpenAI-compat generate) + live integration test"
```

---

## Task 3: FixedSizeChunker (citation metadata captured at chunk time)

**Files:**
- Create: `src/genacademy_rag/core/chunker.py`
- Create: `tests/core/test_chunker.py`

**Design note (accuracy trap):** `all-MiniLM-L6-v2` truncates at **256 tokens**. The design says "~512/64 tok behind a `Chunker` interface" but a 512-token chunk would be half-ignored by the embedder. Phase 0 therefore chunks by **characters** (`chunk_size=1000`, `overlap=150` ≈ ~250 tokens, under the 256 cap) which also yields exact `char_start/char_end` for citations; `line_start/line_end` are derived from the source text. Log this in `docs/design.md` `## Changelog vs source`. Section-aware / token-exact chunking is the Phase-2 eval axis.

- [ ] **Step 1: Write the failing test** `tests/core/test_chunker.py`

```python
from genacademy_rag.core.chunker import FixedSizeChunker
from genacademy_rag.core.types import Document


def _doc(text):
    return Document(doc_id="d1", title="README.md", source_type="github", text=text,
                    repo="r", file_path="README.md", commit_hash="abc123")


def test_short_doc_is_one_chunk_with_full_span():
    doc = _doc("line one\nline two\n")
    chunks = FixedSizeChunker(chunk_size=1000, overlap=150).chunk(doc)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "d1::0"
    assert c.text == doc.text
    assert c.citation.char_start == 0
    assert c.citation.char_end == len(doc.text)
    assert c.citation.line_start == 1 and c.citation.line_end == 2
    assert c.citation.commit_hash == "abc123"  # commit_hash chain preserved


def test_long_doc_splits_with_overlap_and_monotonic_spans():
    text = "\n".join(f"sentence number {i} about retrieval" for i in range(400))
    chunks = FixedSizeChunker(chunk_size=300, overlap=60).chunk(_doc(text))
    assert len(chunks) > 1
    # spans cover the document and overlap (next start < prev end)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.citation.char_start < prev.citation.char_end
        assert nxt.ordinal == prev.ordinal + 1


def test_line_numbers_are_one_based_and_correct():
    doc = _doc("a\nb\nc\nd\ne\n" * 50)  # many lines
    chunks = FixedSizeChunker(chunk_size=40, overlap=0).chunk(doc)
    assert chunks[0].citation.line_start == 1
    assert all(c.citation.line_start >= 1 for c in chunks)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.chunker`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/chunker.py`**

```python
"""Chunker seam. FixedSizeChunker = character windows with overlap, capturing exact char spans
and 1-based line spans for citations. Char-based (≈250 tok at size 1000) respects the embedder's
256-token cap; token-exact/section-aware chunking is the Phase-2 eval axis."""
from __future__ import annotations

from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation, Document


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        n = len(text)
        if n == 0:
            return []
        # Precompute char-offset -> line number (1-based) for span citations.
        line_at = [1] * (n + 1)
        line = 1
        for i, ch in enumerate(text):
            line_at[i] = line
            if ch == "\n":
                line += 1
        line_at[n] = line

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        ordinal = 0
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            piece = text[start:end]
            citation = Citation(
                doc_id=doc.doc_id, title=doc.title, source_type=doc.source_type,
                repo=doc.repo, file_path=doc.file_path, commit_hash=doc.commit_hash,
                line_start=line_at[start], line_end=line_at[max(start, end - 1)],
                char_start=start, char_end=end,
            )
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{ordinal}", doc_id=doc.doc_id,
                                ordinal=ordinal, text=piece, citation=citation))
            ordinal += 1
            if end == n:
                break
            start += step
        return chunks
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_chunker.py -v && uv run ruff check src tests`
Expected: 3 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/chunker.py tests/core/test_chunker.py
git commit -m "feat(core): FixedSizeChunker with char/line-span citations (respects 256-tok embed cap)"
```

---

## Task 4: Loaders — GitHub fetcher (pinned SHA + Week-2 firewall), Markdown, Jupyter

**Files:**
- Create: `src/genacademy_rag/core/loaders/__init__.py`, `github_fetcher.py`, `markdown_loader.py`, `jupyter_loader.py`
- Create: `tests/core/test_loaders.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_loaders.py`

```python
import json

import pytest

from genacademy_rag.core.loaders import EVAL_CORPUS, assert_allowed
from genacademy_rag.core.loaders.markdown_loader import load_markdown
from genacademy_rag.core.loaders.jupyter_loader import load_notebook


def test_eval_corpus_pins_two_repos_to_exact_shas():
    repos = {r["repo"]: r["sha"] for r in EVAL_CORPUS}
    assert repos["awesome-agentic-ai-resources"] == "5dfb8691180dc4956107e86839998ba3a2ebd94f"
    assert repos["Mastering-Agentic-AI-Week1"] == "3aa31dfede8c76422be91f2ecdbc59eddc690b1d"


def test_week2_repo_is_firewalled_out():
    # The sample solution must never be fetchable. Allowlist enforcement, not convention.
    assert "Mastering-Agentic-AI-Week2" not in {r["repo"] for r in EVAL_CORPUS}
    with pytest.raises(ValueError, match="not in the eval allowlist"):
        assert_allowed("Mastering-Agentic-AI-Week2")


def test_markdown_loader_builds_document_with_provenance():
    doc = load_markdown(
        repo="awesome-agentic-ai-resources", file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        raw_text="# Title\n\n| Resource | Covers |\n|---|---|\n| QLoRA | finetuning |\n",
    )
    assert doc.source_type == "github"
    assert doc.title == "README.md"
    assert doc.commit_hash.startswith("5dfb869")
    assert "QLoRA" in doc.text


def test_jupyter_loader_keeps_markdown_and_code_cells():
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# Langchain Fundamentals\n"]},
        {"cell_type": "code", "source": ["from langchain import PromptTemplate\n"]},
    ], "nbformat": 4, "nbformat_minor": 5}
    doc = load_notebook(
        repo="Mastering-Agentic-AI-Week1",
        file_path="Langchain Basics/Langchain_Fundamentals.ipynb",
        commit_hash="3aa31dfede8c76422be91f2ecdbc59eddc690b1d",
        raw_bytes=json.dumps(nb).encode(),
    )
    assert "Langchain Fundamentals" in doc.text
    assert "PromptTemplate" in doc.text
    assert doc.source_type == "github"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.loaders`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/loaders/__init__.py`** (the allowlist + firewall)

```python
"""Loader registry + the commit-pinned eval-corpus allowlist. The allowlist IS the Week-2
firewall: only these two repos+SHAs are ever fetched. Mastering-Agentic-AI-Week2 (the sample
solution) is absent by construction; reading it is disqualifying (AGENTS.md §5)."""
from __future__ import annotations

# Pinned SHAs from docs/spike-findings.md §4 (verified 2026-06-08).
EVAL_CORPUS: list[dict] = [
    {
        "repo": "awesome-agentic-ai-resources",
        "owner": "The-Gen-Academy",
        "sha": "5dfb8691180dc4956107e86839998ba3a2ebd94f",
        "files": [{"path": "README.md", "kind": "markdown"}],
    },
    {
        "repo": "Mastering-Agentic-AI-Week1",
        "owner": "The-Gen-Academy",
        "sha": "3aa31dfede8c76422be91f2ecdbc59eddc690b1d",
        "files": [
            {"path": "Langchain Basics/Langchain_Fundamentals.ipynb", "kind": "jupyter"},
            {"path": "Langchain Basics/README.md", "kind": "markdown"},
            {"path": "Langchain Basics/langchain_prompts.py", "kind": "markdown"},  # treat .py as text
        ],
    },
]

_ALLOWED = {r["repo"] for r in EVAL_CORPUS}


def assert_allowed(repo: str) -> None:
    if repo not in _ALLOWED:
        raise ValueError(f"repo {repo!r} is not in the eval allowlist {_ALLOWED} "
                         f"(Mastering-Agentic-AI-Week2 is the sample solution and is firewalled)")
```

- [ ] **Step 4: Implement `markdown_loader.py` and `jupyter_loader.py`**

`src/genacademy_rag/core/loaders/markdown_loader.py`:
```python
"""Markdown/text → Document. Raw text kept verbatim (tables matter for the catalog questions)."""
from __future__ import annotations

from genacademy_rag.core.types import Document


def load_markdown(*, repo: str, file_path: str, commit_hash: str, raw_text: str) -> Document:
    doc_id = f"{repo}/{file_path}@{commit_hash[:7]}"
    return Document(doc_id=doc_id, title=file_path.split("/")[-1], source_type="github",
                    text=raw_text, repo=repo, file_path=file_path, commit_hash=commit_hash)
```

`src/genacademy_rag/core/loaders/jupyter_loader.py`:
```python
"""Jupyter .ipynb → Document. Flattens markdown + code cells into text (nbformat parse)."""
from __future__ import annotations

import nbformat

from genacademy_rag.core.types import Document


def load_notebook(*, repo: str, file_path: str, commit_hash: str, raw_bytes: bytes) -> Document:
    nb = nbformat.reads(raw_bytes.decode("utf-8"), as_version=4)
    parts: list[str] = []
    for cell in nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "code":
            parts.append(f"```python\n{src}\n```")
        else:
            parts.append(src)
    doc_id = f"{repo}/{file_path}@{commit_hash[:7]}"
    return Document(doc_id=doc_id, title=file_path.split("/")[-1], source_type="github",
                    text="\n\n".join(parts), repo=repo, file_path=file_path, commit_hash=commit_hash)
```

- [ ] **Step 5: Implement `github_fetcher.py`** (fetch raw blobs at the pinned SHA)

```python
"""Fetch raw file bytes at a pinned commit SHA via raw.githubusercontent.com. Public repos,
no auth needed. Every fetch goes through assert_allowed() — the firewall."""
from __future__ import annotations

import requests

from genacademy_rag.core.loaders import assert_allowed

RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"


def fetch_raw(*, owner: str, repo: str, sha: str, path: str, timeout: int = 30) -> bytes:
    assert_allowed(repo)
    url = RAW_URL.format(owner=owner, repo=repo, sha=sha, path=requests.utils.quote(path))
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content
```

- [ ] **Step 6: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_loaders.py -v && uv run ruff check src tests`
Expected: 4 passed; ruff clean. (Loaders are tested with inline raw text/bytes — no network. `fetch_raw` is exercised live by the ingest script in Task 10.)

- [ ] **Step 7: Commit**

```bash
git add src/genacademy_rag/core/loaders tests/core/test_loaders.py
git commit -m "feat(core): pinned-SHA GitHub fetcher + Markdown/Jupyter loaders + Week-2 firewall allowlist"
```

---

## Task 5: ChromaStore (VectorStore seam, precomputed embeddings)

**Files:**
- Create: `src/genacademy_rag/core/vectorstore.py`
- Create: `tests/core/test_vectorstore.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_vectorstore.py`

```python
import pytest

from genacademy_rag.core.types import Chunk, Citation
from genacademy_rag.core.vectorstore import ChromaStore


def _chunk(i, text):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="r", file_path="README.md", commit_hash="abc123",
                   line_start=i, line_end=i, char_start=i, char_end=i + 1)
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=text, citation=cit)


def test_upsert_then_query_returns_nearest_chunk_ids(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="t")
    chunks = [_chunk(0, "retrieval augmented generation"), _chunk(1, "banana bread recipe")]
    embs = fake_provider.embed([c.text for c in chunks])
    store.upsert(chunks, embs)
    qvec = fake_provider.embed(["retrieval augmented generation"])[0]
    results = store.query(qvec, top_k=2)               # list[(chunk_id, cosine_similarity)]
    ids = [cid for cid, _ in results]
    assert ids[0] == "d1::0"  # exact-text match ranks first under the deterministic fake embed
    assert set(ids) == {"d1::0", "d1::1"}
    assert results[0][1] >= results[1][1]              # similarity descending
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)  # query == chunk text -> sim ~1.0


def test_get_chunk_round_trips_citation(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="t")
    chunks = [_chunk(0, "alpha")]
    store.upsert(chunks, fake_provider.embed(["alpha"]))
    got = store.get_chunk("d1::0")
    assert got.text == "alpha"
    assert got.citation.commit_hash == "abc123"
    assert got.citation.line_start == 0


def test_get_all_chunks_returns_every_upserted_chunk(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="t")
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    store.upsert(chunks, fake_provider.embed([c.text for c in chunks]))
    got = {c.chunk_id for c in store.get_all_chunks()}
    assert got == {"d1::0", "d1::1"}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_vectorstore.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.vectorstore`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/vectorstore.py`**

```python
"""VectorStore seam. ChromaStore = raw chromadb PersistentClient holding precomputed embeddings
(we embed via ModelProvider, Chroma just stores+searches). Phase-2 swap target: PineconeStore."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]: ...
    def get_chunk(self, chunk_id: str) -> Chunk: ...
    def get_all_chunks(self) -> list[Chunk]: ...


class ChromaStore:
    def __init__(self, persist_dir, collection: str = "genacademy"):
        import chromadb
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # cosine space; we pass our own normalized embeddings.
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{**c.citation.to_metadata(), "ordinal": c.ordinal} for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        # Return (chunk_id, cosine_similarity). Chroma's cosine space gives DISTANCE; sim = 1 - dist.
        # The similarity is the confidence signal the grader's cosine fallback uses (see Task 7);
        # RRF (Task 6) handles ranking. Returning raw IDs only would leave the fallback no signal.
        res = self._col.query(query_embeddings=[list(map(float, query_embedding))],
                              n_results=top_k, include=["distances"])
        if not res["ids"] or not res["ids"][0]:
            return []
        ids, dists = res["ids"][0], res["distances"][0]
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists)]

    def get_chunk(self, chunk_id: str) -> Chunk:
        res = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        text = res["documents"][0]
        meta = dict(res["metadatas"][0])
        ordinal = int(meta.pop("ordinal", 0))
        return Chunk(chunk_id=chunk_id, doc_id=meta["doc_id"], ordinal=ordinal,
                     text=text, citation=Citation.from_metadata(meta))

    def get_all_chunks(self) -> list[Chunk]:
        """Public accessor so callers never reach into `_col` (keeps the Pinecone swap clean)."""
        res = self._col.get(include=["documents", "metadatas"])
        out: list[Chunk] = []
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            m = dict(meta)
            ordinal = int(m.pop("ordinal", 0))
            out.append(Chunk(chunk_id=cid, doc_id=m["doc_id"], ordinal=ordinal,
                             text=doc, citation=Citation.from_metadata(m)))
        return out
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_vectorstore.py -v && uv run ruff check src tests`
Expected: 2 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/vectorstore.py tests/core/test_vectorstore.py
git commit -m "feat(core): ChromaStore VectorStore impl over precomputed embeddings"
```

---

## Task 6: HybridRetriever (dense + BM25 + hand-rolled RRF)

**Files:**
- Create: `src/genacademy_rag/core/retriever.py`
- Create: `tests/core/test_retriever.py`

**Design note:** we hand-roll RRF over `rank-bm25` + Chroma rather than using LangChain's `EnsembleRetriever` — the spec names `rank-bm25` explicitly (design §4, tech-stack), it keeps the core pure and avoids LangChain import churn, and RRF is ~10 lines. The exact-match gold question (BM25 wins where dense misses) is what turns this hybrid decision into a reported eval number (design §7).

**Known fallback-path interaction (document during calibration, Task 7):** a chunk surfaced *only* by BM25 (not in the dense top-`candidate_k`) carries `score = 0.0`, because cosine similarity is only known for dense hits. The eval runs on the **primary** JSON-mode grader (spike-confirmed reliable), so eval numbers are unaffected. But if the JSON grader ever fails and the cosine-threshold *fallback* fires on an **exact-match** question whose answer is a BM25-only hit, `max(score)=0.0 < threshold` would wrongly refuse it. Two mitigations, both already in place: `candidate_k=20` over the tiny ~50–100-chunk eval corpus means most chunks get a real dense score anyway, and the fallback threshold is calibrated on held-out questions (design §7). Note this when calibrating; do not "fix" it by weakening the refusal path.

- [ ] **Step 1: Write the failing test** `tests/core/test_retriever.py`

```python
from genacademy_rag.core.retriever import rrf_fuse, HybridRetriever
from genacademy_rag.core.types import Chunk, Citation


def _chunk(i, text):
    cit = Citation(doc_id="d1", title="t", source_type="github", repo="r",
                   file_path="f", commit_hash="abc123", line_start=i, line_end=i)
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=text, citation=cit)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = rrf_fuse([dense, sparse], k=60)
    # "a" (ranks 0,1) and "b" (ranks 1,0) outrank "c"/"d" (appear once).
    top2 = sorted(fused, key=fused.get, reverse=True)[:2]
    assert set(top2) == {"a", "b"}


def test_hybrid_retriever_surfaces_exact_keyword_via_bm25(fake_provider):
    # A rare proper noun the dense fake-embed would scatter; BM25 must catch it.
    chunks = [
        _chunk(0, "general notes about retrieval and embeddings"),
        _chunk(1, "the QLoRA technique quantizes weights for finetuning"),
        _chunk(2, "more general notes about vector databases"),
    ]

    class _Store:
        def __init__(self, chunks):
            self._by_id = {c.chunk_id: c for c in chunks}
            self._embs = {c.chunk_id: fake_provider.embed([c.text])[0] for c in chunks}

        def query(self, qvec, top_k):
            import math
            def cos(a, b):
                num = sum(x * y for x, y in zip(a, b))
                da = math.sqrt(sum(x * x for x in a)); db = math.sqrt(sum(y * y for y in b))
                return num / (da * db + 1e-9)
            ranked = sorted(self._embs, key=lambda cid: cos(qvec, self._embs[cid]), reverse=True)
            return [(cid, cos(qvec, self._embs[cid])) for cid in ranked[:top_k]]  # (id, cosine_sim)

        def get_chunk(self, cid):
            return self._by_id[cid]

    retr = HybridRetriever(store=_Store(chunks), provider=fake_provider, all_chunks=chunks, top_k=2)
    results = retr.retrieve("QLoRA")
    assert any("QLoRA" in r.chunk.text for r in results)


def test_retrieved_score_is_cosine_similarity_not_rrf(fake_provider):
    # Regression guard: score must be the cosine sim (usable by the grader's threshold fallback),
    # NOT the tiny RRF score (~0.03) that would make the cosine fallback refuse everything.
    chunk = _chunk(0, "retrieval augmented generation")

    class _Store:
        def query(self, qvec, top_k):
            return [("d1::0", 0.91)]      # high cosine similarity

        def get_chunk(self, cid):
            return chunk

    retr = HybridRetriever(store=_Store(), provider=fake_provider, all_chunks=[chunk], top_k=1)
    [r] = retr.retrieve("retrieval augmented generation")
    assert r.score == 0.91               # cosine sim carried through, not 2/61
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.retriever`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/retriever.py`**

```python
"""Retriever seam. HybridRetriever = dense (VectorStore) + sparse (rank-bm25) fused via RRF.
Phase-2 swap target: + cross-encoder rerank."""
from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from genacademy_rag.core.types import Chunk, RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = sum over lists of 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, *, store, provider, all_chunks: list[Chunk], top_k: int = 5,
                 candidate_k: int = 20, rrf_k: int = 60):
        self._store = store
        self._provider = provider
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._chunks_by_id = {c.chunk_id: c for c in all_chunks}
        self._ids = [c.chunk_id for c in all_chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in all_chunks])

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        qvec = self._provider.embed([query])[0]
        dense_hits = self._store.query(qvec, top_k=self._candidate_k)   # list[(id, cosine_sim)]
        dense_ids = [cid for cid, _ in dense_hits]
        sim_by_id = {cid: sim for cid, sim in dense_hits}
        scores = self._bm25.get_scores(_tokenize(query))
        sparse_ids = [self._ids[i] for i in sorted(range(len(scores)),
                                                   key=lambda j: scores[j], reverse=True)][:self._candidate_k]
        fused = rrf_fuse([dense_ids, sparse_ids], k=self._rrf_k)        # ranking signal
        ranked = sorted(fused, key=fused.get, reverse=True)[:self._top_k]
        # score = cosine similarity (the grader's confidence signal); 0.0 for BM25-only hits.
        # RRF decides ORDER; cosine sim is carried separately so the grader fallback is meaningful.
        return [RetrievedChunk(chunk=self._chunks_by_id[cid], score=sim_by_id.get(cid, 0.0))
                for cid in ranked]
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_retriever.py -v && uv run ruff check src tests`
Expected: 2 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/retriever.py tests/core/test_retriever.py
git commit -m "feat(core): HybridRetriever (dense + BM25 + RRF)"
```

---

## Task 7: Grader — JSON-mode answerability + cosine fallback

**Files:**
- Create: `src/genacademy_rag/core/grader.py`
- Create: `tests/core/test_grader.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_grader.py`

```python
from tests.conftest import FakeModelProvider
from genacademy_rag.core.grader import grade_answerability, cosine_fallback_grade
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk


def _rc(text, score=0.5):
    cit = Citation(doc_id="d1", title="t", source_type="github")
    return RetrievedChunk(chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text=text, citation=cit),
                          score=score)


def test_json_grader_parses_answerable_true():
    p = FakeModelProvider(canned_json='{"answerable": true, "confidence": 5}')
    g = grade_answerability("what is RAG?", [_rc("RAG retrieves then generates.")], p)
    assert g.answerable is True and g.confidence == 5


def test_json_grader_parses_refusal():
    p = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')
    g = grade_answerability("who won the 2050 world cup?", [_rc("unrelated text")], p)
    assert g.answerable is False


def test_grader_falls_back_to_cosine_on_malformed_json():
    p = FakeModelProvider(canned_json="not json at all")
    g = grade_answerability("q", [_rc("x", score=0.9)], p, cosine_threshold=0.2)
    assert g.answerable is True            # fell back, top score 0.9 >= 0.2
    assert g.used_fallback is True


def test_cosine_fallback_refuses_when_below_threshold():
    g = cosine_fallback_grade([_rc("x", score=0.05)], threshold=0.2)
    assert g.answerable is False
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_grader.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.grader`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/grader.py`**

The grader prompt is copied verbatim into `specs/grader/requirements.md` (tech-stack §5) — keep this string and that file in sync.

```python
"""Refusal grader. Primary: JSON-mode LLM call (spike confirmed it works on the open model).
Fallback: max cosine similarity of the retrieved set vs a calibrated threshold. Load-bearing:
low confidence ⇒ refuse, never answer from priors (AGENTS.md §3)."""
from __future__ import annotations

import json
from dataclasses import dataclass

from genacademy_rag.core.types import RetrievedChunk

GRADER_SYSTEM = "You are a strict grader. Reply ONLY with a JSON object."
GRADER_USER_TMPL = (
    "Question:\n{question}\n\n"
    "Retrieved context (the ONLY allowed source):\n{context}\n\n"
    'Decide if the question can be answered FROM THIS CONTEXT ALONE. '
    'Return exactly {{"answerable": <true|false>, "confidence": <1-5 integer>}}. '
    "answerable=false if the context does not contain the answer."
)


@dataclass(frozen=True)
class Grade:
    answerable: bool
    confidence: int
    used_fallback: bool = False


def cosine_fallback_grade(retrieved: list[RetrievedChunk], threshold: float) -> Grade:
    # r.score is the cosine similarity carried by HybridRetriever (Task 6), in [-1, 1] — NOT the
    # RRF rank score. Calibrate `threshold` (design §7) on 3-5 held-out questions against THIS signal.
    top = max((r.score for r in retrieved), default=0.0)
    answerable = top >= threshold
    # Map the top score into a 1-5 confidence bucket for the report.
    confidence = max(1, min(5, int(round(top * 5)))) if answerable else 1
    return Grade(answerable=answerable, confidence=confidence, used_fallback=True)


def grade_answerability(question: str, retrieved: list[RetrievedChunk], provider, *,
                        cosine_threshold: float = 0.2) -> Grade:
    context = "\n---\n".join(r.chunk.text for r in retrieved)
    try:
        raw = provider.generate(
            [{"role": "system", "content": GRADER_SYSTEM},
             {"role": "user", "content": GRADER_USER_TMPL.format(question=question, context=context)}],
            json_mode=True, max_tokens=64,
        )
        parsed = json.loads(raw)
        return Grade(answerable=bool(parsed["answerable"]),
                     confidence=int(parsed.get("confidence", 3)))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return cosine_fallback_grade(retrieved, threshold=cosine_threshold)
```

- [ ] **Step 4: Write `specs/grader/requirements.md`** with the verbatim prompt strings (so reference text is copied, not paraphrased — tech-stack §5). Create the file containing `GRADER_SYSTEM` and `GRADER_USER_TMPL` exactly as above plus a one-line note: "Source of truth for the grader prompt; keep in sync with `core/grader.py`."

- [ ] **Step 5: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_grader.py -v && uv run ruff check src tests`
Expected: 4 passed; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/genacademy_rag/core/grader.py tests/core/test_grader.py specs/grader/requirements.md
git commit -m "feat(core): refusal grader (JSON-mode primary + cosine fallback)"
```

---

## Task 8: LangGraph graph — retrieve → grade → {answer | refuse}

**Files:**
- Create: `src/genacademy_rag/core/graph.py`
- Create: `tests/core/test_graph.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_graph.py`

```python
from tests.conftest import FakeModelProvider
from genacademy_rag.core.graph import build_graph
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk

REFUSAL = "I could not find this in the course materials."


class _Retriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query):
        return self._chunks


def _rc(text):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="r", file_path="README.md", commit_hash="abc123", line_start=1, line_end=2)
    return RetrievedChunk(chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text=text, citation=cit),
                          score=0.8)


def test_answerable_path_returns_answer_and_citations():
    provider = FakeModelProvider(canned_json='{"answerable": true, "confidence": 5}',
                                 canned_answer="RAG retrieves then generates.")
    graph = build_graph(retriever=_Retriever([_rc("RAG retrieves then generates.")]), provider=provider)
    out = graph.invoke({"question": "what is RAG?"})
    assert out["refused"] is False
    assert out["answer"] == "RAG retrieves then generates."
    assert out["citations"][0].commit_hash == "abc123"


def test_unanswerable_path_refuses_without_calling_answer():
    provider = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')
    graph = build_graph(retriever=_Retriever([_rc("unrelated")]), provider=provider)
    out = graph.invoke({"question": "who won the 2050 world cup?"})
    assert out["refused"] is True
    assert out["answer"] == REFUSAL
    # the answer-generation path must NOT have been taken (no non-json generate call)
    assert all(c["json_mode"] for c in provider.calls)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.graph`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/graph.py`**

```python
"""The one LangGraph graph: retrieve → grade → {answer + citations | refuse}. Dependencies
(retriever, provider) are injected so the graph is unit-tested against fakes."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from genacademy_rag.core.grader import grade_answerability
from genacademy_rag.core.types import GraphState

REFUSAL_MESSAGE = "I could not find this in the course materials."
ANSWER_SYSTEM = (
    "You answer ONLY from the provided course context. If the context does not contain the "
    "answer, say you could not find it. Never use outside knowledge. Be concise."
)


def build_graph(*, retriever, provider, cosine_threshold: float = 0.2):
    def retrieve_node(state: GraphState) -> dict:
        return {"retrieved": retriever.retrieve(state["question"])}

    def grade_node(state: GraphState) -> dict:
        g = grade_answerability(state["question"], state["retrieved"], provider,
                                cosine_threshold=cosine_threshold)
        return {"answerable": g.answerable, "confidence": g.confidence}

    def answer_node(state: GraphState) -> dict:
        context = "\n---\n".join(r.chunk.text for r in state["retrieved"])
        answer = provider.generate(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"}],
            json_mode=False, max_tokens=512,
        )
        citations = [r.chunk.citation for r in state["retrieved"]]
        return {"answer": answer, "citations": citations, "refused": False}

    def refuse_node(state: GraphState) -> dict:
        return {"answer": REFUSAL_MESSAGE,
                "citations": [r.chunk.citation for r in state["retrieved"]], "refused": True}

    def route(state: GraphState) -> str:
        return "answer" if state["answerable"] else "refuse"

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("answer", answer_node)
    g.add_node("refuse", refuse_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"answer": "answer", "refuse": "refuse"})
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_graph.py -v && uv run ruff check src tests`
Expected: 2 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/graph.py tests/core/test_graph.py
git commit -m "feat(core): LangGraph refusal graph (retrieve→grade→answer|refuse)"
```

---

## Task 9: SQLiteDatastore (users, documents, chunks_meta)

**Files:**
- Create: `src/genacademy_rag/data/__init__.py`, `src/genacademy_rag/data/datastore.py`
- Create: `tests/data/__init__.py`, `tests/data/test_datastore.py`

- [ ] **Step 1: Write the failing test** `tests/data/test_datastore.py`

```python
from genacademy_rag.data.datastore import SQLiteDatastore
from genacademy_rag.core.types import Chunk, Citation


def _chunk(i):
    cit = Citation(doc_id="d1", title="README.md", source_type="github", repo="r",
                   file_path="README.md", commit_hash="abc123", line_start=i, line_end=i + 1,
                   char_start=i, char_end=i + 5)
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=f"chunk {i} preview", citation=cit)


def test_seed_users_and_lookup(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.seed_users()
    admin = ds.get_user_by_email("admin@genacademy.local")
    member = ds.get_user_by_email("member@genacademy.local")
    assert admin["role"] == "admin"
    assert member["role"] == "member"


def test_record_document_and_chunks(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.add_document(doc_id="d1", title="README.md", source_type="github",
                    repo="r", file_path="README.md", commit_hash="abc123", n_chunks=2)
    ds.add_chunks_meta([_chunk(0), _chunk(1)])
    doc = ds.get_document("d1")
    assert doc["commit_hash"] == "abc123" and doc["n_chunks"] == 2
    metas = ds.get_chunks_for_doc("d1")
    assert len(metas) == 2 and metas[0]["line_start"] == 0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/data/test_datastore.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.data.datastore`.

- [ ] **Step 3: Implement `src/genacademy_rag/data/datastore.py`**

```python
"""Datastore seam (SQLite, Phase 0). Holds users, documents, chunks_meta. Vectors live in Chroma;
everything relational here. Phase-2 swap target: Postgres. usage_log is Phase 1."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Protocol

from genacademy_rag.core.types import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin','member')),
    password TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, title TEXT, source_type TEXT, repo TEXT, file_path TEXT,
    commit_hash TEXT, filename TEXT, uploaded_by TEXT, status TEXT DEFAULT 'indexed',
    n_chunks INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks_meta (
    id TEXT PRIMARY KEY, doc_id TEXT, ordinal INTEGER, page_or_section TEXT,
    line_start INTEGER, line_end INTEGER, char_start INTEGER, char_end INTEGER,
    text_preview TEXT);
"""


class Datastore(Protocol):
    def seed_users(self) -> None: ...
    def get_user_by_email(self, email: str) -> Optional[dict]: ...
    def add_document(self, **kwargs) -> None: ...
    def add_chunks_meta(self, chunks: list[Chunk]) -> None: ...


class SQLiteDatastore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def seed_users(self) -> None:
        # Phase 0: 2 seeded users, plaintext dev passwords (RBAC/hashing = Phase 1).
        self._conn.executemany(
            "INSERT OR IGNORE INTO users(email, role, password) VALUES (?,?,?)",
            [("admin@genacademy.local", "admin", "admin"),
             ("member@genacademy.local", "member", "member")])
        self._conn.commit()

    def get_user_by_email(self, email: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

    def add_document(self, *, doc_id, title, source_type, repo=None, file_path=None,
                     commit_hash=None, filename=None, uploaded_by=None, n_chunks=0) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents(id,title,source_type,repo,file_path,commit_hash,"
            "filename,uploaded_by,n_chunks) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, title, source_type, repo, file_path, commit_hash, filename, uploaded_by, n_chunks))
        self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def add_chunks_meta(self, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks_meta(id,doc_id,ordinal,page_or_section,line_start,"
            "line_end,char_start,char_end,text_preview) VALUES (?,?,?,?,?,?,?,?,?)",
            [(c.chunk_id, c.doc_id, c.ordinal, c.citation.page_or_section, c.citation.line_start,
              c.citation.line_end, c.citation.char_start, c.citation.char_end, c.text[:200])
             for c in chunks])
        self._conn.commit()

    def get_chunks_for_doc(self, doc_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks_meta WHERE doc_id=? ORDER BY ordinal", (doc_id,)).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/data/test_datastore.py -v && uv run ruff check src tests`
Expected: 2 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/data tests/data
git commit -m "feat(data): SQLiteDatastore (users, documents, chunks_meta) + seeded users"
```

---

## Task 10: Ingest pipeline + commit-pinned eval-corpus script

**Files:**
- Create: `src/genacademy_rag/core/pipeline.py` (the `IngestPipeline` half)
- Create: `scripts/ingest_eval_corpus.py`
- Create: `tests/core/test_ingest_pipeline.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_ingest_pipeline.py`

```python
from genacademy_rag.core.chunker import FixedSizeChunker
from genacademy_rag.core.pipeline import IngestPipeline
from genacademy_rag.core.types import Document
from genacademy_rag.core.vectorstore import ChromaStore


def test_ingest_chunks_embeds_stores_and_records(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="t")

    class _DS:
        def __init__(self):
            self.docs, self.chunks = [], []
        def add_document(self, **kw): self.docs.append(kw)
        def add_chunks_meta(self, chunks): self.chunks.extend(chunks)

    ds = _DS()
    pipe = IngestPipeline(chunker=FixedSizeChunker(chunk_size=50, overlap=10),
                          provider=fake_provider, store=store, datastore=ds)
    doc = Document(doc_id="d1", title="README.md", source_type="github",
                   text="x" * 200, repo="r", file_path="README.md", commit_hash="abc123")
    n = pipe.ingest([doc])
    assert n > 1                                   # multiple chunks
    assert ds.docs[0]["n_chunks"] == n
    assert ds.docs[0]["commit_hash"] == "abc123"
    # stored & queryable
    qvec = fake_provider.embed(["x" * 50])[0]
    assert store.query(qvec, top_k=1)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_ingest_pipeline.py -v`
Expected: FAIL — `cannot import name 'IngestPipeline'`.

- [ ] **Step 3: Implement the `IngestPipeline` in `src/genacademy_rag/core/pipeline.py`**

```python
"""Pipelines. IngestPipeline (offline): Document → chunk → embed → store + record metadata.
QueryPipeline (online, Task 11): question → graph → {answer, citations}. Both pure."""
from __future__ import annotations

from genacademy_rag.core.types import Chunk, Document


class IngestPipeline:
    def __init__(self, *, chunker, provider, store, datastore):
        self._chunker = chunker
        self._provider = provider
        self._store = store
        self._datastore = datastore

    def ingest(self, docs: list[Document]) -> int:
        total = 0
        for doc in docs:
            chunks: list[Chunk] = self._chunker.chunk(doc)
            if not chunks:
                continue
            embeddings = self._provider.embed([c.text for c in chunks])
            self._store.upsert(chunks, embeddings)
            self._datastore.add_document(
                doc_id=doc.doc_id, title=doc.title, source_type=doc.source_type,
                repo=doc.repo, file_path=doc.file_path, commit_hash=doc.commit_hash,
                filename=doc.filename, n_chunks=len(chunks))
            self._datastore.add_chunks_meta(chunks)
            total += len(chunks)
        return total
```

- [ ] **Step 4: Run the unit test, verify pass**

Run: `uv run pytest tests/core/test_ingest_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write `scripts/ingest_eval_corpus.py`** (the commit-pinned entry point — live)

```python
"""Ingest the commit-pinned eval corpus into Chroma + SQLite. Fetches ONLY the allowlisted
repos+SHAs (docs/spike-findings.md §4); Week-2 is firewalled by the allowlist. Run once before
the eval. Idempotent (upsert by chunk_id)."""
from genacademy_rag.config import Settings
from genacademy_rag.core.chunker import FixedSizeChunker
from genacademy_rag.core.loaders import EVAL_CORPUS
from genacademy_rag.core.loaders.github_fetcher import fetch_raw
from genacademy_rag.core.loaders.markdown_loader import load_markdown
from genacademy_rag.core.loaders.jupyter_loader import load_notebook
from genacademy_rag.core.pipeline import IngestPipeline
from genacademy_rag.core.providers import build_provider
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.data.datastore import SQLiteDatastore


def main():
    s = Settings.from_env()
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    ds = SQLiteDatastore(s.sqlite_path)
    ds.seed_users()
    pipe = IngestPipeline(chunker=FixedSizeChunker(s.chunk_size, s.chunk_overlap),
                          provider=provider, store=store, datastore=ds)

    docs = []
    for repo in EVAL_CORPUS:
        for f in repo["files"]:
            raw = fetch_raw(owner=repo["owner"], repo=repo["repo"], sha=repo["sha"], path=f["path"])
            if f["kind"] == "jupyter":
                docs.append(load_notebook(repo=repo["repo"], file_path=f["path"],
                                          commit_hash=repo["sha"], raw_bytes=raw))
            else:
                docs.append(load_markdown(repo=repo["repo"], file_path=f["path"],
                                          commit_hash=repo["sha"], raw_text=raw.decode("utf-8")))
            print(f"fetched {repo['repo']}/{f['path']} @ {repo['sha'][:7]}")
    n = pipe.ingest(docs)
    print(f"ingested {len(docs)} docs -> {n} chunks into {s.chroma_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the live ingest** (needs network; embeds locally — no gen key required)

Run: `uv run python scripts/ingest_eval_corpus.py`
Expected: prints `fetched …` lines for the allowlisted files and `ingested 4 docs -> N chunks`. Confirms the pinned-SHA fetch + full ingest path end-to-end.

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check src scripts tests
git add src/genacademy_rag/core/pipeline.py scripts/ingest_eval_corpus.py tests/core/test_ingest_pipeline.py
git commit -m "feat: IngestPipeline + commit-pinned eval-corpus ingest script"
```

---

## Task 11: QueryPipeline (online: question → answer + citations)

**Files:**
- Modify: `src/genacademy_rag/core/pipeline.py` (add `QueryResult` + `QueryPipeline`)
- Create: `tests/core/test_query_pipeline.py`

- [ ] **Step 1: Write the failing test** `tests/core/test_query_pipeline.py`

```python
from tests.conftest import FakeModelProvider
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk


class _Retriever:
    def retrieve(self, q):
        cit = Citation(doc_id="d1", title="README.md", source_type="github",
                       repo="r", file_path="README.md", commit_hash="abc123", line_start=1, line_end=2)
        return [RetrievedChunk(chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0,
                                           text="RAG retrieves then generates.", citation=cit), score=0.8)]


def test_answerable_query_returns_answer_with_citations():
    provider = FakeModelProvider(canned_json='{"answerable": true, "confidence": 5}',
                                 canned_answer="RAG retrieves then generates.")
    qp = QueryPipeline(retriever=_Retriever(), provider=provider)
    result = qp.answer("what is RAG?")
    assert result.refused is False
    assert result.answer == "RAG retrieves then generates."
    assert result.citations[0].file_path == "README.md"


def test_unanswerable_query_refuses():
    provider = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')
    qp = QueryPipeline(retriever=_Retriever(), provider=provider)
    result = qp.answer("who won the 2050 world cup?")
    assert result.refused is True
    assert "could not find" in result.answer.lower()
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_query_pipeline.py -v`
Expected: FAIL — `cannot import name 'QueryPipeline'`.

- [ ] **Step 3: Add `QueryPipeline` to `src/genacademy_rag/core/pipeline.py`**

Append:
```python
from dataclasses import dataclass

from genacademy_rag.core.graph import build_graph
from genacademy_rag.core.types import Citation


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list  # list[Citation]
    refused: bool
    confidence: int


class QueryPipeline:
    def __init__(self, *, retriever, provider, cosine_threshold: float = 0.2):
        self._graph = build_graph(retriever=retriever, provider=provider,
                                  cosine_threshold=cosine_threshold)

    def answer(self, question: str) -> QueryResult:
        out = self._graph.invoke({"question": question})
        return QueryResult(answer=out["answer"], citations=out.get("citations", []),
                           refused=out["refused"], confidence=out.get("confidence", 0))
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/core/test_query_pipeline.py -v && uv run ruff check src tests`
Expected: 2 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/pipeline.py tests/core/test_query_pipeline.py
git commit -m "feat(core): QueryPipeline wrapping the refusal graph into {answer, citations}"
```

---

## Task 12: Gold-set schema + annotation (start Day 1, in parallel)

**This is the #1 risk (~6 h manual reading) — begin it the moment Task 1's types exist, alongside Tasks 2–6.** The schema must precede both the annotation and the scorer (Task 13).

**Files:**
- Create: `src/genacademy_rag/eval/__init__.py`, `src/genacademy_rag/eval/gold_schema.py`
- Create: `src/genacademy_rag/eval/gold/gold_set.yaml`
- Create: `tests/eval/__init__.py`, `tests/eval/test_gold_schema.py`

- [ ] **Step 1: Write the failing test** `tests/eval/test_gold_schema.py`

```python
import pytest

from genacademy_rag.eval.gold_schema import load_gold_set, GoldQuestion, CATEGORIES


def test_categories_cover_the_required_buckets():
    assert {"answerable", "exact_match", "chunking_stress", "multi_document",
            "ambiguous", "unanswerable"} == set(CATEGORIES)


def test_load_gold_set_parses_and_validates(tmp_path):
    yaml_text = """
- id: q1
  question: "What does the catalog say covers QLoRA?"
  category: exact_match
  answerable: true
  gold:
    - repo: awesome-agentic-ai-resources
      file_path: README.md
      commit_hash: 5dfb8691180dc4956107e86839998ba3a2ebd94f
      line_start: 40
      line_end: 41
- id: q2
  question: "What does the course say about week 8?"
  category: unanswerable
  answerable: false
  gold: []
"""
    p = tmp_path / "g.yaml"
    p.write_text(yaml_text)
    gold = load_gold_set(p)
    assert len(gold) == 2
    assert isinstance(gold[0], GoldQuestion)
    assert gold[0].category == "exact_match"
    assert gold[1].answerable is False and gold[1].gold == []


def test_answerable_question_must_have_gold_spans(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text('- id: q1\n  question: "x"\n  category: answerable\n  answerable: true\n  gold: []\n')
    with pytest.raises(ValueError, match="answerable.*requires.*gold"):
        load_gold_set(bad)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/eval/test_gold_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.eval.gold_schema`.

- [ ] **Step 3: Implement `src/genacademy_rag/eval/gold_schema.py`**

```python
"""Gold-set schema + validator. Each question pins gold spans by repo+file_path+commit_hash
(the provenance chain), so the scorer can confirm a retrieved chunk is the gold source AND that
its commit_hash matches (production content never satisfies a gold marker)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = ["answerable", "exact_match", "chunking_stress", "multi_document",
              "ambiguous", "unanswerable"]


@dataclass(frozen=True)
class GoldSpan:
    repo: str
    file_path: str
    commit_hash: str
    line_start: int | None = None
    line_end: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    category: str
    answerable: bool
    gold: list[GoldSpan] = field(default_factory=list)


def load_gold_set(path) -> list[GoldQuestion]:
    raw = yaml.safe_load(Path(path).read_text())
    out: list[GoldQuestion] = []
    for item in raw:
        if item["category"] not in CATEGORIES:
            raise ValueError(f"q{item['id']}: unknown category {item['category']!r}")
        spans = [GoldSpan(**s) for s in item.get("gold", [])]
        if item["answerable"] and not spans:
            raise ValueError(f"q{item['id']}: answerable=true requires at least one gold span")
        if not item["answerable"] and spans:
            raise ValueError(f"q{item['id']}: unanswerable question must have empty gold")
        out.append(GoldQuestion(id=item["id"], question=item["question"],
                                category=item["category"], answerable=item["answerable"], gold=spans))
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/eval/test_gold_schema.py -v`
Expected: 3 passed.

- [ ] **Step 5: ANNOTATION (manual, ~6 h) — produce `gold_set.yaml` with all 15 questions**

Start from Kimchi's 15 drafts in `docs/design-review.md` Part 7. Distribution (design §7): **4 answerable · 2 exact_match · 2 chunking_stress · 2 multi_document · 2 ambiguous · 3 unanswerable** (= 15). Annotate against the **pinned** corpus only (the two SHAs in `EVAL_CORPUS`).

**Pre-determined by the design review — apply these, do NOT rediscover (they will cost annotation time otherwise):**
- **Q8 (ShopEasy KB) is DEAD — do not annotate it.** Its `shopeasy_knowledge_base.json` lives in the **excluded** `Mastering-Agentic-AI-Week2` repo (the firewalled sample solution), so it is inaccessible by rule. **Replace that chunking-stress slot** with a *split-table* question over the `awesome-agentic-ai-resources` README catalog — one whose answer straddles a Markdown table-row boundary (the chunk-boundary stress the slot exists to test). Ref: `design.md` §7; `design-review.md` Part 7 Q8.
- **Q3 ("Attention Is All You Need") and Q6 ("QLoRA") fail the substance gate as written** — the README only *catalogs* (links) them, it does not *describe* them. The design review already determined this. **Either reword each to "which resource covers X?"** (answerable from the catalog text) **or move it to the unanswerable bucket** — do not leave them as "what does X cover?" Ref: `design.md` §7 annotation gate.

**Apply the §7 annotation gate to every remaining answerable question** — substance must be *in* the corpus text, not merely *catalogued*:
- For each answerable/exact_match/chunking_stress/multi_document question, open the pinned file and confirm the answer's substance is present. A catalog **link** appearing only as a table row is **not** answerable content. If the README only lists a resource, either **reword to what the catalog states** ("which resource covers X?") or **move it to the unanswerable bucket**.
- Confirm the 2 multi_document questions truly span **two docs** (README + the Week-1 notebook — the only two substantive docs), not two sections of one README. Otherwise the multi-doc claim is hollow.
- The 3 unanswerable: one about something the corpus never mentions (the catalog spans Weeks 1–7 → "Week 8" works), one related-but-not-covered, one adversarially close to corpus terms but absent.
- Record exact `line_start/line_end` (or `section`) and the matching `commit_hash` for every gold span.
- Re-balance back to **4+2+2+2+2+3** after the gate moves any questions.

Write the result to `src/genacademy_rag/eval/gold/gold_set.yaml`. Then run `uv run python -c "from genacademy_rag.eval.gold_schema import load_gold_set; g=load_gold_set('src/genacademy_rag/eval/gold/gold_set.yaml'); print(len(g), 'questions OK')"` and confirm it prints `15 questions OK`.

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check src tests
git add src/genacademy_rag/eval/__init__.py src/genacademy_rag/eval/gold_schema.py \
        src/genacademy_rag/eval/gold/gold_set.yaml tests/eval
git commit -m "feat(eval): gold-set schema + 15 annotated questions (annotation gate applied)"
```

---

## Task 13: Retrieval eval — recall@k, precision@k, MRR (THE PROTECTED ARTIFACT)

**This is the handout-graded spine — never cut.** It is deterministic, no LLM. Eval green here = the Day-2 gate met.

**Files:**
- Create: `src/genacademy_rag/eval/retrieval_eval.py`
- Create: `tests/eval/test_retrieval_eval.py`
- Create: `scripts/eval_retrieval.py` (the Day-2 gate runner)

- [ ] **Step 1: Write the failing test** `tests/eval/test_retrieval_eval.py`

```python
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan
from genacademy_rag.eval.retrieval_eval import score_question, chunk_matches_span, aggregate
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk


def _rc(line_start, line_end, commit="abc123", text="x"):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="awesome-agentic-ai-resources", file_path="README.md",
                   commit_hash=commit, line_start=line_start, line_end=line_end)
    return RetrievedChunk(chunk=Chunk(chunk_id=f"d1::{line_start}", doc_id="d1",
                                      ordinal=line_start, text=text, citation=cit), score=1.0)


def test_chunk_matches_span_requires_overlap_and_commit():
    span = GoldSpan(repo="awesome-agentic-ai-resources", file_path="README.md",
                    commit_hash="abc123", line_start=10, line_end=20)
    assert chunk_matches_span(_rc(8, 15).chunk, span)            # overlaps 10-15
    assert not chunk_matches_span(_rc(30, 40).chunk, span)       # no overlap
    assert not chunk_matches_span(_rc(8, 15, commit="WRONG").chunk, span)  # commit mismatch -> no leak


def test_recall_precision_mrr_on_a_hit_at_rank_2():
    q = GoldQuestion(id="q1", question="x", category="answerable", answerable=True,
                     gold=[GoldSpan("awesome-agentic-ai-resources", "README.md", "abc123", 10, 20)])
    retrieved = [_rc(30, 40), _rc(12, 18), _rc(50, 60)]   # gold is at rank 2
    s = score_question(q, retrieved, k=5)
    assert s["recall"] == 1.0
    assert s["precision"] == 1 / 5           # 1 relevant of k=5 slots
    assert s["mrr"] == 1 / 2                 # first relevant at rank 2


def test_unanswerable_question_excluded_from_retrieval_metrics():
    q = GoldQuestion(id="q2", question="x", category="unanswerable", answerable=False, gold=[])
    s = score_question(q, [_rc(1, 2)], k=5)
    assert s["recall"] is None               # not a retrieval-scored question
    agg = aggregate([s])
    assert agg["n_retrieval_questions"] == 0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/eval/test_retrieval_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.eval.retrieval_eval`.

- [ ] **Step 3: Implement `src/genacademy_rag/eval/retrieval_eval.py`**

```python
"""Deterministic retrieval eval (handout-mandatory). recall@k / precision@k / MRR over gold spans.
A retrieved chunk counts only if it overlaps a gold span AND its commit_hash matches the gold's —
so production content (different/missing commit_hash) can never satisfy a gold marker."""
from __future__ import annotations

from genacademy_rag.core.types import Chunk
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan


def chunk_matches_span(chunk: Chunk, span: GoldSpan) -> bool:
    c = chunk.citation
    if c.repo != span.repo or c.file_path != span.file_path:
        return False
    if c.commit_hash != span.commit_hash:   # provenance gate: no production leak into gold
        return False
    if span.section is not None:
        return (c.page_or_section or "") == span.section
    if span.line_start is None or c.line_start is None:
        return True
    return not (c.line_end < span.line_start or c.line_start > span.line_end)  # overlap


def score_question(q: GoldQuestion, retrieved, k: int) -> dict:
    if not q.answerable:
        return {"id": q.id, "category": q.category, "recall": None, "precision": None, "mrr": None}
    topk = retrieved[:k]
    hits = [any(chunk_matches_span(r.chunk, s) for s in q.gold) for r in topk]
    gold_found = sum(1 for s in q.gold if any(chunk_matches_span(r.chunk, s) for r in topk))
    recall = gold_found / len(q.gold) if q.gold else 0.0
    precision = sum(hits) / k
    first = next((i for i, h in enumerate(hits) if h), None)
    mrr = 1.0 / (first + 1) if first is not None else 0.0
    return {"id": q.id, "category": q.category, "recall": recall, "precision": precision, "mrr": mrr}


def aggregate(scores: list[dict]) -> dict:
    retr = [s for s in scores if s["recall"] is not None]
    n = len(retr)
    mean = lambda key: (sum(s[key] for s in retr) / n) if n else 0.0
    return {"n_retrieval_questions": n, "recall@k": mean("recall"),
            "precision@k": mean("precision"), "mrr": mean("mrr")}
```

- [ ] **Step 4: Run tests, verify pass + lint**

Run: `uv run pytest tests/eval/test_retrieval_eval.py -v && uv run ruff check src tests`
Expected: 3 passed; ruff clean.

- [ ] **Step 5: Write `scripts/eval_retrieval.py`** — the **Day-2 gate runner** (retrieval only, NO LLM)

This makes "eval green by Day 2" verifiable *now*, before the full report/faithfulness runner (Task 15). It needs only the ingested Chroma collection + the gold set — no generation key.

```python
"""Day-2 gate: deterministic retrieval eval over the ingested pinned corpus. Prints recall@k /
precision@k / MRR. No LLM, no generation key. (Full report + faithfulness = scripts/run_eval.py.)"""
from genacademy_rag.config import Settings
from genacademy_rag.core.providers import STEmbedder
from genacademy_rag.core.retriever import HybridRetriever
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.eval.gold_schema import load_gold_set
from genacademy_rag.eval.retrieval_eval import score_question, aggregate

GOLD = "src/genacademy_rag/eval/gold/gold_set.yaml"


def main():
    s = Settings.from_env()
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()
    # Embeddings only — no generate() — so this runs with zero provider key.
    embedder = STEmbedder(s.embed_model)
    retriever = HybridRetriever(store=store, provider=embedder, all_chunks=chunks, top_k=s.top_k)
    scores = [score_question(q, retriever.retrieve(q.question), k=s.top_k)
              for q in load_gold_set(GOLD)]
    agg = aggregate(scores)
    print(f"RETRIEVAL EVAL  recall@k={agg['recall@k']:.2f}  precision@k={agg['precision@k']:.2f}  "
          f"mrr={agg['mrr']:.2f}  (n={agg['n_retrieval_questions']})")
    for row in scores:
        if row["recall"] is not None:
            print(f"  {row['id']:<5} {row['category']:<16} recall={row['recall']:.2f} mrr={row['mrr']:.2f}")


if __name__ == "__main__":
    main()
```

`STEmbedder` satisfies the `ModelProvider.embed` half the retriever needs; the retriever never calls `generate()`, so no key is required.

- [ ] **Step 6: Run the Day-2 gate** (needs a prior `ingest_eval_corpus.py` run + the annotated gold set)

Run: `uv run python scripts/eval_retrieval.py`
Expected: prints a `RETRIEVAL EVAL recall@k=… precision@k=… mrr=…` line + per-question rows. **This is the Day-2 "eval green" check** — it gates UI work and needs no generation key. Investigate any `recall < 1` row before proceeding (likely `RetrievalRecallFailure`/`TopKTooSmall`/`ChunkingBoundary`).

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check src scripts tests
git add src/genacademy_rag/eval/retrieval_eval.py tests/eval/test_retrieval_eval.py scripts/eval_retrieval.py
git commit -m "feat(eval): deterministic retrieval eval + Day-2 gate runner (recall/precision/MRR, commit-hash provenance gate)"
```

---

## Task 14: Faithfulness eval — LLM-judge + citation-grounding fallback (CUTTABLE)

Cut order: if the Nebius free tier throttles, drop the LLM-judge and ship the citation-grounding floor; the retrieval eval (Task 13) ships regardless. Structure so cutting the judge never breaks the report.

**Files:**
- Create: `src/genacademy_rag/eval/faithfulness_eval.py`
- Create: `tests/eval/test_faithfulness_eval.py`

- [ ] **Step 1: Write the failing test** `tests/eval/test_faithfulness_eval.py`

```python
from tests.conftest import FakeModelProvider
from genacademy_rag.eval.faithfulness_eval import (
    citation_grounding_score, llm_judge_score, FAITHFULNESS_JUDGE_SYSTEM)
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk


def _rc(text):
    cit = Citation(doc_id="d1", title="t", source_type="github")
    return RetrievedChunk(chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text=text, citation=cit),
                          score=1.0)


def test_citation_grounding_true_when_answer_overlaps_chunks():
    retrieved = [_rc("RAG retrieves documents then generates an answer.")]
    assert citation_grounding_score("RAG retrieves documents", retrieved) is True


def test_citation_grounding_false_when_answer_is_fabricated():
    retrieved = [_rc("This text is entirely about cooking pasta.")]
    assert citation_grounding_score("The capital of France is Paris.", retrieved) is False


def test_llm_judge_parses_pinned_json():
    p = FakeModelProvider(canned_json='{"faithful": true, "hallucinated_claims": [], "score": 5}')
    out = llm_judge_score("q", "a", [_rc("ctx")], p)
    assert out["faithful"] is True and out["score"] == 5
    assert "ONLY" in FAITHFULNESS_JUDGE_SYSTEM  # verbatim pinned prompt present
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/eval/test_faithfulness_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.eval.faithfulness_eval`.

- [ ] **Step 3: Implement `src/genacademy_rag/eval/faithfulness_eval.py`**

```python
"""Faithfulness eval (depth add-on, cuttable). llm_judge_score = pinned-prompt LLM-as-judge at
temp 0 (raw outputs saved by the report). citation_grounding_score = the zero-LLM fallback that
always ships: do the answer's content words actually appear in the retrieved chunks?"""
from __future__ import annotations

import json
import re

from genacademy_rag.core.types import RetrievedChunk

FAITHFULNESS_JUDGE_SYSTEM = "You are a strict faithfulness judge. Reply ONLY with a JSON object."
FAITHFULNESS_JUDGE_USER = (
    "Question:\n{question}\n\nAnswer to judge:\n{answer}\n\nRetrieved context (ground truth):\n"
    "{context}\n\nIs every claim in the answer supported by the context? Return exactly "
    '{{"faithful": <true|false>, "hallucinated_claims": [<strings>], "score": <1-5 integer>}}.'
)

_STOP = {"the", "a", "an", "of", "to", "and", "is", "are", "in", "on", "for", "it", "that",
         "this", "with", "as", "be", "by", "or", "what", "which", "does", "do"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def citation_grounding_score(answer: str, retrieved: list[RetrievedChunk],
                             min_overlap: float = 0.6) -> bool:
    ans = _content_words(answer)
    if not ans:
        return True
    ctx = set()
    for r in retrieved:
        ctx |= _content_words(r.chunk.text)
    return len(ans & ctx) / len(ans) >= min_overlap


def llm_judge_score(question: str, answer: str, retrieved: list[RetrievedChunk], provider) -> dict:
    context = "\n---\n".join(r.chunk.text for r in retrieved)
    raw = provider.generate(
        [{"role": "system", "content": FAITHFULNESS_JUDGE_SYSTEM},
         {"role": "user", "content": FAITHFULNESS_JUDGE_USER.format(
             question=question, answer=answer, context=context)}],
        json_mode=True, max_tokens=256, temperature=0.0)
    parsed = json.loads(raw)
    return {"faithful": bool(parsed["faithful"]),
            "hallucinated_claims": parsed.get("hallucinated_claims", []),
            "score": int(parsed.get("score", 0)), "raw": raw}
```

- [ ] **Step 4: Mirror the verbatim judge prompt** into `specs/eval-judge/requirements.md` (tech-stack §5: judge prompt copied from source, kept in sync with code).

- [ ] **Step 5: Run tests, verify pass + lint**

Run: `uv run pytest tests/eval/test_faithfulness_eval.py -v && uv run ruff check src tests`
Expected: 3 passed; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/genacademy_rag/eval/faithfulness_eval.py tests/eval/test_faithfulness_eval.py specs/eval-judge/requirements.md
git commit -m "feat(eval): faithfulness — LLM-judge (pinned) + citation-grounding fallback"
```

---

## Task 15: Eval report generator (scores table + failure-analysis table → markdown)

**Files:**
- Create: `src/genacademy_rag/eval/report.py`
- Create: `tests/eval/test_report.py`
- Create: `scripts/run_eval.py` (the live end-to-end eval runner)

- [ ] **Step 1: Write the failing test** `tests/eval/test_report.py`

```python
from genacademy_rag.eval.report import render_report


def test_report_has_scores_table_and_failure_table():
    agg = {"n_retrieval_questions": 12, "recall@k": 0.83, "precision@k": 0.31, "mrr": 0.71}
    per_q = [
        {"id": "q1", "category": "answerable", "recall": 1.0, "precision": 0.2, "mrr": 1.0,
         "refused": False, "refusal_correct": True, "faithful": True},
        {"id": "q13", "category": "unanswerable", "recall": None, "precision": None, "mrr": None,
         "refused": True, "refusal_correct": True, "faithful": None},
    ]
    failures = [{"symptom": "missed gold chunk", "cause": "RetrievalRecallFailure",
                 "fix": "raise candidate_k / inspect BM25 tokenization", "qid": "q7"}]
    md = render_report(agg, per_q, failures, faithfulness_pct=0.92, judge_used=True)
    assert "recall@k" in md and "0.83" in md
    assert "Symptom" in md and "Cause" in md and "Fix" in md      # FIX column required
    assert "RetrievalRecallFailure" in md
    assert "refusal" in md.lower()
    assert "92" in md  # faithfulness %
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/eval/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.eval.report`.

- [ ] **Step 3: Implement `src/genacademy_rag/eval/report.py`**

```python
"""Render the eval report markdown: a scores table (retrieval columns ALWAYS; faithfulness % from
whichever scorer survived) + a failure-analysis table (Symptom → Cause[taxonomy] → Fix)."""
from __future__ import annotations

TAXONOMY = ["ChunkingBoundary", "RetrievalRecallFailure", "FaithfulnessHallucination",
            "RefusalFalsePositive", "RefusalFalseNegative", "TopKTooSmall"]


def render_report(agg: dict, per_q: list[dict], failures: list[dict],
                  *, faithfulness_pct: float | None, judge_used: bool) -> str:
    lines = ["# GenAcademy RAG — Evaluation Report", ""]
    lines += ["## Scores", "",
              "| Metric | Value |", "|---|---|",
              f"| Retrieval questions | {agg['n_retrieval_questions']} |",
              f"| recall@k | {agg['recall@k']:.2f} |",
              f"| precision@k | {agg['precision@k']:.2f} |",
              f"| MRR | {agg['mrr']:.2f} |"]
    refusal_correct = [q for q in per_q if "refusal_correct" in q]
    if refusal_correct:
        rate = sum(q["refusal_correct"] for q in refusal_correct) / len(refusal_correct)
        lines.append(f"| refusal correctness | {rate:.2f} |")
    if faithfulness_pct is not None:
        src = "LLM-judge" if judge_used else "citation-grounding fallback"
        lines.append(f"| faithfulness % ({src}) | {faithfulness_pct * 100:.0f}% |")
    lines += ["", "## Per-question", "",
              "| id | category | recall | precision | mrr | refused | faithful |",
              "|---|---|---|---|---|---|---|"]
    for q in per_q:
        fmt = lambda v: "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {q['id']} | {q['category']} | {fmt(q.get('recall'))} | "
                     f"{fmt(q.get('precision'))} | {fmt(q.get('mrr'))} | "
                     f"{q.get('refused', '—')} | {fmt(q.get('faithful'))} |")
    lines += ["", "## Failure analysis", "",
              "| Symptom | Cause | Fix | Question |", "|---|---|---|---|"]
    for f in failures:
        lines.append(f"| {f['symptom']} | {f['cause']} | {f['fix']} | {f.get('qid', '—')} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the unit test, verify pass**

Run: `uv run pytest tests/eval/test_report.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write `scripts/run_eval.py`** — the live end-to-end runner

```python
"""Run the 15-question eval over the ingested pinned corpus and write the report.
Retrieval eval always runs (protected). LLM-judge runs unless --no-judge or throttling forces
the citation-grounding fallback. Saves raw judge outputs to eval/runs/ for auditability."""
import argparse
import json
from pathlib import Path

from genacademy_rag.config import Settings
from genacademy_rag.core.providers import build_provider
from genacademy_rag.core.retriever import HybridRetriever
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.eval.gold_schema import load_gold_set
from genacademy_rag.eval.retrieval_eval import score_question, aggregate
from genacademy_rag.eval.faithfulness_eval import llm_judge_score, citation_grounding_score
from genacademy_rag.eval.report import render_report

GOLD = "src/genacademy_rag/eval/gold/gold_set.yaml"
REFUSAL = "I could not find this in the course materials."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    s = Settings.from_env()
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()                  # public accessor, not store._col
    retriever = HybridRetriever(store=store, provider=provider, all_chunks=chunks, top_k=s.top_k)
    qp = QueryPipeline(retriever=retriever, provider=provider)
    gold = load_gold_set(GOLD)

    runs_dir = Path("eval/runs"); runs_dir.mkdir(parents=True, exist_ok=True)
    per_q, faith_flags, judge_used = [], [], not args.no_judge
    for q in gold:
        retrieved = retriever.retrieve(q.question)
        row = score_question(q, retrieved, k=s.top_k)
        result = qp.answer(q.question)
        row["refused"] = result.refused
        row["refusal_correct"] = (result.refused != q.answerable)
        if q.answerable:
            if judge_used:
                try:
                    j = llm_judge_score(q.question, result.answer, retrieved, provider)
                    (runs_dir / f"judge_{q.id}.json").write_text(json.dumps(j, indent=2))
                    row["faithful"] = j["faithful"]
                except Exception:           # noqa: BLE001
                    # Deliberate: disable the judge run-wide on first failure (throttling/parse) and
                    # fall back to citation-grounding for ALL questions. A report mixing two
                    # faithfulness scorers is incoherent; one labeled scorer is the honest output.
                    judge_used = False
            if not judge_used:
                row["faithful"] = citation_grounding_score(result.answer, retrieved)
            faith_flags.append(bool(row["faithful"]))
        else:
            row["faithful"] = None
        per_q.append(row)

    agg = aggregate(per_q)
    faith_pct = (sum(faith_flags) / len(faith_flags)) if faith_flags else None
    failures = []  # fill from per_q rows where recall<1 / refusal_correct is False during analysis
    md = render_report(agg, per_q, failures, faithfulness_pct=faith_pct, judge_used=judge_used)
    out = Path("eval/REPORT.md"); out.write_text(md)
    print(f"wrote {out} | recall@k={agg['recall@k']:.2f} precision@k={agg['precision@k']:.2f} "
          f"mrr={agg['mrr']:.2f} judge_used={judge_used}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the live eval** (this is the Day-2 gate — needs a generation key + a prior `ingest_eval_corpus.py` run)

Run: `uv run python scripts/run_eval.py`
Expected: writes `eval/REPORT.md`, prints a line with `recall@k=… precision@k=… mrr=…`. **This is "eval green."** Inspect `eval/REPORT.md`: scores table present, 15 per-question rows, refusal correctness on the 3 unanswerables, and (if judge ran) a faithfulness %. Then hand-fill the failure-analysis table by analysing rows with `recall < 1` or `refusal_correct = False`, tagging each with a `TAXONOMY` cause and a concrete FIX.

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check src scripts tests
git add src/genacademy_rag/eval/report.py tests/eval/test_report.py scripts/run_eval.py eval/REPORT.md
git commit -m "feat(eval): report generator + live eval runner (retrieval green, judge-or-fallback)"
```

> **GATE: the Day-2 hard rule is the *retrieval* eval green — `scripts/eval_retrieval.py` (Task 13 Step 6) printing recall/precision/MRR with no generation key. That must pass before any Task 16 UI work. `eval/REPORT.md` (this task) adds refusal correctness + faithfulness on top.**

---

## Task 16: Web view — session auth + non-streaming chat UI with source cards

**Files:**
- Create: `src/genacademy_rag/web/__init__.py`, `auth.py`, `app.py`, `templates/login.html`, `templates/chat.html`
- Create: `tests/web/__init__.py`, `tests/web/test_app.py`

- [ ] **Step 1: Write the failing test** `tests/web/test_app.py`

```python
from starlette.testclient import TestClient


def _client(monkeypatch, tmp_path, refused=False):
    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    from genacademy_rag.web.app import create_app
    from genacademy_rag.data.datastore import SQLiteDatastore
    from tests.conftest import FakeModelProvider

    class _Retriever:
        def retrieve(self, q):
            from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
            cit = Citation(doc_id="d1", title="README.md", source_type="github",
                           repo="r", file_path="README.md", commit_hash="abc123",
                           line_start=1, line_end=2)
            return [RetrievedChunk(chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0,
                                               text="RAG retrieves then generates.", citation=cit), score=0.8)]

    canned = '{"answerable": false, "confidence": 1}' if refused else '{"answerable": true, "confidence": 5}'
    provider = FakeModelProvider(canned_json=canned, canned_answer="RAG retrieves then generates.")
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
    assert "README.md" in r.text                      # source card rendered
    assert "details" in r.text.lower()


def test_refusal_is_rendered_not_an_answer(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, refused=True)
    c.post("/login", data={"email": "member@genacademy.local", "password": "member"})
    r = c.post("/ask", data={"question": "who won the 2050 world cup?"})
    assert "could not find" in r.text.lower()
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/web/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.web.app`.

- [ ] **Step 3: Implement `src/genacademy_rag/web/auth.py`**

```python
"""Thin session auth. Phase 0: 2 seeded users, plaintext dev passwords, session cookie holds email.
RBAC + hashing + invite-code = Phase 1."""
from __future__ import annotations

from genacademy_rag.data.datastore import SQLiteDatastore


def authenticate(datastore: SQLiteDatastore, email: str, password: str) -> dict | None:
    user = datastore.get_user_by_email(email)
    if user and user["password"] == password:
        return user
    return None
```

- [ ] **Step 4: Implement `src/genacademy_rag/web/app.py`** (the only HTTP/template layer)

```python
"""Thin FastAPI view. ALL RAG logic is injected (QueryPipeline); no core logic here. Non-streaming
form-post (HTMX-ready). create_app(retriever, provider) lets tests inject fakes; the real wiring
(local embed + provider preset + ingested Chroma) happens in __main__."""
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
    datastore.seed_users()                       # injected, not constructed here (pluggability seam)
    qp = QueryPipeline(retriever=retriever, provider=provider)

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

    def current_user(request: Request) -> str | None:
        return request.session.get("email")

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        return TEMPLATES.TemplateResponse("login.html", {"request": request, "error": None})

    @app.post("/login")
    def login(request: Request, email: str = Form(...), password: str = Form(...)):
        user = authenticate(datastore, email, password)
        if not user:
            return TEMPLATES.TemplateResponse("login.html",
                                              {"request": request, "error": "Invalid credentials"}, status_code=401)
        request.session["email"] = user["email"]
        request.session["role"] = user["role"]
        return RedirectResponse("/", status_code=303)  # PRG: force GET after POST

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        if not current_user(request):
            return RedirectResponse("/login", status_code=302)
        return TEMPLATES.TemplateResponse("chat.html", {"request": request, "result": None, "question": None})

    @app.post("/ask", response_class=HTMLResponse)
    def ask(request: Request, question: str = Form(...)):
        if not current_user(request):
            return RedirectResponse("/login", status_code=303)
        result = qp.answer(question)
        return TEMPLATES.TemplateResponse("chat.html",
                                          {"request": request, "result": result, "question": question})

    return app
```

- [ ] **Step 5: Implement the templates**

`src/genacademy_rag/web/templates/login.html`:
```html
<!doctype html><html><head><meta charset="utf-8"><title>GenAcademy RAG — Login</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center">
<form method="post" action="/login" class="bg-white p-8 rounded-xl shadow w-80 space-y-4">
  <h1 class="text-xl font-semibold">GenAcademy RAG</h1>
  {% if error %}<p class="text-red-600 text-sm">{{ error }}</p>{% endif %}
  <input name="email" placeholder="email" class="w-full border rounded px-3 py-2" value="member@genacademy.local">
  <input name="password" type="password" placeholder="password" class="w-full border rounded px-3 py-2">
  <button class="w-full bg-slate-900 text-white rounded py-2">Sign in</button>
</form></body></html>
```

`src/genacademy_rag/web/templates/chat.html`:
```html
<!doctype html><html><head><meta charset="utf-8"><title>GenAcademy RAG</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/htmx.org@1.9.12"></script></head>
<body class="bg-slate-50 min-h-screen">
<div class="max-w-2xl mx-auto p-6 space-y-4">
  <h1 class="text-2xl font-semibold">Ask the cohort materials</h1>
  <form method="post" action="/ask" class="flex gap-2">
    <input name="question" value="{{ question or '' }}" placeholder="What did the course say about…?"
           class="flex-1 border rounded px-3 py-2">
    <button class="bg-slate-900 text-white rounded px-4">Ask</button>
  </form>
  {% if result %}
  <div class="bg-white rounded-xl shadow p-5 space-y-3">
    {% if result.refused %}
      <p class="text-amber-700 font-medium">{{ result.answer }}</p>
    {% else %}
      <p class="whitespace-pre-wrap">{{ result.answer }}</p>
      <div class="pt-2 border-t">
        <p class="text-xs uppercase tracking-wide text-slate-500 mb-1">Sources</p>
        {% for c in result.citations %}
        <details class="text-sm mb-1">
          <summary class="cursor-pointer">{{ c.title }}{% if c.line_start %} (lines {{ c.line_start }}–{{ c.line_end }}){% endif %}</summary>
          <div class="text-slate-600 pl-4">{{ c.repo }}/{{ c.file_path }} @ {{ c.commit_hash[:7] if c.commit_hash }}</div>
        </details>
        {% endfor %}
      </div>
    {% endif %}
  </div>
  {% endif %}
</div></body></html>
```

- [ ] **Step 6: Run tests, verify pass + lint**

Run: `uv run pytest tests/web/test_app.py -v && uv run ruff check src tests`
Expected: 3 passed; ruff clean. (Uses Starlette's `TestClient`; ensure `httpx` is available — `uv add --dev httpx` if `TestClient` import fails.)

- [ ] **Step 7: Add the real wiring entry point** to `src/genacademy_rag/web/app.py`

Append:
```python
def build_default_app() -> FastAPI:
    """Real wiring: local embed + active provider preset + the ingested eval Chroma collection."""
    from genacademy_rag.core.providers import build_provider
    from genacademy_rag.core.retriever import HybridRetriever
    from genacademy_rag.core.vectorstore import ChromaStore
    s = Settings.from_env()
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()                  # public accessor, not store._col
    retriever = HybridRetriever(store=store, provider=provider, all_chunks=chunks, top_k=s.top_k)
    datastore = SQLiteDatastore(s.sqlite_path)
    return create_app(retriever=retriever, provider=provider, datastore=datastore)
```

- [ ] **Step 8: Live run — demonstrate the bot** (evidence before done, AGENTS.md §6)

Run: `uv run uvicorn "genacademy_rag.web.app:build_default_app" --factory --port 8000`
Then in a browser: log in as `member@genacademy.local` / `member`, ask a known-answerable course question → cited answer with a `<details>` source card; ask an unanswerable one → the refusal message. Capture a screenshot of each for the demo/write-up.

- [ ] **Step 9: Commit**

```bash
git add src/genacademy_rag/web tests/web
git commit -m "feat(web): session auth + non-streaming chat UI with citation source cards"
```

---

## Task 17 (SHOULD — only after eval green): PDF loader + minimal admin upload endpoint

Production-corpus content. Sequence **after** Task 16; never let it delay the graded spine. Eval ships on the pinned GitHub corpus regardless. Uses the corrected path `../../CuratedRAGMaterials/`.

**Files:**
- Create: `src/genacademy_rag/core/loaders/pdf_loader.py`
- Create: `tests/core/test_pdf_loader.py`
- Modify: `src/genacademy_rag/core/chunker.py` (derive `page_or_section` from form-feed page breaks)
- Modify: `tests/core/test_chunker.py` (add the page-citation test)
- Modify: `src/genacademy_rag/web/app.py` (admin-only `/upload`: persist file, ingest, **rebuild retriever**)

- [ ] **Step 1: Write the failing test** `tests/core/test_pdf_loader.py`

```python
from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes


def test_pdf_loader_extracts_text_with_page_citations(tmp_path):
    # Build a tiny 1-page PDF on the fly so the test needs no fixture file.
    from pypdf import PdfWriter
    import io
    w = PdfWriter(); w.add_blank_page(width=200, height=200)
    buf = io.BytesIO(); w.write(buf)
    doc = load_pdf_bytes(filename="g.pdf", raw_bytes=buf.getvalue(), uploaded_by="admin@genacademy.local")
    assert doc.source_type == "pdf"
    assert doc.filename == "g.pdf"
    assert doc.title == "g.pdf"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/core/test_pdf_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: genacademy_rag.core.loaders.pdf_loader`.

- [ ] **Step 3: Implement `src/genacademy_rag/core/loaders/pdf_loader.py`**

```python
"""PDF → Document (production corpus). pypdf only — spike confirmed the guidebook needs no OCR
(printable ratio 0.994). Per-page text concatenated; page boundaries kept for page citations via
a form-feed marker the chunker preserves in char spans."""
from __future__ import annotations

import hashlib

from pypdf import PdfReader

from genacademy_rag.core.types import Document


def load_pdf_bytes(*, filename: str, raw_bytes: bytes, uploaded_by: str | None = None) -> Document:
    import io
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\f\n".join(pages)  # form-feed separates pages
    doc_id = "pdf/" + hashlib.sha256(raw_bytes).hexdigest()[:12]
    return Document(doc_id=doc_id, title=filename, source_type="pdf", text=text,
                    filename=filename)
```

- [ ] **Step 4: Run the unit test, verify pass**

Run: `uv run pytest tests/core/test_pdf_loader.py -v`
Expected: 1 passed.

- [ ] **Step 5: Page citations for PDFs (B5)** — extend `FixedSizeChunker` to derive `page_or_section` from the form-feed page breaks `load_pdf_bytes` inserts. Add this test to `tests/core/test_chunker.py`:

```python
def test_pdf_pages_become_page_citations():
    text = "page one text\n\f\npage two text\n\f\npage three text"
    doc = Document(doc_id="g", title="g.pdf", source_type="pdf", text=text, filename="g.pdf")
    chunks = FixedSizeChunker(chunk_size=12, overlap=0).chunk(doc)
    pages = {c.citation.page_or_section for c in chunks}
    assert "page 1" in pages and "page 3" in pages          # spans land on the right page
```

Then, in `FixedSizeChunker.chunk`, after the `line_at` map, add a page count and set `page_or_section` on the citation (general — no `source_type` branch; docs without `\f` get `None`):

```python
        has_pages = "\f" in text
        # ... inside the while-loop, when building `citation`, add:
                page_or_section=(f"page {text.count(chr(12), 0, start) + 1}" if has_pages else None),
```

Run: `uv run pytest tests/core/test_chunker.py -v` → all pass (the existing GitHub tests still pass; `page_or_section` stays `None` for them).

- [ ] **Step 6: Make uploads searchable without a restart (B3)** — add a `reindex` method to `HybridRetriever` (the running instance rebuilds its BM25 index + chunk map in place; the LangGraph node calls `retriever.retrieve()` each request, so the live pipeline picks it up). Add to `tests/core/test_retriever.py`:

```python
def test_reindex_makes_new_chunks_searchable(fake_provider):
    c0 = _chunk(0, "original chunk about embeddings")

    class _Store:
        def __init__(self): self.chunks = [c0]
        def query(self, qvec, top_k): return [(c.chunk_id, 0.5) for c in self.chunks][:top_k]
        def get_chunk(self, cid): return next(c for c in self.chunks if c.chunk_id == cid)

    store = _Store()
    retr = HybridRetriever(store=store, provider=fake_provider, all_chunks=[c0], top_k=5)
    new = _chunk(1, "uploaded chunk about Pinecone")
    store.chunks.append(new)
    retr.reindex(store.chunks)                               # rebuild BM25 + maps
    assert any("Pinecone" in r.chunk.text for r in retr.retrieve("Pinecone"))
```

Implement on `HybridRetriever`:
```python
    def reindex(self, all_chunks: list[Chunk]) -> None:
        self._chunks_by_id = {c.chunk_id: c for c in all_chunks}
        self._ids = [c.chunk_id for c in all_chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in all_chunks])
```
Run: `uv run pytest tests/core/test_retriever.py -v` → passes.

- [ ] **Step 7: Admin-only `/upload` — persist the file (S2), ingest into a `serving` collection (never `eval`), rebuild the retriever (B3)**

The eval collection (`collection="eval"`) stays pinned and pristine — the eval scripts read it, so uploads must NOT land there (or precision@k would be distorted by production chunks ranking into top-k; the commit_hash gate stops gold *scoring* leakage but not rank displacement). Serving therefore reads a separate `serving` collection seeded once from the eval chunks; uploads append to `serving`.

Add to `create_app` two optional params and the route (upload is enabled only when wiring is supplied, so the Task-16 tests — which don't pass it — are unaffected):
```python
def create_app(*, retriever, provider, datastore, ingest_upload=None, uploads_dir=None) -> FastAPI:
    ...
    @app.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...)):
        if request.session.get("role") != "admin" or ingest_upload is None:
            return RedirectResponse("/login", status_code=303)
        raw = await file.read()
        if uploads_dir is not None:                                  # S2: keep the original for re-index
            (uploads_dir / file.filename).write_bytes(raw)
        from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes
        doc = load_pdf_bytes(filename=file.filename, raw_bytes=raw,
                             uploaded_by=request.session.get("email"))
        ingest_upload(doc)                                           # ingest + reindex (B3)
        return RedirectResponse("/", status_code=303)
```
Add `from fastapi import File, UploadFile` to the imports (`python-multipart` is already a dep).

Then modify `build_default_app` to wire serving + uploads (replaces its eval-only retriever):
```python
    from genacademy_rag.core.chunker import FixedSizeChunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.config import DATA_DIR
    serving = ChromaStore(persist_dir=s.chroma_dir, collection="serving")
    if not serving.get_all_chunks():                       # seed once from the pinned eval chunks
        serving.upsert(chunks, provider.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store=serving, provider=provider,
                                all_chunks=serving.get_all_chunks(), top_k=s.top_k)
    pipe = IngestPipeline(chunker=FixedSizeChunker(s.chunk_size, s.chunk_overlap),
                          provider=provider, store=serving, datastore=datastore)
    uploads_dir = DATA_DIR / "uploads"; uploads_dir.mkdir(parents=True, exist_ok=True)

    def ingest_upload(doc):
        pipe.ingest([doc])
        retriever.reindex(serving.get_all_chunks())        # uploaded doc immediately searchable

    return create_app(retriever=retriever, provider=provider, datastore=datastore,
                      ingest_upload=ingest_upload, uploads_dir=uploads_dir)
```

- [ ] **Step 8: Add an upload integration test + verify eval stays pristine**

In `tests/web/test_app.py`, add a test that logs in as admin, POSTs a small in-memory PDF to `/upload` with a real `ingest_upload` closure wired (build a `serving` `ChromaStore` in `tmp_path`), then asks a question whose answer is only in the uploaded doc and asserts it is retrieved — and asserts the `eval` collection is untouched (`ChromaStore(collection="eval").get_all_chunks() == []` in the tmp dir). Run: `uv run pytest tests/web/test_app.py -v`.

- [ ] **Step 9: Lint + commit**

```bash
uv run ruff check src tests && uv run pytest -q
git add src/genacademy_rag/core/loaders/pdf_loader.py src/genacademy_rag/core/chunker.py \
        src/genacademy_rag/core/retriever.py src/genacademy_rag/web/app.py tests/
git commit -m "feat: PDF loader + admin upload (page citations, persisted files, live reindex, eval collection kept pristine)"
```

---

## Final Phase-0 verification

- [ ] **Full suite green:** `uv run pytest -q` → all unit tests pass. `uv run pytest -m integration -v` (with a key) → live provider test passes.
- [ ] **Lint clean:** `uv run ruff check src tests scripts`.
- [ ] **Eval green:** `eval/REPORT.md` exists with the scores table (recall@k, precision@k, MRR, refusal correctness) + the failure-analysis table with the FIX column.
- [ ] **The mandatory Nebius call:** once credit lands, set `GENACADEMY_PROVIDER=nebius` + `NEBIUS_MODEL`, re-run `scripts/run_eval.py` to confirm JSON-mode + latency hold on Nebius (one config line, not a re-architecture), and record it in the write-up.
- [ ] **Model-swap demo:** flip `GENACADEMY_PROVIDER` between `openrouter`/`openai`/`nebius` and show the same query answered — the graded "swap models/providers" moment, zero code change.
- [ ] **Update `docs/design.md` `## Changelog vs source`** with the Phase-0 divergences: char-based chunking (256-tok embed cap) vs the "~512/64 tok" prose; the corrected `../../CuratedRAGMaterials/` path.

---

## Forward pointer (Phases 1–2) — separate plans, written once Phase 0 is green

Do **not** build these ahead of a finished Phase 0 (`AGENTS.md` §5). They plug into the seams above:

- **Phase 1 (product layer):** real admin/member **RBAC** + **invite-code** signup (extends `web/auth.py` + the `users` table); **admin content management** UI (list/delete/re-index over the `documents` table + the `production` collection); **`usage_log` + dashboard** (new `UsageStore`, ungraded). Web-page loader joins the loader registry.
- **Phase 2 (depth & deploy, each an eval delta):** **PineconeStore** (second `VectorStore`; the "Chroma → Pinecone, one config line" demo — its tasks need the Phase-0 baseline numbers from `eval/REPORT.md` to report the before/after); **cross-encoder rerank** + **section-aware chunker** (each a `retrieval_eval` before/after); **Nebius embeddings** preset (second `ModelProvider.embed`); **deploy** Docker → HF Space + Postgres `Datastore` + auth hardening + live-URL smoke check.

Each Phase-2 item is reported as a before/after delta against the Phase-0 eval, so write its plan only after Task 15 produces baseline numbers.

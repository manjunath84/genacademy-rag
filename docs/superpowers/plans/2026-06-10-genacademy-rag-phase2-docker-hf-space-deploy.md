# GenAcademy RAG Phase 2 Docker Hugging Face Space Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing FastAPI RAG app reproducibly runnable as a Docker-based Hugging Face
Space with first-boot corpus seeding, HTTPS cookie hardening, and local/live smoke checks.

**Architecture:** Keep the app local-first and preserve pure core boundaries. Add a thin ASGI
entrypoint, deployment bootstrap code outside core, Docker/HF Space packaging, and a smoke script
that checks the HTTP surface without requiring model generation. Keep Postgres out of this slice;
it is a separate datastore implementation plan because it touches persistence semantics across users,
documents, usage logs, and chunk metadata.

**Tech Stack:** Python 3.12, uv, FastAPI, Starlette SessionMiddleware, Docker, Hugging Face Docker
Spaces (`sdk: docker`, `app_port: 7860`), Chroma, SQLite.

---

## Source Docs Checked

- Context7 `/huggingface/hub-docs`: Docker Space FastAPI example runs uvicorn on `0.0.0.0:7860`;
  Space metadata lives in README YAML; variables/secrets are configured through Space settings.
- Hugging Face docs connector: Docker Spaces use `sdk: docker`; `app_port` defaults to `7860` and can
  be set in README YAML.
- Context7 `/kludex/starlette`: `SessionMiddleware` supports `https_only=True`, `same_site`, `max_age`,
  and uses signed cookie sessions.

## Approval Gate

This plan is documentation only. Do not edit implementation code until the user approves this plan or
explicitly asks to execute it.

When approved, execute with `superpowers:executing-plans`:

1. Write each focused failing test first.
2. Run the focused test and confirm the expected failure.
3. Implement the smallest source change.
4. Run the focused test and confirm pass.
5. Commit after each task-sized working slice.

Stop after implementation evidence is collected. Do not self-approve or merge; a separate fresh
context reviews the diff.

## Scope

In scope:

- Dockerfile and `.dockerignore` for a CPU-only Docker Space.
- Hugging Face Space README metadata with `sdk: docker` and `app_port: 7860`.
- Thin ASGI module for `uvicorn genacademy_rag.web.main:app`.
- Runtime data directory override with `GENACADEMY_DATA_DIR`.
- First-boot pinned eval corpus bootstrap into the configured data directory.
- HTTPS-only session cookie flag via `GENACADEMY_SECURE_COOKIES`.
- HTTP smoke script for local container or live Space URL.
- Deployment runbook documenting required Space secrets and variables.

Out of scope:

- Postgres datastore preset.
- Multi-worker shared retriever index.
- Query streaming or UI polish.
- Changing retrieval, chunking, rerank, Pinecone, or Nebius embedding behavior.

## File Structure

- Create `src/genacademy_rag/web/main.py`: ASGI entrypoint for uvicorn and Docker.
- Modify `src/genacademy_rag/config.py`: data directory env helper and secure-cookie setting.
- Modify `src/genacademy_rag/web/app.py`: use secure-cookie setting and deploy data dir helper.
- Create `src/genacademy_rag/deploy/__init__.py`.
- Create `src/genacademy_rag/deploy/bootstrap.py`: first-boot eval corpus seeding helper.
- Create `tests/deploy/test_bootstrap.py`.
- Modify `tests/test_config.py`: env parsing for deploy data and secure cookies.
- Modify `tests/web/test_app.py`: session middleware receives secure-cookie config.
- Create `scripts/smoke_http.py`: local/live HTTP smoke probe.
- Create `tests/deploy/test_smoke_http.py`.
- Create `Dockerfile`.
- Create `.dockerignore`.
- Create `README.md`: Hugging Face Space metadata plus concise project entrypoint.
- Create `scripts/start_hf_space.sh`.
- Create `tests/deploy/test_deploy_files.py`.
- Create `docs/deploy.md`: operator runbook and secrets/variables table.
- Modify `.env.example`: deployment environment variables.

## Task 0: Branch And Baseline

**Files:** none

- [ ] **Step 1: Create the implementation branch**

Run:

```bash
git checkout main
git pull --ff-only
git checkout -b feat/genacademy-rag-phase2-docker-hf-space-deploy
```

Expected: branch is created from current `main`.

- [ ] **Step 2: Confirm clean working tree**

Run:

```bash
git status --short --branch
```

Expected:

```text
## feat/genacademy-rag-phase2-docker-hf-space-deploy
```

- [ ] **Step 3: Run baseline tests**

Run:

```bash
uv run pytest
```

Expected: existing tests pass before deploy edits.

- [ ] **Step 4: Run baseline lint**

Run:

```bash
uv run ruff check .
```

Expected: `All checks passed!`

No commit for Task 0.

## Task 1: ASGI Entrypoint

**Files:**

- Create: `src/genacademy_rag/web/main.py`
- Create: `tests/web/test_main_entrypoint.py`

- [ ] **Step 1: Add failing entrypoint test**

Create `tests/web/test_main_entrypoint.py`:

```python
def test_main_entrypoint_builds_default_app(monkeypatch):
    import sys

    import genacademy_rag.web.app as app_module

    built = object()
    monkeypatch.setattr(app_module, "build_default_app", lambda: built)

    import importlib
    import genacademy_rag.web.main as main_module

    main_module = importlib.reload(main_module)

    try:
        assert main_module.app is built
    finally:
        sys.modules.pop("genacademy_rag.web.main", None)
```

- [ ] **Step 2: Run focused test and verify failure**

Run:

```bash
uv run pytest tests/web/test_main_entrypoint.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `genacademy_rag.web.main`.

- [ ] **Step 3: Add ASGI entrypoint**

Create `src/genacademy_rag/web/main.py`:

```python
"""ASGI entrypoint for Docker and Hugging Face Spaces."""

from genacademy_rag.web.app import build_default_app

app = build_default_app()
```

- [ ] **Step 4: Run focused test**

Run:

```bash
uv run pytest tests/web/test_main_entrypoint.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/web/main.py tests/web/test_main_entrypoint.py
git commit -m "feat: add ASGI deploy entrypoint"
```

## Task 2: Deploy Data Directory And Secure Cookies

**Files:**

- Modify: `src/genacademy_rag/config.py`
- Modify: `src/genacademy_rag/web/app.py`
- Modify: `tests/test_config.py`
- Modify: `tests/web/test_app.py`

- [ ] **Step 1: Add failing config tests**

Append to `tests/test_config.py`:

```python
def test_deploy_data_dir_drives_default_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("GENACADEMY_DATA_DIR", str(tmp_path / "deploy-data"))
    monkeypatch.delenv("GENACADEMY_CHROMA_DIR", raising=False)
    monkeypatch.delenv("GENACADEMY_SQLITE", raising=False)

    s = Settings.from_env()

    assert s.chroma_dir == tmp_path / "deploy-data" / "chroma"
    assert s.sqlite_path == tmp_path / "deploy-data" / "genacademy.sqlite"


def test_secure_cookies_default_false_and_env_parse(monkeypatch):
    monkeypatch.delenv("GENACADEMY_SECURE_COOKIES", raising=False)
    assert Settings.from_env().secure_cookies is False

    monkeypatch.setenv("GENACADEMY_SECURE_COOKIES", "true")
    assert Settings.from_env().secure_cookies is True
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
uv run pytest tests/test_config.py::test_deploy_data_dir_drives_default_paths tests/test_config.py::test_secure_cookies_default_false_and_env_parse -q
```

Expected: FAIL because `secure_cookies` and `GENACADEMY_DATA_DIR` are not implemented.

- [ ] **Step 3: Implement config helper and secure-cookie setting**

Modify `src/genacademy_rag/config.py`.

Remove the unused `CURATED_MATERIALS_DIR` constant, then add below `DATA_DIR`:

```python
def data_dir_from_env() -> Path:
    return Path(os.environ.get("GENACADEMY_DATA_DIR", str(DATA_DIR)))
```

Add a trailing default field to `Settings`:

```python
    secure_cookies: bool = False
```

At the start of `Settings.from_env()`, after embedding validation, add:

```python
        data_dir = data_dir_from_env()
```

Change default path construction:

```python
            chroma_dir=Path(os.environ.get("GENACADEMY_CHROMA_DIR", str(data_dir / "chroma"))),
            sqlite_path=Path(
                os.environ.get("GENACADEMY_SQLITE", str(data_dir / "genacademy.sqlite"))
            ),
```

Add to the returned settings:

```python
            secure_cookies=_env_bool("GENACADEMY_SECURE_COOKIES", False),
```

- [ ] **Step 4: Add failing web middleware test**

Append to `tests/web/test_app.py`:

```python
def test_create_app_uses_secure_cookie_setting(monkeypatch, tmp_path):
    from genacademy_rag.config import Settings
    from genacademy_rag.data.datastore import SQLiteDatastore
    import genacademy_rag.web.app as app_module

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
```

- [ ] **Step 5: Run web middleware test and verify failure**

Run:

```bash
uv run pytest tests/web/test_app.py::test_create_app_uses_secure_cookie_setting -q
```

Expected: FAIL because `https_only` is not passed to `SessionMiddleware`.

- [ ] **Step 6: Wire secure cookies and deploy upload directory**

Modify `src/genacademy_rag/web/app.py`.

Change imports:

```python
from genacademy_rag.config import Settings, data_dir_from_env
```

Change session middleware setup:

```python
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        same_site="lax",
        https_only=settings.secure_cookies,
    )
```

Delete the inner `from genacademy_rag.config import DATA_DIR` import in `build_default_app()`, and
change the uploads directory:

```python
    uploads_dir = data_dir_from_env() / "uploads"
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_config.py tests/web/test_app.py::test_create_app_uses_secure_cookie_setting -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/genacademy_rag/config.py src/genacademy_rag/web/app.py tests/test_config.py tests/web/test_app.py
git commit -m "feat: add deploy data dir and secure cookies"
```

## Task 3: First-Boot Corpus Bootstrap

**Files:**

- Create: `src/genacademy_rag/deploy/__init__.py`
- Create: `src/genacademy_rag/deploy/bootstrap.py`
- Create: `tests/deploy/__init__.py`
- Create: `tests/deploy/test_bootstrap.py`

- [ ] **Step 1: Add failing bootstrap tests**

Create empty `tests/deploy/__init__.py`.

Create `tests/deploy/test_bootstrap.py`:

```python
from genacademy_rag.config import Settings


def _settings(tmp_path):
    return Settings(
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
        sqlite_path=tmp_path / "genacademy.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
    )


def test_bootstrap_skips_when_eval_collection_has_chunks(monkeypatch, tmp_path, capsys):
    import genacademy_rag.deploy.bootstrap as bootstrap

    settings = _settings(tmp_path)
    state = {"ingest_called": False}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["collection"] = collection

        def get_all_chunks(self):
            return [object()]

    monkeypatch.setattr(bootstrap.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(bootstrap, "ChromaStore", _Store)
    monkeypatch.setattr(
        bootstrap,
        "_run_ingest",
        lambda: state.__setitem__("ingest_called", True),
    )

    bootstrap.main([])

    assert state["collection"] == "eval"
    assert state["ingest_called"] is False
    assert "eval collection already seeded" in capsys.readouterr().out


def test_bootstrap_runs_ingest_when_eval_collection_empty(monkeypatch, tmp_path):
    import genacademy_rag.deploy.bootstrap as bootstrap

    settings = _settings(tmp_path)
    state = {"reset": None}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            pass

        def get_all_chunks(self):
            return []

    def fake_ingest(*, reset_collection):
        state["reset"] = reset_collection

    monkeypatch.setattr(bootstrap.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(bootstrap, "ChromaStore", _Store)
    monkeypatch.setattr(bootstrap, "_run_ingest", fake_ingest)

    bootstrap.main([])

    assert state["reset"] is False


def test_bootstrap_force_reseeds_existing_eval_collection(monkeypatch, tmp_path):
    import genacademy_rag.deploy.bootstrap as bootstrap

    settings = _settings(tmp_path)
    state = {"reset": None}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            pass

        def get_all_chunks(self):
            return [object()]

    monkeypatch.setattr(bootstrap.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(bootstrap, "ChromaStore", _Store)
    monkeypatch.setattr(
        bootstrap,
        "_run_ingest",
        lambda *, reset_collection: state.__setitem__("reset", reset_collection),
    )

    bootstrap.main(["--force"])

    assert state["reset"] is True


def test_run_ingest_uses_module_subprocess(monkeypatch):
    import genacademy_rag.deploy.bootstrap as bootstrap

    state = {}
    monkeypatch.setattr(bootstrap.sys, "executable", "/python")
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda cmd, cwd, check: state.update({"cmd": cmd, "cwd": cwd, "check": check}),
    )

    bootstrap._run_ingest(reset_collection=True)

    assert state["cmd"] == [
        "/python",
        "-m",
        "scripts.ingest_eval_corpus",
        "--chunker",
        "fixed",
        "--reset-collection",
    ]
    assert state["cwd"] == bootstrap.REPO_ROOT
    assert state["check"] is True
```

- [ ] **Step 2: Run bootstrap tests and verify failure**

Run:

```bash
uv run pytest tests/deploy/test_bootstrap.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `genacademy_rag.deploy`.

- [ ] **Step 3: Implement bootstrap module**

Create `src/genacademy_rag/deploy/__init__.py`:

```python
"""Deployment helpers for packaging and bootstrapping the app."""
```

Create `src/genacademy_rag/deploy/bootstrap.py`:

```python
"""Seed deploy data on first boot if the pinned eval corpus is absent."""

from __future__ import annotations

import argparse
import subprocess
import sys

from genacademy_rag.config import REPO_ROOT, Settings
from genacademy_rag.core.vectorstore import ChromaStore


def _run_ingest(*, reset_collection: bool = False) -> None:
    cmd = [sys.executable, "-m", "scripts.ingest_eval_corpus", "--chunker", "fixed"]
    if reset_collection:
        cmd.append("--reset-collection")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    store = ChromaStore(persist_dir=settings.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()
    if chunks and not args.force:
        print("deploy bootstrap: eval collection already seeded")
        return
    if chunks:
        print("deploy bootstrap: forcing eval collection re-seed")
    else:
        print("deploy bootstrap: seeding eval collection")
    _run_ingest(reset_collection=args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run bootstrap tests**

Run:

```bash
uv run pytest tests/deploy/test_bootstrap.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/deploy tests/deploy/__init__.py tests/deploy/test_bootstrap.py
git commit -m "feat: bootstrap deploy corpus data"
```

## Task 4: HTTP Smoke Script

**Files:**

- Create: `scripts/smoke_http.py`
- Create: `tests/deploy/test_smoke_http.py`

- [ ] **Step 1: Add failing smoke tests**

Create `tests/deploy/test_smoke_http.py`:

```python
import pytest
import requests


def test_smoke_http_checks_login_page(monkeypatch, capsys):
    import scripts.smoke_http as smoke_http

    class _Response:
        status_code = 200
        text = '<form action="/login">member@genacademy.local</form>'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(smoke_http.requests, "get", lambda url, timeout: _Response())

    smoke_http.main(["--base-url", "http://127.0.0.1:7860"])

    assert "HTTP SMOKE OK" in capsys.readouterr().out


def test_smoke_http_fails_when_login_marker_missing(monkeypatch):
    import scripts.smoke_http as smoke_http

    class _Response:
        status_code = 200
        text = "not the login page"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(smoke_http.requests, "get", lambda url, timeout: _Response())

    with pytest.raises(SystemExit) as exc:
        smoke_http.main(["--base-url", "http://127.0.0.1:7860"])

    assert "login marker not found" in str(exc.value)
```

- [ ] **Step 2: Run smoke tests and verify failure**

Run:

```bash
uv run pytest tests/deploy/test_smoke_http.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `scripts.smoke_http`.

- [ ] **Step 3: Implement smoke script**

Create `scripts/smoke_http.py`:

```python
"""Smoke-check a local container or live Hugging Face Space HTTP URL."""

from __future__ import annotations

import argparse

import requests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    response = requests.get(f"{base_url}/login", timeout=args.timeout)
    response.raise_for_status()
    if "member@genacademy.local" not in response.text or 'name="csrf_token"' not in response.text:
        raise SystemExit("login marker not found")
    print(f"HTTP SMOKE OK  base_url={base_url}")


if __name__ == "__main__":
    main()
```

Update the first smoke test response text to include the CSRF marker:

```python
text = '<form action="/login"><input name="csrf_token">member@genacademy.local</form>'
```

- [ ] **Step 4: Run smoke tests**

Run:

```bash
uv run pytest tests/deploy/test_smoke_http.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/smoke_http.py tests/deploy/test_smoke_http.py
git commit -m "feat: add HTTP deploy smoke check"
```

## Task 5: Docker And Hugging Face Space Files

**Files:**

- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `README.md`
- Create: `scripts/start_hf_space.sh`
- Create: `tests/deploy/test_deploy_files.py`

- [ ] **Step 1: Add failing deploy-file tests**

Create `tests/deploy/test_deploy_files.py`:

```python
from pathlib import Path


def test_dockerfile_runs_uvicorn_on_hf_space_port():
    dockerfile = Path("Dockerfile").read_text()

    assert "EXPOSE 7860" in dockerfile
    assert "GENACADEMY_DATA_DIR=/data" in dockerfile
    assert "HF_HOME=/app/.cache/huggingface" in dockerfile
    assert "SentenceTransformer('all-MiniLM-L6-v2')" in dockerfile
    assert "scripts/start_hf_space.sh" in dockerfile


def test_space_readme_declares_docker_sdk_and_port():
    readme = Path("README.md").read_text()

    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert "GenAcademy RAG" in readme


def test_dockerignore_excludes_local_state():
    dockerignore = Path(".dockerignore").read_text()

    assert ".env" in dockerignore
    assert ".venv" in dockerignore
    assert ".github/" in dockerignore
    assert "data/" in dockerignore
    assert "docs/" in dockerignore
    assert "eval/runs/" in dockerignore
    assert "tests/" in dockerignore
```

- [ ] **Step 2: Run deploy-file tests and verify failure**

Run:

```bash
uv run pytest tests/deploy/test_deploy_files.py -q
```

Expected: FAIL because the deploy files do not exist.

- [ ] **Step 3: Add Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    GENACADEMY_DATA_DIR=/data \
    GENACADEMY_SECURE_COOKIES=true \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN useradd -m -u 1000 user

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

RUN mkdir -p /app/.cache/huggingface /data && chown -R user:user /app /data
USER user

# Pre-download the offline embedding model so first boot does not depend on a model download.
RUN uv run --no-sync python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860

CMD ["bash", "scripts/start_hf_space.sh"]
```

- [ ] **Step 4: Add `.dockerignore`**

Create `.dockerignore`:

```text
.git/
.github/
.venv/
.pytest_cache/
.ruff_cache/
__pycache__/
*.pyc
.env
.env.*
data/
docs/
eval/runs/
models/
spike/
tests/
```

- [ ] **Step 5: Add HF Space README**

Create `README.md`:

```markdown
---
title: GenAcademy RAG
sdk: docker
app_port: 7860
---

# GenAcademy RAG

Knowledge assistant for Gen Academy cohort materials. It retrieves from a pinned corpus, answers with
citations, and refuses when the course materials do not support an answer.

## Local Run

```bash
uv run python scripts/ingest_eval_corpus.py
uv run uvicorn genacademy_rag.web.main:app --host 0.0.0.0 --port 7860
```

## Deploy

See `docs/deploy.md` for Hugging Face Space variables, secrets, first-boot corpus seeding, and smoke
checks.
```

- [ ] **Step 6: Add start script**

Create `scripts/start_hf_space.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${GENACADEMY_DATA_DIR:-/data}"
uv run --no-sync python -m genacademy_rag.deploy.bootstrap
exec uv run --no-sync uvicorn genacademy_rag.web.main:app --host 0.0.0.0 --port "${PORT:-7860}"
```

Run:

```bash
chmod +x scripts/start_hf_space.sh
```

- [ ] **Step 7: Run deploy-file tests**

Run:

```bash
uv run pytest tests/deploy/test_deploy_files.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add Dockerfile .dockerignore README.md scripts/start_hf_space.sh tests/deploy/test_deploy_files.py
git commit -m "feat: add docker space packaging"
```

## Task 6: Deploy Documentation And Env Example

**Files:**

- Create: `docs/deploy.md`
- Modify: `.env.example`

- [ ] **Step 1: Add deploy runbook**

Create `docs/deploy.md`:

```markdown
# Deployment Runbook

## Target

Docker-based Hugging Face Space serving `genacademy_rag.web.main:app` on port `7860`.

## Required Space Secrets

| Name | Purpose |
| --- | --- |
| `GENACADEMY_SESSION_SECRET` | Stable signed-session secret. Use a long random value. |
| `NEBIUS_API_KEY` | Generation key when `GENACADEMY_PROVIDER=nebius`. |
| `PINECONE_API_KEY` | Required only when `GENACADEMY_VECTORSTORE=pinecone`. |

## Recommended Space Variables

| Name | Value |
| --- | --- |
| `GENACADEMY_PROVIDER` | `nebius` for the mandatory-provider demo, or `openrouter` for dev fallback. |
| `NEBIUS_BASE_URL` | `https://api.studio.nebius.com/v1` |
| `NEBIUS_MODEL` | The validated generation model used for the demo. |
| `GENACADEMY_DATA_DIR` | `/data` |
| `GENACADEMY_SECURE_COOKIES` | `true` |
| `GENACADEMY_VECTORSTORE` | `chroma` for the first deploy slice. |
| `GENACADEMY_EMBEDDINGS` | `local` for deterministic first-boot corpus seeding. |

## First Boot

`scripts/start_hf_space.sh` runs `python -m genacademy_rag.deploy.bootstrap` before uvicorn. The
bootstrap checks the `eval` Chroma collection and runs `scripts/ingest_eval_corpus.py --chunker fixed`
only when the collection is empty.

The first boot fetches the pinned eval corpus from GitHub, so outbound HTTPS must be available. With
`set -euo pipefail` in `scripts/start_hf_space.sh`, the container exits if that fetch or ingest fails.

If a previous boot was killed during ingest, run:

```bash
uv run --no-sync python -m genacademy_rag.deploy.bootstrap --force
```

`--force` resets the `eval` Chroma collection before re-ingesting the pinned fixed-chunker corpus.

## Local Docker Smoke

```bash
docker build -t genacademy-rag .
docker run --rm -p 7860:7860 --env-file .env genacademy-rag
uv run python scripts/smoke_http.py --base-url http://127.0.0.1:7860
```

## Live Space Smoke

```bash
uv run python scripts/smoke_http.py --base-url https://<namespace>-<space>.hf.space
```

The HTTP smoke checks `/login` only. It proves the container booted, templates render, sessions are
initialized, and CSRF is present. It does not spend generation tokens.

## Local HTTP Login Testing

When testing browser login over plain HTTP, set `GENACADEMY_SECURE_COOKIES=false`. The Docker image
defaults this to `true` for the HTTPS Space, and browsers will not send `Secure` cookies over local
HTTP.

## Known Restrictions

- `/data` persists only when the Space has persistent storage attached. Without persistent storage,
  SQLite users, invites, usage logs, uploads, and the Chroma collection are lost on each restart; the
  bootstrap re-fetches and re-embeds the corpus on every cold boot.
- Keep uvicorn single-worker. The app holds an in-process retriever snapshot, and Chroma/SQLite are
  not a multi-process serving target in this slice.
- The offline embedding model is baked into the Docker image under `HF_HOME=/app/.cache/huggingface`.
  Rebuild the image when changing `GENACADEMY_EMBED_MODEL`.

## Postgres

Postgres is intentionally outside this Docker/HF Space slice. It needs a separate plan because the
current `SQLiteDatastore` owns users, documents, chunk metadata, invites, and usage logs.
```

- [ ] **Step 2: Update `.env.example`**

Append:

```dotenv
# Deployment / Hugging Face Space
# Set GENACADEMY_SESSION_SECRET as a Space secret, not a public variable.
# Prefer an absolute path; relative paths resolve from the process working directory.
GENACADEMY_DATA_DIR=./data
GENACADEMY_SECURE_COOKIES=false
```

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/deploy.md .env.example
git commit -m "docs: add deploy runbook"
```

## Task 7: Local Docker Verification

**Files:** none unless verification reveals a defect

- [ ] **Step 1: Run full Python verification**

Run:

```bash
uv run ruff check .
uv run pytest
```

Expected: lint passes and tests pass.

- [ ] **Step 2: Build Docker image**

Run:

```bash
docker build -t genacademy-rag .
```

Expected: image builds successfully, including the `SentenceTransformer('all-MiniLM-L6-v2')`
pre-download layer.

- [ ] **Step 3: Run local container**

Run in one terminal:

```bash
docker run --rm -p 7860:7860 --env-file .env genacademy-rag
```

Expected: bootstrap prints either `deploy bootstrap: seeding eval collection` on first boot or
`deploy bootstrap: eval collection already seeded` on later boots, then uvicorn starts on port `7860`.

- [ ] **Step 4: Run HTTP smoke**

Run in another terminal:

```bash
uv run python scripts/smoke_http.py --base-url http://127.0.0.1:7860
```

Expected:

```text
HTTP SMOKE OK  base_url=http://127.0.0.1:7860
```

- [ ] **Step 5: Confirm fixed eval baseline remains intact outside Docker**

Run:

```bash
GENACADEMY_RERANK_ENABLED=false GENACADEMY_CHUNKER=fixed uv run python scripts/eval_retrieval.py --collection eval
```

Expected:

```text
RETRIEVAL EVAL  recall@k=0.67  precision@k=0.22  mrr=0.55  (n=12)
```

## Task 8: Publish For Review

**Files:** all files touched by earlier tasks

- [ ] **Step 1: Check working tree**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
```

Expected: branch is ahead of `origin/main` with only deploy-slice commits.

- [ ] **Step 2: Push branch**

Run:

```bash
git push -u origin feat/genacademy-rag-phase2-docker-hf-space-deploy
```

- [ ] **Step 3: Open draft PR**

Run:

```bash
gh pr create --draft --base main --head feat/genacademy-rag-phase2-docker-hf-space-deploy \
  --title "feat: add Docker Hugging Face Space deploy slice" \
  --body "## Summary
- adds Docker/Hugging Face Space packaging for the FastAPI app
- bootstraps the pinned eval corpus on first deploy boot
- adds HTTPS session cookie config and HTTP smoke checks

## Verification
- uv run ruff check .
- uv run pytest
- docker build -t genacademy-rag .
- uv run python scripts/smoke_http.py --base-url http://127.0.0.1:7860
- fixed eval baseline remains recall@k=0.67 precision@k=0.22 mrr=0.55"
```

Expected: draft PR is open for fresh-context review.

## Self-Review Checklist

- Spec coverage: Docker Space packaging, ASGI entrypoint, deploy data directory, first-boot corpus
  bootstrap, secure cookies, HTTP smoke, deployment runbook.
- Scope control: no Postgres implementation, no retrieval changes, no provider changes, no UI redesign.
- Current-doc alignment: `sdk: docker`, `app_port: 7860`, uvicorn on `0.0.0.0:7860`, Space secrets via
  environment variables, Starlette `https_only=True`.
- Baseline preservation: eval collection remains fixed/local and the known retrieval line still
  reproduces after implementation.

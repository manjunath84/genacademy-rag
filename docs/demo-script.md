# GenAcademy RAG Demo Script

**Target length:** under 5 minutes
**Live app:** <https://Manjunath84-genacademy-rag.hf.space>
**Repo:** <https://github.com/manjunath84/genacademy-rag>

## Pre-Recording Checklist

Run this from the local repo:

```bash
cd /Users/manjunathans/projects/GenAcademy/Week2-RAG_ContextEngineering/genacademy-rag
uv run python scripts/smoke_http.py --base-url https://Manjunath84-genacademy-rag.hf.space
```

Expected:

```text
HTTP SMOKE OK  base_url=https://Manjunath84-genacademy-rag.hf.space
```

In Hugging Face Space settings, confirm:

```text
GENACADEMY_PROVIDER=nebius
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEBIUS_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
GENACADEMY_VECTORSTORE=pinecone
GENACADEMY_PINECONE_INDEX=genacademy-rag
GENACADEMY_EMBEDDINGS=local
GENACADEMY_EMBED_DIM=384
GENACADEMY_RERANK_ENABLED=true
GENACADEMY_RERANK_POOL=20
GENACADEMY_SECURE_COOKIES=true
```

Do not show secret values in the video.

## Recording Flow

### 0:00-0:30 - Open

Show the live Space URL in the browser.

Say:

> This is my GenAcademy Week 2 RAG project deployed as a Docker Hugging Face Space. It answers course-material questions with citations and refuses questions that are not supported by the corpus.

### 0:30-1:00 - Login

Open:

```text
https://Manjunath84-genacademy-rag.hf.space
```

Log in:

```text
email: member@genacademy.local
password: member
```

Say:

> The app has session login with seeded demo users, and the production path is deployed behind HTTPS with secure cookies enabled.

### 1:00-2:00 - Cited Answer

Ask:

```text
Which resource in the Gen Academy catalog covers chunking strategies for RAG, and what chunking types does it address?
```

Expected answer shape:

```text
The resource is Chunking Strategies for RAG by Weaviate.
It covers fixed, recursive, sentence, and semantic chunking, plus overlap trade-offs.
```

Point at the citations.

Say:

> The answer includes line-level provenance. The app stores citation metadata during ingest, so citations are a data model feature rather than a UI-only decoration.

### 2:00-2:40 - Refusal

Ask:

```text
How much does the Mastering Agentic AI certification cost?
```

Expected:

```text
I could not find this in the course materials.
```

Say:

> The refusal path is intentional. If retrieval does not support the answer, the graph refuses instead of guessing.

### 2:40-3:30 - Admin/Product Layer

Log out or open a new session if needed, then log in:

```text
email: admin@genacademy.local
password: admin
```

Briefly show the admin pages:

```text
/admin/invites
/admin/documents
/admin/dashboard
```

Say:

> Beyond the core Q&A flow, the project includes a small product layer: admin invites, document management, and usage analytics. These are separate from the deterministic eval corpus.

### 3:30-4:20 - Evaluation And Architecture

Show the repo files:

```text
eval/REPORT.md
src/genacademy_rag/core/
src/genacademy_rag/web/
src/genacademy_rag/deploy/
```

Say:

> The project separates pure retrieval and generation logic from the web layer. Retrieval quality is measured with a pinned 15-question eval set using recall, precision, MRR, and failure analysis. The live deploy uses Nebius for generation and Pinecone for the serving vector store through clean provider and vector-store seams.

### 4:20-4:50 - Close

Say:

> The main design choice was to be scale-aware without overbuilding: local embeddings, Pinecone for the live serving corpus, local Chroma for deterministic eval, and a Docker/Hugging Face deploy that preserves the same app path.

End by showing:

```text
https://Manjunath84-genacademy-rag.hf.space
```

## If Something Goes Wrong

- If the answer request errors, check the Hugging Face logs for `NEBIUS_API_KEY` or `NEBIUS_MODEL`.
- If startup fails before the app runs, check `PINECONE_API_KEY` and the Pinecone index dimension
  (`384` for the local embedder).
- If login fails after a restart, remember that `/data` is ephemeral without persistent storage; the seeded demo users should be recreated at boot.
- If the Space is waking from sleep, wait until it shows **Running**, then rerun the smoke command.
- If the demo is running long, skip the admin section and keep the cited-answer/refusal/eval sections.

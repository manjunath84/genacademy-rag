# Deployment Runbook

## Target

Docker-based Hugging Face Space serving `genacademy_rag.web.main:app` on port `7860`.

## Beginner Hugging Face Space Setup

Use this path for the first Week 2 course deployment, especially if you do not have a Hugging Face
Pro plan.

### 1. Create The Space

1. Go to <https://huggingface.co/spaces>.
2. Click **Create new Space**.
3. Fill the form:
   - **Space name:** `genacademy-rag`
   - **License:** any course-appropriate license, or leave the default if unsure
   - **SDK:** `Docker`
   - **Visibility:** `Public` if you need to submit a link; otherwise `Private`
   - **Hardware:** `CPU Basic` / free CPU
   - **Storage bucket / persistent storage:** none for the first deploy
   - **Dev Mode:** disabled
4. Click **Create Space**.

Dev Mode is for debugging inside the Space container and is not needed for a normal Git-based
deployment. Persistent storage is optional and may require a paid plan. Without persistent storage,
`/data` is wiped on restart; this app will re-fetch and re-ingest the eval corpus during cold boot,
which is slower but acceptable for the first deploy.

### 2. Add Secrets

In the Space page, go to **Settings -> Variables and secrets**.

Add these as **Secrets**:

```text
GENACADEMY_SESSION_SECRET=<generated-secret>
NEBIUS_API_KEY=<your-nebius-api-key>
```

Generate the session secret locally:

```bash
openssl rand -hex 32
```

Do not paste API keys or secrets into chat, docs, commits, or public issue/PR comments.

### 3. Add Variables

Add these as **Variables**, not secrets:

```text
GENACADEMY_PROVIDER=nebius
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1
NEBIUS_MODEL=<your-validated-nebius-model>
GENACADEMY_DATA_DIR=/data
GENACADEMY_SECURE_COOKIES=true
GENACADEMY_VECTORSTORE=chroma
GENACADEMY_EMBEDDINGS=local
GENACADEMY_RERANK_ENABLED=false
```

Use the same `NEBIUS_MODEL` you validated locally or were given for the course. Do not set
`GENACADEMY_VECTORSTORE=pinecone` for the first deployment; keep the first deploy simple with Chroma.

### 4. Push Code To The Space

From the local repo:

```bash
cd /Users/manjunathans/projects/GenAcademy/Week2-RAG_ContextEngineering/genacademy-rag
git checkout main
git pull --ff-only origin main
```

Add the Hugging Face Space as a remote:

```bash
git remote add hf https://huggingface.co/spaces/<your-hf-username>/genacademy-rag
```

Push `main` to the Space:

```bash
git push hf main:main
```

If this is a brand-new empty Space and Git rejects because the Space already has its own initial
README, use:

```bash
git push --force-with-lease hf main:main
```

Only use that force command for a fresh Space you just created.

### 5. Watch Build Logs

On the Hugging Face Space page, open the **Logs** tab.

Expected signs:

```text
deploy bootstrap: seeding eval collection
ingested 4 docs -> 53 chunks into /data/chroma collection=eval
Uvicorn running on http://0.0.0.0:7860
```

The first build can take a while because Docker installs dependencies and downloads the embedding
model.

### 6. Smoke Test

Once the Space says it is running, copy its app URL. It usually looks like:

```text
https://<your-hf-username>-genacademy-rag.hf.space
```

Then run locally:

```bash
uv run python scripts/smoke_http.py --base-url https://<your-hf-username>-genacademy-rag.hf.space
```

Expected:

```text
HTTP SMOKE OK  base_url=https://...
```

### 7. If It Fails

- `GENACADEMY_SESSION_SECRET` error: add the secret in Space settings and restart/rebuild.
- Nebius auth/model error: check `NEBIUS_API_KEY` and `NEBIUS_MODEL`.
- App keeps rebuilding slowly: normal for the first Docker build.
- Data disappears after restart: expected without persistent storage; the app re-seeds the eval
  corpus on boot.

After the smoke test passes, the next task is live login/query testing.

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
| `GENACADEMY_EMBEDDINGS` | `local`; first-boot eval corpus seeding refuses non-local embeddings. |

## First Boot

`scripts/start_hf_space.sh` runs `python -m genacademy_rag.deploy.bootstrap` before uvicorn. The
bootstrap checks the `eval` Chroma collection and runs `scripts/ingest_eval_corpus.py --chunker fixed`
only when the collection is empty.

The first boot fetches the pinned eval corpus from GitHub, so outbound HTTPS must be available. With
`set -euo pipefail` in `scripts/start_hf_space.sh`, the container exits if that fetch or ingest fails.

If a previous boot was killed during ingest, run this inside the Space/container shell:

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

Docker `--env-file` entries must be plain `KEY=value` lines. Keep comments on separate lines; Docker
does not strip inline comments after values.

Values in `--env-file .env` override Dockerfile defaults such as `GENACADEMY_DATA_DIR=/data` and
`GENACADEMY_SECURE_COOKIES=true`. For a local browser login over plain HTTP, override secure cookies
to `false`; for a Space-like local smoke, keep `GENACADEMY_DATA_DIR=/data`.

## Live Space Smoke

```bash
uv run python scripts/smoke_http.py --base-url https://<namespace>-<space>.hf.space
```

The HTTP smoke checks `/login` only. It proves the container booted, templates render, and the CSRF
render path works. It does not prove a cookie round-trip and does not spend generation tokens.

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
- The rerank model is not baked into this Docker image. Leave `GENACADEMY_RERANK_ENABLED=false`
  unless the rerank model is separately provisioned inside the image/cache.

## Postgres

Postgres is intentionally outside this Docker/HF Space slice. It needs a separate plan because the
current `SQLiteDatastore` owns users, documents, chunk metadata, invites, and usage logs.

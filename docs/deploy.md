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
deployment. Persistent storage / storage buckets are optional and may depend on account or product
availability. Without persistent storage, `/data` is wiped on restart; this app will re-fetch and
re-ingest the eval corpus during cold boot, which is slower but acceptable for the first deploy.

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
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
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
git remote add hf https://huggingface.co/spaces/Manjunath84/genacademy-rag
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

If the first push stalls before uploading objects, create a Hugging Face access token with write
permission and force Git to prompt in the terminal:

```bash
GIT_ASKPASS= SSH_ASKPASS= git push --verbose --progress --force-with-lease hf main:main
```

Use `Manjunath84` as the username and the Hugging Face token as the password. If
`--force-with-lease` reports `stale info` on the first sync and `git fetch hf main` fails with a
protocol error, it is acceptable to replace the brand-new Space README once:

```bash
GIT_ASKPASS= SSH_ASKPASS= git -c protocol.version=1 push --verbose --progress --force hf main:main
```

After the first successful Space sync, use normal pushes again.

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
https://Manjunath84-genacademy-rag.hf.space
```

Then run locally:

```bash
uv run python scripts/smoke_http.py --base-url https://Manjunath84-genacademy-rag.hf.space
```

Expected:

```text
HTTP SMOKE OK  base_url=https://...
```

Confirmed live smoke for this Space on 2026-06-10:

```bash
uv run python scripts/smoke_http.py --base-url https://Manjunath84-genacademy-rag.hf.space
```

```text
HTTP SMOKE OK  base_url=https://Manjunath84-genacademy-rag.hf.space
```

The documented Nebius Token Factory URL includes a trailing slash. With the pinned OpenAI SDK, that
base URL resolves chat completions as `https://api.tokenfactory.nebius.com/v1/chat/completions`.

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
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1/` |
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
uv run python scripts/smoke_http.py --base-url https://Manjunath84-genacademy-rag.hf.space
```

The HTTP smoke checks `/login` only. It proves the container booted, templates render, and the CSRF
render path works. It does not prove a cookie round-trip and does not spend generation tokens.

## Live End-To-End Test Checklist

Run these checks after the Space is **Running** and the HTTP smoke passes. Use the live URL:

```text
https://Manjunath84-genacademy-rag.hf.space
```

### 1. Member Login

1. Open the live URL in a browser.
2. Log in with the seeded member:

   ```text
   email: member@genacademy.local
   password: member
   ```

3. Expected: the chat screen loads with the heading **Ask the cohort materials**.

There is no logout button in this slice. To switch users, use an incognito/private browser window,
another browser profile, or clear the Space's cookies.

### 2. Cited Course-Material Answer

Ask:

```text
Which resource in the Gen Academy catalog covers chunking strategies for RAG, and what chunking types does it address?
```

Expected:

- The answer identifies Weaviate's **Chunking Strategies for RAG** resource.
- The answer mentions fixed, recursive, sentence, semantic chunking, and overlap trade-offs.
- The **Sources** section is visible and includes course-material citations.

This proves browser login, secure cookies, `/ask`, retrieval, generation, citations, and usage
logging all work on the live deployment.

### 3. Honest Refusal

Ask:

```text
How much does the Mastering Agentic AI certification cost?
```

Expected:

```text
I could not find this in the course materials.
```

This proves the refusal path still works instead of forcing an unsupported answer.

### 4. Admin Login And Pages

Open a private/incognito window and log in with the seeded admin:

```text
email: admin@genacademy.local
password: admin
```

Visit these paths:

```text
/admin/invites
/admin/documents
/admin/dashboard
```

Expected:

- `/admin/invites` shows invite generation controls.
- `/admin/documents` shows upload, re-index, and document table controls.
- `/admin/dashboard` shows usage summary and recent questions.

This proves admin RBAC and the product/admin layer are live.

### 5. Admin PDF Upload Becomes Searchable

1. Create a one-page PDF with a unique sentence, for example:

   ```text
   GenAcademy upload smoke test: the private demo keyword is saffron-orbit.
   ```

   On macOS, TextEdit -> New Document -> type the sentence -> File -> Export as PDF works.

2. In the admin session, open:

   ```text
   /admin/documents
   ```

3. Upload the PDF.
4. Expected: the document table shows the PDF with `source_type=pdf`, `status=indexed`, and a
   non-zero chunk count.
5. In a member session, ask:

   ```text
   What is the private demo keyword in the uploaded smoke test document?
   ```

6. Expected: the answer says `saffron-orbit` and cites the uploaded PDF.

This proves production uploads join the serving corpus without touching the deterministic eval
corpus.

### 6. Admin Dashboard Shows Usage

After the member queries above, refresh:

```text
/admin/dashboard
```

Expected:

- Total query count increased.
- Recent questions include the cited-answer/refusal/upload smoke-test questions.
- Refusal rate and latency cards render.

This proves `/ask` writes usage rows and the admin dashboard reads them.

### 7. Delete Uploaded PDF And Re-Test

1. In `/admin/documents`, delete the uploaded PDF.
2. Click **Re-index serving corpus**.
3. In the member session, ask again:

   ```text
   What is the private demo keyword in the uploaded smoke test document?
   ```

4. Expected: the app refuses or at least does not cite the deleted uploaded PDF.

This proves uploaded documents can be removed from the serving corpus.

### 8. Invite Signup

1. In `/admin/invites`, generate a member invite.
2. Copy the invite code when it is shown. It is shown once.
3. Open a private/incognito window and go to:

   ```text
   /signup
   ```

4. Sign up with a new test email, password, and the invite code.
5. Expected: signup redirects to the chat screen, and the new member can ask a course-material
   question.

This proves invite-gated user creation works.

### 9. Persistence Reminder

If no storage bucket / persistent storage is attached, these live-test side effects are temporary:

- uploaded PDFs
- invite codes
- new signed-up users
- usage rows
- Chroma collections under `/data`

The Space still works after restart because bootstrapping re-seeds the eval corpus, but admin-created
production state is lost.

## Why Hugging Face Spaces

We deployed this as a Docker Hugging Face Space because it is the smallest course-friendly path to a
public, reproducible ML demo:

- It gives a public HTTPS URL that is easy to submit and record in the demo video.
- It can run the same Docker image and FastAPI app used locally, with no special Gradio rewrite.
- It has built-in Git-based deploys, logs, variables, and secrets.
- It sits close to the ML ecosystem and handles the local embedding-model download/cache path cleanly.
- The free CPU path is enough for this Week 2 slice because generation is delegated to Nebius and
  embeddings are local/small.

This is not a hard architectural dependency. The app can be deployed on another Docker-capable host
because it is a normal FastAPI service started by `scripts/start_hf_space.sh` and configured by
environment variables.

Another host is reasonable later if you need stronger persistence, custom domains, autoscaling,
background workers, managed Postgres, or managed vector storage. A separate app platform would also
need its own decisions for HTTPS, secrets, a persistent `/data` equivalent or external database,
health checks, build cache, container startup command, and cost controls. For the Week 2 submission,
Hugging Face Spaces avoids that extra deployment scope while still proving the real app works online.

Relevant Hugging Face docs:

- Docker Spaces: <https://huggingface.co/docs/hub/spaces-sdks-docker>
- Spaces configuration reference: <https://huggingface.co/docs/hub/spaces-config-reference>
- Storage bucket / `/data` mounting example: <https://huggingface.co/docs/hub/spaces-sdks-docker-label-studio>

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

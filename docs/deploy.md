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

Docker `--env-file` entries must be plain `KEY=value` lines. Keep comments on separate lines; Docker
does not strip inline comments after values.

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

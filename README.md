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

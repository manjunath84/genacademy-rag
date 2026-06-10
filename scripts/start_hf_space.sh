#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${GENACADEMY_DATA_DIR:-/data}"
uv run --no-sync python -m genacademy_rag.deploy.bootstrap
exec uv run --no-sync uvicorn genacademy_rag.web.main:app --host 0.0.0.0 --port "${PORT:-7860}"

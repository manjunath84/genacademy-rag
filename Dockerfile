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

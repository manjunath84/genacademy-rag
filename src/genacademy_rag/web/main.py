"""ASGI entrypoint for Docker and Hugging Face Spaces."""

from genacademy_rag.web.app import build_default_app

app = build_default_app()

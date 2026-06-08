"""Markdown/text → Document. Raw text kept verbatim (tables matter for the catalog questions)."""
from __future__ import annotations

from genacademy_rag.core.types import Document


def load_markdown(*, repo: str, file_path: str, commit_hash: str, raw_text: str) -> Document:
    doc_id = f"{repo}/{file_path}@{commit_hash[:7]}"
    return Document(doc_id=doc_id, title=file_path.split("/")[-1], source_type="github",
                    text=raw_text, repo=repo, file_path=file_path, commit_hash=commit_hash)

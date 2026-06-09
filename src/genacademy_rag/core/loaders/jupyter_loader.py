"""Jupyter .ipynb → Document. Flattens markdown + code cells into text (nbformat parse)."""
from __future__ import annotations

import json
import warnings

import nbformat

from genacademy_rag.core.types import Document


def _normalize_nb_dict(nb_dict: dict) -> dict:
    """Add minimal required fields that nbformat.reads needs but minimal fixtures omit."""
    nb_dict.setdefault("metadata", {})
    for cell in nb_dict.get("cells", []):
        cell.setdefault("metadata", {})
        if cell.get("cell_type") == "code":
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)
    return nb_dict


def load_notebook(*, repo: str, file_path: str, commit_hash: str, raw_bytes: bytes) -> Document:
    nb_dict = _normalize_nb_dict(json.loads(raw_bytes.decode("utf-8")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = nbformat.reads(json.dumps(nb_dict), as_version=4)
    parts: list[str] = []
    for cell in nb.cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if cell.cell_type == "code":
            parts.append(f"```python\n{src}\n```")
        else:
            parts.append(src)
    doc_id = f"{repo}/{file_path}@{commit_hash[:7]}"
    return Document(
        doc_id=doc_id, title=file_path.split("/")[-1], source_type="github",
        text="\n\n".join(parts), repo=repo, file_path=file_path, commit_hash=commit_hash,
    )

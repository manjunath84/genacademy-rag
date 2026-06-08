import json

import pytest

from genacademy_rag.core.loaders import EVAL_CORPUS, assert_allowed
from genacademy_rag.core.loaders.jupyter_loader import load_notebook
from genacademy_rag.core.loaders.markdown_loader import load_markdown


def test_eval_corpus_pins_two_repos_to_exact_shas():
    repos = {r["repo"]: r["sha"] for r in EVAL_CORPUS}
    assert repos["awesome-agentic-ai-resources"] == "5dfb8691180dc4956107e86839998ba3a2ebd94f"
    assert repos["Mastering-Agentic-AI-Week1"] == "3aa31dfede8c76422be91f2ecdbc59eddc690b1d"


def test_week2_repo_is_firewalled_out():
    # The sample solution must never be fetchable. Allowlist enforcement, not convention.
    assert "Mastering-Agentic-AI-Week2" not in {r["repo"] for r in EVAL_CORPUS}
    with pytest.raises(ValueError, match="not in the eval allowlist"):
        assert_allowed("Mastering-Agentic-AI-Week2")


def test_markdown_loader_builds_document_with_provenance():
    doc = load_markdown(
        repo="awesome-agentic-ai-resources", file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        raw_text="# Title\n\n| Resource | Covers |\n|---|---|\n| QLoRA | finetuning |\n",
    )
    assert doc.source_type == "github"
    assert doc.title == "README.md"
    assert doc.commit_hash.startswith("5dfb869")
    assert "QLoRA" in doc.text


def test_jupyter_loader_keeps_markdown_and_code_cells():
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# Langchain Fundamentals\n"]},
        {"cell_type": "code", "source": ["from langchain import PromptTemplate\n"]},
    ], "nbformat": 4, "nbformat_minor": 5}
    doc = load_notebook(
        repo="Mastering-Agentic-AI-Week1",
        file_path="Langchain Basics/Langchain_Fundamentals.ipynb",
        commit_hash="3aa31dfede8c76422be91f2ecdbc59eddc690b1d",
        raw_bytes=json.dumps(nb).encode(),
    )
    assert "Langchain Fundamentals" in doc.text
    assert "PromptTemplate" in doc.text
    assert doc.source_type == "github"

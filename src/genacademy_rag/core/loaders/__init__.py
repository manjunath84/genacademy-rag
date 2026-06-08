"""Loader registry + the commit-pinned eval-corpus allowlist. The allowlist IS the Week-2
firewall: only these two repos+SHAs are ever fetched. Mastering-Agentic-AI-Week2 (the sample
solution) is absent by construction; reading it is disqualifying (AGENTS.md §5)."""
from __future__ import annotations

# Pinned SHAs from docs/spike-findings.md §4 (verified 2026-06-08).
EVAL_CORPUS: list[dict] = [
    {
        "repo": "awesome-agentic-ai-resources",
        "owner": "The-Gen-Academy",
        "sha": "5dfb8691180dc4956107e86839998ba3a2ebd94f",
        "files": [{"path": "README.md", "kind": "markdown"}],
    },
    {
        "repo": "Mastering-Agentic-AI-Week1",
        "owner": "The-Gen-Academy",
        "sha": "3aa31dfede8c76422be91f2ecdbc59eddc690b1d",
        "files": [
            {"path": "Langchain Basics/Langchain_Fundamentals.ipynb", "kind": "jupyter"},
            {"path": "Langchain Basics/README.md", "kind": "markdown"},
            {"path": "Langchain Basics/langchain_prompts.py", "kind": "markdown"},  # treat .py as text  # noqa: E501
        ],
    },
]

_ALLOWED = {r["repo"] for r in EVAL_CORPUS}


def assert_allowed(repo: str) -> None:
    if repo not in _ALLOWED:
        raise ValueError(f"repo {repo!r} is not in the eval allowlist {_ALLOWED} "
                         f"(Mastering-Agentic-AI-Week2 is the sample solution and is firewalled)")

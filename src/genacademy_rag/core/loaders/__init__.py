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
_ALLOWED_TRIPLES = {(r["owner"], r["repo"], r["sha"]) for r in EVAL_CORPUS}


def assert_allowed(repo: str, *, owner: str | None = None, sha: str | None = None) -> None:
    """Firewall gate. Always rejects a repo name outside the allowlist. When owner+sha are supplied
    (the real fetch path), additionally requires the exact pinned (owner, repo, sha) triple — so a
    fork under a different owner that reuses an allowlisted repo name, or an unpinned SHA, is also
    blocked, not just Mastering-Agentic-AI-Week2."""
    if repo not in _ALLOWED:
        raise ValueError(f"repo {repo!r} is not in the eval allowlist {_ALLOWED} "
                         f"(Mastering-Agentic-AI-Week2 is the sample solution and is firewalled)")
    if owner is not None and sha is not None and (owner, repo, sha) not in _ALLOWED_TRIPLES:
        raise ValueError(f"({owner}/{repo}@{sha}) is not a pinned eval-corpus entry — only the "
                         f"exact commit-pinned triples in EVAL_CORPUS are fetchable")

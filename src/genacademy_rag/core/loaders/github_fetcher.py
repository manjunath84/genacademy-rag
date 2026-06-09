"""Fetch raw file bytes at a pinned commit SHA via raw.githubusercontent.com. Public repos,
no auth needed. Every fetch goes through assert_allowed() — the firewall."""
from __future__ import annotations

import requests

from genacademy_rag.core.loaders import assert_allowed

RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"


def fetch_raw(*, owner: str, repo: str, sha: str, path: str, timeout: int = 30) -> bytes:
    assert_allowed(repo, owner=owner, sha=sha)
    url = RAW_URL.format(owner=owner, repo=repo, sha=sha, path=requests.utils.quote(path))
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

"""Gold-set schema + validator. Each question pins gold spans by repo+file_path+commit_hash
(the provenance chain), so the scorer can confirm a retrieved chunk is the gold source AND that
its commit_hash matches (production content never satisfies a gold marker)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = ["answerable", "exact_match", "chunking_stress", "multi_document",
              "ambiguous", "unanswerable"]


@dataclass(frozen=True)
class GoldSpan:
    repo: str
    file_path: str
    commit_hash: str
    line_start: int | None = None
    line_end: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    category: str
    answerable: bool
    gold: list[GoldSpan] = field(default_factory=list)


def load_gold_set(path) -> list[GoldQuestion]:
    raw = yaml.safe_load(Path(path).read_text())
    out: list[GoldQuestion] = []
    for item in raw:
        if item["category"] not in CATEGORIES:
            raise ValueError(f"q{item['id']}: unknown category {item['category']!r}")
        spans = [GoldSpan(**s) for s in item.get("gold", [])]
        if item["answerable"] and not spans:
            raise ValueError(f"q{item['id']}: answerable=true requires at least one gold span")
        if not item["answerable"] and spans:
            raise ValueError(f"q{item['id']}: unanswerable question must have empty gold")
        out.append(GoldQuestion(
            id=item["id"], question=item["question"],
            category=item["category"], answerable=item["answerable"], gold=spans,
        ))
    return out

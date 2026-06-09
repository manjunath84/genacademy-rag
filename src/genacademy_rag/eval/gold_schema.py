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

    def __post_init__(self):
        # Enforce the invariant at construction so no invalid GoldQuestion can exist in memory,
        # regardless of how it's built (load_gold_set or a direct call).
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.answerable and not self.gold:
            raise ValueError(f"{self.id}: answerable=true requires at least one gold span")
        if not self.answerable and self.gold:
            raise ValueError(f"{self.id}: unanswerable question must have empty gold")


def load_gold_set(path) -> list[GoldQuestion]:
    raw = yaml.safe_load(Path(path).read_text())
    return [
        GoldQuestion(
            id=item["id"], question=item["question"], category=item["category"],
            answerable=item["answerable"],
            gold=[GoldSpan(**s) for s in item.get("gold", [])],
        )
        for item in raw
    ]

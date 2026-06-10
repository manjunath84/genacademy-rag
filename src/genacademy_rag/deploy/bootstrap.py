"""Seed deploy data on first boot if the pinned eval corpus is absent."""

from __future__ import annotations

import argparse
import subprocess
import sys

from genacademy_rag.config import REPO_ROOT, Settings
from genacademy_rag.core.vectorstore import ChromaStore


def _run_ingest(*, reset_collection: bool = False) -> None:
    cmd = [sys.executable, "-m", "scripts.ingest_eval_corpus", "--chunker", "fixed"]
    if reset_collection:
        cmd.append("--reset-collection")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    store = ChromaStore(persist_dir=settings.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()
    if chunks and not args.force:
        print("deploy bootstrap: eval collection already seeded")
        return
    if chunks:
        print("deploy bootstrap: forcing eval collection re-seed")
    else:
        print("deploy bootstrap: seeding eval collection")
    _run_ingest(reset_collection=args.force)


if __name__ == "__main__":
    main()

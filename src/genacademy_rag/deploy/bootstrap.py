"""Seed deploy data on first boot if the pinned eval corpus is absent."""

from __future__ import annotations

import argparse
import subprocess
import sys

from genacademy_rag.config import REPO_ROOT, Settings
from genacademy_rag.core.loaders import EVAL_CORPUS
from genacademy_rag.core.vectorstore import ChromaStore


def _expected_eval_doc_ids() -> set[str]:
    return {
        f"{repo['repo']}/{file_spec['path']}@{repo['sha'][:7]}"
        for repo in EVAL_CORPUS
        for file_spec in repo["files"]
    }


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
        expected_doc_ids = _expected_eval_doc_ids()
        present_doc_ids = {chunk.doc_id for chunk in chunks}
        if present_doc_ids != expected_doc_ids:
            missing = sorted(expected_doc_ids - present_doc_ids)
            unexpected = sorted(present_doc_ids - expected_doc_ids)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise SystemExit(
                "deploy bootstrap: eval collection is incomplete "
                f"({len(present_doc_ids)}/{len(expected_doc_ids)} expected docs, "
                f"{len(chunks)} chunks; {'; '.join(details)}). "
                "Run python -m genacademy_rag.deploy.bootstrap --force to reset and re-ingest."
            )
        print(
            "deploy bootstrap: eval collection already seeded "
            f"({len(present_doc_ids)} docs, {len(chunks)} chunks)"
        )
        return
    if chunks:
        print("deploy bootstrap: forcing eval collection re-seed")
    else:
        print("deploy bootstrap: seeding eval collection")
    try:
        _run_ingest(reset_collection=args.force)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "deploy bootstrap: eval ingest failed. Ensure GENACADEMY_EMBEDDINGS=local, "
            "outbound HTTPS is available, and rerun with --force after fixing any partial ingest."
        ) from exc


if __name__ == "__main__":
    main()

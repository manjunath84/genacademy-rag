"""Ingest the commit-pinned eval corpus into Chroma + SQLite. Fetches ONLY the allowlisted
repos+SHAs (docs/spike-findings.md §4); Week-2 is firewalled by the allowlist. Run once before
the eval. Idempotent (upsert by chunk_id)."""
import argparse
from pathlib import Path

from genacademy_rag.config import Settings
from genacademy_rag.core.chunker import build_chunker
from genacademy_rag.core.loaders import EVAL_CORPUS
from genacademy_rag.core.loaders.github_fetcher import fetch_raw
from genacademy_rag.core.loaders.jupyter_loader import load_notebook
from genacademy_rag.core.loaders.markdown_loader import load_markdown
from genacademy_rag.core.pipeline import IngestPipeline
from genacademy_rag.core.providers import build_provider
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.data.datastore import SQLiteDatastore


def reset_chroma_collection(persist_dir: Path, collection: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    names = {getattr(item, "name", item) for item in client.list_collections()}
    if collection in names:
        client.delete_collection(collection)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="eval")
    parser.add_argument("--chunker", choices=("fixed", "section"), default=None)
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument("--reset-collection", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    s = Settings.from_env()
    chunker_name = args.chunker or s.chunker
    sqlite_path = args.sqlite_path or s.sqlite_path
    if args.collection == "eval" and chunker_name != "fixed":
        raise SystemExit(
            "refusing to ingest collection='eval' with chunker="
            f"{chunker_name!r}; use --collection for alternate chunker experiments"
        )
    if args.collection != "eval" and args.sqlite_path is None:
        sqlite_path = s.sqlite_path.with_name(f"{s.sqlite_path.stem}-{args.collection}.sqlite")
    if args.reset_collection:
        reset_chroma_collection(s.chroma_dir, args.collection)
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection=args.collection)
    ds = SQLiteDatastore(sqlite_path)
    ds.seed_users()
    pipe = IngestPipeline(
        chunker=build_chunker(
            chunker_name,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            section_max_chars=s.section_chunk_max_chars,
            section_overlap=s.section_chunk_overlap,
        ),
        provider=provider,
        store=store,
        datastore=ds,
    )

    docs = []
    for repo in EVAL_CORPUS:
        for f in repo["files"]:
            raw = fetch_raw(owner=repo["owner"], repo=repo["repo"], sha=repo["sha"], path=f["path"])
            if f["kind"] == "jupyter":
                docs.append(load_notebook(
                    repo=repo["repo"], file_path=f["path"],
                    commit_hash=repo["sha"], raw_bytes=raw,
                ))
            else:
                docs.append(load_markdown(
                    repo=repo["repo"], file_path=f["path"],
                    commit_hash=repo["sha"], raw_text=raw.decode("utf-8"),
            ))
            print(f"fetched {repo['repo']}/{f['path']} @ {repo['sha'][:7]}")
    n = pipe.ingest(docs)
    print(
        f"ingested {len(docs)} docs -> {n} chunks into "
        f"{s.chroma_dir} collection={args.collection} chunker={chunker_name} sqlite={sqlite_path}"
    )


if __name__ == "__main__":
    main()

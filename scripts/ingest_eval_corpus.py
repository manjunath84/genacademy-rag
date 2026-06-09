"""Ingest the commit-pinned eval corpus into Chroma + SQLite. Fetches ONLY the allowlisted
repos+SHAs (docs/spike-findings.md §4); Week-2 is firewalled by the allowlist. Run once before
the eval. Idempotent (upsert by chunk_id)."""
from genacademy_rag.config import Settings
from genacademy_rag.core.chunker import FixedSizeChunker
from genacademy_rag.core.loaders import EVAL_CORPUS
from genacademy_rag.core.loaders.github_fetcher import fetch_raw
from genacademy_rag.core.loaders.jupyter_loader import load_notebook
from genacademy_rag.core.loaders.markdown_loader import load_markdown
from genacademy_rag.core.pipeline import IngestPipeline
from genacademy_rag.core.providers import build_provider
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.data.datastore import SQLiteDatastore


def main():
    s = Settings.from_env()
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    ds = SQLiteDatastore(s.sqlite_path)
    ds.seed_users()
    pipe = IngestPipeline(
        chunker=FixedSizeChunker(s.chunk_size, s.chunk_overlap),
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
    print(f"ingested {len(docs)} docs -> {n} chunks into {s.chroma_dir}")


if __name__ == "__main__":
    main()

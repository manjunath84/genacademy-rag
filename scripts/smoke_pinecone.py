"""Live smoke for the Pinecone preset (the "Chroma -> Pinecone, one config line" demo check).
Requires PINECONE_API_KEY. Writes ONLY to a throwaway `smoke-test` namespace and deletes its
doc afterwards. Not part of the deterministic eval — that always runs on local Chroma."""
import time

from genacademy_rag.config import Settings
from genacademy_rag.core.providers import STEmbedder
from genacademy_rag.core.types import Chunk, Citation
from genacademy_rag.core.vectorstore import PineconeStore

NAMESPACE = "smoke-test"


def main():
    s = Settings.from_env()
    if not s.pinecone_api_key:
        raise SystemExit("PINECONE_API_KEY not set; export it (env-only secret) and rerun")
    store = PineconeStore(
        api_key=s.pinecone_api_key, index_name=s.pinecone_index, namespace=NAMESPACE,
        cloud=s.pinecone_cloud, region=s.pinecone_region,
    )
    embedder = STEmbedder(s.embed_model)
    cit = Citation(doc_id="smoke", title="smoke.md", source_type="github",
                   repo="r", file_path="smoke.md", commit_hash="deadbeef",
                   line_start=1, line_end=2, char_start=0, char_end=10)
    chunks = [
        Chunk(chunk_id="smoke::0", doc_id="smoke", ordinal=0,
              text="retrieval augmented generation", citation=cit),
        Chunk(chunk_id="smoke::1", doc_id="smoke", ordinal=1,
              text="banana bread recipe", citation=cit),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    qvec = embedder.embed(["retrieval augmented generation"])[0]
    results: list[tuple[str, float]] = []
    deadline = time.time() + 60
    while time.time() < deadline:           # serverless upserts are eventually consistent
        results = store.query(qvec, top_k=2)
        if len(results) == 2:
            break
        time.sleep(3)
    assert results and results[0][0] == "smoke::0", f"unexpected query result: {results}"
    assert results[0][1] > results[1][1], "similarity must be descending"

    got = store.get_chunk("smoke::0")
    assert got.text == "retrieval augmented generation"
    assert got.citation.commit_hash == "deadbeef"
    assert isinstance(got.citation.line_start, int)      # float->int coercion on read

    all_ids = [c.chunk_id for c in store.get_all_chunks()]
    assert all_ids == ["smoke::0", "smoke::1"], all_ids   # (doc_id, ordinal) sort, both present
    store.delete_doc("smoke")
    print(
        f"PINECONE SMOKE OK  index={s.pinecone_index} namespace={NAMESPACE} "
        f"top=({results[0][0]}, {results[0][1]:.3f}) chunks_seen={len(all_ids)}"
    )


if __name__ == "__main__":
    main()

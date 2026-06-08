"""Spike: local embedding latency for all-MiniLM-L6-v2 (384-dim).

Part of the < 8 s end-to-end ceiling (design.md §9): embed-local + Nebius-generate.
Measures cold load, single-query embed, and a small corpus batch embed.
"""
import time

from sentence_transformers import SentenceTransformer

t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
load_s = time.time() - t0

dim = model.get_sentence_embedding_dimension()

# single query (the online path that counts against the 8 s ceiling)
q = "What does the course say about chunking strategies for RAG?"
# warm up
model.encode([q])
t0 = time.time()
v = model.encode([q])
query_ms = (time.time() - t0) * 1000

# batch ingest (offline path; just to confirm throughput)
docs = [f"chunk number {i} about retrieval augmented generation and embeddings" for i in range(256)]
t0 = time.time()
model.encode(docs, batch_size=32)
batch_s = time.time() - t0

print(f"model            : all-MiniLM-L6-v2")
print(f"embedding dim    : {dim}   (Pinecone index dim MUST match this)")
print(f"cold load        : {load_s:.1f} s (one-time)")
print(f"single query     : {query_ms:.0f} ms  <- counts against the 8 s online ceiling")
print(f"batch 256 chunks : {batch_s:.1f} s  ({256/batch_s:.0f} chunks/s, ingest path)")
print()
print(f"VERDICT: dim={dim} confirmed; online embed ~{query_ms:.0f} ms leaves the bulk")
print(f"         of the 8 s budget to Nebius generation.")

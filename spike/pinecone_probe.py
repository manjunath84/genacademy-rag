"""Spike: Pinecone probe (item 6 from design.md §9) — optional this pass.

Reads spike/.env. Confirms credentials work and a 384-dim index can be created
(dimension MUST match all-MiniLM-L6-v2). Cleans up the test index.
"""
import os
import time
from pathlib import Path

envp = Path(__file__).parent / ".env"
if envp.exists():
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

key = os.environ.get("PINECONE_API_KEY")
if not key:
    raise SystemExit("PINECONE_API_KEY not set (optional this pass).")

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=key)
print("=== existing indexes ===")
for ix in pc.list_indexes():
    print(" ", ix["name"], ix.get("dimension"), ix.get("metric"))

name = "genacademy-spike-384"
print(f"\n=== creating {name} (dim=384, cosine, serverless) ===")
try:
    pc.create_index(
        name=name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(2)
    desc = pc.describe_index(name)
    print("created:", desc["name"], "dim", desc["dimension"], "status", desc.get("status"))
    print("VERDICT: Pinecone free-tier OK; 384-dim serverless index works.")
finally:
    try:
        pc.delete_index(name)
        print("cleaned up test index.")
    except Exception as e:
        print("cleanup note:", e)

"""Spike: Nebius generation probe (items 1, 2, 3-generate from design.md §9).

Reads spike/.env. Checks: chat model id, JSON mode / structured output support,
throughput over ~10 sequential calls, per-call generation latency vs the 8 s ceiling.
"""
import json
import os
import time
from pathlib import Path

# minimal .env loader (no extra dep)
envp = Path(__file__).parent / ".env"
if envp.exists():
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

key = os.environ.get("NEBIUS_API_KEY")
base = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")
if not key:
    raise SystemExit("NEBIUS_API_KEY not set — copy spike/.env.example to spike/.env and fill it in.")

client = OpenAI(api_key=key, base_url=base)

# --- item 1a: list models, pick a chat model ---
model = os.environ.get("NEBIUS_MODEL") or ""
print("=== /v1/models (first 20) ===")
try:
    models = client.models.list()
    ids = [m.id for m in models.data]
    for mid in ids[:20]:
        print(" ", mid)
    if not model:
        # prefer an instruct/chat model
        pref = [m for m in ids if any(t in m.lower() for t in ("instruct", "chat", "llama", "qwen", "mistral"))]
        model = (pref or ids)[0]
except Exception as e:
    print("  could not list models:", e)
    if not model:
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
print(f"\nchosen model: {model}")

# --- item 3: generation latency (single call) ---
print("\n=== single generation latency ===")
t0 = time.time()
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"}],
    temperature=0,
    max_tokens=120,
)
gen_s = time.time() - t0
print(f"latency: {gen_s:.2f} s   (ceiling 8 s, minus ~12 ms local embed)")
print("sample:", r.choices[0].message.content[:160].replace("\n", " "))

# --- item 1b: JSON mode / structured output (decides grader + judge format) ---
print("\n=== JSON mode probe (response_format=json_object) ===")
json_ok = False
try:
    rj = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Reply ONLY with JSON."},
            {"role": "user", "content": 'Is "the sky is green" grounded in the context "the sky is blue"? '
                                        'Return {"grounded": bool, "reason": str}.'},
        ],
        temperature=0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )
    raw = rj.choices[0].message.content
    parsed = json.loads(raw)
    json_ok = isinstance(parsed, dict) and "grounded" in parsed
    print(f"json_object supported: {json_ok}   parsed={parsed}")
except Exception as e:
    print(f"json_object NOT supported / errored: {type(e).__name__}: {e}")
    print("  -> grader falls back to plain-prompt parse + cosine-similarity threshold (design §7).")

# --- item 2: throughput / rate limits (10 sequential) ---
print("\n=== throughput: 10 sequential calls ===")
lat = []
throttled = 0
for i in range(10):
    try:
        t0 = time.time()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Reply with the number {i}."}],
            temperature=0,
            max_tokens=8,
        )
        lat.append(time.time() - t0)
    except Exception as e:
        throttled += 1
        print(f"  call {i}: {type(e).__name__}: {str(e)[:100]}")
        time.sleep(2)
if lat:
    import statistics
    print(f"ok={len(lat)}/10  throttled={throttled}  "
          f"mean={statistics.mean(lat):.2f}s  max={max(lat):.2f}s")

print("\n=== SUMMARY ===")
print(f"model            : {model}")
print(f"json mode        : {'YES' if json_ok else 'NO -> cosine-threshold grader fallback'}")
print(f"single gen        : {gen_s:.2f} s")
print(f"throttling        : {'none observed' if throttled == 0 else f'{throttled}/10 failed'}")

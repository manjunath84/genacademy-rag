"""Spike: generation probe across OpenAI-compatible providers (design.md §9 items 1-3).

Reads spike/.env. For every provider with a key set, checks:
  - model id (auto-pick from /v1/models if not pinned)
  - JSON mode / structured output  -> decides grader + LLM-judge format
  - single-call generation latency -> vs the < 8 s ceiling (minus ~12 ms local embed)
  - throughput over 10 sequential calls -> rate-limit / throttling signal

OpenRouter (open models) is the representative stand-in for Nebius until credit lands.
"""
import json
import os
import statistics
import time
from pathlib import Path

envp = Path(__file__).parent / ".env"
if envp.exists():
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().split("#")[0].strip())

from openai import OpenAI

PROVIDERS = [
    ("nebius", "NEBIUS_API_KEY", "NEBIUS_BASE_URL", "NEBIUS_MODEL"),
    ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"),
    ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"),
]


def pick_model(client, hint):
    if hint:
        return hint
    try:
        ids = [m.id for m in client.models.list().data]
        pref = [m for m in ids if any(t in m.lower() for t in ("instruct", "llama", "qwen", "mistral", "chat"))]
        return (pref or ids)[0]
    except Exception:
        return "(unknown — set *_MODEL in .env)"


def probe(name, key, base, model_hint):
    client = OpenAI(api_key=key, base_url=base)
    model = pick_model(client, model_hint)
    print(f"\n========== {name}  ({base}) ==========")
    print(f"model: {model}")

    # latency (single)
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"}],
        temperature=0, max_tokens=120,
    )
    gen_s = time.time() - t0
    print(f"single gen latency : {gen_s:.2f} s   (ceiling 8 s)")
    print(f"sample             : {r.choices[0].message.content[:140].strip()!r}")

    # JSON mode
    json_ok = False
    try:
        rj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply ONLY with JSON."},
                {"role": "user", "content": 'Is "the sky is green" grounded in the context "the sky is blue"? '
                                            'Return {"grounded": bool, "reason": str}.'},
            ],
            temperature=0, max_tokens=120,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(rj.choices[0].message.content)
        json_ok = isinstance(parsed, dict) and "grounded" in parsed
        print(f"json_object        : {'YES' if json_ok else 'partial'}  parsed={parsed}")
    except Exception as e:
        print(f"json_object        : NO ({type(e).__name__}: {str(e)[:80]}) -> cosine-threshold grader")

    # throughput
    lat, throttled = [], 0
    for i in range(10):
        try:
            t0 = time.time()
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Reply with the number {i}."}],
                temperature=0, max_tokens=8,
            )
            lat.append(time.time() - t0)
        except Exception as e:
            throttled += 1
            print(f"  call {i}: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)
    tp = (f"ok={len(lat)}/10 throttled={throttled} mean={statistics.mean(lat):.2f}s max={max(lat):.2f}s"
          if lat else f"all 10 failed (throttled={throttled})")
    print(f"throughput (10 seq): {tp}")
    return dict(provider=name, model=model, gen_s=gen_s, json_ok=json_ok, throttled=throttled,
               mean=statistics.mean(lat) if lat else None)


def main():
    results = []
    for name, kk, bk, mk in PROVIDERS:
        key = os.environ.get(kk)
        if not key:
            print(f"\n-- {name}: no key, skipped --")
            continue
        base = os.environ.get(bk)
        try:
            results.append(probe(name, key, base, os.environ.get(mk)))
        except Exception as e:
            print(f"\n!! {name} failed: {type(e).__name__}: {e}")
    if results:
        print("\n================= SUMMARY =================")
        print(f"{'provider':<12}{'json':<6}{'gen_s':<8}{'mean_s':<8}{'throttle':<9}model")
        for r in results:
            mean_s = f"{r['mean']:.2f}" if r['mean'] else "-"
            print(f"{r['provider']:<12}{('Y' if r['json_ok'] else 'n'):<6}"
                  f"{r['gen_s']:<8.2f}{mean_s:<8}{r['throttled']:<9}{r['model']}")


if __name__ == "__main__":
    main()

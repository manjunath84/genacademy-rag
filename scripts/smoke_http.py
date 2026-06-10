"""Smoke-check a local container or live Hugging Face Space HTTP URL."""

from __future__ import annotations

import argparse

import requests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    response = requests.get(f"{base_url}/login", timeout=args.timeout)
    response.raise_for_status()
    if "member@genacademy.local" not in response.text or 'name="csrf_token"' not in response.text:
        raise SystemExit("login marker not found")
    print(f"HTTP SMOKE OK  base_url={base_url}")


if __name__ == "__main__":
    main()

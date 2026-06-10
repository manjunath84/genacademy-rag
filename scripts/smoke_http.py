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
    # Coupled to the login template's prefilled demo-member email plus CSRF hidden input.
    if "member@genacademy.local" not in response.text or 'name="csrf_token"' not in response.text:
        snippet = response.text[:200].replace("\n", " ")
        final_url = getattr(response, "url", f"{base_url}/login")
        raise SystemExit(
            "login marker not found "
            f"status={response.status_code} url={final_url} body={snippet!r}"
        )
    print(f"HTTP SMOKE OK  base_url={base_url}")


if __name__ == "__main__":
    main()

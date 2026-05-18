#!/usr/bin/env python3
"""
CI tool: verify that CDN resources referenced in base.html still match their
pinned SRI (sha384) hashes.  Exits non-zero on any mismatch or fetch failure.

Usage:
    python scripts/check_cdn_integrity.py
"""
import base64
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

TIMEOUT = 10
MAX_RETRIES = 2

BASE_HTML = Path(__file__).parent.parent / "web_app" / "templates" / "base.html"

# Regex: grab (url, hash) pairs from  src="..."  integrity="sha384-..."
_PATTERN = re.compile(
    r'(?:src|href)="(https://[^"]+)"[^>]*integrity="sha384-([^"]+)"',
    re.DOTALL,
)


def _fetch(url: str) -> bytes:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def _sha384_b64(data: bytes) -> str:
    digest = hashlib.sha384(data).digest()
    return base64.b64encode(digest).decode()


def main() -> int:
    html = BASE_HTML.read_text(encoding="utf-8")
    pairs = _PATTERN.findall(html)

    if not pairs:
        print("ERROR: no SRI-hashed CDN resources found in base.html", file=sys.stderr)
        return 1

    failures = []
    for url, pinned_hash in pairs:
        print(f"  checking {url} ...", end=" ", flush=True)
        try:
            content = _fetch(url)
            actual = _sha384_b64(content)
            if actual == pinned_hash:
                print("OK")
            else:
                print(f"MISMATCH\n    pinned: {pinned_hash}\n    actual: {actual}")
                failures.append(url)
        except RuntimeError as exc:
            print(f"FETCH ERROR: {exc}")
            failures.append(url)

    if failures:
        print(f"\nFAILED: {len(failures)} resource(s) did not match pinned hash:", file=sys.stderr)
        for url in failures:
            print(f"  {url}", file=sys.stderr)
        return 1

    print(f"\nAll {len(pairs)} SRI hashes verified OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

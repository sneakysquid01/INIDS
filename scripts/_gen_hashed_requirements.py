#!/usr/bin/env python3
"""Generate requirements.txt with SHA-256 hashes from locally downloaded wheels.

Run after: pip download -r requirements.in --no-deps --dest <wheels_dir>
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    wheels_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(os.environ["TEMP"]) / "wheels"
    out_path = pathlib.Path(__file__).parent.parent / "requirements.txt"

    freeze = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout

    entries: list[str] = [
        "# This file is auto-generated. Edit requirements.in and re-run this script.",
        "# Install: pip install --require-hashes --no-deps -r requirements.txt",
        "",
    ]
    missing: list[str] = []

    for line in sorted(freeze.strip().splitlines()):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)==(.+)$", line)
        if not m:
            continue
        pkg_name, pkg_ver = m.group(1), m.group(2)

        safe_name = re.sub(r"[-_.]+", r"[-_.]", pkg_name)
        safe_ver = re.escape(pkg_ver)
        pattern = re.compile(
            rf"^{safe_name}-{safe_ver}[^-]*-.*\.(whl|tar\.gz)$", re.IGNORECASE
        )
        matched = sorted(w for w in wheels_dir.iterdir() if pattern.match(w.name))

        if not matched:
            missing.append(f"{pkg_name}=={pkg_ver}")
            continue

        hash_lines = [f"    --hash=sha256:{sha256_file(whl)}" for whl in matched]
        entry = f"{pkg_name}=={pkg_ver} \\\n" + " \\\n".join(hash_lines)
        entries.append(entry)

    out_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Written {len(entries) - 3} entries to {out_path}")
    if missing:
        print(f"WARNING: {len(missing)} packages had no local wheel (skipped): {missing[:5]}")


if __name__ == "__main__":
    main()

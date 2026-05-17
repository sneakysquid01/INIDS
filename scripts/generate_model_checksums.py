#!/usr/bin/env python3
"""Generate SHA-256 checksums for all model .pkl files.

PLAN.md Phase A Step 6 (A-06): Run this script against known-good models
to produce models/checksums.sha256. Commit the output to the repo.

Usage:
    python scripts/generate_model_checksums.py [models_dir]
"""
import hashlib
import pathlib
import sys


def compute_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    models_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "models"

    if not models_dir.exists():
        print(f"ERROR: models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    pkl_files = sorted(models_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"WARNING: no .pkl files found in {models_dir}")
        sys.exit(0)

    out = models_dir / "checksums.sha256"
    with out.open("w", encoding="utf-8") as f:
        for pkl in pkl_files:
            digest = compute_sha256(pkl)
            f.write(f"{digest}  {pkl.name}\n")
            print(f"  {digest[:16]}...  {pkl.name}")

    print(f"\nChecksums written to {out}")
    print("Commit checksums.sha256 to the repository to enable integrity verification.")


if __name__ == "__main__":
    main()

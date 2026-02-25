"""Backward-compatible entrypoint for full-suite model training."""

import sys

from src.train_cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--suite", "full", "--include-multiclass", *sys.argv[1:]]
    main()

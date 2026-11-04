"""Download the pinned SSA authors' mean-pooling NSA baseline (no remote code)."""

import argparse
from pathlib import Path

from download_checkpoint import FILES, download_checkpoint

REPOSITORY = "zen-E/NSA-1B"
REVISION = "3dc3694d2f873820122b094276ae626a44f00a36"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/NSA-1B"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in 1..32")
    download_checkpoint(args.output, REPOSITORY, REVISION, FILES, workers=args.workers)


if __name__ == "__main__":
    main()

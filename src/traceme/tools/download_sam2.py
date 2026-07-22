#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from traceme.sam2.checkpoints import (
    download_all_checkpoints,
    download_checkpoint,
    checkpoint_filename,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM2 checkpoints for TraceME.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Target directory for checkpoints (default: user cache).",
    )
    parser.add_argument(
        "--model",
        choices=("all", "tiny", "small", "base_plus", "large"),
        default="all",
        help="Which checkpoint to download (default: all).",
    )
    args = parser.parse_args()

    if args.model == "all":
        target_dir = download_all_checkpoints(checkpoint_dir=args.dir)
        print(f"Downloaded checkpoints to: {target_dir}")
    else:
        path = download_checkpoint(args.model, checkpoint_dir=args.dir)
        print(f"Downloaded {checkpoint_filename(args.model)} to: {path}")


if __name__ == "__main__":
    main()

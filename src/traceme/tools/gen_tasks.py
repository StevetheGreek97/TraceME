#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

def find_yaml_for_folder(folder: Path) -> Path | None:
    """Prefer <folder>/<folder.name>.yaml; fallback to the only *.yaml present."""
    preferred = folder / f"{folder.name}.yaml"
    if preferred.exists():
        return preferred
    yamls = sorted(folder.glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0]
    return None  # ambiguous or missing

def main():
    p = argparse.ArgumentParser(description="Generate tasks.tsv from root directory.")
    p.add_argument("root", type=Path, help="Root dir containing N subfolders (e.g., Control1_A7_100925/).")
    p.add_argument("-o", "--out", type=Path, default=Path("tasks.tsv"), help="Output TSV path (default: tasks.tsv)")
    p.add_argument("--recursive", action="store_true", help="Recurse beyond immediate children.")
    args = p.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    # Choose search depth
    if args.recursive:
        candidates = [d for d in root.rglob("*") if d.is_dir()]
    else:
        candidates = [d for d in root.iterdir() if d.is_dir()]

    rows = []
    skipped = 0
    for d in sorted(candidates):
        yaml_path = find_yaml_for_folder(d)
        if yaml_path is None:
            # skip folders without a unique/matching yaml
            skipped += 1
            continue

        # Heuristic: treat any directory with image frames as valid.
        # If you want to be strict, require at least one image file:
        # has_frames = any(d.glob("*.png")) or any(d.glob("*.jpg")) or any(d.glob("*.jpeg"))
        # if not has_frames: continue

        rows.append((str(d.resolve()), str(yaml_path.resolve())))

    if not rows:
        print("No valid (folder, yaml) pairs found.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for frame_dir, yaml_file in rows:
            f.write(f"{frame_dir}\t{yaml_file}\n")

    print(f"Wrote {len(rows)} tasks to {args.out}")
    if skipped:
        print(f"(Skipped {skipped} folders without an unambiguous YAML)", file=sys.stderr)

if __name__ == "__main__":
    main()

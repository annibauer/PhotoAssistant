from __future__ import annotations

import argparse
from pathlib import Path


def delete_empty_folders(root: Path, dry_run: bool = False) -> list[Path]:
    """Delete all empty subfolders inside root and return deleted paths.

    Traverses bottom-up so nested empty folders are removed correctly.
    The root folder itself is not deleted.
    """
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid folder: {root}")

    removed: list[Path] = []

    # Bottom-up traversal ensures children are handled before parents.
    for folder in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            if any(folder.iterdir()):
                continue
            removed.append(folder)
            if not dry_run:
                folder.rmdir()
        except OSError:
            # Skip folders that become non-empty or are inaccessible.
            continue

    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete all empty folders within a specific folder")
    parser.add_argument("root", type=Path, help="Root folder to clean")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which folders would be removed without deleting them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    removed = delete_empty_folders(root, dry_run=args.dry_run)

    if args.dry_run:
        print(f"[dry-run] {len(removed)} empty folders would be removed under: {root}")
    else:
        print(f"Removed {len(removed)} empty folders under: {root}")

    for folder in removed:
        print(folder)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def _sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _list_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file()])


def find_unique_files(source: Path, reference: Path) -> list[Path]:
    """Return files in `source` whose SHA-256 hash is not present in `reference`."""
    if not source.exists() or not source.is_dir():
        raise ValueError(f"Invalid source folder: {source}")
    if not reference.exists() or not reference.is_dir():
        raise ValueError(f"Invalid reference folder: {reference}")

    reference_hashes: set[str] = set()
    for path in _list_files(reference):
        reference_hashes.add(_sha256(path))

    unique: list[Path] = []
    for path in _list_files(source):
        if _sha256(path) not in reference_hashes:
            unique.append(path)
    return unique


def _destination_for(file_path: Path, source: Path, destination: Path) -> Path:
    """Map a source file into the destination, preserving relative folders.

    If a name collision occurs, append a counter before the suffix.
    """
    try:
        relative = file_path.relative_to(source)
    except ValueError:
        relative = Path(file_path.name)

    target = destination / relative
    if not target.exists():
        return target

    counter = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{counter}{target.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_unique_files(source: Path, reference: Path, destination: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Move files from `source` that are not present in `reference` into `destination`.

    Returns a list of (source, destination) pairs for every moved (or would-be moved) file.
    """
    unique_files = find_unique_files(source, reference)

    moved: list[tuple[Path, Path]] = []
    for file_path in unique_files:
        target = _destination_for(file_path, source, destination)
        moved.append((file_path, target))
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(target))

    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move files from a source folder that are NOT present in a reference folder (SHA-256 comparison)"
    )
    parser.add_argument("source", type=Path, help="Folder to scan for unique files")
    parser.add_argument("reference", type=Path, help="Folder to compare against")
    parser.add_argument("destination", type=Path, help="Folder to move unique files into")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be moved without moving anything",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    moved = move_unique_files(args.source, args.reference, args.destination, dry_run=args.dry_run)

    action = "Would move" if args.dry_run else "Moved"
    for src, dst in moved:
        print(f"{action}: {src} -> {dst}")
    print(f"{action} {len(moved)} file(s)")


if __name__ == "__main__":
    main()

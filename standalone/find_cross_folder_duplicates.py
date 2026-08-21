from __future__ import annotations

import argparse
import csv
import hashlib
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


def find_cross_folder_duplicates(folder_a: Path, folder_b: Path) -> list[dict[str, str]]:
    """Return files that are duplicates across folder_a and folder_b.

    The function compares SHA-256 hashes from all files in folder_a and folder_b,
    and only reports matches where one file is from A and the other from B.
    """
    if not folder_a.exists() or not folder_a.is_dir():
        raise ValueError(f"Invalid folder_a: {folder_a}")
    if not folder_b.exists() or not folder_b.is_dir():
        raise ValueError(f"Invalid folder_b: {folder_b}")

    hash_to_a: dict[str, list[Path]] = {}
    for path in _list_files(folder_a):
        file_hash = _sha256(path)
        hash_to_a.setdefault(file_hash, []).append(path)

    matches: list[dict[str, str]] = []
    for path_b in _list_files(folder_b):
        file_hash = _sha256(path_b)
        if file_hash not in hash_to_a:
            continue
        for path_a in hash_to_a[file_hash]:
            matches.append(
                {
                    "sha256": file_hash,
                    "original_file_path": str(path_a),
                    "matching_file_path": str(path_b),
                    "same_filename": str(path_a.name == path_b.name),
                }
            )

    return matches


def _write_csv(matches: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "original_file_path", "matching_file_path", "same_filename"],
        )
        writer.writeheader()
        writer.writerows(matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate files between two folders using SHA-256")
    parser.add_argument("folder_a", type=Path, help="First folder")
    parser.add_argument("folder_b", type=Path, help="Second folder")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("cross_folder_duplicates.csv"),
        help="CSV output path (default: ./cross_folder_duplicates.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = find_cross_folder_duplicates(args.folder_a.expanduser().resolve(), args.folder_b.expanduser().resolve())
    out_csv = args.csv.expanduser().resolve()
    _write_csv(matches, out_csv)
    print(f"Wrote {len(matches)} rows to CSV: {out_csv}")


if __name__ == "__main__":
    main()
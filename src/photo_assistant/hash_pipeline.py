from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import ExifTags, Image

from .models import DuplicateGroup, PhotoAsset

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
    ".bmp",
}


def list_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_capture_time(image: Image.Image) -> datetime | None:
    try:
        exif = image.getexif()
        if not exif:
            return None

        tag_map = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}
        value = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        if not value:
            return None
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def build_assets(image_paths: list[Path]) -> list[PhotoAsset]:
    assets: list[PhotoAsset] = []
    for path in image_paths:
        try:
            sha = compute_sha256(path)
            with Image.open(path) as image:
                phash = imagehash.phash(image)
                captured_at = _extract_capture_time(image)
            phash_hex = str(phash)
            phash_int = int(phash_hex, 16)
            assets.append(
                PhotoAsset(
                    path=str(path),
                    sha256=sha,
                    phash_hex=phash_hex,
                    phash_int=phash_int,
                    captured_at=captured_at,
                )
            )
        except Exception:
            # Skip unreadable files so the pipeline can continue.
            continue
    return assets


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_exact_duplicate_groups(assets: list[PhotoAsset]) -> list[DuplicateGroup]:
    buckets: dict[str, list[PhotoAsset]] = defaultdict(list)
    for asset in assets:
        buckets[asset.sha256].append(asset)

    groups: list[DuplicateGroup] = []
    for idx, members in enumerate(buckets.values(), start=1):
        if len(members) > 1:
            groups.append(DuplicateGroup(group_id=f"exact-{idx}", group_type="exact", members=members))
    return groups


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _remaining_assets_after_exact(assets: list[PhotoAsset], exact_groups: list[DuplicateGroup]) -> list[PhotoAsset]:
    exact_paths = {member.path for group in exact_groups for member in group.members}
    return [asset for asset in assets if asset.path not in exact_paths]


def find_near_duplicate_groups(
    assets: list[PhotoAsset],
    exact_groups: list[DuplicateGroup],
    max_hamming_distance: int = 8,
) -> list[DuplicateGroup]:
    candidates = _remaining_assets_after_exact(assets, exact_groups)
    if len(candidates) < 2:
        return []

    uf = _UnionFind(len(candidates))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if hamming_distance(candidates[i].phash_int, candidates[j].phash_int) <= max_hamming_distance:
                uf.union(i, j)

    clusters: dict[int, list[PhotoAsset]] = defaultdict(list)
    for idx, asset in enumerate(candidates):
        clusters[uf.find(idx)].append(asset)

    groups: list[DuplicateGroup] = []
    near_idx = 1
    for members in clusters.values():
        if len(members) > 1:
            groups.append(DuplicateGroup(group_id=f"near-{near_idx}", group_type="near", members=members))
            near_idx += 1
    return groups

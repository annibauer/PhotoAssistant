from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from photo_assistant.hash_pipeline import (
    build_assets,
    compute_sha256,
    find_exact_duplicate_groups,
    find_near_duplicate_groups,
    hamming_distance,
    list_images,
)
from photo_assistant.models import PhotoAsset


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (80, 80), color=color)
    img.save(path)


def test_list_images_filters_extensions(tmp_path: Path) -> None:
    _make_image(tmp_path / "a.jpg", (255, 0, 0))
    _make_image(tmp_path / "b.png", (0, 255, 0))
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    images = list_images(tmp_path)
    names = [p.name for p in images]

    assert names == ["a.jpg", "b.png"]


def test_compute_sha256_changes_when_file_changes(tmp_path: Path) -> None:
    file_path = tmp_path / "file.bin"
    file_path.write_bytes(b"abc")
    digest_a = compute_sha256(file_path)

    file_path.write_bytes(b"abcd")
    digest_b = compute_sha256(file_path)

    assert digest_a != digest_b


def test_find_exact_duplicate_groups_by_sha() -> None:
    a = PhotoAsset("/x/a.jpg", "same", "1", 1, None)
    b = PhotoAsset("/x/b.jpg", "same", "2", 2, None)
    c = PhotoAsset("/x/c.jpg", "diff", "3", 3, None)

    groups = find_exact_duplicate_groups([a, b, c])

    assert len(groups) == 1
    assert groups[0].group_type == "exact"
    assert sorted(m.path for m in groups[0].members) == ["/x/a.jpg", "/x/b.jpg"]


def test_hamming_distance_bit_count() -> None:
    assert hamming_distance(0b0000, 0b1111) == 4
    assert hamming_distance(0b1010, 0b1000) == 1


def test_find_near_duplicate_groups_excludes_exact_members() -> None:
    # These two are exact duplicates and should be removed from near-duplicate candidate set.
    exact_a = PhotoAsset("/x/exact_a.jpg", "sha1", "00", 0b0000, None)
    exact_b = PhotoAsset("/x/exact_b.jpg", "sha1", "00", 0b0000, None)
    # These two are near duplicates by pHash distance.
    near_a = PhotoAsset("/x/near_a.jpg", "sha2", "00", 0b0000, None)
    near_b = PhotoAsset("/x/near_b.jpg", "01", "01", 0b0001, None)
    far_c = PhotoAsset("/x/far_c.jpg", "sha4", "ff", 0b1111, None)

    exact_groups = find_exact_duplicate_groups([exact_a, exact_b, near_a, near_b, far_c])
    near_groups = find_near_duplicate_groups(
        [exact_a, exact_b, near_a, near_b, far_c], exact_groups, max_hamming_distance=2
    )

    assert len(exact_groups) == 1
    assert len(near_groups) == 1
    near_paths = sorted(m.path for m in near_groups[0].members)
    assert near_paths == ["/x/near_a.jpg", "/x/near_b.jpg"]


def test_build_assets_reads_real_images(tmp_path: Path) -> None:
    _make_image(tmp_path / "img1.jpg", (200, 100, 50))
    _make_image(tmp_path / "img2.jpg", (200, 100, 50))

    assets = build_assets(sorted(tmp_path.glob("*.jpg")))

    assert len(assets) == 2
    assert all(asset.sha256 for asset in assets)
    assert all(asset.phash_hex for asset in assets)
    assert all(isinstance(asset.phash_int, int) for asset in assets)
    assert all(asset.captured_at is None or isinstance(asset.captured_at, datetime) for asset in assets)

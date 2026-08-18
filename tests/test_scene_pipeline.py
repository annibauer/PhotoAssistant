from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from photo_assistant.models import PhotoAsset
from photo_assistant.scene_pipeline import find_scene_groups


def _asset(path: str, phash_int: int, captured_at: datetime | None) -> PhotoAsset:
    return PhotoAsset(
        path=path,
        sha256=path,
        phash_hex=f"{phash_int:x}",
        phash_int=phash_int,
        captured_at=captured_at,
    )


def test_scene_group_confidence_boost_from_time_and_phash() -> None:
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    a = _asset("/x/a.jpg", phash_int=0b0000, captured_at=base_time)
    b = _asset("/x/b.jpg", phash_int=0b0001, captured_at=base_time + timedelta(minutes=30))

    assets = [a, b]
    # Similarity starts at 0.72; with +0.10 boost total should be >= 0.82 and pass 0.8 threshold.
    # cosine distance = 1 - similarity
    target_similarity = 0.72
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [target_similarity, (1 - target_similarity**2) ** 0.5, 0.0],
        ],
        dtype=float,
    )

    groups = find_scene_groups(assets, embeddings, similarity_threshold=0.80, neighbors_k=2)

    assert len(groups) == 1
    assert groups[0].avg_confidence >= 0.80
    assert sorted(m.path for m in groups[0].members) == ["/x/a.jpg", "/x/b.jpg"]


def test_scene_groups_sorted_by_confidence_desc() -> None:
    t = datetime(2026, 1, 1, 12, 0, 0)
    a1 = _asset("/x/a1.jpg", phash_int=0, captured_at=t)
    a2 = _asset("/x/a2.jpg", phash_int=1, captured_at=t + timedelta(minutes=5))
    b1 = _asset("/x/b1.jpg", phash_int=0, captured_at=None)
    b2 = _asset("/x/b2.jpg", phash_int=15, captured_at=None)

    assets = [a1, a2, b1, b2]
    # Pair a1/a2 has higher base similarity and gets boosts.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.88, (1 - 0.88**2) ** 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.83, (1 - 0.83**2) ** 0.5],
        ],
        dtype=float,
    )

    groups = find_scene_groups(assets, embeddings, similarity_threshold=0.80, neighbors_k=4)

    assert len(groups) == 2
    assert groups[0].avg_confidence >= groups[1].avg_confidence

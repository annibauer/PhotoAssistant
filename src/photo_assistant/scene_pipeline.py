from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from .hash_pipeline import hamming_distance
from .models import PhotoAsset, SceneGroup


def generate_embeddings(
    assets: list[PhotoAsset],
    thumb_size: int = 64,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []

    for asset in assets:
        with Image.open(asset.path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((thumb_size, thumb_size))

            arr = np.asarray(rgb, dtype=np.float32) / 255.0

            # Combine low-resolution grayscale signal with simple color statistics.
            gray = arr.mean(axis=2)
            gray_small = Image.fromarray((gray * 255).astype(np.uint8)).resize((16, 16), Image.BILINEAR)
            gray_vec = np.asarray(gray_small, dtype=np.float32).reshape(-1) / 255.0

            channel_mean = arr.mean(axis=(0, 1))
            channel_std = arr.std(axis=(0, 1))
            feat = np.concatenate([gray_vec, channel_mean, channel_std]).astype(np.float32)

            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            embeddings.append(feat)

    if not embeddings:
        return np.zeros((0, 262), dtype=np.float32)

    return np.vstack(embeddings)


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


def _confidence_boost(a: PhotoAsset, b: PhotoAsset) -> float:
    boost = 0.0
    if a.captured_at and b.captured_at:
        if abs(a.captured_at - b.captured_at) <= timedelta(hours=2):
            boost += 0.05
    if hamming_distance(a.phash_int, b.phash_int) <= 12:
        boost += 0.05
    return boost


def find_scene_groups(
    assets: list[PhotoAsset],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.78,
    neighbors_k: int = 12,
) -> list[SceneGroup]:
    if len(assets) < 2:
        return []

    nbrs = NearestNeighbors(n_neighbors=min(neighbors_k, len(assets)), metric="cosine")
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    uf = _UnionFind(len(assets))
    pair_confidences: dict[tuple[int, int], float] = {}

    for i in range(len(assets)):
        for d, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = 1.0 - float(d)
            confidence = min(0.99, sim + _confidence_boost(assets[i], assets[j]))
            if confidence >= similarity_threshold:
                uf.union(i, j)
                key = (min(i, int(j)), max(i, int(j)))
                pair_confidences[key] = max(pair_confidences.get(key, 0.0), confidence)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(assets)):
        clusters[uf.find(idx)].append(idx)

    groups: list[SceneGroup] = []
    scene_idx = 1
    for member_indexes in clusters.values():
        if len(member_indexes) < 2:
            continue

        member_assets = [assets[i] for i in member_indexes]
        scores: list[float] = []
        output_pair_conf: dict[tuple[str, str], float] = {}

        for a_pos, a_idx in enumerate(member_indexes):
            for b_idx in member_indexes[a_pos + 1 :]:
                key = (min(a_idx, b_idx), max(a_idx, b_idx))
                if key in pair_confidences:
                    score = pair_confidences[key]
                    scores.append(score)
                    output_pair_conf[(assets[a_idx].path, assets[b_idx].path)] = score

        if not scores:
            continue

        groups.append(
            SceneGroup(
                group_id=f"scene-{scene_idx}",
                members=member_assets,
                avg_confidence=float(np.mean(scores)),
                pair_confidence=output_pair_conf,
            )
        )
        scene_idx += 1

    groups.sort(key=lambda g: g.avg_confidence, reverse=True)
    return groups

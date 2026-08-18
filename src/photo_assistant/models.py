from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class PhotoAsset:
    path: str
    sha256: str
    phash_hex: str
    phash_int: int
    captured_at: datetime | None
    embedding: list[float] | None = None


@dataclass(slots=True)
class DuplicateGroup:
    group_id: str
    group_type: str
    members: list[PhotoAsset]


@dataclass(slots=True)
class SceneGroup:
    group_id: str
    members: list[PhotoAsset]
    avg_confidence: float
    pair_confidence: dict[tuple[str, str], float] = field(default_factory=dict)

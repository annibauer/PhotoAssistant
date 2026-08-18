from __future__ import annotations

import pytest

from photo_assistant.hash_pipeline import hamming_distance
from photo_assistant.models import PhotoAsset
from photo_assistant.review import build_decision_table


@pytest.fixture
def sample_members() -> dict[str, list[str]]:
    # Fixture provides reusable, deterministic input data.
    return {"g1": ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"]}


@pytest.fixture
def sample_asset() -> PhotoAsset:
    # Fixture returns a domain model object used across tests.
    return PhotoAsset(path="/x/a.jpg", sha256="sha", phash_hex="0", phash_int=0, captured_at=None)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (0b0000, 0b0000, 0),
        (0b0000, 0b1111, 4),
        (0b1010, 0b1000, 1),
    ],
)
def test_hamming_distance_parametrized(left: int, right: int, expected: int) -> None:
    # Parametrization runs one test body over multiple input cases.
    assert hamming_distance(left, right) == expected


def test_build_decision_table_with_fixture(sample_members: dict[str, list[str]]) -> None:
    choices = {"g1": "/x/b.jpg"}

    decision_df = build_decision_table(choices, sample_members)
    by_path = {row["photo_path"]: row["decision"] for _, row in decision_df.iterrows()}

    assert by_path["/x/b.jpg"] == "keep"
    assert by_path["/x/a.jpg"] == "discard"
    assert by_path["/x/c.jpg"] == "discard"


def test_sample_asset_fixture_shape(sample_asset: PhotoAsset) -> None:
    # This test shows how a model fixture is injected by name.
    assert sample_asset.path == "/x/a.jpg"
    assert sample_asset.sha256 == "sha"
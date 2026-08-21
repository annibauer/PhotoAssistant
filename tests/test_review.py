from __future__ import annotations

from pathlib import Path

import pandas as pd

from photo_assistant.models import DuplicateGroup, PhotoAsset, SceneGroup
from photo_assistant.review import (
    build_decision_table,
    build_group_table,
    build_scene_table,
    export_analysis,
    move_discards_to_folder,
    move_keeps_to_folder,
)


def _asset(path: str) -> PhotoAsset:
    return PhotoAsset(path=path, sha256=path, phash_hex="0", phash_int=0, captured_at=None)


def test_build_group_table_and_scene_table() -> None:
    exact = DuplicateGroup(group_id="exact-1", group_type="exact", members=[_asset("/x/a.jpg"), _asset("/x/b.jpg")])
    near = DuplicateGroup(group_id="near-1", group_type="near", members=[_asset("/x/c.jpg")])
    scene = SceneGroup(group_id="scene-1", members=[_asset("/x/d.jpg")], avg_confidence=0.91)

    duplicate_df = build_group_table([exact], [near])
    scene_df = build_scene_table([scene])

    assert set(duplicate_df.columns) == {"group_id", "group_type", "photo_path"}
    assert len(duplicate_df) == 3
    assert set(scene_df.columns) == {"group_id", "avg_confidence", "photo_path"}
    assert len(scene_df) == 1


def test_build_decision_table_labels_keep_and_discard() -> None:
    members = {
        "g1": ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"],
    }
    choices = {"g1": "/x/b.jpg"}

    decision_df = build_decision_table(choices, members)
    by_path = {row["photo_path"]: row["decision"] for _, row in decision_df.iterrows()}

    assert by_path["/x/b.jpg"] == "keep"
    assert by_path["/x/a.jpg"] == "discard"
    assert by_path["/x/c.jpg"] == "discard"


def test_build_decision_table_allows_multiple_keeps() -> None:
    members = {
        "g1": ["/x/a.jpg", "/x/b.jpg", "/x/c.jpg"],
    }
    choices = {"g1": ["/x/a.jpg", "/x/c.jpg"]}

    decision_df = build_decision_table(choices, members)
    by_path = {row["photo_path"]: row["decision"] for _, row in decision_df.iterrows()}

    assert by_path["/x/a.jpg"] == "keep"
    assert by_path["/x/b.jpg"] == "discard"
    assert by_path["/x/c.jpg"] == "keep"


def test_export_analysis_writes_csvs(tmp_path: Path) -> None:
    duplicate_df = pd.DataFrame([{"group_id": "g1", "group_type": "exact", "photo_path": "/x/a.jpg"}])
    scene_df = pd.DataFrame([{"group_id": "s1", "avg_confidence": 0.8, "photo_path": "/x/b.jpg"}])
    decision_df = pd.DataFrame([{"group_id": "g1", "photo_path": "/x/a.jpg", "decision": "keep", "kept_path": "/x/a.jpg"}])

    export_analysis(tmp_path, duplicate_df, scene_df, decision_df)

    assert (tmp_path / "duplicate_groups.csv").exists()
    assert (tmp_path / "scene_groups.csv").exists()
    assert (tmp_path / "review_decisions.csv").exists()


def test_move_discards_to_folder_moves_only_discard_rows(tmp_path: Path) -> None:
    keep_file = tmp_path / "keep.jpg"
    discard_file = tmp_path / "discard.jpg"
    keep_file.write_bytes(b"keep")
    discard_file.write_bytes(b"discard")

    decision_df = pd.DataFrame(
        [
            {"group_id": "g1", "photo_path": str(keep_file), "decision": "keep", "kept_path": str(keep_file)},
            {
                "group_id": "g1",
                "photo_path": str(discard_file),
                "decision": "discard",
                "kept_path": str(keep_file),
            },
        ]
    )

    target_dir = tmp_path / "discarded"
    moved_df = move_discards_to_folder(decision_df, target_dir)

    assert len(moved_df) == 1
    assert keep_file.exists()
    assert not discard_file.exists()
    moved_to = Path(str(moved_df.iloc[0]["moved_to"]))
    assert moved_to.exists()
    assert moved_to.parent == target_dir


def test_move_keeps_to_folder_moves_only_keep_rows(tmp_path: Path) -> None:
    keep_file = tmp_path / "keep.jpg"
    discard_file = tmp_path / "discard.jpg"
    keep_file.write_bytes(b"keep")
    discard_file.write_bytes(b"discard")

    decision_df = pd.DataFrame(
        [
            {"group_id": "g1", "photo_path": str(keep_file), "decision": "keep", "kept_path": str(keep_file)},
            {
                "group_id": "g1",
                "photo_path": str(discard_file),
                "decision": "discard",
                "kept_path": str(keep_file),
            },
        ]
    )

    target_dir = tmp_path / "kept"
    moved_df = move_keeps_to_folder(decision_df, target_dir)

    assert len(moved_df) == 1
    assert not keep_file.exists()
    assert discard_file.exists()
    moved_to = Path(str(moved_df.iloc[0]["moved_to"]))
    assert moved_to.exists()
    assert moved_to.parent == target_dir
    assert moved_df.iloc[0]["decision"] == "keep"
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .models import DuplicateGroup, SceneGroup


def build_group_table(exact_groups: list[DuplicateGroup], near_groups: list[DuplicateGroup]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group in [*exact_groups, *near_groups]:
        for member in group.members:
            rows.append(
                {
                    "group_id": group.group_id,
                    "group_type": group.group_type,
                    "photo_path": member.path,
                }
            )
    return pd.DataFrame(rows)


def build_scene_table(scene_groups: list[SceneGroup]) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for group in scene_groups:
        for member in group.members:
            rows.append(
                {
                    "group_id": group.group_id,
                    "avg_confidence": group.avg_confidence,
                    "photo_path": member.path,
                }
            )
    return pd.DataFrame(rows)


def build_decision_table(choices: dict[str, str], group_members: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group_id, keep_path in choices.items():
        for member_path in group_members.get(group_id, []):
            rows.append(
                {
                    "group_id": group_id,
                    "photo_path": member_path,
                    "decision": "keep" if member_path == keep_path else "discard",
                    "kept_path": keep_path,
                }
            )
    return pd.DataFrame(rows)


def export_analysis(
    out_dir: Path,
    duplicate_table: pd.DataFrame,
    scene_table: pd.DataFrame,
    decision_table: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    duplicate_table.to_csv(out_dir / "duplicate_groups.csv", index=False)
    scene_table.to_csv(out_dir / "scene_groups.csv", index=False)
    if decision_table is not None:
        decision_table.to_csv(out_dir / "review_decisions.csv", index=False)


def move_discards_to_folder(decision_table: pd.DataFrame, target_dir: Path) -> pd.DataFrame:
    target_dir.mkdir(parents=True, exist_ok=True)
    moved_rows: list[dict[str, str]] = []

    discard_rows = decision_table[decision_table["decision"] == "discard"]
    for _, row in discard_rows.iterrows():
        source = Path(str(row["photo_path"]))
        if not source.exists():
            continue

        destination = target_dir / source.name
        if destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            count = 1
            while destination.exists():
                destination = target_dir / f"{stem}_{count}{suffix}"
                count += 1

        shutil.move(str(source), str(destination))
        moved_rows.append(
            {
                "group_id": str(row["group_id"]),
                "original_path": str(source),
                "moved_to": str(destination),
            }
        )

    return pd.DataFrame(moved_rows)

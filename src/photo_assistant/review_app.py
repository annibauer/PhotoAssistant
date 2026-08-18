from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from photo_assistant.hash_pipeline import (
    build_assets,
    find_exact_duplicate_groups,
    find_near_duplicate_groups,
    list_images,
)
from photo_assistant.review import (
    build_decision_table,
    build_group_table,
    build_scene_table,
    export_analysis,
    move_discards_to_folder,
)
from photo_assistant.scene_pipeline import find_scene_groups, generate_embeddings

st.set_page_config(page_title="Photo Assistant", page_icon="🖼️", layout="wide")

st.title("Photo Assistant")
st.caption("Detect exact duplicates, near duplicates, and same-scene photos.")


@st.cache_data(show_spinner=False)
def _analyze(input_dir: str, phash_threshold: int, scene_threshold: float) -> dict:
    root = Path(input_dir).expanduser().resolve()
    image_paths = list_images(root)
    assets = build_assets(image_paths)

    exact_groups = find_exact_duplicate_groups(assets)
    near_groups = find_near_duplicate_groups(assets, exact_groups, max_hamming_distance=phash_threshold)

    dedupe_paths = {member.path for group in [*exact_groups, *near_groups] for member in group.members}
    remaining = [asset for asset in assets if asset.path not in dedupe_paths]

    if remaining:
        embeddings = generate_embeddings(remaining)
        scene_groups = find_scene_groups(remaining, embeddings, similarity_threshold=scene_threshold)
    else:
        scene_groups = []

    return {
        "assets": assets,
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "scene_groups": scene_groups,
    }


def _render_group(group_id: str, member_paths: list[str], group_label: str, choices: dict[str, str]) -> None:
    st.subheader(f"{group_label} - {group_id}")

    cols = st.columns(min(4, len(member_paths)))
    for idx, path in enumerate(member_paths):
        with cols[idx % len(cols)]:
            st.image(path, caption=Path(path).name, use_container_width=True)

    default_idx = member_paths.index(choices[group_id]) if group_id in choices and choices[group_id] in member_paths else 0
    selected = st.radio(
        f"Choose one photo to keep for {group_id}",
        options=member_paths,
        index=default_idx,
        format_func=lambda p: Path(p).name,
        key=f"choice-{group_id}",
        horizontal=True,
    )
    choices[group_id] = selected
    st.divider()


with st.sidebar:
    input_dir = st.text_input("Photo folder", value="")
    phash_threshold = st.slider("Near-duplicate pHash distance", min_value=1, max_value=16, value=8)
    scene_threshold = st.slider("Scene similarity threshold", min_value=0.60, max_value=0.95, value=0.78)
    analyze_clicked = st.button("Analyze photos", type="primary")

if analyze_clicked:
    if not input_dir:
        st.error("Please provide a photo folder path.")
    elif not Path(input_dir).expanduser().exists():
        st.error("Folder does not exist.")
    else:
        with st.spinner("Analyzing photos. This can take a while for large folders..."):
            st.session_state["analysis"] = _analyze(input_dir, phash_threshold, scene_threshold)

analysis = st.session_state.get("analysis")

if analysis:
    exact_groups = analysis["exact_groups"]
    near_groups = analysis["near_groups"]
    scene_groups = analysis["scene_groups"]

    st.success(
        f"Found {len(exact_groups)} exact groups, {len(near_groups)} near groups, and {len(scene_groups)} scene groups."
    )

    st.header("1) Duplicate Review")
    duplicate_choices = st.session_state.setdefault("duplicate_choices", {})
    group_members_map: dict[str, list[str]] = {}

    for group in [*exact_groups, *near_groups]:
        member_paths = [m.path for m in group.members]
        group_members_map[group.group_id] = member_paths
        if group.group_id not in duplicate_choices:
            duplicate_choices[group.group_id] = member_paths[0]
        _render_group(group.group_id, member_paths, group.group_type, duplicate_choices)

    st.header("2) Similar Scene Review")
    scene_choices = st.session_state.setdefault("scene_choices", {})
    for group in scene_groups:
        member_paths = [m.path for m in group.members]
        group_members_map[group.group_id] = member_paths
        if group.group_id not in scene_choices:
            scene_choices[group.group_id] = member_paths[0]
        _render_group(group.group_id, member_paths, f"scene (avg {group.avg_confidence:.2f})", scene_choices)

    all_choices = {**duplicate_choices, **scene_choices}

    decision_df = build_decision_table(all_choices, group_members_map)
    st.header("3) Decisions")
    st.dataframe(decision_df, use_container_width=True)

    csv_data = decision_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download decisions CSV",
        data=csv_data,
        file_name="review_decisions.csv",
        mime="text/csv",
    )

    export_dir = st.text_input("Export analysis directory", value="analysis_output")
    if st.button("Export all analysis CSVs"):
        duplicate_df = build_group_table(exact_groups, near_groups)
        scene_df = build_scene_table(scene_groups)
        export_analysis(Path(export_dir), duplicate_df, scene_df, decision_df)
        st.success(f"Exported CSV files to {Path(export_dir).resolve()}")

    st.subheader("Optional: Apply discard decisions")
    discard_dir = st.text_input("Move discarded photos to", value="analysis_output/discarded_review")
    if st.button("Move discard photos"):
        moved_df = move_discards_to_folder(decision_df, Path(discard_dir))
        if moved_df.empty:
            st.warning("No files were moved. Check whether discard files still exist at original locations.")
        else:
            st.success(f"Moved {len(moved_df)} photos to {Path(discard_dir).resolve()}")
            st.dataframe(moved_df, use_container_width=True)
else:
    st.info("Set a folder in the sidebar and click Analyze photos.")
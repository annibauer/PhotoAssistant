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
    move_keeps_to_folder,
)
from photo_assistant.scene_pipeline import find_scene_groups, generate_embeddings

st.set_page_config(page_title="Photo Assistant", page_icon="🖼️", layout="wide")

st.title("Photo Assistant")
st.caption("Detect exact duplicates, near duplicates, and same-scene photos.")

PRIORITIZED_PREFIXES = [
    # str(Path("/Users/annibauer/Documents/__Photos.nosync/__Photo-Albums.nosync/China Praktikum 2019").resolve()),
    # str(Path("/Users/annibauer/Documents/__Photos.nosync/20").resolve()),
    # str(Path("/Users/annibauer/Documents/__Photos.nosync/2023/2023_Vietnam_HagiangLoop").resolve()),
    # str(Path("/Users/annibauer/Documents/__Photos.nosync/2023/2023_BackpackTrip_Kashgar_ChongChing").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2024/2024_Berlin").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2024/2024_Bodensee").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2024/2024_Nuernberg").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2016/2016_July").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2016/2016_Ha").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2016/2016_Amst").resolve()),

]

DEPRIORITIZED_PREFIXES = [
    str(Path("/Users/annibauer/Documents/__Photos.nosync/Extra/Mareike-Graben").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/Extra/Geneva Visits").resolve()), 
    str(Path("/Users/annibauer/Documents/__Photos.nosync/2010 - 2015 Facebook_HighSchool").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/Edits_Highschool").resolve()),
    str(Path('/Users/annibauer/Documents/__Photos.nosync/2010 - 2015 Facebook_HighSchool').resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/Extra/Uni").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/Extra/Uni_CameraRoll").resolve()),
    str(Path("/Users/annibauer/Documents/__Photos.nosync/Profile Pictures Childhood").resolve()),
    
]


def _analyze(
    input_dir: str,
    phash_threshold: int,
    scene_threshold: float,
    enable_scene_grouping: bool,
    max_scene_assets: int,
) -> dict:
    root = Path(input_dir).expanduser().resolve()
    image_paths = list_images(root)
    assets = build_assets(image_paths)

    exact_groups = find_exact_duplicate_groups(assets)
    near_groups = find_near_duplicate_groups(assets, exact_groups, max_hamming_distance=phash_threshold)

    dedupe_paths = {member.path for group in [*exact_groups, *near_groups] for member in group.members}
    remaining = [asset for asset in assets if asset.path not in dedupe_paths]

    scene_candidates = remaining
    skipped_scene_assets = 0
    if enable_scene_grouping and max_scene_assets > 0 and len(remaining) > max_scene_assets:
        scene_candidates = sorted(remaining, key=lambda a: a.path)[:max_scene_assets]
        skipped_scene_assets = len(remaining) - len(scene_candidates)

    if enable_scene_grouping and scene_candidates:
        embeddings = generate_embeddings(scene_candidates)
        scene_groups = find_scene_groups(scene_candidates, embeddings, similarity_threshold=scene_threshold)
    else:
        scene_groups = []

    return {
        "assets": assets,
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "scene_groups": scene_groups,
        "scene_candidate_count": len(scene_candidates),
        "skipped_scene_assets": skipped_scene_assets,
    }


def _render_group(group_id: str, member_paths: list[str], group_label: str, choices: dict[str, list[str]]) -> None:
    st.subheader(f"{group_label} - {group_id}")

    cols = st.columns(min(4, len(member_paths)))
    for idx, path in enumerate(member_paths):
        with cols[idx % len(cols)]:
            st.image(path, caption=Path(path).name, use_container_width=True)

    existing_selection = choices.get(group_id, [])
    default_selection = [path for path in existing_selection if path in member_paths]
    if not default_selection:
        default_selection = [member_paths[0]]

    option_tokens: list[str] = []
    token_to_path: dict[str, str] = {}
    token_to_label: dict[str, str] = {}
    for idx, path in enumerate(member_paths):
        token = f"{idx}:{path}"
        option_tokens.append(token)
        token_to_path[token] = path
        token_to_label[token] = _display_path_with_two_levels(path)

    default_tokens = [token for token, path in token_to_path.items() if path in default_selection]

    selected = st.multiselect(
        f"Choose one or more photos to keep for {group_id}",
        options=option_tokens,
        default=default_tokens,
        format_func=lambda token: token_to_label[token],
        key=f"choice-{group_id}",
        help="You can keep multiple photos in a group. One is selected by default.",
    )

    if not selected:
        selected = [option_tokens[0]]

    choices[group_id] = [token_to_path[token] for token in selected]
    st.divider()


def _preferred_order(paths: list[str]) -> list[str]:
    def _priority_rank(path: str) -> int:
        candidate = str(Path(path).expanduser().resolve())
        if any(candidate.startswith(prefix) for prefix in PRIORITIZED_PREFIXES):
            return 0
        if any(candidate.startswith(prefix) for prefix in DEPRIORITIZED_PREFIXES):
            return 2
        return 1

    # Prioritize configured folders first, keep neutral paths in the middle, then deprioritize sync/archive folders.
    return sorted(paths, key=lambda p: (_priority_rank(p), str(Path(p).name).lower(), p.lower()))


def _display_path_with_two_levels(path: str) -> str:
    p = Path(path)
    parent_1 = p.parent.name if p.parent.name else "."
    parent_2 = p.parent.parent.name if p.parent.parent.name else "."
    return f"{parent_2}/{parent_1}/{p.name}"


with st.sidebar:
    input_dir = st.text_input("Photo folder", value="")
    phash_threshold = st.slider("Near-duplicate pHash distance", min_value=1, max_value=16, value=8)
    scene_threshold = st.slider("Scene similarity threshold", min_value=0.60, max_value=0.95, value=0.78)
    enable_scene_grouping = st.checkbox(
        "Enable scene grouping (more CPU/RAM)",
        value=True,
        help="Disable this if you only want exact + near duplicate review with lower memory usage.",
    )
    max_scene_assets = st.number_input(
        "Max photos used for scene grouping",
        min_value=100,
        max_value=10000,
        value=1500,
        step=100,
        help="If remaining photos exceed this limit, only the first N (sorted by path) are used for scene grouping.",
    )
    analyze_clicked = st.button("Analyze photos", type="primary")

if analyze_clicked:
    if not input_dir:
        st.error("Please provide a photo folder path.")
    elif not Path(input_dir).expanduser().exists():
        st.error("Folder does not exist.")
    else:
        with st.spinner("Analyzing photos. This can take a while for large folders..."):
            st.session_state["analysis"] = _analyze(
                input_dir,
                phash_threshold,
                scene_threshold,
                enable_scene_grouping,
                int(max_scene_assets),
            )

analysis = st.session_state.get("analysis")

if analysis:
    exact_groups = analysis["exact_groups"]
    near_groups = analysis["near_groups"]
    scene_groups = analysis["scene_groups"]
    skipped_scene_assets = analysis.get("skipped_scene_assets", 0)
    scene_candidate_count = analysis.get("scene_candidate_count", 0)

    st.success(
        f"Found {len(exact_groups)} exact groups, {len(near_groups)} near groups, and {len(scene_groups)} scene groups."
    )
    if skipped_scene_assets > 0:
        st.warning(
            f"Scene grouping used {scene_candidate_count} photos and skipped {skipped_scene_assets} to reduce memory usage."
        )

    st.header("1) Duplicate Review")
    duplicate_choices = st.session_state.setdefault("duplicate_choices", {})
    group_members_map: dict[str, list[str]] = {}

    for group in [*exact_groups, *near_groups]:
        member_paths = _preferred_order([m.path for m in group.members])
        group_members_map[group.group_id] = member_paths
        if group.group_id not in duplicate_choices:
            duplicate_choices[group.group_id] = [member_paths[0]]
        elif isinstance(duplicate_choices[group.group_id], str):
            duplicate_choices[group.group_id] = [duplicate_choices[group.group_id]]
        _render_group(group.group_id, member_paths, group.group_type, duplicate_choices)

    st.header("2) Similar Scene Review")
    scene_choices = st.session_state.setdefault("scene_choices", {})
    for group in scene_groups:
        member_paths = _preferred_order([m.path for m in group.members])
        group_members_map[group.group_id] = member_paths
        if group.group_id not in scene_choices:
            scene_choices[group.group_id] = [member_paths[0]]
        elif isinstance(scene_choices[group.group_id], str):
            scene_choices[group.group_id] = [scene_choices[group.group_id]]
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

    st.subheader("Optional: Apply keep decisions")
    keep_dir = st.text_input("Move kept photos to", value="analysis_output/kept_review")
    if st.button("Move keep photos"):
        moved_keep_df = move_keeps_to_folder(decision_df, Path(keep_dir))
        if moved_keep_df.empty:
            st.warning("No files were moved. Check whether keep files still exist at original locations.")
        else:
            st.success(f"Moved {len(moved_keep_df)} photos to {Path(keep_dir).resolve()}")
            st.dataframe(moved_keep_df, use_container_width=True)
else:
    st.info("Set a folder in the sidebar and click Analyze photos.")
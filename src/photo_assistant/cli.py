from __future__ import annotations

import argparse
from pathlib import Path

from .hash_pipeline import build_assets, find_exact_duplicate_groups, find_near_duplicate_groups, list_images
from .review import build_group_table, build_scene_table, export_analysis
from .scene_pipeline import find_scene_groups, generate_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze photos for duplicates and similar scenes")
    parser.add_argument("--input", required=True, type=Path, help="Root photo directory")
    parser.add_argument("--out", default=Path("analysis_output"), type=Path, help="Output directory")
    parser.add_argument("--phash-threshold", default=8, type=int, help="Near-duplicate pHash distance")
    parser.add_argument("--scene-threshold", default=0.78, type=float, help="Scene confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.input)
    assets = build_assets(image_paths)

    exact_groups = find_exact_duplicate_groups(assets)
    near_groups = find_near_duplicate_groups(assets, exact_groups, max_hamming_distance=args.phash_threshold)

    dedupe_member_paths = {member.path for group in [*exact_groups, *near_groups] for member in group.members}
    remaining_assets = [asset for asset in assets if asset.path not in dedupe_member_paths]

    if remaining_assets:
        embeddings = generate_embeddings(remaining_assets)
        scene_groups = find_scene_groups(remaining_assets, embeddings, similarity_threshold=args.scene_threshold)
    else:
        scene_groups = []
        
    scene_groups = []

    duplicate_table = build_group_table(exact_groups, near_groups)
    scene_table = build_scene_table(scene_groups)
    export_analysis(args.out, duplicate_table, scene_table)

    print(f"Analyzed {len(assets)} photos")
    print(f"Exact duplicate groups: {len(exact_groups)}")
    print(f"Near duplicate groups: {len(near_groups)}")
    print(f"Scene groups: {len(scene_groups)}")
    print(f"Saved output in: {args.out}")


if __name__ == "__main__":
    main()
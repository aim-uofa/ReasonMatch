#!/usr/bin/env python3
"""End-to-end buffer demo that visualizes point selection across curriculum stages.

Synthetic data is generated per step:
  • two Gaussian-noise images
  • ~30 random correspondences

For each stage we run the point filtering/assignment pipeline and dump:
  • generated images (A/B) with annotations
  • JSON metadata describing core/distractor labels
  • CSV snapshot of raw matches & the chosen subsets

Example:
  python scripts/test_matching_buffer.py --preset basic --output out/test_buffer
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from my_recipe.buffer.dynamic_task_buffer import (
    BufferConfig,
    MatchingCurriculumConfig,
    matching_curriculum_from_dict,
)
from my_recipe.mydatasets.anno_db import annotate_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="basic", help="Curriculum preset (basic / text_first / ...).")
    parser.add_argument("--config", help="Optional curriculum YAML path; overrides --preset.")
    parser.add_argument("--output", required=True, help="Directory to save visualizations and logs.")
    parser.add_argument(
        "--samples-per-stage",
        type=int,
        default=2,
        help="How many synthetic samples to draw for each curriculum stage.",
    )
    parser.add_argument("--matches", type=int, default=30, help="Number of raw matches per sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", help="Optional log file for DynamicTaskBuffer JSON logs.")
    return parser.parse_args()


def load_curriculum(args: argparse.Namespace) -> MatchingCurriculumConfig:
    if args.config:
        cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    else:
        cfg_path = Path(__file__).resolve().parents[1] / "my_recipe" / "config" / "matching_curriculum" / f"{args.preset}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    if "matching_curriculum" in cfg:
        cfg = cfg["matching_curriculum"]

    return matching_curriculum_from_dict(cfg)


def gaussian_noise_image(size: int = 512) -> Image.Image:
    arr = (np.random.normal(0.5, 0.2, (size, size, 3)).clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_matches(num: int) -> list[dict]:
    coords = np.random.uniform(0, 1000, size=(num, 4)).astype(int)
    matches = []
    for i in range(num):
        matches.append({"x1": int(coords[i][0]), "y1": int(coords[i][1]), "x2": int(coords[i][2]), "y2": int(coords[i][3]), "is_relative": True})
    return matches


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    curriculum = load_curriculum(args)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Curriculum stages:")
    for idx, stage in enumerate(curriculum.stages):
        print(f"  {idx + 1}. {stage.stage_id}")

    from my_recipe.buffer.dynamic_task_buffer import DynamicTaskBuffer, MatchingFilterConfig, GroundingPointConfig

    buffer = DynamicTaskBuffer(
        BufferConfig(
            task_mode="matching",
            matching_curriculum=curriculum,
            matching_filter=MatchingFilterConfig(strategy="static"),
            grounding_points=GroundingPointConfig(),
            log_file=args.log_file,
        ),
        raw_dataset=None,
    )

    global_step = 0
    for stage in curriculum.stages:
        stage_metric = 0.0
        stage_cfg = stage.reference_config()
        params = buffer._suggest_matching_params(
            stage_metric,
            stage_cfg=stage_cfg,
            aggregator_stage_id=stage.stage_id,
        )
        buffer.current_matching_params = params

        for sample_idx in range(args.samples_per_stage):
            global_step += 1

            image1 = gaussian_noise_image()
            image2 = gaussian_noise_image()
            matches = random_matches(args.matches)

            raw_sample = {
                "db_idx": global_step,
                "image1": image1,
                "image2": image2,
                "matches": matches,
                "height": 512,
                "width": 512,
                "processor": None,
                "tokenizer": None,
                "min_pixels": 0,
                "max_pixels": 0,
                "config": {},
            }

            matches_abs = buffer._normalize_matches_to_absolute(raw_sample)
            safe_matches = buffer._collect_safe_matches(raw_sample, matches_abs)

            if len(safe_matches) < params.get("min_points", 1):
                raise RuntimeError(
                    f"Safe match pool ({len(safe_matches)}) below min_points={params.get('min_points', 1)} "
                    f"for stage {stage.stage_id}"
                )

            core_matches, distractor_pool = buffer._select_core_matches(raw_sample, safe_matches, params, stage_cfg)
            effective_min_distance = max(float(params.get("min_distance", 10.0)), buffer._annotation_guard_px)
            distractors_a, distractors_b = buffer._sample_stage_distractors(
                raw_sample, core_matches, distractor_pool, stage_cfg, effective_min_distance
            )
            assignment = buffer._build_label_assignment(core_matches, distractors_a, distractors_b, stage_cfg)

            payload_a = buffer._prepare_view_payload(raw_sample, assignment, stage_cfg, "image1")
            payload_b = buffer._prepare_view_payload(raw_sample, assignment, stage_cfg, "image2")

            stage_dir = out_dir / f"stage_{stage.stage_id}" / f"sample_{sample_idx:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            (stage_dir / "raw_matches.json").write_text(json.dumps(matches, indent=2), encoding="utf-8")
            save_json(assignment, stage_dir / "assignment.json")

            if payload_a["image"]:
                payload_a["image"].save(stage_dir / "viewA.jpg")
            if payload_b["image"]:
                payload_b["image"].save(stage_dir / "viewB.jpg")

            save_json({"points": payload_a["image_points"]}, stage_dir / "viewA_points.json")
            save_json({"points": payload_b["image_points"]}, stage_dir / "viewB_points.json")

            csv_rows = []
            for entry in assignment["core"]:
                csv_rows.append(
                    {
                        "type": "core",
                        "label_a": entry["label_a"],
                        "label_b": entry["label_b"],
                        "x1": entry["match"]["x1"],
                        "y1": entry["match"]["y1"],
                        "x2": entry["match"]["x2"],
                        "y2": entry["match"]["y2"],
                    }
                )
            for entry in assignment["distractors_a"]:
                point = entry["point"]
                is_relative = point.get("is_relative", False)
                x = point.get("x", 0.0)
                y = point.get("y", 0.0)
                if not is_relative:
                    x = int(round(float(x)))
                    y = int(round(float(y)))
                csv_rows.append(
                    {
                        "type": "distractor_a",
                        "label": entry["label"],
                        "x1": x,
                        "y1": y,
                        "x2": None,
                        "y2": None,
                    }
                )
            for entry in assignment["distractors_b"]:
                point = entry["point"]
                is_relative = point.get("is_relative", False)
                x = point.get("x", 0.0)
                y = point.get("y", 0.0)
                if not is_relative:
                    x = int(round(float(x)))
                    y = int(round(float(y)))
                csv_rows.append(
                    {
                        "type": "distractor_b",
                        "label": entry["label"],
                        "x1": None,
                        "y1": None,
                        "x2": x,
                        "y2": y,
                    }
                )
            save_csv(csv_rows, stage_dir / "selection.csv")

            history = {
                "stage": stage.stage_id,
                "sample_idx": sample_idx,
                "core_count": len(core_matches),
                "distractor_a": len(distractors_a),
                "distractor_b": len(distractors_b),
                "params": params,
            }
            save_json(history, stage_dir / "summary.json")

    print(f"Artifacts saved under {out_dir}")
    if args.log_file:
        print(f"Buffer log written to {args.log_file}")


if __name__ == "__main__":
    run(parse_args())

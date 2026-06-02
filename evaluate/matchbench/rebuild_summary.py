#!/usr/bin/env python3
"""Recompute summary.json from an existing samples.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_metrics import recompute_from_samples, load_scene_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=str, required=True, help="Path to samples.jsonl to process.")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write summary.json (defaults to samples_dir/new_summary.json).",
    )
    parser.add_argument(
        "--scene-map",
        type=str,
        help="Optional JSON file with per-sample scene labels (indoor/outdoor/object).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    if not samples_path.exists():
        raise FileNotFoundError(f"{samples_path} not found.")
    scene_map = load_scene_map(Path(args.scene_map)) if args.scene_map else None
    summary = recompute_from_samples(samples_path, scene_map=scene_map)
    output_path = Path(args.output) if args.output else samples_path.parent / "new_summary.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Wrote recomputed summary to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize evaluation metrics for a given run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from utils import compare_mappings

OVERLAP_BUCKETS = [
    ("low", 0.0, 0.33),
    ("medium", 0.33, 0.66),
    ("high", 0.66, 1.01),
]


def bucket_from_rank(rank: float | None) -> str | None:
    if rank is None:
        return None
    for name, low, high in OVERLAP_BUCKETS:
        if low <= rank < high:
            return name
    return OVERLAP_BUCKETS[-1][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a run directory under eval_results.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore summary.json and recompute metrics from samples.jsonl.",
    )
    parser.add_argument(
        "--scene_map",
        type=str,
        help="Optional JSON mapping sample_id -> scene label (indoor/outdoor/object) for scene metrics.",
    )
    return parser.parse_args()


def _update(
    stats: Dict[str, dict], key: str, precision: float, recall: float, f1: float
) -> dict:
    entry = stats.setdefault(key, {"count": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0})
    entry["count"] += 1
    entry["prec"] += precision
    entry["rec"] += recall
    entry["f1"] += f1
    return entry


def _finalize(stats: Dict[str, dict]) -> Dict[str, dict]:
    for data in stats.values():
        count = max(data.get("count", 0), 1)
        data["avg_precision"] = data["prec"] / count
        data["avg_recall"] = data["rec"] / count
        data["avg_f1"] = data["f1"] / count
        data.pop("prec", None)
        data.pop("rec", None)
        data.pop("f1", None)
    return stats


def _relative_bucket_stats(samples: List[dict]) -> Dict[str, dict]:
    buckets = {
        "low": {"count": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0},
        "medium": {"count": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0},
        "high": {"count": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0},
    }
    valid = [sample for sample in samples if sample.get("rank") is not None]
    if not valid:
        return {
            name: {"count": 0, "avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0}
            for name in buckets
        }
    ranks = [sample["rank"] for sample in valid]
    min_rank = min(ranks)
    max_rank = max(ranks)
    span = max(max_rank - min_rank, 1e-9)
    for sample in valid:
        relative = (sample["rank"] - min_rank) / span
        if relative <= 1 / 3:
            bucket = "low"
        elif relative <= 2 / 3:
            bucket = "medium"
        else:
            bucket = "high"
        buckets[bucket]["count"] += 1
        buckets[bucket]["prec"] += sample["prec"]
        buckets[bucket]["rec"] += sample["rec"]
        buckets[bucket]["f1"] += sample["f1"]
    finalized: Dict[str, dict] = {}
    for name, data in buckets.items():
        count = max(data["count"], 1)
        finalized[name] = {
            "count": data["count"],
            "avg_precision": data["prec"] / count,
            "avg_recall": data["rec"] / count,
            "avg_f1": data["f1"] / count,
        }
    return finalized


SCENE_LABELS = {"indoor", "outdoor", "object"}
SCENE_DEFAULTS = {
    "uco3d": "object",
    "scannet": "indoor",
}


def load_scene_map(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Scene map {path} not found")
    raw = json.loads(path.read_text(encoding="utf-8"))
    scene_map: Dict[str, str] = {}

    def _extract_scene(value) -> Optional[str]:
        label: Optional[str] = None
        if isinstance(value, str):
            label = value
        elif isinstance(value, dict):
            label = value.get("scene") or value.get("label")
        if label is None:
            return None
        label = label.strip().lower()
        return label if label in SCENE_LABELS else None

    if isinstance(raw, dict):
        for sample_id, payload in raw.items():
            scene = _extract_scene(payload)
            if not scene:
                raise ValueError(f"Invalid scene entry for {sample_id}: {payload}")
            scene_map[str(sample_id)] = scene
    elif isinstance(raw, list):
        for entry in raw:
            sample_id = entry.get("sample_id") if isinstance(entry, dict) else None
            if not sample_id:
                continue
            scene = _extract_scene(entry)
            if not scene:
                raise ValueError(f"Invalid scene entry for {sample_id}: {entry}")
            scene_map[str(sample_id)] = scene
    else:
        raise ValueError("Scene map must be a dict or list of entries")
    return scene_map


def recompute_from_samples(
    samples_path: Path, scene_map: Dict[str, str] | None = None
) -> dict:
    total = 0
    prec_sum = 0.0
    rec_sum = 0.0
    f1_sum = 0.0
    dataset_stats: Dict[str, dict] = {}
    stage_stats: Dict[str, dict] = {}
    difficulty_stats: Dict[str, dict] = {}
    overlap_bucket_stats: Dict[str, dict] = {}
    scene_stats: Dict[str, dict] = {}
    scene_map = scene_map or {}
    json_quality = {
        "total": 0,
        "json_absent": 0,
        "json_parse_fail": 0,
        "request_errors": 0,
    }

    with samples_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            gt = record.get("gt") or {}
            pred = (
                record.get("prediction_parsed")
                if isinstance(record.get("prediction_parsed"), dict)
                else None
            )
            prec, rec, f1, _ = compare_mappings(gt, pred)
            prec_sum += prec
            rec_sum += rec
            f1_sum += f1

            meta = record.get("meta", {}) or {}
            dataset = meta.get("dataset_key") or "unknown"
            stage = meta.get("stage") or "unknown"
            difficulty = meta.get("difficulty") or "unknown"
            overlap = meta.get("overlap")
            rank = meta.get("overlap_rank")
            bucket = bucket_from_rank(rank)
            sample_id = meta.get("sample_id")

            ds_entry = dataset_stats.setdefault(
                dataset,
                {
                    "count": 0,
                    "prec_sum": 0.0,
                    "rec_sum": 0.0,
                    "f1_sum": 0.0,
                    "overlap_sum": 0.0,
                    "overlap_rank_sum": 0.0,
                    "overlap_min": None,
                    "overlap_max": None,
                    "overlap_rank_min": None,
                    "overlap_rank_max": None,
                    "stages": {},
                    "overlap_buckets": {},
                    "difficulties": {},
                },
            )
            ds_entry["count"] += 1
            ds_entry["prec_sum"] += prec
            ds_entry["rec_sum"] += rec
            ds_entry["f1_sum"] += f1
            if overlap is not None:
                ds_entry["overlap_sum"] += overlap
                ds_entry["overlap_min"] = (
                    overlap
                    if ds_entry["overlap_min"] is None
                    else min(ds_entry["overlap_min"], overlap)
                )
                ds_entry["overlap_max"] = (
                    overlap
                    if ds_entry["overlap_max"] is None
                    else max(ds_entry["overlap_max"], overlap)
                )
            if rank is not None:
                ds_entry["overlap_rank_sum"] += rank
                ds_entry["overlap_rank_min"] = (
                    rank
                    if ds_entry["overlap_rank_min"] is None
                    else min(ds_entry["overlap_rank_min"], rank)
                )
                ds_entry["overlap_rank_max"] = (
                    rank
                    if ds_entry["overlap_rank_max"] is None
                    else max(ds_entry["overlap_rank_max"], rank)
                )

            _update(stage_stats, stage, prec, rec, f1)
            _update(difficulty_stats, difficulty, prec, rec, f1)
            _update(ds_entry["stages"], stage, prec, rec, f1)
            _update(ds_entry["difficulties"], difficulty, prec, rec, f1)
            if bucket:
                _update(overlap_bucket_stats, bucket, prec, rec, f1)
                _update(ds_entry["overlap_buckets"], bucket, prec, rec, f1)

            scene_label = scene_map.get(sample_id) if sample_id else None
            if not scene_label:
                scene_label = SCENE_DEFAULTS.get((dataset or "").lower())
            if scene_label:
                scene_entry = scene_stats.setdefault(
                    scene_label,
                    {
                        "count": 0,
                        "prec_sum": 0.0,
                        "rec_sum": 0.0,
                        "f1_sum": 0.0,
                        "stages": {},
                    },
                )
                scene_entry["count"] += 1
                scene_entry["prec_sum"] += prec
                scene_entry["rec_sum"] += rec
                scene_entry["f1_sum"] += f1
                _update(scene_entry["stages"], stage, prec, rec, f1)
                scene_entry.setdefault("samples", []).append(
                    {"rank": rank, "prec": prec, "rec": rec, "f1": f1}
                )

            json_quality["total"] += 1
            if record.get("error"):
                json_quality["request_errors"] += 1
            if record.get("prediction_json_raw") is None:
                json_quality["json_absent"] += 1
            elif not isinstance(record.get("prediction_parsed"), dict):
                json_quality["json_parse_fail"] += 1

    for ds_data in dataset_stats.values():
        count = max(ds_data.get("count", 0), 1)
        ds_data["avg_precision"] = ds_data["prec_sum"] / count
        ds_data["avg_recall"] = ds_data["rec_sum"] / count
        ds_data["avg_f1"] = ds_data["f1_sum"] / count
        ds_data["avg_overlap"] = (
            ds_data["overlap_sum"] / count
            if ds_data["overlap_min"] is not None
            else None
        )
        ds_data["avg_overlap_rank"] = (
            ds_data["overlap_rank_sum"] / count
            if ds_data["overlap_rank_min"] is not None
            else None
        )
        for key in ("prec_sum", "rec_sum", "f1_sum", "overlap_sum", "overlap_rank_sum"):
            ds_data.pop(key, None)
        _finalize(ds_data["stages"])
        _finalize(ds_data["difficulties"])
        _finalize(ds_data["overlap_buckets"])

    overlap_bucket_stats = _finalize(overlap_bucket_stats)
    finalized_scene_stats: Dict[str, dict] = {}
    for label in SCENE_LABELS:
        entry = scene_stats.get(label, None)
        if entry:
            count = max(entry.get("count", 0), 1)
            finalized_scene_stats[label] = {
                "count": entry.get("count", 0),
                "avg_precision": entry.get("prec_sum", 0.0) / count,
                "avg_recall": entry.get("rec_sum", 0.0) / count,
                "avg_f1": entry.get("f1_sum", 0.0) / count,
                "stages": _finalize(entry.get("stages", {})),
                "overlap_buckets": _relative_bucket_stats(entry.get("samples", [])),
            }
        else:
            finalized_scene_stats[label] = {
                "count": 0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "stages": {},
                "overlap_buckets": {},
            }
    json_quality["json_success"] = (
        json_quality["total"]
        - json_quality["json_absent"]
        - json_quality["json_parse_fail"]
    )

    return {
        "total_samples": total,
        "avg_precision": prec_sum / max(total, 1),
        "avg_recall": rec_sum / max(total, 1),
        "avg_f1": f1_sum / max(total, 1),
        "dataset_metrics": dataset_stats,
        "stage_metrics": _finalize(stage_stats),
        "difficulty_metrics": _finalize(difficulty_stats),
        "overlap_buckets": overlap_bucket_stats,
        "scene_metrics": finalized_scene_stats,
        "json_quality": json_quality,
    }


def load_summary(
    run_dir: Path, recompute: bool, scene_map: Dict[str, str] | None = None
) -> dict:
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "samples.jsonl"
    if not summary_path.exists() or not summary_path.is_file() or recompute:
        if not samples_path.exists():
            raise FileNotFoundError(f"{samples_path} not found; cannot recompute.")
        summary = recompute_from_samples(samples_path, scene_map=scene_map)
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary


def print_table(title: str, stats: Dict[str, dict]) -> None:
    if not stats:
        return
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'Key':<20}{'Count':>8}{'Prec':>10}{'Recall':>10}{'F1':>10}"
    print(header)
    for key, data in stats.items():
        print(
            f"{key:<20}{data.get('count', 0):>8}"
            f"{data.get('avg_precision', 0.0):>10.3f}"
            f"{data.get('avg_recall', 0.0):>10.3f}"
            f"{data.get('avg_f1', 0.0):>10.3f}"
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} not found.")
    scene_map = load_scene_map(Path(args.scene_map)) if args.scene_map else {}
    summary = load_summary(
        run_dir, args.recompute, scene_map if args.recompute else None
    )

    print(f"Run: {run_dir.name}")
    print(f"Total samples: {summary.get('total_samples', 0)}")
    print(
        f"Avg precision: {summary.get('avg_precision', 0.0):.3f} | "
        f"Avg recall: {summary.get('avg_recall', 0.0):.3f} | "
        f"Avg F1: {summary.get('avg_f1', 0.0):.3f}"
    )

    print_table("Dataset metrics", summary.get("dataset_metrics", {}))
    print_table("Stage metrics", summary.get("stage_metrics", {}))
    print_table("Difficulty metrics", summary.get("difficulty_metrics", {}))
    print_table("Overlap buckets", summary.get("overlap_buckets", {}))
    print_table("Scene metrics", summary.get("scene_metrics", {}))


if __name__ == "__main__":
    main()

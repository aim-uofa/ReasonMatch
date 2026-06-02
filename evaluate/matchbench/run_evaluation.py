from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import requests
from PIL import Image
from tqdm import tqdm

from model_runners import BaseRunner, OpenAIChatRunner, VLLMChatRunner
from utils import (
    build_prompt,
    compare_mappings,
    extract_json_from_text,
    load_annotation_index,
    read_metadata,
    safe_parse_json,
)

OVERLAP_BUCKETS = [
    ("low", 0.0, 0.33),
    ("medium", 0.33, 0.66),
    ("high", 0.66, 1.01),
]


def encode_image(image: str | Image.Image) -> str:
    buffer = BytesIO()
    if isinstance(image, str):
        Image.open(image).save(buffer, format="JPEG")
    elif isinstance(image, Image.Image):
        image.save(buffer, format="JPEG")
    else:
        raise ValueError("Image should be a file path or PIL Image object.")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model evaluations on the auto-labeled test set.")
    parser.add_argument("--annotation_index", type=str, required=True, help="Path to annotation index JSON.")
    parser.add_argument("--testset_root", type=str, required=True, help="Root folder that stores the testset outputs.")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to store evaluation logs.")
    parser.add_argument("--model_name", type=str, required=True, help="Human-readable model alias (used in file names).")
    parser.add_argument("--runner", choices=["openai", "vllm"], required=True, help="Backend runner type.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier passed to the backend.")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for OpenAI-compatible endpoints (vLLM).")
    parser.add_argument("--api_key", type=str, default=None, help="API key (defaults to OPENAI_API_KEY env).")
    parser.add_argument("--dataset_filter", nargs="*", help="Optional dataset keys to keep (e.g., uco3d dl3dv).")
    parser.add_argument("--stage_filter", nargs="*", help="Optional stage IDs to keep.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests (seconds).")
    parser.add_argument("--concurrency", type=int, default=64, help="Max number of concurrent requests.")
    parser.add_argument("--no_think", action="store_false", dest="think", help="Disable thinking mode")
    return parser.parse_args()


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.runner != "vllm":
        return args.model_id
    if not args.base_url:
        raise RuntimeError("--base_url is required for the vllm runner")

    model_id = args.model_id or ""
    if model_id and model_id.lower() != "auto":
        return model_id

    headers = {
        "Authorization": f"Bearer {(args.api_key or os.environ.get('OPENAI_API_KEY', '') or 'EMPTY')}",
    }
    endpoint = args.base_url.rstrip("/") + "/models"
    response = requests.get(endpoint, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    models = [m["id"] for m in data.get("data", []) if "id" in m]
    if not models:
        raise RuntimeError("No models available on vLLM server.")
    resolved = models[0]
    print(f"[evaluate] Using vLLM model '{resolved}'.")
    return resolved


def make_runner(args: argparse.Namespace, model_id: str) -> BaseRunner:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if args.runner == "openai":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai runner")
        return OpenAIChatRunner(model=model_id, api_key=api_key, base_url=args.base_url)
    if args.runner == "vllm":
        if not args.base_url:
            raise RuntimeError("--base_url is required for the vllm runner")
        return VLLMChatRunner(
            model=model_id,
            base_url=args.base_url,
            api_key=api_key or "EMPTY"
        )
    raise ValueError(f"Unsupported runner: {args.runner}")


def sanitise_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


_THREAD_LOCAL = threading.local()


def get_thread_runner(runner_factory) -> BaseRunner:
    runner = getattr(_THREAD_LOCAL, "runner", None)
    if runner is None:
        runner = runner_factory()
        _THREAD_LOCAL.runner = runner
    return runner


def bucket_from_rank(rank: float | None) -> str | None:
    if rank is None:
        return None
    for name, low, high in OVERLAP_BUCKETS:
        if low <= rank < high:
            return name
    return OVERLAP_BUCKETS[-1][0]


def evaluate_entry(
    entry: dict,
    runner_factory,
    testset_root: Path,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    sleep: float,
    think: bool
) -> tuple[dict, float, float, float]:
    metadata = None
    error_message = None

    try:
        metadata = read_metadata(testset_root, entry["metadata"])
    except Exception as exc:
        error_message = f"metadata_error: {exc}"

    if metadata is None:
        record = {
            "meta": {
                "dataset": entry.get("dataset"),
                "dataset_key": entry.get("dataset_key"),
                "stage": entry.get("stage"),
                "sample_id": entry.get("sample_id"),
                "metadata_path": entry.get("metadata"),
                "viewA_path": entry.get("viewA"),
                "viewB_path": entry.get("viewB"),
            },
            "gt": {},
            "prediction": "",
            "prediction_json_raw": None,
            "prediction_parsed": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "latency": None,
            "error": error_message or "metadata_missing",
        }
        return record, 0.0, 0.0, 0.0

    prompt = build_prompt(metadata, think=think)

    base64_image1 = encode_image(metadata["image1"])
    base64_image2 = encode_image(metadata["image2"])

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"},
                },
                {"type": "text", "text": prompt},
            ]
        },
    ]

    response_text = ""
    candidate_json = None
    parsed_json = None
    latency = None
    error = None

    try:
        runner = get_thread_runner(runner_factory)
        start = time.time()
        response_text = runner.run(messages, temperature=temperature, max_tokens=max_tokens)
        latency = time.time() - start
        candidate_json, _ = extract_json_from_text(response_text)
        parsed_json = safe_parse_json(candidate_json)
    except Exception as exc:
        error = str(exc)
        response_text = f"[error] {error}"
        candidate_json = None
        parsed_json = None

    if sleep:
        time.sleep(sleep)

    gt_answer = metadata.get("answer", {})
    precision, recall, f1, _ = compare_mappings(gt_answer, parsed_json if isinstance(parsed_json, dict) else None)

    meta_info = metadata.get("meta", {}) or {}
    record = {
        "meta": {
            "dataset": entry.get("dataset"),
            "dataset_key": entry.get("dataset_key"),
            "stage": meta_info.get("stage") or entry.get("stage"),
            "difficulty": meta_info.get("difficulty") or entry.get("difficulty"),
            "sample_id": entry.get("sample_id") or entry.get("id"),
            "metadata_path": entry.get("metadata"),
            "viewA_path": entry.get("viewA"),
            "viewB_path": entry.get("viewB"),
            "overlap": meta_info.get("overlap"),
            "overlap_rank": meta_info.get("overlap_rank"),
        },
        "gt": gt_answer,
        "prediction": response_text,
        "prediction_json_raw": candidate_json,
        "prediction_parsed": parsed_json,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency": latency,
        "error": error,
    }

    return record, precision, recall, f1


def _update_metric(target: dict, key: str, precision: float, recall: float, f1: float) -> dict:
    data = target.setdefault(key, {"count": 0, "prec_sum": 0.0, "rec_sum": 0.0, "f1_sum": 0.0})
    data["count"] += 1
    data["prec_sum"] += precision
    data["rec_sum"] += recall
    data["f1_sum"] += f1
    return data


def _finalize_metric_dict(metrics: dict) -> dict:
    for key, data in metrics.items():
        count = max(data.get("count", 0), 1)
        data["avg_precision"] = data.get("prec_sum", 0.0) / count
        data["avg_recall"] = data.get("rec_sum", 0.0) / count
        data["avg_f1"] = data.get("f1_sum", 0.0) / count
        data.pop("prec_sum", None)
        data.pop("rec_sum", None)
        data.pop("f1_sum", None)
    return metrics


def main() -> None:
    args = parse_args()
    resolved_model_id = resolve_model_id(args)
    runner_factory = lambda: make_runner(args, resolved_model_id)

    index_entries = load_annotation_index(Path(args.annotation_index))
    if args.dataset_filter:
        index_entries = [e for e in index_entries if e.get("dataset_key") in args.dataset_filter]
    if args.stage_filter:
        index_entries = [e for e in index_entries if e.get("stage") in args.stage_filter]

    if args.max_samples and args.max_samples > 0:
        index_entries = index_entries[: args.max_samples]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{sanitise_name(args.model_name)}__{args.runner}__{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": args.model_name,
        "runner": args.runner,
        "model_id": args.model_id,
        "resolved_model_id": resolved_model_id,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system_prompt": args.system_prompt,
        "dataset_filter": args.dataset_filter,
        "stage_filter": args.stage_filter,
        "annotation_index": args.annotation_index,
        "testset_root": args.testset_root,
        "concurrency": args.concurrency,
        "sleep": args.sleep,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    samples_path = run_dir / "samples.jsonl"

    metrics_precision = []
    metrics_recall = []
    metrics_f1 = []
    total = 0
    stage_metrics: Dict[str, dict] = {}
    dataset_metrics: Dict[str, dict] = {}
    overlap_bucket_metrics: Dict[str, dict] = {}
    difficulty_metrics: Dict[str, dict] = {}
    json_quality = {
        "total": 0,
        "json_absent": 0,
        "json_parse_fail": 0,
        "request_errors": 0,
    }

    testset_root = Path(args.testset_root)

    with samples_path.open("w", encoding="utf-8") as samples_fh, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                evaluate_entry,
                entry,
                runner_factory,
                testset_root,
                args.system_prompt,
                args.temperature,
                args.max_tokens,
                args.sleep,
                args.think
            )
            for entry in index_entries
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            record, precision, recall, f1 = future.result()
            samples_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_precision.append(precision)
            metrics_recall.append(recall)
            metrics_f1.append(f1)
            total += 1

            meta = record.get("meta", {})
            dataset_key = meta.get("dataset_key") or "unknown"
            stage_id = meta.get("stage") or "unknown"
            difficulty_name = meta.get("difficulty") or "unknown"
            overlap = meta.get("overlap")
            overlap_rank = meta.get("overlap_rank")
            bucket_name = bucket_from_rank(overlap_rank)

            _update_metric(stage_metrics, stage_id, precision, recall, f1)
            _update_metric(difficulty_metrics, difficulty_name, precision, recall, f1)

            dataset_entry = dataset_metrics.setdefault(
                dataset_key,
                {
                    "dataset": meta.get("dataset"),
                    "count": 0,
                    "prec_sum": 0.0,
                    "recall_sum": 0.0,
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
            dataset_entry["count"] += 1
            dataset_entry["prec_sum"] += precision
            dataset_entry["recall_sum"] += recall
            dataset_entry["f1_sum"] += f1
            if overlap is not None:
                dataset_entry["overlap_sum"] += overlap
                dataset_entry["overlap_min"] = overlap if dataset_entry["overlap_min"] is None else min(dataset_entry["overlap_min"], overlap)
                dataset_entry["overlap_max"] = overlap if dataset_entry["overlap_max"] is None else max(dataset_entry["overlap_max"], overlap)
            if overlap_rank is not None:
                dataset_entry["overlap_rank_sum"] += overlap_rank
                dataset_entry["overlap_rank_min"] = (
                    overlap_rank if dataset_entry["overlap_rank_min"] is None else min(dataset_entry["overlap_rank_min"], overlap_rank)
                )
                dataset_entry["overlap_rank_max"] = (
                    overlap_rank if dataset_entry["overlap_rank_max"] is None else max(dataset_entry["overlap_rank_max"], overlap_rank)
                )

            _update_metric(dataset_entry["stages"], stage_id, precision, recall, f1)
            _update_metric(dataset_entry["difficulties"], difficulty_name, precision, recall, f1)
            if bucket_name:
                _update_metric(dataset_entry["overlap_buckets"], bucket_name, precision, recall, f1)
                _update_metric(overlap_bucket_metrics, bucket_name, precision, recall, f1)

            json_quality["total"] += 1
            if record.get("error"):
                json_quality["request_errors"] += 1
            if record.get("prediction_json_raw") is None:
                json_quality["json_absent"] += 1
            elif not isinstance(record.get("prediction_parsed"), dict):
                json_quality["json_parse_fail"] += 1

    _finalize_metric_dict(stage_metrics)
    for dataset_entry in dataset_metrics.values():
        count = max(dataset_entry.get("count", 0), 1)
        dataset_entry["avg_precision"] = dataset_entry["prec_sum"] / count
        dataset_entry["avg_recall"] = dataset_entry["recall_sum"] / count
        dataset_entry["avg_f1"] = dataset_entry["f1_sum"] / count
        if dataset_entry["overlap_min"] is not None:
            dataset_entry["avg_overlap"] = dataset_entry["overlap_sum"] / count
        else:
            dataset_entry["avg_overlap"] = None
        if dataset_entry["overlap_rank_min"] is not None:
            dataset_entry["avg_overlap_rank"] = dataset_entry["overlap_rank_sum"] / count
        else:
            dataset_entry["avg_overlap_rank"] = None
        dataset_entry.pop("prec_sum", None)
        dataset_entry.pop("recall_sum", None)
        dataset_entry.pop("f1_sum", None)
        dataset_entry.pop("overlap_sum", None)
        dataset_entry.pop("overlap_rank_sum", None)
        _finalize_metric_dict(dataset_entry["stages"])
        _finalize_metric_dict(dataset_entry["overlap_buckets"])
        _finalize_metric_dict(dataset_entry["difficulties"])

    _finalize_metric_dict(overlap_bucket_metrics)
    _finalize_metric_dict(difficulty_metrics)
    json_quality["json_success"] = (
        json_quality["total"] - json_quality["json_absent"] - json_quality["json_parse_fail"]
    )

    summary = {
        "total_samples": total,
        "avg_precision": sum(metrics_precision) / max(len(metrics_precision), 1),
        "avg_recall": sum(metrics_recall) / max(len(metrics_recall), 1),
        "avg_f1": sum(metrics_f1) / max(len(metrics_f1), 1),
        "config": config,
        "samples_file": str(samples_path),
        "stage_metrics": stage_metrics,
        "dataset_metrics": dataset_metrics,
        "overlap_buckets": overlap_bucket_metrics,
        "difficulty_metrics": difficulty_metrics,
        "json_quality": json_quality,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()

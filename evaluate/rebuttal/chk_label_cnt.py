from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from model_runners import (
    BaseRunner,
    dataset_group_from_key,
    encode_image,
    make_runner,
    resolve_model_id,
    sanitise_name,
)
from utils import (
    extract_json_from_text,
    load_annotation_index,
    read_metadata,
    safe_parse_json,
    scan_dataset_index,
)

_INT_RE = re.compile(r"-?\d+")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick check: count visible markers in images."
    )
    parser.add_argument("--testset_root", type=str, required=True)
    parser.add_argument(
        "--annotation_index",
        type=str,
        default=None,
        help="Optional JSON index. If omitted/missing, scan testset_root.",
    )
    parser.add_argument("--output_dir", type=str, default="eval_visibility_results")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--runner", choices=["openai", "vllm"], required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dataset_filter", nargs="*")
    parser.add_argument("--stage_filter", nargs="*")
    return parser.parse_args()



_THREAD_LOCAL = threading.local()


def get_thread_runner(runner_factory) -> BaseRunner:
    runner = getattr(_THREAD_LOCAL, "runner", None)
    if runner is None:
        runner = runner_factory()
        _THREAD_LOCAL.runner = runner
    return runner


def gt_count(view: dict | None) -> int:
    if not isinstance(view, dict):
        return 0
    image_points = view.get("image_points") or []
    text_points = view.get("text_points") or []
    return int(len(image_points) + len(text_points))


def parse_pred_counts(text: str) -> Tuple[int | None, int | None, str | None]:
    candidate, _ = extract_json_from_text(text or "")
    parsed = safe_parse_json(candidate)
    if isinstance(parsed, dict):
        a = parsed.get("A", parsed.get("a", None))
        b = parsed.get("B", parsed.get("b", None))
        try:
            a_value = int(a) if a is not None else None
        except Exception:
            a_value = None
        try:
            b_value = int(b) if b is not None else None
        except Exception:
            b_value = None
        return a_value, b_value, candidate

    numbers = _INT_RE.findall(text or "")
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1]), None
    return None, None, None


def build_count_prompt() -> str:
    return (
        "You will see two images (Image A and Image B) of the same scene.\n"
        "Each image contains several circular markers with IDs (letters or numbers).\n"
        "Task: count how many markers are visible in Image A and in Image B.\n"
        'Return JSON ONLY in the format: {"A": <int>, "B": <int>}.\n'
        "Do not include any other text."
    )


def evaluate_one(
    entry: dict,
    runner_factory,
    testset_root: Path,
    temperature: float,
    max_tokens: int,
    sleep: float,
) -> Tuple[dict, int, int, int, int]:
    meta_path = entry.get("metadata")
    record_error = None

    try:
        meta = read_metadata(testset_root, meta_path or "")
    except Exception as exc:  # pragma: no cover - depends on filesystem state
        record_error = f"metadata_error: {exc}"
        meta = None

    dataset_key = entry.get("dataset_key") or "unknown"
    dataset = entry.get("dataset") or dataset_key
    dataset_group = dataset_group_from_key(dataset_key, dataset)

    if meta is None:
        record = {
            "meta": {
                "dataset": dataset,
                "dataset_key": dataset_key,
                "dataset_group": dataset_group,
                "label_scheme": entry.get("label_scheme"),
                "stage": entry.get("stage"),
                "sample_id": entry.get("sample_id"),
                "metadata_path": meta_path,
            },
            "gt": {"A": 0, "B": 0},
            "pred": {"A": None, "B": None},
            "raw": "",
            "latency": None,
            "error": record_error or "metadata_missing",
            "recall": {"A": 0.0, "B": 0.0},
        }
        return record, 0, 0, 0, 0

    gt_a = gt_count(meta.get("viewA"))
    gt_b = gt_count(meta.get("viewB"))

    base64_a = encode_image(meta["image1"])
    base64_b = encode_image(meta["image2"])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_a}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_b}"},
                },
                {"type": "text", "text": build_count_prompt()},
            ],
        },
    ]

    raw = ""
    latency = None
    error = None
    pred_a = None
    pred_b = None

    try:
        runner = get_thread_runner(runner_factory)
        start = time.time()
        raw = runner.run(messages, temperature=temperature, max_tokens=max_tokens)
        latency = time.time() - start
        pred_a, pred_b, _ = parse_pred_counts(raw)
    except Exception as exc:  # pragma: no cover - depends on external APIs
        error = str(exc)
        raw = f"[error] {error}"

    if sleep:
        time.sleep(sleep)

    recognized_a = 0
    recognized_b = 0
    if pred_a is not None:
        recognized_a = min(max(int(pred_a), 0), gt_a)
    if pred_b is not None:
        recognized_b = min(max(int(pred_b), 0), gt_b)

    recall_a = 1.0 if gt_a == 0 else recognized_a / gt_a
    recall_b = 1.0 if gt_b == 0 else recognized_b / gt_b

    meta_info = meta.get("meta", {}) or {}
    record = {
        "meta": {
            "dataset": dataset,
            "dataset_key": dataset_key,
            "dataset_group": dataset_group,
            "label_scheme": meta_info.get("label_scheme")
            or entry.get("label_scheme")
            or "unknown",
            "stage": meta_info.get("stage") or entry.get("stage") or "unknown",
            "sample_id": meta_info.get("sample_id") or entry.get("sample_id"),
            "metadata_path": meta_path,
        },
        "gt": {"A": gt_a, "B": gt_b},
        "pred": {"A": pred_a, "B": pred_b},
        "raw": raw,
        "latency": latency,
        "error": error,
        "recall": {"A": recall_a, "B": recall_b},
        "abs_err": {
            "A": None if pred_a is None else abs(pred_a - gt_a),
            "B": None if pred_b is None else abs(pred_b - gt_b),
        },
    }
    return record, recognized_a, recognized_b, gt_a, gt_b


def update_bucket(
    bucket: dict,
    key: str,
    recognized_a: int,
    recognized_b: int,
    gt_a: int,
    gt_b: int,
    err_a: int | None,
    err_b: int | None,
) -> None:
    data = bucket.setdefault(
        key,
        {
            "count": 0,
            "recognizedA_sum": 0,
            "recognizedB_sum": 0,
            "gtA_sum": 0,
            "gtB_sum": 0,
            "maeA_sum": 0,
            "maeB_sum": 0,
            "maeA_n": 0,
            "maeB_n": 0,
        },
    )
    data["count"] += 1
    data["recognizedA_sum"] += recognized_a
    data["recognizedB_sum"] += recognized_b
    data["gtA_sum"] += gt_a
    data["gtB_sum"] += gt_b
    if err_a is not None:
        data["maeA_sum"] += err_a
        data["maeA_n"] += 1
    if err_b is not None:
        data["maeB_sum"] += err_b
        data["maeB_n"] += 1


def finalize(bucket: dict) -> dict:
    for _, data in bucket.items():
        gt_a_sum = int(data.get("gtA_sum", 0))
        gt_b_sum = int(data.get("gtB_sum", 0))
        rec_a_sum = int(data.get("recognizedA_sum", 0))
        rec_b_sum = int(data.get("recognizedB_sum", 0))
        total_gt = gt_a_sum + gt_b_sum
        data["recallA"] = 1.0 if gt_a_sum == 0 else float(rec_a_sum) / gt_a_sum
        data["recallB"] = 1.0 if gt_b_sum == 0 else float(rec_b_sum) / gt_b_sum
        data["recallBoth"] = (
            1.0 if total_gt == 0 else float(rec_a_sum + rec_b_sum) / total_gt
        )
        data["maeA"] = (
            float(data["maeA_sum"]) / max(int(data["maeA_n"]), 1)
            if data.get("maeA_n")
            else None
        )
        data["maeB"] = (
            float(data["maeB_sum"]) / max(int(data["maeB_n"]), 1)
            if data.get("maeB_n")
            else None
        )
        data.pop("maeA_sum", None)
        data.pop("maeB_sum", None)
        data.pop("maeA_n", None)
        data.pop("maeB_n", None)
        data.pop("recognizedA_sum", None)
        data.pop("recognizedB_sum", None)
        data.pop("gtA_sum", None)
        data.pop("gtB_sum", None)
    return bucket


def main() -> None:
    args = parse_args()
    testset_root = Path(args.testset_root)

    resolved_model_id = resolve_model_id(args)
    runner_factory = lambda: make_runner(args, resolved_model_id)

    entries = None
    if args.annotation_index:
        index_path = Path(args.annotation_index)
        if index_path.exists():
            entries = load_annotation_index(index_path)
        else:
            print(
                f"[count] annotation_index not found: {index_path}. Scanning {testset_root}."
            )

    if entries is None:
        print(f"[count] Scanning {testset_root} for metadata.json ...")
        entries = scan_dataset_index(testset_root)

    if args.dataset_filter:
        entries = [e for e in entries if e.get("dataset_key") in args.dataset_filter]
    if args.stage_filter:
        entries = [e for e in entries if e.get("stage") in args.stage_filter]
    if args.max_samples and args.max_samples > 0:
        entries = entries[: args.max_samples]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{sanitise_name(args.model_name)}__count__{args.runner}__{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, ensure_ascii=False)

    samples_path = run_dir / "samples.jsonl"

    total = 0
    rec_a_sum = 0
    rec_b_sum = 0
    gt_a_sum = 0
    gt_b_sum = 0
    mae_a_sum = 0
    mae_b_sum = 0
    mae_a_n = 0
    mae_b_n = 0

    group_split: Dict[str, dict] = {}
    key_split: Dict[str, dict] = {}

    with samples_path.open("w", encoding="utf-8") as samples_fh, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                entry,
                runner_factory,
                testset_root,
                args.temperature,
                args.max_tokens,
                args.sleep,
            )
            for entry in entries
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating",
        ):
            record, recognized_a, recognized_b, gt_a, gt_b = future.result()
            samples_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            rec_a_sum += recognized_a
            rec_b_sum += recognized_b
            gt_a_sum += gt_a
            gt_b_sum += gt_b

            err_a = record.get("abs_err", {}).get("A")
            err_b = record.get("abs_err", {}).get("B")
            if isinstance(err_a, int):
                mae_a_sum += err_a
                mae_a_n += 1
            if isinstance(err_b, int):
                mae_b_sum += err_b
                mae_b_n += 1

            meta = record.get("meta", {}) or {}
            dataset_group = meta.get("dataset_group") or "unknown"
            dataset_key = meta.get("dataset_key") or "unknown"
            label_scheme = meta.get("label_scheme") or "unknown"

            update_bucket(
                group_split.setdefault(dataset_group, {}),
                label_scheme,
                recognized_a,
                recognized_b,
                gt_a,
                gt_b,
                err_a,
                err_b,
            )
            update_bucket(
                key_split.setdefault(dataset_key, {}),
                label_scheme,
                recognized_a,
                recognized_b,
                gt_a,
                gt_b,
                err_a,
                err_b,
            )
            if dataset_group != dataset_key:
                update_bucket(
                    key_split.setdefault(dataset_group, {}),
                    label_scheme,
                    recognized_a,
                    recognized_b,
                    gt_a,
                    gt_b,
                    err_a,
                    err_b,
                )

    for split_metrics in group_split.values():
        finalize(split_metrics)
    for split_metrics in key_split.values():
        finalize(split_metrics)

    total_gt = gt_a_sum + gt_b_sum
    summary = {
        "total_samples": total,
        "recallA": 1.0 if gt_a_sum == 0 else float(rec_a_sum) / gt_a_sum,
        "recallB": 1.0 if gt_b_sum == 0 else float(rec_b_sum) / gt_b_sum,
        "recallBoth": 1.0 if total_gt == 0 else float(rec_a_sum + rec_b_sum) / total_gt,
        "maeA": (float(mae_a_sum) / max(mae_a_n, 1)) if mae_a_n else None,
        "maeB": (float(mae_b_sum) / max(mae_b_n, 1)) if mae_b_n else None,
        "group_split_metrics": group_split,
        "key_split_metrics": key_split,
        "samples_file": str(samples_path),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"[count] Done. Results saved to {run_dir}")
    print("[count] Split metrics by dataset group:")
    for group_name in sorted(summary["group_split_metrics"].keys()):
        for scheme_name in sorted(summary["group_split_metrics"][group_name].keys()):
            data = summary["group_split_metrics"][group_name][scheme_name]
            print(
                f"  {group_name} [{scheme_name}]"
                f" recallBoth={data.get('recallBoth'):.4f}"
                f" recallA={data.get('recallA'):.4f}"
                f" recallB={data.get('recallB'):.4f}"
                f" count={data.get('count')}"
            )
    print("[count] Split metrics by dataset key:")
    for key_name in sorted(summary["key_split_metrics"].keys()):
        for scheme_name in sorted(summary["key_split_metrics"][key_name].keys()):
            data = summary["key_split_metrics"][key_name][scheme_name]
            print(
                f"  {key_name} [{scheme_name}]"
                f" recallBoth={data.get('recallBoth'):.4f}"
                f" recallA={data.get('recallA'):.4f}"
                f" recallB={data.get('recallB'):.4f}"
                f" count={data.get('count')}"
            )


if __name__ == "__main__":
    main()

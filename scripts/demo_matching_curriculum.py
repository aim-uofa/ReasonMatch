#!/usr/bin/env python3
"""Simulate matching curriculum progression with dummy rewards.

Run this script to watch the curriculum controller step through stages without
loading any dataset. Rewards are synthetic and can be tuned via CLI flags.

Example usage:

```bash
python scripts/demo_matching_curriculum.py \\ 
  --preset basic \\
  --min-count 4 \\
  --delta 0.08
```

Use `--config` to point to a custom YAML file (same structure as the presets in
`my_recipe/config/matching_curriculum`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import OmegaConf

from my_recipe.buffer.dynamic_task_buffer import (
    BufferConfig,
    CurriculumStage,
    MatchingCurriculumConfig,
    DynamicTaskBuffer,
    matching_curriculum_from_dict,
)


def resolve_curriculum(args: argparse.Namespace) -> MatchingCurriculumConfig:
    """Load curriculum config from preset name or YAML path."""

    # if args.config and args.preset:
    #     raise ValueError("Use either --config or --preset, not both.")

    if args.config:
        cfg_path = Path(args.config)
        cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    else:
        preset = args.preset or "basic"
        preset_dir = Path(__file__).resolve().parents[1] / "my_recipe" / "config" / "matching_curriculum"
        cfg_path = preset_dir / f"{preset}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Preset '{preset}' not found at {cfg_path}")
        cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    if "matching_curriculum" in cfg_dict:
        cfg_dict = cfg_dict["matching_curriculum"]

    curriculum_cfg = matching_curriculum_from_dict(cfg_dict)

    # Clamp min_count for quicker demo if requested.
    max_count = args.min_count
    if max_count is not None:
        max_count = max(1, max_count)
        for stage in curriculum_cfg.stages:
            if stage.promotion is not None:
                stage.promotion.min_count = min(stage.promotion.min_count, max_count)
            if stage.demotion is not None:
                stage.demotion.min_count = min(stage.demotion.min_count, max_count)
        curriculum_cfg.metric_window = min(curriculum_cfg.metric_window, max(max_count, 32))

    return curriculum_cfg


def stage_reward_schedule(stage: CurriculumStage, delta: float, steps: int) -> Iterable[float]:
    """Generate a simple low→high reward pattern for a stage."""

    stage_cfg = stage.reference_config()
    promotion = stage.promotion or (stage_cfg.promotion if stage_cfg else None)
    base_threshold = promotion.threshold if promotion else 0.75

    ratio_targets: list[float] = []
    reward_targets: list[float] = [base_threshold]

    if promotion and promotion.success_ratio_threshold is not None:
        ratio_targets.append(float(promotion.success_ratio_threshold))
        reward_targets.append(float(promotion.success_reward_threshold or promotion.threshold))

    filter_schedule = stage_cfg.filter_schedule if stage_cfg else []
    for entry in filter_schedule:
        ratio = entry.get("success_ratio_threshold")
        reward = entry.get("success_reward_threshold")
        if ratio is not None and reward is not None:
            ratio_targets.append(float(ratio))
            reward_targets.append(float(reward))

    target_ratio = max(ratio_targets) if ratio_targets else None
    target_reward = max(reward_targets) if reward_targets else base_threshold

    eps = 1e-3

    promo_min = promotion.min_count if promotion else None
    failures = max(1, min(steps // 2 or 1, (promo_min // 2) if promo_min else steps // 3 or 1))
    successes = max(1, max(steps - failures, (promo_min or failures) + 1))

    low_reward = max(0.0, base_threshold - max(abs(delta) * 2, 0.2))
    high_reward = base_threshold + max(abs(delta), 0.2)
    high_reward = max(high_reward, target_reward + eps)

    for _ in range(failures):
        yield min(low_reward, target_reward - eps)

    for _ in range(successes):
        yield high_reward


def simulate(curriculum: MatchingCurriculumConfig, delta: float, steps: int, log_file: Optional[str]) -> None:
    """Simulate curriculum progression with synthetic rewards and buffer logging."""

    config = BufferConfig(
        task_mode="matching",
        matching_curriculum=curriculum,
        log_file=log_file,
    )
    buffer = DynamicTaskBuffer(config, raw_dataset=None)

    print("Loaded curriculum stages:")
    for idx, stage in enumerate(curriculum.stages):
        promo = stage.promotion.threshold if stage.promotion else None
        demo = stage.demotion.threshold if stage.demotion else None
        print(
            f"  {idx + 1}. {stage.stage_id} | promotion={promo} | demotion={demo}"
        )
    print("\nBeginning simulation...\n")

    log_rows = []

    for stage in curriculum.stages:
        for reward in stage_reward_schedule(stage, delta=delta, steps=steps):
            # Update buffer's internal tracking to mimic training loop
            buffer.stage_seen_counts[stage.stage_id] += 1
            buffer.stage_reward_history[stage.stage_id].append(reward)
            buffer.buffer.append(
                {
                    "task": "matching",
                    "reward": float(reward),
                    "db_idx": None,
                    "stage": stage.stage_id,
                }
            )

            task_metrics = buffer.get_task_metrics()
            assessment = {"task_metrics": task_metrics}

            prev_stage_id = buffer.current_matching_stage_id
            buffer._apply_curriculum_update(assessment)

            current_stage = buffer.matching_curriculum_manager.get_current_stage()
            current_id = current_stage.stage_id if current_stage else "<none>"

            stage_stats = task_metrics.get("matching_stages", {}).get(stage.stage_id, {})
            stage_mean = stage_stats.get("mean_reward", 0.0)
            stage_total = stage_stats.get("count", buffer.stage_seen_counts[stage.stage_id])

            transition = None
            if prev_stage_id != buffer.current_matching_stage_id:
                transition = {
                    "from": prev_stage_id,
                    "to": buffer.current_matching_stage_id,
                }

            row = {
                "update_stage": stage.stage_id,
                "reward": reward,
                "mean_reward": stage_mean,
                "total_count": stage_total,
                "controller_stage": current_id,
                "transition": transition,
            }
            log_rows.append(row)

            print(
                f"step={len(log_rows):02d} | feed={stage.stage_id:20s} "
                f"reward={reward:.3f} | mean={stage_mean:.3f} | "
                f"controller={current_id}"
            )
            if transition:
                print(
                    f"    >>> stage transition from {transition['from']} to {transition['to']}"
                )

                # Once controller moves to a new stage, advance to the next loop iteration.
                if transition["to"] != stage.stage_id:
                    break

    print("\nSimulation complete. Summary (JSON lines):")
    for row in log_rows:
        print(json.dumps(row))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="basic",
        help="Name of matching curriculum preset (basic, text_first, distractor_heavy, ...).",
    )
    parser.add_argument(
        "--config",
        help="Path to a custom curriculum YAML file (overrides --preset).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=6,
        help="Clamp min_count requirements to this value for quicker demos.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Amount to offset rewards around each stage threshold.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of low/high reward samples to emit per stage phase.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file to test DynamicTaskBuffer logging output.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    curriculum_cfg = resolve_curriculum(args)
    simulate(curriculum_cfg, delta=args.delta, steps=args.steps, log_file=args.log_file)


if __name__ == "__main__":
    main()

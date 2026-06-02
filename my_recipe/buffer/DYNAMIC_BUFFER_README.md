# Dynamic Task-Switching Buffer (v2)

A flexible buffer system for dynamically switching between matching and grounding tasks during MLLM training based on model performance.

## Overview

The Dynamic Task Buffer enables curriculum learning by automatically adjusting task difficulty during training:

- **Matching Task** (easier): Identify region ID correspondences across images
- **Grounding Task** (harder): Locate 2D coordinates of regions across images

The buffer maintains a fixed-size cache of recent samples with their rewards, and uses this information to decide which task to use for each new batch.

## Architecture (v2 - Cleaner Design)

### Key Improvement

**v2 eliminates the need for duplicate dataset instances!** Instead of creating separate `DL3DV` and `DL3DVG` datasets:

- **v1 (old)**: Two full datasets → Buffer selects between them
- **v2 (new)**: One raw dataset → Buffer processes with chosen task

This is cleaner because:
- Single LMDB reader (no duplication)
- Buffer owns task logic (better separation of concerns)
- Easier to add new tasks
- Less memory overhead

### Components

1. **`AnnoRawDataset`** (`mydatasets/anno_raw.py`) - NEW!
   - Reads from LMDB database
   - Decodes images and rescales coordinates
   - Returns raw samples WITHOUT task-specific formatting
   - Lightweight and efficient

2. **`DynamicTaskBuffer`** (`dynamic_task_buffer.py`) - ENHANCED!
   - Maintains a deque of recent samples with rewards
   - Tracks performance metrics separately for matching and grounding tasks
   - Makes task-switching decisions based on configurable thresholds
   - **Integrates task-specific processing logic** (matching/grounding)
   - Processes raw samples into training-ready samples

3. **`BufferedDataLoader`** (`buffered_dataloader.py`)
   - Wraps the standard PyTorch DataLoader
   - Intercepts batches and regenerates them with dynamically chosen tasks
   - Maintains compatibility with existing dataloader interface

4. Internal **`RayDAPOTrainer`** integration (`workers/dapo_ray_trainer.py`)
   - Initializes buffer with single raw dataset
   - Updates buffer with rewards after each training step
   - Logs buffer metrics for monitoring

## Workflow (v2)

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  AnnoRawDataset      │
                    │  • Read from LMDB    │
                    │  • Decode images     │
                    │  • Rescale coords    │
                    └──────────┬───────────┘
                               │ (raw samples)
                               ▼
                    ┌──────────────────────┐
                    │  Dynamic Buffer      │
                    │  • Check metrics     │
                    │  • Decide task       │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
          ┌─────────▼────────┐  ┌────────▼─────────┐
          │ process_matching │  │ process_grounding│
          │ (if perf > θ₁)   │  │ (if perf < θ₂)   │
          └─────────┬────────┘  └────────┬─────────┘
                    │ (formatted)        │ (formatted)
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Training Sample     │
                    │  • Prompt messages   │
                    │  • Model inputs      │
                    │  • Ground truth      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Normal Training     │
                    │  • generate_seq      │
                    │  • compute_reward    │
                    │  • update_actor      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Add to Buffer       │
                    │  with Rewards        │
                    └──────────────────────┘
```

**Key Difference from v1:**
- v1: Buffer selected between two pre-formatted datasets
- v2: Buffer receives raw samples and applies formatting based on task decision

## Configuration

### Basic Configuration

Add to your training config (e.g., `config/dapo_trainer.yaml`):

```yaml
dynamic_buffer:
  enable: true
  buffer_size: 100
  warmup_proportion: 0.5      # only adapt after buffer is 50% full
  task_switch_metric: "mean_reward"
  matching_threshold: 0.7
  grounding_threshold: 0.3
  min_samples_for_switch: 20
  task_mode: dynamic        # or "matching"/"grounding" for direct task creation
  matching_filter:
    min_distance: 40
    min_points: 3
    max_points: 6
    strategy: adaptive
    schedule:
      - threshold: 0.0
        min_distance: 80
        min_points: 2
        max_points: 3
      - threshold: 0.8
        min_distance: 25
        min_points: 4
        max_points: 8
  grounding_points:
    min_points: 1
    max_points: 3
    strategy: adaptive
    schedule:
      - threshold: 0.0
        min_points: 1
        max_points: 2
      - threshold: 0.75
        min_points: 3
        max_points: 5
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | true | Enable/disable dynamic task switching |
| `buffer_size` | int | 100 | Number of recent samples to maintain |
| `task_switch_metric` | str | "mean_reward" | Metric for task switching decisions |
| `matching_threshold` | float | 0.7 | Switch to grounding if matching reward > this |
| `grounding_threshold` | float | 0.3 | Switch to matching if grounding reward < this |
| `min_samples_for_switch` | int | 20 | Minimum samples before switching |
| `warmup_proportion` | float | 0.5 | Buffer fill ratio required before the dynamic routine adjusts tasks/params |
| `task_mode` | str | "dynamic" | `"dynamic"` (default), `"matching"` or `"grounding"` for direct task control |
| `matching_filter` | dict | see config | Controls filtering strategy for `filter_sparse_points` (supports `min_distance`, `min_points`, `max_points`) |
| `matching_curriculum` | dict | see config | Multi-stage matching curriculum (image → hybrid → distractor) |
| `grounding_points` | dict | see config | Controls how many correspondences grounding samples request |
| `log_file` | str | null | Optional JSONL file path for per-step buffer logs |

#### Direct Task Creation (`task_mode`)

- `dynamic`: keep adaptive switching logic (default).
- `matching`: always build matching samples (useful for ablations or curriculum warmup).
- `grounding`: always build grounding samples.

#### Adaptive Difficulty via Filtering

- `matching_filter` forwards knobs to `draw_img()`/`filter_sparse_points()`.  
  Larger `min_distance` + smaller `max_points` ⇒ easier samples.  
  Set `strategy: adaptive` plus a `schedule` (ordered by `threshold`) to automatically tighten the filter as the buffer's mean reward grows.
  `min_points` guarantees at least N matches are kept after filtering.

- `grounding_points` controls the min/max number of matches sampled for grounding.  
  Similarly, `strategy: adaptive` lets you request more points only after the model performs well on grounding samples.

Schedules accept entries like:

```yaml
schedule:
  - threshold: 0.4   # mean reward
    min_distance: 60
    max_points: 5
    success_ratio_threshold: 0.65      # optional: % of samples
    success_reward_threshold: 0.75     # optional: reward considered "success"
```

The buffer selects the highest entry whose `threshold` and (if provided) success-ratio constraint are both satisfied.  
`success_ratio_threshold` looks at the fraction of recent rewards that meet `success_reward_threshold`; omit either field to fall back to mean-only gating.

### BufferedDataLoader Difficulty Bins

In addition to the in-buffer curriculum, the dataloader can now stage raw samples by *overlap difficulty* before the buffer sees them.  
This is useful when the LMDB has been pre-sorted by overlap so harder viewpoints appear later in the file.

Enable it via:

```yaml
dynamic_buffer:
  buffered_loader:
    enable_overlap_bins: true
    overlap_bin_count: 4
    promotion_reward_threshold: 1.9
    promotion_window: 3
    min_batches_per_bin: 6
```

How it works:

- Split dataset indices into `overlap_bin_count` contiguous bins (easy → hard).
- Within each bin, shuffle indices and feed them repeatedly to keep training localized.
- After every batch, the trainer reports the mean reward to the dataloader.
- Once the last `promotion_window` batches within a bin average above `promotion_reward_threshold`
  *and* at least `min_batches_per_bin` have been seen, the dataloader unlocks the next bin.
- Final bin loops indefinitely, so the entire dataset remains available once the curriculum completes.

This binning logic is orthogonal to the dynamic buffer's own stage manager—the buffer still runs matching/grounding curricula as before.  
Use the loader bins when you want a coarse-grained “easy→hard” ordering based purely on overlap ranks without re-enabling global dataset shuffling.

#### Matching Curriculum (Image → Text → Distractors)

Set `matching_curriculum` to orchestrate a staged curriculum over a single raw dataset. The default pipeline is:

1. **`image_only`** – two annotated images, progressing from sparse to dense clusters via `filter_schedule`.
2. **`hybrid_one_to_one`** – annotated images plus JSON coordinate snippets (relative 0–1000 by default for Qwen3). Ratios let you decide how many IDs appear visually vs. text-only.
3. **`hybrid_multi_to_multi`** – mixed inputs with distractor IDs on both images; unmatched Image-A IDs must map to `null` in the answer.

Key knobs per stage:

- `representation.image1|image2`
  - `render_image` / `include_text`
  - `annotate_ratio` (fraction drawn on the image), `text_ratio` (fraction of the **remaining** points converted to text)
  - `min_annotated_points`, `min_text_points`
  - `text_format` (`json` or `list`), `text_header`, `coordinate_scale` (`auto`, `relative`, `absolute`)
- `point_filter`: point sampling mode (`cluster`, `greedy`, `random`, `dense`) plus `cluster_eps_scale` and `greedy_spacing`.
- `distractors`: how many unmatched points to add on each image; set `mode: synthetic` to spawn random distractors (`synthetic_radius` controls jitter) instead of sampling leftovers.
- `filter_overrides` / `filter_schedule`: stage-specific overrides for `filter_sparse_points` (now supporting `min_points`, `min_distance`, `max_points`). Each schedule row may also define `success_ratio_threshold` + `success_reward_threshold` for success-rate gating.
- `promotion` / `demotion`: thresholds (mean reward) and sample counts required to advance or fall back. They likewise accept the optional success-ratio fields to demand (for example) “≥70 % of recent samples scored ≥0.8” before promoting.

#### Stage Templates & Aggregate Stages

Curriculum files now optionally expose a `stage_library` where you define reusable stage templates (representation, filter knobs, distractors, etc.).  
Each entry under `stages` can then either:

- `use: <template_name>` – reference a template directly (legacy inline definitions still work).
- `aggregate: [{use: foo, weight: 0.6}, {use: bar, weight: 0.4}]` – mix multiple templates. When the controller reaches this curriculum stage it will sample one of the listed variants per sample according to the provided weights.

Metrics, promotion, and demotion thresholds are tracked on the aggregate stage id, while the concrete variant is logged as `matching_stage_variant` in `sample_meta`/`data_source`.

Example:

```yaml
stage_library:
  dense_multi: {...}
  single_reference: {...}

stages:
  - stage_id: advanced_mix
    description: "Blend dense matching with single-reference distractor tasks."
    aggregate:
      - use: dense_multi
        weight: 0.7
      - use: single_reference
        weight: 0.3
    promotion:
      threshold: 1.85
      min_count: 1024
    demotion:
      threshold: 1.3
      min_count: 512
```

Example override:

```yaml
matching_curriculum:
  enable: true
  stages:
    - stage_id: image_only
    promotion:
      threshold: 0.9
      min_count: 120
      point_filter:
        mode: cluster
        cluster_eps_scale: 1.0
    - stage_id: hybrid_one_to_one
      representation:
        image1:
          render_image: true
          include_text: true
          annotate_ratio: 0.5
          text_ratio: 1.0
        image2:
          render_image: false
          include_text: true
          text_format: list
      point_filter:
        mode: cluster
        cluster_eps_scale: 0.6
      promotion:
        threshold: 0.9
        min_count: 150
    - stage_id: hybrid_multi_to_multi
      distractors:
        enable: true
        min_extra_image1: 2
        max_extra_image1: 3
        min_extra_image2: 1
        max_extra_image2: 2
      point_filter:
        mode: greedy
```

Any field you omit falls back to the defaults defined in `dynamic_task_buffer.py`.

Ready-to-use curriculum presets live under `my_recipe/config/matching_curriculum/`:

- `basic.yaml` – matches the default progression with gentle thresholds.
- `text_first.yaml` – jumps into text+image inputs immediately, then removes images for diagnostics.
- `distractor_heavy.yaml` – assumes a stronger policy and increases distractor density aggressively.

Select one by overriding the Hydra group, e.g.:

```bash
python -m my_recipe.main_dcrl \
  dynamic_buffer.@matching_curriculum=text_first
```

You can also point to your own file by placing it in the same directory and referencing its stem name.

Want a quick visual sanity check? Run:

```bash
python scripts/test_matching_buffer.py --output /tmp/buffer_viz
```

The script synthesizes noise images and random matches for each stage, then saves annotated views, JSON metadata, and CSV summaries so you can inspect core vs distractor selection end-to-end.

#### Warmup Proportion

`warmup_proportion` gates the adaptive routine. While the buffer is below the specified ratio, the buffer simply caches rewards and keeps the initial sampling/task configuration. Once the threshold is met, `_assess_buffer_state()` and `_apply_curriculum_update()` are invoked after every training step (only when `task_mode: dynamic`). You can customize the actual assessment/update strategy inside `dynamic_task_buffer.py`.

### Task Switching Logic

```python
if current_task == "matching":
    if mean_reward > matching_threshold:
        switch to "grounding"  # Model is doing well, increase difficulty

elif current_task == "grounding":
    if mean_reward < grounding_threshold:
        switch to "matching"  # Model is struggling, decrease difficulty
```

## Usage

### 1. Enable in Training

The buffer is automatically initialized if configured. Just add the config section to your training yaml:

```yaml
# config/your_training_config.yaml
dynamic_buffer:
  enable: true
  buffer_size: 100
  matching_threshold: 0.7
  grounding_threshold: 0.3
```

### 2. Run Training

```bash
python -m my_recipe.main_dcrl
```

### 3. Monitor Metrics

The following metrics are logged during training:

- `buffer/current_task`: Current task ("matching" or "grounding")
- `buffer/total_samples`: Total samples in buffer
- `buffer/matching_count`: Number of matching samples
- `buffer/matching_mean_reward`: Average reward for matching task
- `buffer/grounding_count`: Number of grounding samples
- `buffer/grounding_mean_reward`: Average reward for grounding task
- `buffer/num_switches`: Total number of task switches
- `buffer/current_matching_stage`: Current curriculum stage id (if enabled)
- `buffer/stage/<stage_id>/mean_reward`, `recent_count`, `total_count`: rolling stats per matching stage

## Customization

### Adjusting Task Difficulty Balance

If the model switches too frequently:
```yaml
min_samples_for_switch: 50  # Require more samples before switching
```

If the model stays on matching too long:
```yaml
matching_threshold: 0.6  # Lower threshold = switch earlier
```

If the model falls back to matching too quickly:
```yaml
grounding_threshold: 0.2  # Lower threshold = tolerate more difficulty
```

### Custom Task Switching Metrics

You can extend the buffer to use different metrics by modifying `get_task_metrics()` in `dynamic_task_buffer.py`:

```python
def get_task_metrics(self) -> dict:
    metrics = {
        "matching": {
            "mean_reward": np.mean(self.matching_rewards),
            "success_rate": np.mean([r > 0.5 for r in self.matching_rewards]),  # Custom metric
            # Add more custom metrics
        },
        # ...
    }
    return metrics
```

## Implementation Details

### Buffer Storage

Each buffer entry contains:
```python
{
    "task": "matching" | "grounding",
    "reward": float,  # Sequence-level reward
    "db_idx": str,    # Dataset index for reproducibility
}
```

### Reward Computation

Sequence-level rewards are computed by summing token-level rewards:
```python
seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1)
```

### Sample Processing (v2)

The buffer processes raw samples through task-specific methods:

```python
# Get raw sample from dataset
raw_sample = raw_dataset[idx]  # Reads from LMDB, decodes, rescales

# Buffer applies task-specific processing
if task == "matching":
    processed = buffer.process_matching_task(raw_sample)
else:  # grounding
    processed = buffer.process_grounding_task(raw_sample)

# Result: Training-ready sample with prompts, images, and answers
```

**Benefits:**
- Single LMDB access (not duplicated)
- Task logic centralized in buffer
- Easy to add new tasks (just add a new `process_xxx_task` method)

## Files Modified/Created

### New Files (v2)
- `my_recipe/dynamic_task_buffer.py` - Core buffer logic with integrated task processing
- `my_recipe/mydatasets/anno_raw.py` - **NEW!** Raw LMDB dataset reader
- `my_recipe/buffered_dataloader.py` - DataLoader wrapper
- `my_recipe/config/dynamic_buffer.yaml` - Example config
- `my_recipe/DYNAMIC_BUFFER_README.md` - This documentation
- `my_recipe/example_buffer_usage.py` - Usage examples

### Modified Files
- `my_recipe/workers/dapo_ray_trainer.py` - Buffer integration
  - Added `__init__` with buffer initialization
  - Added `_init_dynamic_buffer()` method (creates single raw dataset)
  - Updated `fit()` to add samples with rewards
  - Added buffer metrics logging
- `my_recipe/main_dcrl.py` - Load buffer config
- `my_recipe/mydatasets/anno_grounding.py` - Removed debug breakpoint

## Examples

### Example 1: Aggressive Curriculum (Quick Progression)

```yaml
dynamic_buffer:
  enable: true
  buffer_size: 50
  matching_threshold: 0.6  # Switch earlier
  grounding_threshold: 0.4  # Tolerate more difficulty
  min_samples_for_switch: 10  # Fast switching
```

### Example 2: Conservative Curriculum (Gradual Progression)

```yaml
dynamic_buffer:
  enable: true
  buffer_size: 200
  matching_threshold: 0.8  # Require high performance before advancing
  grounding_threshold: 0.2  # Only fall back if really struggling
  min_samples_for_switch: 50  # Require more evidence before switching
```

### Example 3: Disabled (Static Task)

```yaml
dynamic_buffer:
  enable: false
  # Buffer will not affect training
```

## Debugging

### Print Buffer State

The buffer prints task switches automatically:
```
[DynamicBuffer] Switching to GROUNDING (matching reward: 0.752)
[DynamicBuffer] Switching to MATCHING (grounding reward: 0.281)
```

### Check Buffer Metrics in Logs

Look for these metrics in your training logs:
```python
{
    "buffer/current_task": "grounding",
    "buffer/matching_mean_reward": 0.75,
    "buffer/grounding_mean_reward": 0.32,
    "buffer/num_switches": 3
}
```

### Verify Task Distribution

Monitor `buffer/matching_count` vs `buffer/grounding_count` to ensure balanced sampling.

## Future Enhancements

Potential improvements you can implement:

1. **Multiple Difficulty Levels**
   - Add intermediate tasks between matching and grounding
   - Implement multi-level curriculum with >2 tasks

2. **Adaptive Thresholds**
   - Dynamically adjust thresholds based on training progress
   - Use learning rate schedule-like decay for thresholds

3. **Per-Sample Difficulty**
   - Track difficulty of individual samples (e.g., number of regions)
   - Select samples based on both task type and sample difficulty

4. **Exploration Strategy**
   - Add epsilon-greedy exploration to occasionally try harder tasks
   - Implement UCB-style selection for task exploration

5. **Multi-Metric Decision**
   - Combine multiple metrics (reward, success rate, confidence)
   - Use weighted combination for more robust switching

## Troubleshooting

### Buffer not switching tasks

**Issue**: Task stays on matching despite high performance

**Solutions**:
- Lower `matching_threshold`
- Reduce `min_samples_for_switch`
- Check if rewards are being computed correctly

### Too frequent switching

**Issue**: Task switches every few steps

**Solutions**:
- Increase `min_samples_for_switch`
- Widen the gap between thresholds
- Increase `buffer_size` for more stable metrics

### Performance degradation

**Issue**: Model performs worse with buffer enabled

**Solutions**:
- Start with conservative thresholds
- Increase `buffer_size` for smoother transitions
- Consider disabling temporarily to establish baseline

## Contact

For questions or issues related to the dynamic buffer implementation, please refer to the code comments or modify the implementation to fit your specific needs.

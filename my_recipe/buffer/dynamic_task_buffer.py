"""
Dynamic Task-Switching Buffer for MLLM Training (v2)

This version integrates task-specific processing logic directly into the buffer,
eliminating the need for separate matching and grounding dataset instances.
The buffer receives raw samples and applies task processing based on performance.
"""

import copy
import json
import logging
import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from my_recipe.mydatasets import safe_collate_fn

try:  # Optional import for type checks
    from my_recipe.mydatasets.anno_raw import AnnoRawDataset
except Exception:  # pragma: no cover - avoid hard dependency during docs/build
    AnnoRawDataset = None  # type: ignore
from my_recipe.mydatasets.anno_db import (
    ANNO_GROUND_TEMPLATE,
    ANNO_MATCH_MULTI_TEMPLATE,
    ANNO_MATCH_TEMPLATE,  # noqa: F401
    annotate_image,
    draw_img,
    filter_sparse_points,
    greedy_max_spacing_selection,
)
from my_recipe.mydatasets.utils import build_model_inputs

logger = logging.getLogger(__name__)

MAX_SPACING_ATTEMPTS = 6

_DEFAULT_ANNOTATE_STYLE = {
    "radius": 8,
    "label_padding": 2,
    "font_scale": 0.5,
    "font_thickness": 1,
    "font": cv2.FONT_HERSHEY_SIMPLEX if cv2 is not None else None,
}


def _compute_annotation_guard_px(labels: Optional[list[str]] = None, style: Optional[dict] = None) -> float:
    """Estimate the diagonal footprint (in pixels) of rendered annotations."""
    if cv2 is None:
        return 32.0  # Conservative fallback when OpenCV is unavailable

    params = dict(_DEFAULT_ANNOTATE_STYLE)
    if style:
        params.update(style)

    font = params.get("font")
    if font is None:
        return 32.0

    radius = float(params.get("radius", 8))
    padding = float(params.get("label_padding", 2))
    font_scale = float(params.get("font_scale", 0.5))
    font_thickness = int(params.get("font_thickness", 1))

    label_samples = labels or ["999", "88"]
    max_diag_sq = 0.0
    for label in label_samples:
        (text_w, text_h), baseline = cv2.getTextSize(str(label), font, font_scale, font_thickness)
        text_w = max(1, text_w)
        text_h = max(1, text_h)
        width = max(2 * radius, text_w + 2 * padding)
        height = 2 * radius + text_h + baseline + 3 * padding
        max_diag_sq = max(max_diag_sq, float(width * width + height * height))

    return math.sqrt(max_diag_sq)


ANNOTATION_GUARD_PX = _compute_annotation_guard_px()


class InsufficientMatchesError(RuntimeError):
    """Raised when a sample does not contain enough matches for the configured stage."""

    def __init__(self, stage_id: str | None, required: int, available: int, db_idx: int | None = None):
        msg = (
            f"Sample has {available} matches but requires at least {required}"
            f"{f' for stage {stage_id}' if stage_id else ''}"
            f"{f' (db_idx={db_idx})' if db_idx is not None else ''}."
        )
        super().__init__(msg)
        self.stage_id = stage_id
        self.required = required
        self.available = available
        self.db_idx = db_idx


@dataclass
class MatchingFilterConfig:
    """Configuration for matching point filtering difficulty."""

    min_distance: float = 20.0
    min_points: int = 1
    max_points: int = 6
    strategy: Literal["static", "adaptive"] = "static"
    schedule: list[dict] = field(default_factory=list)


@dataclass
class StageTransitionRule:
    """Promotion/Demotion rule for curriculum stages."""

    threshold: float = 0.85
    min_count: int = 60
    use_recent_window: bool = True
    success_ratio_threshold: Optional[float] = None
    success_reward_threshold: Optional[float] = None


@dataclass
class MatchingDistractorConfig:
    """Control how many unmatched distractor points each image receives."""

    enable: bool = False
    min_extra_image1: int = 0
    max_extra_image1: int = 0
    min_extra_image2: int = 0
    max_extra_image2: int = 0
    mode: Literal["sample", "synthetic"] = "sample"
    synthetic_radius: float = 80.0


@dataclass
class RepresentationConfig:
    """How a single image view is represented to the model."""

    render_image: bool = True
    include_text: bool = False
    annotate_ratio: float = 1.0  # fraction of available points to annotate on the image
    text_ratio: float = 1.0  # fraction of the *remaining* (non-annotated) points to describe in text
    min_annotated_points: int = 0
    min_text_points: int = 0
    text_format: Literal["json", "list"] = "json"
    text_header: str = ""
    coordinate_scale: Literal["auto", "relative", "absolute"] = "auto"
    text_precision: int = 0
    ensure_core_cover: bool = True


@dataclass
class MatchingStageRepresentation:
    image1: RepresentationConfig = field(default_factory=RepresentationConfig)
    image2: RepresentationConfig = field(default_factory=RepresentationConfig)


@dataclass
class PointFilteringStrategy:
    """Stage-specific point sampling strategy."""

    mode: Literal["cluster", "greedy", "random", "dense"] = "cluster"
    cluster_eps_scale: float = 0.8
    greedy_spacing: bool = True


@dataclass
class MatchingStageConfig:
    """Curriculum stage definition for matching tasks."""

    stage_id: str
    description: str = ""
    representation: MatchingStageRepresentation = field(default_factory=MatchingStageRepresentation)
    filter_overrides: dict = field(default_factory=dict)
    filter_schedule: list[dict] = field(default_factory=list)
    distractors: MatchingDistractorConfig = field(default_factory=MatchingDistractorConfig)
    promotion: StageTransitionRule | None = field(default_factory=StageTransitionRule)
    demotion: StageTransitionRule | None = None
    prompt_appendix: str = ""
    shuffle_image2_labels: bool = True
    point_filter: PointFilteringStrategy | None = None


@dataclass
class StageVariant:
    """Concrete stage variant that can be sampled within an aggregate curriculum stage."""

    name: str
    weight: float
    config: MatchingStageConfig


@dataclass
class CurriculumStage:
    """A stage node in the curriculum containing one or more variants."""

    stage_id: str
    description: str = ""
    promotion: StageTransitionRule | None = None
    demotion: StageTransitionRule | None = None
    variants: list[StageVariant] = field(default_factory=list)

    def reference_config(self) -> Optional[MatchingStageConfig]:
        return self.variants[0].config if self.variants else None


def _default_matching_curriculum_stages() -> list[MatchingStageConfig]:
    """Generate default curriculum covering image-only, hybrid, and distractor stages."""

    stage_image_only = MatchingStageConfig(
        stage_id="image_only",
        description="Image annotations only. Progressively denser spatial layouts.",
        filter_overrides={"min_distance": 70.0, "max_points": 5},
        filter_schedule=[
            {"threshold": 0.0, "min_distance": 90.0, "max_points": 3},
            {"threshold": 0.7, "min_distance": 55.0, "max_points": 5},
            {"threshold": 0.85, "min_distance": 35.0, "max_points": 7},
        ],
        representation=MatchingStageRepresentation(
            image1=RepresentationConfig(render_image=True, include_text=False),
            image2=RepresentationConfig(render_image=True, include_text=False),
        ),
        promotion=StageTransitionRule(threshold=0.86, min_count=80),
        demotion=None,
        prompt_appendix="Stage 1 — Image-only: rely purely on annotated visuals. Expect point clusters to gradually become denser as training succeeds.",
        point_filter=PointFilteringStrategy(mode="cluster", cluster_eps_scale=1.0, greedy_spacing=True),
    )

    stage_hybrid_one_to_one = MatchingStageConfig(
        stage_id="hybrid_one_to_one",
        description="Hybrid visual + text one-to-one matching.",
        filter_overrides={"min_distance": 50.0, "max_points": 6},
        representation=MatchingStageRepresentation(
            image1=RepresentationConfig(
                render_image=True,
                include_text=True,
                annotate_ratio=1.0,
                text_ratio=0.5,
                text_header="Image A coordinate snippets (0-1000 scale).",
            ),
            image2=RepresentationConfig(
                render_image=True,
                include_text=True,
                annotate_ratio=0.6,
                text_ratio=1.0,
                text_header="Image B coordinate snippets (0-1000 scale).",
            ),
        ),
        promotion=StageTransitionRule(threshold=0.88, min_count=110),
        demotion=StageTransitionRule(threshold=0.68, min_count=60),
        prompt_appendix=(
            "Stage 2 — Mixed modalities: some regions include JSON coordinate hints. "
            "Cross-check visual IDs with coordinate tables to maintain perfect one-to-one mappings."
        ),
        point_filter=PointFilteringStrategy(mode="cluster", cluster_eps_scale=0.6, greedy_spacing=True),
    )

    stage_hybrid_multi = MatchingStageConfig(
        stage_id="hybrid_multi_to_multi",
        description="Hybrid input with distractor points on both images.",
        filter_overrides={"min_distance": 40.0, "max_points": 7},
        representation=MatchingStageRepresentation(
            image1=RepresentationConfig(
                render_image=True,
                include_text=True,
                annotate_ratio=0.7,
                text_ratio=1.0,
                text_header="Image A (may include distractor IDs).",
            ),
            image2=RepresentationConfig(
                render_image=True,
                include_text=True,
                annotate_ratio=0.7,
                text_ratio=1.0,
                text_header="Image B (may include distractor IDs).",
            ),
        ),
        distractors=MatchingDistractorConfig(
            enable=True,
            min_extra_image1=1,
            max_extra_image1=2,
            min_extra_image2=1,
            max_extra_image2=2,
        ),
        promotion=None,
        demotion=StageTransitionRule(threshold=0.62, min_count=90),
        prompt_appendix=(
            "Stage 3 — Distractor-aware: either image can show extra IDs with no true counterpart. "
            "When an Image A ID has no match, output null for that key. Ignore distractor-only IDs from Image B."
        ),
        point_filter=PointFilteringStrategy(mode="greedy", cluster_eps_scale=0.4, greedy_spacing=True),
    )

    return [stage_image_only, stage_hybrid_one_to_one, stage_hybrid_multi]


def _default_curriculum_stages() -> list[CurriculumStage]:
    """Wrap the default stage configs as single-variant curriculum nodes."""

    curriculum: list[CurriculumStage] = []
    for cfg in _default_matching_curriculum_stages():
        curriculum.append(
            CurriculumStage(
                stage_id=cfg.stage_id,
                description=cfg.description,
                promotion=cfg.promotion,
                demotion=cfg.demotion,
                variants=[StageVariant(name=cfg.stage_id, weight=1.0, config=cfg)],
            )
        )
    return curriculum


@dataclass
class MatchingCurriculumConfig:
    """Top-level curriculum settings for matching tasks."""

    enable: bool = True
    metric_window: int = 120
    start_stage: Optional[str] = None
    stages: list[CurriculumStage] = field(default_factory=_default_curriculum_stages)


def _build_stage_from_dict(stage_dict: dict) -> MatchingStageConfig:
    stage_id = stage_dict.get("stage_id")
    if not stage_id:
        raise ValueError("matching curriculum stage must include 'stage_id'")

    representation_dict = stage_dict.get("representation", {})
    rep_image1 = representation_dict.get("image1", {})
    rep_image2 = representation_dict.get("image2", {})

    representation = MatchingStageRepresentation(
        image1=RepresentationConfig(**rep_image1),
        image2=RepresentationConfig(**rep_image2),
    )

    distractor_cfg = MatchingDistractorConfig(**stage_dict.get("distractors", {}))

    promotion_rule = stage_dict.get("promotion")
    promotion = StageTransitionRule(**promotion_rule) if promotion_rule else None

    demotion_rule = stage_dict.get("demotion")
    demotion = StageTransitionRule(**demotion_rule) if demotion_rule else None

    point_filter_cfg = stage_dict.get("point_filter")
    point_filter = PointFilteringStrategy(**point_filter_cfg) if point_filter_cfg else None

    return MatchingStageConfig(
        stage_id=stage_id,
        description=stage_dict.get("description", ""),
        representation=representation,
        filter_overrides=stage_dict.get("filter_overrides", {}),
        filter_schedule=stage_dict.get("filter_schedule", []),
        distractors=distractor_cfg,
        promotion=promotion,
        demotion=demotion,
        prompt_appendix=stage_dict.get("prompt_appendix", ""),
        shuffle_image2_labels=stage_dict.get("shuffle_image2_labels", True),
        point_filter=point_filter,
    )


def _deep_update_dict(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (in-place) and return base."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def _build_stage_variant_from_spec(
    spec: dict,
    stage_library: dict[str, dict],
    fallback_stage_id: Optional[str] = None,
) -> StageVariant:
    """Create a StageVariant from a spec referencing either a template or inline stage."""

    weight = float(spec.get("weight", spec.get("prob", 1.0)))
    template_name = spec.get("use") or spec.get("template")

    if template_name:
        if template_name not in stage_library:
            raise ValueError(f"Unknown stage template '{template_name}' referenced in curriculum.")
        stage_dict = copy.deepcopy(stage_library[template_name])
        overrides = spec.get("stage_overrides")
        if overrides:
            if not isinstance(overrides, dict):
                raise ValueError(f"stage_overrides for template '{template_name}' must be a dict.")
            _deep_update_dict(stage_dict, copy.deepcopy(overrides))
        variant_stage_id = spec.get("variant_stage_id")
        if variant_stage_id:
            stage_dict["stage_id"] = variant_stage_id
    else:
        stage_dict = copy.deepcopy(spec)

    if "stage_id" not in stage_dict and fallback_stage_id:
        stage_dict["stage_id"] = fallback_stage_id

    stage_cfg = _build_stage_from_dict(stage_dict)
    variant_name = spec.get("name") or stage_cfg.stage_id
    return StageVariant(name=variant_name, weight=weight, config=stage_cfg)


def _build_curriculum_stage(
    stage_entry: dict, stage_library: dict[str, dict]
) -> CurriculumStage:
    """Construct a CurriculumStage (with variants) from configuration."""

    stage_id = stage_entry.get("stage_id")
    if not stage_id:
        raise ValueError("Each curriculum stage must define 'stage_id'.")

    aggregate_specs = stage_entry.get("aggregate")
    variants: list[StageVariant] = []

    if aggregate_specs:
        if not isinstance(aggregate_specs, list):
            raise ValueError(f"Stage '{stage_id}' aggregate must be a list.")
        for spec in aggregate_specs:
            if not isinstance(spec, dict):
                raise ValueError(f"Aggregate entries for stage '{stage_id}' must be dictionaries.")
            variants.append(_build_stage_variant_from_spec(spec, stage_library))
    elif "use" in stage_entry or "template" in stage_entry:
        variants.append(_build_stage_variant_from_spec(stage_entry, stage_library, fallback_stage_id=stage_id))
    else:
        # Legacy inline definition: treat the entire entry as the stage config.
        variants.append(_build_stage_variant_from_spec(stage_entry, stage_library, fallback_stage_id=stage_id))

    if not variants:
        raise ValueError(f"Stage '{stage_id}' did not define any variants.")

    promotion_rule = stage_entry.get("promotion")
    demotion_rule = stage_entry.get("demotion")

    if promotion_rule:
        promotion = StageTransitionRule(**promotion_rule)
    elif aggregate_specs:
        promotion = None
    else:
        promotion = copy.deepcopy(variants[0].config.promotion)

    if demotion_rule:
        demotion = StageTransitionRule(**demotion_rule)
    elif aggregate_specs:
        demotion = None
    else:
        demotion = copy.deepcopy(variants[0].config.demotion)

    description = stage_entry.get("description") or variants[0].config.description

    return CurriculumStage(
        stage_id=stage_id,
        description=description,
        promotion=promotion,
        demotion=demotion,
        variants=variants,
    )


def matching_curriculum_from_dict(cfg: Optional[dict]) -> MatchingCurriculumConfig:
    """Convert raw config (e.g., OmegaConf -> dict) into curriculum dataclass."""

    default_cfg = MatchingCurriculumConfig()
    if cfg is None:
        return default_cfg

    stage_library_cfg = cfg.get("stage_library", {})
    stage_library: dict[str, dict] = {}
    for name, stage_dict in stage_library_cfg.items():
        if not isinstance(stage_dict, dict):
            raise ValueError(f"Stage template '{name}' must be a dict.")
        entry = copy.deepcopy(stage_dict)
        entry.setdefault("stage_id", name)
        stage_library[name] = entry

    stages_data = cfg.get("stages")
    if stages_data:
        stages = [_build_curriculum_stage(stage_dict, stage_library) for stage_dict in stages_data]
    else:
        stages = default_cfg.stages

    return MatchingCurriculumConfig(
        enable=cfg.get("enable", default_cfg.enable),
        metric_window=int(cfg.get("metric_window", default_cfg.metric_window)),
        start_stage=cfg.get("start_stage", default_cfg.start_stage),
        stages=stages,
    )


class MatchingCurriculumManager:
    """Stateful helper that tracks and updates the current matching stage."""

    def __init__(self, config: MatchingCurriculumConfig):
        self.config = config
        self.stage_map = {stage.stage_id: stage for stage in config.stages}
        self.stage_order = [stage.stage_id for stage in config.stages]
        self.variant_parent_map: dict[str, str] = {}
        for stage in config.stages:
            for variant in stage.variants:
                self.variant_parent_map[variant.config.stage_id] = stage.stage_id
        self._completed = False

        if not self.stage_order:
            self.current_stage_id: Optional[str] = None
        else:
            start_stage = config.start_stage or self.stage_order[0]
            if start_stage not in self.stage_map:
                start_stage = self.stage_order[0]
            self.current_stage_id = start_stage

        self.history: list[tuple[str, str, float]] = []
        self._last_stage_change: Optional[tuple[str, float]] = None

    def get_current_stage(self) -> Optional[CurriculumStage]:
        if self.current_stage_id is None:
            return None
        return self.stage_map.get(self.current_stage_id)

    def get_current_stage_config(self) -> Optional[MatchingStageConfig]:
        stage = self.get_current_stage()
        return stage.reference_config() if stage else None

    def sample_current_variant(self) -> Optional[StageVariant]:
        stage = self.get_current_stage()
        if stage is None or not stage.variants:
            return None
        weights = [max(0.0, variant.weight) for variant in stage.variants]
        if not any(weights):
            weights = [1.0 for _ in stage.variants]
        return random.choices(stage.variants, weights=weights, k=1)[0]

    def _stage_index(self, stage_id: str) -> int:
        return self.stage_order.index(stage_id)

    def get_next_stage(self) -> Optional[CurriculumStage]:
        if self.current_stage_id is None:
            return None
        idx = self._stage_index(self.current_stage_id)
        if idx + 1 < len(self.stage_order):
            return self.stage_map[self.stage_order[idx + 1]]
        return None

    def get_prev_stage(self) -> Optional[CurriculumStage]:
        if self.current_stage_id is None:
            return None
        idx = self._stage_index(self.current_stage_id)
        if idx - 1 >= 0:
            return self.stage_map[self.stage_order[idx - 1]]
        return None

    def get_parent_stage_id(self, variant_stage_id: str) -> Optional[str]:
        return self.variant_parent_map.get(variant_stage_id)

    def is_complete(self) -> bool:
        return self._completed

    @staticmethod
    def _satisfy_rule(rule: StageTransitionRule, stats: dict) -> bool:
        if rule is None:
            return False
        count_key = "recent_count" if rule.use_recent_window else "count"
        available = int(stats.get(count_key, 0))
        metric = float(stats.get("mean_reward", 0.0))
        if available < rule.min_count:
            return False
        if metric < rule.threshold:
            return False

        ratio_threshold = rule.success_ratio_threshold
        reward_threshold = rule.success_reward_threshold
        if ratio_threshold is not None and reward_threshold is not None:
            rewards_key = "recent_rewards" if rule.use_recent_window else "recent_rewards"
            rewards: Optional[list[float]] = stats.get(rewards_key)
            if not rewards:
                return False
            success_count = sum(1 for value in rewards if value >= reward_threshold)
            success_ratio = success_count / len(rewards) if rewards else 0.0
            stats_key_prefix = "recent" if rule.use_recent_window else "overall"
            stats[f"{stats_key_prefix}_success_count"] = success_count
            stats[f"{stats_key_prefix}_success_ratio"] = success_ratio
            if success_ratio < ratio_threshold:
                return False
        return True

    def _should_demote(self, rule: StageTransitionRule | None, stats: dict) -> bool:
        if rule is None:
            return False
        count_key = "recent_count" if rule.use_recent_window else "count"
        available = int(stats.get(count_key, 0))
        metric = float(stats.get("mean_reward", 0.0))
        if available < rule.min_count:
            return False
        return metric <= rule.threshold

    def maybe_update(self, stage_metrics: dict[str, dict]) -> Optional[dict]:
        """Update current stage given up-to-date metrics.

        Returns transition info when a stage change occurs.
        """

        current_stage = self.get_current_stage()
        if current_stage is None:
            return None

        stats = stage_metrics.get(current_stage.stage_id, {})

        # Promotion check (progress to harder stage)
        if current_stage.promotion and self._satisfy_rule(current_stage.promotion, stats):
            next_stage = self.get_next_stage()
            if next_stage is not None:
                prev_stage_id = self.current_stage_id
                self.current_stage_id = next_stage.stage_id
                self.history.append((prev_stage_id, self.current_stage_id, float(stats.get("mean_reward", 0.0))))
                self._completed = False
                self._last_stage_change = (self.current_stage_id, float(stats.get("mean_reward", 0.0)))
                return {"transition": "promote", "from": prev_stage_id, "to": self.current_stage_id, "metric": stats}
            else:
                self._completed = True
                self._last_stage_change = (self.current_stage_id, float(stats.get("mean_reward", 0.0)))
                return {
                    "transition": "complete",
                    "from": self.current_stage_id,
                    "to": self.current_stage_id,
                    "metric": stats,
                }

        # Demotion check (fallback to easier stage)
        if current_stage.demotion and self._should_demote(current_stage.demotion, stats):
            prev_stage = self.get_prev_stage()
            if prev_stage is not None:
                prev_stage_id = self.current_stage_id
                self.current_stage_id = prev_stage.stage_id
                self.history.append((prev_stage_id, self.current_stage_id, float(stats.get("mean_reward", 0.0))))
                self._completed = False
                self._last_stage_change = (self.current_stage_id, float(stats.get("mean_reward", 0.0)))
                return {"transition": "demote", "from": prev_stage_id, "to": self.current_stage_id, "metric": stats}

        return None


@dataclass
class GroundingPointConfig:
    """Configuration for grounding point sampling difficulty."""

    min_points: int = 1
    max_points: int = 3
    strategy: Literal["static", "adaptive"] = "static"
    schedule: list[dict] = field(default_factory=list)


@dataclass
class BufferConfig:
    """Configuration for DynamicTaskBuffer."""

    buffer_size: int = 100
    task_switch_metric: str = "mean_reward"
    matching_threshold: float = 0.7  # Switch to grounding if matching performance > this
    grounding_threshold: float = 0.3  # Switch to matching if grounding performance < this
    min_samples_for_switch: int = 20
    enable: bool = True
    task_mode: Literal["dynamic", "matching", "grounding"] = "dynamic"
    matching_filter: MatchingFilterConfig = field(default_factory=MatchingFilterConfig)
    matching_curriculum: MatchingCurriculumConfig = field(default_factory=MatchingCurriculumConfig)
    grounding_points: GroundingPointConfig = field(default_factory=GroundingPointConfig)
    warmup_proportion: float = 0.5
    log_file: Optional[str] = None
    max_resample_attempts: int = 3
    resample_neighbor_window: int = 40


class DynamicTaskBuffer:
    """Buffer that dynamically switches between matching and grounding tasks based on performance.

    This version integrates all task-specific processing logic, eliminating the need
    for separate dataset instances for each task.

    The buffer:
    1. Receives raw samples from a single AnnoRawDataset
    2. Decides task type (matching/grounding) based on recent performance
    3. Applies task-specific processing (prompts, annotations, answers)
    4. Returns formatted samples ready for training
    5. Collects rewards after training to inform future decisions

    Workflow:
        1. Receive raw samples from dataloader
        2. Check buffer's cached performance metrics
        3. Decide task type (matching/grounding) based on metrics
        4. Apply task-specific processing to raw samples
        5. After training, add samples with rewards back to buffer
    """

    def __init__(self, config: BufferConfig, raw_dataset):
        """Initialize the dynamic task buffer.

        Args:
            config: Buffer configuration
            raw_dataset: AnnoRawDataset instance that provides raw LMDB samples
        """
        self.config = config
        self.raw_dataset = raw_dataset

        # Buffer storage: each entry is a dict with sample info and rewards
        self.buffer = deque(maxlen=config.buffer_size)

        # Optional dedicated logger
        self._buffer_logger: Optional[logging.Logger] = None
        if config.log_file:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._buffer_logger = logging.getLogger("dynamic_task_buffer.log")
            self._buffer_logger.setLevel(logging.INFO)
            if not any(
                isinstance(h, logging.FileHandler) and h.baseFilename == Path(config.log_file).resolve()
                for h in self._buffer_logger.handlers
            ):
                file_handler = logging.FileHandler(config.log_file, mode="a", encoding="utf-8")
                formatter = logging.Formatter("%(asctime)s - %(message)s")
                file_handler.setFormatter(formatter)
                self._buffer_logger.addHandler(file_handler)
            self._buffer_logger.propagate = False

        # Matching curriculum manager and stage statistics
        self.matching_curriculum_config = config.matching_curriculum
        if self.matching_curriculum_config and self.matching_curriculum_config.enable:
            self.matching_curriculum_manager = MatchingCurriculumManager(self.matching_curriculum_config)
        else:
            self.matching_curriculum_manager = None

        self.current_matching_stage_id = (
            self.matching_curriculum_manager.current_stage_id if self.matching_curriculum_manager else None
        )

        metric_window = (
            self.matching_curriculum_config.metric_window if self.matching_curriculum_manager is not None else 0
        )
        self.stage_reward_history: dict[str, deque] = {}
        self.stage_seen_counts: defaultdict[str, int] = defaultdict(int)
        self.stage_variant_reward_history: dict[str, deque] = {}
        self.stage_variant_seen_counts: defaultdict[str, int] = defaultdict(int)
        if self.matching_curriculum_manager is not None:
            for stage in self.matching_curriculum_config.stages:
                stage_history = deque(maxlen=metric_window)
                self.stage_reward_history[stage.stage_id] = stage_history
                for variant in stage.variants:
                    if variant.config.stage_id == stage.stage_id:
                        continue
                    self.stage_variant_reward_history[variant.config.stage_id] = deque(maxlen=metric_window)

        # Task statistics and adaptive state
        if self.config.task_mode == "grounding":
            self.current_task = "grounding"
        else:
            self.current_task = "matching"
        self.task_history = []
        stage_cfg = self._get_current_stage_config()
        self.current_matching_params = self._suggest_matching_params(
            metric=0.0,
            stage_cfg=stage_cfg,
            aggregator_stage_id=self.current_matching_stage_id,
        )
        self.current_grounding_params = self._suggest_grounding_params(metric=0.0)
        self._adaptation_ready = False

        # System prompt
        self.system_prompt = "You are a helpful assistant."
        self.sys_msg = {"role": "system", "content": {"text": self.system_prompt}}
        self._annotation_guard_px = float(max(ANNOTATION_GUARD_PX, 1.0))

    def add_samples(self, batch, rewards: torch.Tensor):
        """Add samples with their rewards to the buffer after training.

        Args:
            batch: DataProto batch containing sample information
            rewards: Tensor of shape (batch_size,) containing final rewards for each sample
        """
        # Extract task type from batch
        data_sources = batch.non_tensor_batch.get("data_source", [])

        # recent_samples: list[dict] = []
        for i, reward in enumerate(rewards):
            # Parse task type from data_source
            data_source = json.loads(data_sources[i]) if isinstance(data_sources[i], str) else data_sources[i]
            task_type = "matching" if data_source.get("type") == "ANNO_MATCH" else "grounding"
            stage_name = data_source.get("matching_stage") if task_type == "matching" else None
            variant_name = data_source.get("matching_stage_variant") if task_type == "matching" else None

            sample_meta = batch.non_tensor_batch.get("sample_meta", [None])[i]
            if isinstance(sample_meta, np.ndarray):
                sample_meta = sample_meta.item()
            if isinstance(sample_meta, bytes):
                sample_meta = sample_meta.decode("utf-8")
            if isinstance(sample_meta, str):
                try:
                    sample_meta = json.loads(sample_meta)
                except json.JSONDecodeError:
                    sample_meta = {"db_idx": sample_meta}
            if not isinstance(sample_meta, dict):
                sample_meta = {"db_idx": sample_meta}

            db_idx = sample_meta.get("db_idx")

            # Store in buffer
            # TODO @zhonghao: consider adding more metadata
            sample_info = {
                "task": task_type,
                "reward": float(reward),
                "db_idx": db_idx,
                "stage": stage_name,
            }
            self.buffer.append(sample_info)
            # recent_samples.append(sample_info)

            if stage_name:
                self.stage_seen_counts[stage_name] += 1
                if stage_name not in self.stage_reward_history:
                    window = self.matching_curriculum_config.metric_window if self.matching_curriculum_manager else 120
                    self.stage_reward_history[stage_name] = deque(maxlen=window)
                self.stage_reward_history[stage_name].append(float(reward))
            if variant_name and variant_name != stage_name:
                self.stage_variant_seen_counts[variant_name] += 1
                window = self.matching_curriculum_config.metric_window if self.matching_curriculum_manager else 120
                if variant_name not in self.stage_variant_reward_history:
                    self.stage_variant_reward_history[variant_name] = deque(maxlen=window)
                self.stage_variant_reward_history[variant_name].append(float(reward))

        self._update_adaptation_state()
        # self._log_buffer_snapshot(recent_samples)
        self._log_buffer_snapshot(self.buffer)

    def get_task_metrics(self) -> dict:
        """Compute performance metrics for each task.

        Returns:
            Dictionary containing metrics for matching and grounding tasks
        """
        matching_rewards = [entry["reward"] for entry in self.buffer if entry["task"] == "matching"]
        grounding_rewards = [entry["reward"] for entry in self.buffer if entry["task"] == "grounding"]

        metrics = {
            "matching": {
                "mean_reward": float(np.mean(matching_rewards)) if matching_rewards else 0.0,
                "count": len(matching_rewards),
            },
            "grounding": {
                "mean_reward": float(np.mean(grounding_rewards)) if grounding_rewards else 0.0,
                "count": len(grounding_rewards),
            },
            "buffer_size": len(self.buffer),
        }

        if self.stage_reward_history:
            stage_metrics = {}
            for stage_id, history in self.stage_reward_history.items():
                recent_count = len(history)
                mean_reward = float(np.mean(history)) if recent_count else 0.0
                total_count = int(self.stage_seen_counts.get(stage_id, 0))
                stage_metrics[stage_id] = {
                    "mean_reward": mean_reward,
                    "count": total_count,
                    "recent_count": recent_count,
                    # "recent_rewards": list(history),
                }
            metrics["matching_stages"] = stage_metrics

        return metrics

    @staticmethod
    def _pick_schedule_stage(schedule: list[dict], stats: dict) -> Optional[dict]:
        """Select the schedule stage whose criteria are satisfied by stats."""
        if not schedule:
            return None

        metric = float(stats.get("mean_reward", 0.0))
        rewards = stats.get("recent_rewards") or []

        stage = None
        for candidate in sorted(schedule, key=lambda item: item.get("threshold", 0.0)):
            threshold = float(candidate.get("threshold", 0.0))
            if metric < threshold:
                continue

            ratio_threshold = candidate.get("success_ratio_threshold")
            reward_threshold = candidate.get("success_reward_threshold")
            if ratio_threshold is not None and reward_threshold is not None:
                if not rewards:
                    continue
                success_count = sum(1 for value in rewards if value >= float(reward_threshold))
                if not rewards:
                    continue
                success_ratio = success_count / len(rewards)
                if success_ratio < float(ratio_threshold):
                    continue

            stage = candidate
        return stage

    @staticmethod
    def _merge_matching_params(base: dict, overrides: Optional[dict]) -> dict:
        if not overrides:
            return base

        merged = copy.deepcopy(base)
        if "min_distance" in overrides and overrides["min_distance"] is not None:
            merged["min_distance"] = float(overrides["min_distance"])
        if "min_points" in overrides and overrides["min_points"] is not None:
            merged["min_points"] = int(overrides["min_points"])
        if "max_points" in overrides and overrides["max_points"] is not None:
            merged["max_points"] = int(overrides["max_points"])

        merged["min_distance"] = max(1.0, float(merged.get("min_distance", 1.0)))
        merged["min_points"] = max(1, int(merged.get("min_points", 1)))
        merged["max_points"] = max(1, int(merged.get("max_points", 1)))
        merged["max_points"] = max(merged["max_points"], merged["min_points"])
        return merged

    def _apply_filter_schedule(self, params: dict, schedule: Optional[list[dict]], stats_payload: dict) -> dict:
        if not schedule:
            return params
        schedule_stage = self._pick_schedule_stage(schedule, stats_payload)
        if schedule_stage:
            params = self._merge_matching_params(params, schedule_stage)
        return params

    def _suggest_matching_params(
        self,
        metric: float,
        stage_cfg: MatchingStageConfig | None = None,
        stage_stats: Optional[dict] = None,
        aggregator_stage_id: Optional[str] = None,
    ) -> dict:
        cfg = self.config.matching_filter
        params = {
            "min_distance": float(cfg.min_distance),
            "min_points": int(cfg.min_points),
            "max_points": int(cfg.max_points),
        }

        if stage_stats is None and aggregator_stage_id is not None:
            stage_stats = self._get_stage_stats(aggregator_stage_id)

        stats_payload: dict[str, Any] = {"mean_reward": metric}
        if stage_stats:
            stats_payload["mean_reward"] = stage_stats.get("mean_reward", metric)
            recent = stage_stats.get("recent_rewards")
            if recent is not None:
                stats_payload["recent_rewards"] = list(recent)
            stats_payload["recent_count"] = stage_stats.get("recent_count", 0)

        if cfg.strategy == "adaptive" and cfg.schedule:
            params = self._apply_filter_schedule(params, cfg.schedule, stats_payload)

        params["min_distance"] = max(1.0, params["min_distance"])
        params["min_points"] = max(1, params["min_points"])
        params["max_points"] = max(1, params["max_points"])

        if stage_cfg is not None:
            params = self._merge_matching_params(params, stage_cfg.filter_overrides)
            stage_specific_stats = stats_payload
            params = self._apply_filter_schedule(params, stage_cfg.filter_schedule, stage_specific_stats)

        return params

    def _suggest_grounding_params(self, metric: float) -> dict:
        cfg = self.config.grounding_points
        params = {
            "min_points": int(cfg.min_points),
            "max_points": int(cfg.max_points),
        }
        if cfg.strategy == "adaptive" and cfg.schedule:
            schedule_stage = self._pick_schedule_stage(cfg.schedule, {"mean_reward": metric})
            if schedule_stage:
                params["min_points"] = int(schedule_stage.get("min_points", params["min_points"]))
                params["max_points"] = int(schedule_stage.get("max_points", params["max_points"]))

        params["min_points"] = max(1, params["min_points"])
        params["max_points"] = max(params["min_points"], params["max_points"])
        return params

    def _get_current_stage_config(self) -> Optional[MatchingStageConfig]:
        if self.matching_curriculum_manager:
            return self.matching_curriculum_manager.get_current_stage_config()
        return None

    def _select_current_stage_variant(self) -> tuple[Optional[MatchingStageConfig], Optional[str], Optional[str]]:
        """Sample the active stage variant (aggregated stage id + variant id)."""

        if not self.matching_curriculum_manager:
            return None, self.current_matching_stage_id, None

        variant = self.matching_curriculum_manager.sample_current_variant()
        stage_id = self.matching_curriculum_manager.current_stage_id

        if variant is None:
            fallback = self.matching_curriculum_manager.get_current_stage_config()
            return fallback, stage_id, None

        return variant.config, stage_id, variant.name

    def _get_history_stats(self, history: Optional[deque]) -> dict:
        if not history:
            return {"mean_reward": 0.0, "recent_count": 0, "recent_rewards": []}
        rewards = list(history)
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "recent_count": len(rewards),
            "recent_rewards": rewards,
        }

    def _get_stage_stats(self, stage_id: str | None) -> dict:
        if stage_id is None:
            return {"mean_reward": 0.0, "recent_count": 0, "recent_rewards": []}
        return self._get_history_stats(self.stage_reward_history.get(stage_id))

    def _get_variant_stats(self, variant_id: str | None) -> dict:
        if variant_id is None:
            return {"mean_reward": 0.0, "recent_count": 0, "recent_rewards": []}
        history = self.stage_variant_reward_history.get(variant_id)
        if history:
            return self._get_history_stats(history)
        parent_id = (
            self.matching_curriculum_manager.get_parent_stage_id(variant_id)
            if self.matching_curriculum_manager
            else None
        )
        return self._get_stage_stats(parent_id or variant_id)

    def _get_stage_metric_value(self, stage_cfg: MatchingStageConfig | None, fallback: float = 0.0) -> float:
        if stage_cfg is None:
            return fallback
        stats = self._get_variant_stats(stage_cfg.stage_id)
        if stats["recent_count"] > 0:
            return stats.get("mean_reward", fallback)
        return fallback

    def _log_event(self, event: dict) -> None:
        if self._buffer_logger is not None:
            self._buffer_logger.info(json.dumps(event))

    def _log_buffer_snapshot(self, recent_samples: list[dict]) -> None:
        if self._buffer_logger is None:
            return

        metrics = self.get_task_metrics()
        step_stats: dict[str, dict[str, float]] = {}
        for sample in recent_samples:
            task = sample.get("task", "unknown")
            stat = step_stats.setdefault(task, {"count": 0, "mean_reward": 0.0})
            count = stat["count"] + 1
            stat["mean_reward"] = ((stat["mean_reward"] * stat["count"]) + float(sample.get("reward", 0.0))) / count
            stat["count"] = count

        entry = {
            "event": "add_samples",
            "step_samples": len(recent_samples),
            "step_stats": step_stats,
            "task": self.current_task,
            "buffer_size": metrics.get("buffer_size", len(self.buffer)),
            "matching_count": metrics.get("matching", {}).get("count", 0),
            "grounding_count": metrics.get("grounding", {}).get("count", 0),
            "matching_mean_reward": metrics.get("matching", {}).get("mean_reward", 0.0),
            "grounding_mean_reward": metrics.get("grounding", {}).get("mean_reward", 0.0),
            "curriculum_stage": self.current_matching_stage_id,
            "curriculum_ready": int(self.matching_curriculum_manager.is_complete())
            if self.matching_curriculum_manager
            else 1,
            "matching_params": self.current_matching_params,
            "grounding_params": self.current_grounding_params,
        }
        stage_metrics = metrics.get("matching_stages")
        if stage_metrics:
            summarized_stage_metrics = {}
            for stage_id, stats in stage_metrics.items():
                summarized_stage_metrics[stage_id] = {
                    "mean_reward": stats.get("mean_reward", 0.0),
                    "recent_count": stats.get("recent_count", 0),
                    "total_count": stats.get("count", 0),
                }
            entry["stage_metrics"] = summarized_stage_metrics
        self._buffer_logger.info(json.dumps(entry))

    def decide_task(self) -> Literal["matching", "grounding"]:
        """Return the task selected by the adaptive controller."""
        if self.config.task_mode in ("matching", "grounding"):
            self.current_task = self.config.task_mode
        return self.current_task

    def _is_adaptation_ready(self) -> bool:
        """Check whether the buffer has accumulated enough samples to start adapting."""
        if self.config.task_mode != "dynamic":
            return False
        if self.config.buffer_size <= 0:
            return len(self.buffer) > 0
        threshold = max(0.0, min(1.0, self.config.warmup_proportion))
        current_ratio = len(self.buffer) / self.config.buffer_size
        return current_ratio >= threshold

    def _update_adaptation_state(self):
        """Update adaptive curriculum state when conditions are satisfied."""
        if not self.config.enable or self.config.task_mode != "dynamic":
            return

        self._adaptation_ready = self._is_adaptation_ready()
        if not self._adaptation_ready:
            return

        assessment = self._assess_buffer_state()
        self._apply_curriculum_update(assessment)

    def _assess_buffer_state(self) -> dict:
        """Summarize buffer statistics for curriculum decisions (demo implementation)."""
        metrics = self.get_task_metrics()
        return {"task_metrics": metrics}

    def _apply_curriculum_update(self, assessment: dict):
        """Demo routine to adjust task selection and sampling difficulty.

        Replace or extend this method with custom curriculum logic.
        """
        task_metrics = assessment.get("task_metrics", {})
        if not task_metrics:
            return

        matching_stats = task_metrics.get("matching", {})
        grounding_stats = task_metrics.get("grounding", {})

        matching_metric = float(matching_stats.get("mean_reward", 0.0))
        grounding_metric = float(grounding_stats.get("mean_reward", 0.0))
        matching_count = int(matching_stats.get("count", 0))
        grounding_count = int(grounding_stats.get("count", 0))

        # Demo task switching logic mirroring earlier behavior.
        curriculum_ready = True
        if self.matching_curriculum_manager is not None:
            curriculum_ready = self.matching_curriculum_manager.is_complete()

        if self.current_task == "matching":
            if curriculum_ready and (
                matching_count >= self.config.min_samples_for_switch
                and matching_metric > self.config.matching_threshold
            ):
                self.current_task = "grounding"
                self.task_history.append(("matching", "grounding", matching_metric))
                print(f"[DynamicBuffer] Switching to GROUNDING (matching reward: {matching_metric:.3f})")
        else:
            if (
                grounding_count >= self.config.min_samples_for_switch
                and grounding_metric < self.config.grounding_threshold
            ):
                self.current_task = "matching"
                self.task_history.append(("grounding", "matching", grounding_metric))
                print(f"[DynamicBuffer] Switching to MATCHING (grounding reward: {grounding_metric:.3f})")

        stage_metrics = task_metrics.get("matching_stages", {})

        transition = None
        if self.matching_curriculum_manager is not None:
            transition = self.matching_curriculum_manager.maybe_update(stage_metrics)
            if transition:
                self.current_matching_stage_id = transition["to"]
                metric_val = float(transition.get("metric", {}).get("mean_reward", matching_metric))
                print(
                    f"[DynamicBuffer] Matching curriculum {transition['transition']} -> {self.current_matching_stage_id}"
                    f" (mean reward: {metric_val:.3f})"
                )
                self._log_event(
                    {
                        "event": "stage_transition",
                        "transition": transition,
                        "matching_params": self.current_matching_params,
                        "grounding_params": self.current_grounding_params,
                    }
                )
        else:
            self.current_matching_stage_id = None

        stage_cfg = self._get_current_stage_config()
        aggregator_stage_id = self.current_matching_stage_id
        stage_stats = None
        stage_metric_value = matching_metric
        if aggregator_stage_id:
            stage_stats = self._get_stage_stats(aggregator_stage_id)
            stage_metric_value = float(stage_stats.get("mean_reward", matching_metric))

        previous_matching_params = dict(self.current_matching_params)

        # Difficulty hooks for the active curriculum stage.
        self.current_matching_params = self._suggest_matching_params(
            stage_metric_value,
            stage_cfg=stage_cfg,
            stage_stats=stage_stats,
            aggregator_stage_id=aggregator_stage_id,
        )
        self.current_grounding_params = self._suggest_grounding_params(grounding_metric)

        if self._buffer_logger is not None and previous_matching_params != self.current_matching_params:
            self._log_event(
                {
                    "event": "matching_params_update",
                    "previous": previous_matching_params,
                    "current": self.current_matching_params,
                    "curriculum_stage": self.current_matching_stage_id,
                    "matching_metric": matching_metric,
                }
            )

    @staticmethod
    def _parse_db_idx(sample_meta) -> Optional[int]:
        """Extract integer db_idx from serialized sample_meta."""
        value = sample_meta
        if isinstance(value, np.ndarray):
            value = value.item()
        elif hasattr(value, "item") and not isinstance(value, (dict, str, bytes)):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if isinstance(value, dict):
            value = value.get("db_idx")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # def _extract_db_indices(self, batch_dict: dict) -> Optional[list[int]]:
    #     """Extract LMDB indices for each sample in the batch."""
    #     metas = batch_dict.get("sample_meta")
    #     indices: list[int] = []

    #     if metas is not None:
    #         for meta in metas:
    #             db_idx = self._parse_db_idx(meta)
    #             if db_idx is not None:
    #                 indices.append(db_idx)
    #     else:
    #         raw_ids = batch_dict.get("db_idx")
    #         if raw_ids is None:
    #             return None
    #         for value in raw_ids:
    #             db_idx = self._parse_db_idx(value)
    #             if db_idx is not None:
    #                 indices.append(db_idx)

    #     return indices or None

    # ========================================================================
    # Task-Specific Processing Methods (matching & grounding)
    # ========================================================================

    @staticmethod
    def _match_signature(match: dict) -> tuple:
        return (match.get("x1"), match.get("y1"), match.get("x2"), match.get("y2"))

    @staticmethod
    def _convert_match_to_absolute(match: dict, width: int, height: int) -> dict:
        """Return a copy of match with absolute pixel coordinates."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for coordinate conversion: width={width}, height={height}")

        def _to_abs(value: float | int, limit: int) -> int:
            upper = max(limit, 0)
            return int(round(float(np.clip(value, 0, upper))))

        if match.get("is_relative", False):
            abs_match = dict(match)
            abs_match["x1"] = _to_abs(float(match["x1"]) * width / 1000.0, width - 1)
            abs_match["y1"] = _to_abs(float(match["y1"]) * height / 1000.0, height - 1)
            abs_match["x2"] = _to_abs(float(match["x2"]) * width / 1000.0, width - 1)
            abs_match["y2"] = _to_abs(float(match["y2"]) * height / 1000.0, height - 1)
            abs_match["is_relative"] = False
            return abs_match

        abs_match = dict(match)
        abs_match["x1"] = _to_abs(match["x1"], width - 1)
        abs_match["y1"] = _to_abs(match["y1"], height - 1)
        abs_match["x2"] = _to_abs(match["x2"], width - 1)
        abs_match["y2"] = _to_abs(match["y2"], height - 1)
        abs_match["is_relative"] = False
        return abs_match

    def _normalize_matches_to_absolute(self, raw_sample: dict) -> list[dict]:
        """Ensure matches inside raw_sample are absolute pixel coordinates."""
        matches = raw_sample.get("matches", [])
        if not matches:
            return []

        if raw_sample.get("_matches_absolute", False):
            return matches

        width = int(raw_sample.get("width") or raw_sample["image1"].size[0])
        height = int(raw_sample.get("height") or raw_sample["image2"].size[1])
        normalized = [self._convert_match_to_absolute(match, width, height) for match in matches]

        raw_sample["matches"] = normalized
        raw_sample["width"] = width
        raw_sample["height"] = height
        raw_sample["_matches_absolute"] = True
        return normalized

    def _sample_random_raw(self, difficulty: Optional[float] = None) -> Optional[dict]:
        """Draw a replacement raw sample from the underlying dataset."""
        dataset = self.raw_dataset

        if difficulty is not None and hasattr(dataset, "sample_by_difficulty"):
            try:
                sample = dataset.sample_by_difficulty(difficulty)
                if sample is not None:
                    return sample
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Difficulty-aware sampling failed (%s); falling back to uniform sampling.", exc)

        try:
            total = len(dataset)
        except Exception:
            return None

        if total <= 0:
            return None

        idx = random.randint(0, max(total - 1, 0))
        return dataset[idx]

    def _sample_neighbor_raw(self, reference_sample: Optional[dict], window: int) -> Optional[dict]:
        if not reference_sample:
            return None

        dataset = self.raw_dataset
        db_idx = reference_sample.get("db_idx")
        source_idx = reference_sample.get("source_dataset")

        if hasattr(dataset, "sample_nearby"):
            try:
                sample = dataset.sample_nearby(db_idx, source_dataset=source_idx, window=window)
                if sample is not None:
                    return sample
            except TypeError:
                try:
                    sample = dataset.sample_nearby(db_idx, window=window)  # type: ignore[arg-type]
                    if sample is not None:
                        if source_idx is not None:
                            sample.setdefault("source_dataset", source_idx)
                        return sample
                except Exception:
                    pass
            except Exception:
                pass

        if AnnoRawDataset is not None and isinstance(dataset, AnnoRawDataset):
            try:
                return dataset.sample_nearby(db_idx, window=window)
            except Exception:
                return None

        return None

    @staticmethod
    @staticmethod
    def _rects_overlap(rect1: tuple[float, float, float, float], rect2: tuple[float, float, float, float]) -> bool:
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

    def _match_center(
        self,
        match: dict,
        view_key: Literal["image1", "image2"],
        width: int,
        height: int,
    ) -> tuple[float, float]:
        x_key = "x1" if view_key == "image1" else "x2"
        y_key = "y1" if view_key == "image1" else "y2"
        x = match.get(x_key, 0.0)
        y = match.get(y_key, 0.0)
        if match.get("is_relative", False):
            x = float(x) * width / 1000.0
            y = float(y) * height / 1000.0
        return float(x), float(y)

    def _apply_point_filter(
        self,
        candidates: list[dict],
        max_points: int,
        min_distance: float,
        policy: PointFilteringStrategy | None,
        mode: str,
    ) -> list[dict]:
        ordered = candidates[:]
        if mode == "random":
            random.shuffle(ordered)
            return ordered

        if mode == "dense":
            return ordered

        if mode == "greedy":
            primary = greedy_max_spacing_selection(ordered, min(len(ordered), max_points), min_distance)
            seen = {self._match_signature(m) for m in primary}
            primary.extend([m for m in ordered if self._match_signature(m) not in seen])
            return primary

        shuffled = ordered[:]
        random.shuffle(shuffled)
        eps_scale = policy.cluster_eps_scale if policy else 0.8
        greedy_spacing = policy.greedy_spacing if policy else True
        primary = filter_sparse_points(
            shuffled,
            min_distance=min_distance,
            max_points=len(shuffled),
            eps_scale=eps_scale,
            use_greedy=greedy_spacing,
        )
        seen = {self._match_signature(m) for m in primary}
        primary.extend([m for m in shuffled if self._match_signature(m) not in seen])
        return primary

    @staticmethod
    def _annotation_style() -> dict:
        return dict(_DEFAULT_ANNOTATE_STYLE)

    def _measure_text(self, text: str) -> tuple[int, int, int]:
        style = self._annotation_style()
        font = style.get("font")
        font_scale = float(style.get("font_scale", 0.5))
        font_thickness = int(style.get("font_thickness", 1))
        if cv2 is not None and font is not None:
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        else:
            avg_char_w = 8.0 * font_scale
            text_w = int(max(1.0, len(text) * avg_char_w))
            text_h = int(max(8.0, 14.0 * font_scale))
            baseline = int(4.0 * font_scale)
        return max(1, text_w), max(1, text_h), max(0, baseline)

    def _annotation_bbox(
        self,
        center: tuple[float, float],
        width: int,
        height: int,
        label: str = "999",
    ) -> tuple[float, float, float, float]:
        style = self._annotation_style()
        radius = float(style.get("radius", 8))
        padding = float(style.get("label_padding", 2))
        x, y = center

        circle_left = max(0.0, x - radius)
        circle_right = min(float(width - 1), x + radius)
        circle_top = max(0.0, y - radius)
        circle_bottom = min(float(height - 1), y + radius)

        text_w, text_h, baseline = self._measure_text(label)
        text_x = float(np.clip(x - text_w / 2.0, padding, max(padding, width - text_w - padding)))
        preferred_baseline = y - radius - padding
        if preferred_baseline - text_h - padding >= 0:
            text_y = preferred_baseline
        else:
            text_y = y + radius + padding + text_h
        text_y = float(np.clip(text_y, text_h + padding, height - padding))

        rect_top = max(0.0, text_y - text_h - padding)
        rect_bottom = min(float(height - 1), text_y + baseline + padding)
        rect_left = max(0.0, text_x - padding)
        rect_right = min(float(width - 1), text_x + text_w + padding)

        x_min = min(circle_left, rect_left)
        y_min = min(circle_top, rect_top)
        x_max = max(circle_right, rect_right)
        y_max = max(circle_bottom, rect_bottom)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _match_view_point(match: dict, view_key: Literal["image1", "image2"]) -> dict:
        x_key = "x1" if view_key == "image1" else "x2"
        y_key = "y1" if view_key == "image1" else "y2"
        return {
            "x": float(match.get(x_key, 0.0)),
            "y": float(match.get(y_key, 0.0)),
            "is_relative": match.get("is_relative", False),
        }

    @staticmethod
    def _get_view_dimensions(raw_sample: dict) -> dict[str, tuple[int, int]]:
        image1 = raw_sample.get("image1")
        image2 = raw_sample.get("image2")

        width1 = int(raw_sample.get("width") or (image1.size[0] if image1 is not None else 0))
        height1 = int((image1.size[1] if image1 is not None else raw_sample.get("height", 0)))

        if image2 is not None:
            width2 = int(image2.size[0])
            height2 = int(image2.size[1])
        else:
            width2 = width1
            height2 = height1

        width1 = max(width1, 1)
        height1 = max(height1, 1)
        width2 = max(width2, 1)
        height2 = max(height2, 1)

        return {
            "image1": (width1, height1),
            "image2": (width2, height2),
        }

    def _match_centers(
        self,
        match: dict,
        view_sizes: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[float, float]]:
        centers: dict[str, tuple[float, float]] = {}
        for view in ("image1", "image2"):
            width, height = view_sizes[view]
            centers[view] = self._match_center(match, view, width, height)
        return centers

    @staticmethod
    def _point_center(
        point: dict,
        view_key: Literal["image1", "image2"],
        view_sizes: dict[str, tuple[int, int]],
    ) -> tuple[float, float]:
        width, height = view_sizes[view_key]
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        if point.get("is_relative", False):
            x = x * width / 1000.0
            y = y * height / 1000.0
        return x, y

    def _centers_ok(
        self,
        cache: dict[str, list[tuple[float, float]]],
        centers: dict[str, tuple[float, float]],
        threshold: float,
    ) -> bool:
        guard = max(float(threshold), self._annotation_guard_px)
        for view in ("image1", "image2"):
            cx, cy = centers[view]
            for px, py in cache[view]:
                if math.hypot(cx - px, cy - py) < guard:
                    return False
        return True

    def _select_spaced_subset(
        self,
        ordered: list[dict],
        max_points: int,
        min_points: int,
        min_distance: float,
        view_sizes: dict[str, tuple[int, int]],
    ) -> list[dict]:
        if not ordered:
            return []

        threshold = max(float(min_distance), self._annotation_guard_px)
        cache: dict[str, list[tuple[float, float]]] = {"image1": [], "image2": []}
        selected: list[dict] = []
        for match in ordered:
            if len(selected) >= max_points:
                break
            centers = self._match_centers(match, view_sizes)
            if not self._centers_ok(cache, centers, threshold):
                continue
            selected.append(match)
            for view in ("image1", "image2"):
                cache[view].append(centers[view])
        if len(selected) < min_points:
            return []
        return selected

    def _collect_safe_matches(self, raw_sample: dict, matches: list[dict]) -> list[dict]:
        if not matches:
            return []
        view_sizes = self._get_view_dimensions(raw_sample)
        guard = self._annotation_guard_px
        policy = PointFilteringStrategy(mode="cluster", cluster_eps_scale=1.0, greedy_spacing=True)
        ordered = self._apply_point_filter(matches, len(matches), guard, policy, "cluster")
        safe = self._select_spaced_subset(ordered, len(matches), 1, guard, view_sizes)
        return safe or []

    def _can_place_points(
        self,
        geometry_cache: dict[str, list[tuple[float, float]]],
        view_key: Literal["image1", "image2"],
        points: list[dict],
        min_distance: float,
        view_sizes: dict[str, tuple[int, int]],
    ) -> bool:
        if not points:
            return True
        guard = max(float(min_distance), self._annotation_guard_px)
        existing = list(geometry_cache[view_key])
        for point in points:
            center = self._point_center(point, view_key, view_sizes)
            for px, py in existing:
                if math.hypot(center[0] - px, center[1] - py) < guard:
                    return False
            existing.append(center)
        geometry_cache[view_key] = existing
        return True


    def _select_core_matches(
        self,
        raw_sample: dict,
        matches: list[dict],
        params: dict,
        stage_cfg: MatchingStageConfig | None,
    ) -> tuple[list[dict], list[dict]]:
        db_idx = raw_sample.get("db_idx")
        stage_id = stage_cfg.stage_id if stage_cfg else None

        if not matches:
            raise InsufficientMatchesError(stage_id, 1, 0, db_idx)

        matches = copy.deepcopy(matches)
        max_points = max(1, int(params.get("max_points", len(matches))))
        min_points = max(1, int(params.get("min_points", 1)))
        max_points = max(max_points, min_points)

        if len(matches) < min_points:
            raise InsufficientMatchesError(stage_id, min_points, len(matches), db_idx)

        requested_min_distance = float(params.get("min_distance", 10.0))

        policy = stage_cfg.point_filter if stage_cfg and stage_cfg.point_filter else None
        mode = policy.mode if policy else "cluster"

        view_sizes = self._get_view_dimensions(raw_sample)
        spacing_distance = max(requested_min_distance, self._annotation_guard_px)

        filtered: list[dict] = []
        attempts = 0
        while attempts < MAX_SPACING_ATTEMPTS:
            ordered = self._apply_point_filter(matches, max_points, spacing_distance, policy, mode)
            filtered = self._select_spaced_subset(
                ordered,
                max_points,
                min_points,
                spacing_distance,
                view_sizes,
            )
            if filtered:
                break
            attempts += 1
        else:
            raise InsufficientMatchesError(stage_id, min_points, len(filtered), db_idx)

        selected_signatures = {self._match_signature(m) for m in filtered}
        remaining = [m for m in matches if self._match_signature(m) not in selected_signatures]
        return filtered, remaining

    @staticmethod
    def _draw_extra_matches(pool: list[dict], min_extra: int, max_extra: int) -> list[dict]:
        if not pool:
            return []
        min_extra = max(0, int(min_extra))
        max_extra = max(min_extra, int(max_extra))
        if max_extra == 0:
            return []
        amount = random.randint(min_extra, max_extra)
        amount = min(amount, len(pool))
        if amount <= 0:
            return []
        selected = random.sample(pool, amount)
        for item in selected:
            try:
                pool.remove(item)
            except ValueError:
                pass
        return selected

    def _generate_synthetic_points(
        self,
        reference_points: list[dict],
        min_extra: int,
        max_extra: int,
        radius: float,
        width: int,
        height: int,
        guard: float,
    ) -> list[dict]:
        min_extra = max(0, int(min_extra))
        max_extra = max(min_extra, int(max_extra))
        if max_extra <= 0:
            return []

        count = random.randint(min_extra, max_extra)
        if count <= 0:
            return []

        max_radius = max(guard + radius, guard * 1.5)
        results: list[dict] = []
        for _ in range(count):
            if reference_points:
                base = random.choice(reference_points)
                base_x = float(base.get("x", random.uniform(0, max(width - 1, 0))))
                base_y = float(base.get("y", random.uniform(0, max(height - 1, 0))))
            else:
                base_x = random.uniform(0, max(width - 1, 0))
                base_y = random.uniform(0, max(height - 1, 0))

            sampled = False
            for _ in range(16):
                angle = random.uniform(0.0, 2.0 * math.pi)
                distance = random.uniform(guard, max_radius)
                cand_x = base_x + math.cos(angle) * distance
                cand_y = base_y + math.sin(angle) * distance
                if 0 <= cand_x < width and 0 <= cand_y < height:
                    results.append({"x": float(cand_x), "y": float(cand_y), "is_relative": False})
                    sampled = True
                    break
            if not sampled:
                results.append(
                    {
                        "x": float(random.uniform(0, max(width - 1, 0))),
                        "y": float(random.uniform(0, max(height - 1, 0))),
                        "is_relative": False,
                    }
                )

        return results

    def _sample_stage_distractors(
        self,
        raw_sample: dict,
        core_matches: list[dict],
        remaining_matches: list[dict],
        stage_cfg: MatchingStageConfig | None,
        min_distance: float,
    ) -> tuple[list[dict], list[dict]]:
        if stage_cfg is None or not stage_cfg.distractors.enable:
            return [], []

        cfg = stage_cfg.distractors
        mode = getattr(cfg, "mode", "sample")
        view_sizes = self._get_view_dimensions(raw_sample)
        width_a, height_a = view_sizes["image1"]
        width_b, height_b = view_sizes["image2"]
        guard = max(self._annotation_guard_px, float(min_distance))

        geometry_cache: dict[str, list[tuple[float, float]]] = {"image1": [], "image2": []}
        for match in core_matches:
            centers = self._match_centers(match, view_sizes)
            for view in ("image1", "image2"):
                geometry_cache[view].append(centers[view])

        reference = core_matches + remaining_matches
        reference_points = {
            "image1": [self._match_view_point(match, "image1") for match in reference],
            "image2": [self._match_view_point(match, "image2") for match in reference],
        }

        def _sample_for_view(
            view_key: Literal["image1", "image2"],
            min_extra: int,
            max_extra: int,
        ) -> list[dict]:
            if max_extra <= 0:
                return []
            width, height = view_sizes[view_key]
            for _ in range(MAX_SPACING_ATTEMPTS):
                if mode == "synthetic":
                    radius = float(getattr(cfg, "synthetic_radius", 80.0))
                    candidates = self._generate_synthetic_points(
                        reference_points[view_key],
                        min_extra,
                        max_extra,
                        radius,
                        width,
                        height,
                        guard,
                    )
                else:
                    pool = list(remaining_matches)
                    matches = self._draw_extra_matches(pool, min_extra, max_extra)
                    candidates = [self._match_view_point(match, view_key) for match in matches]

                if not candidates:
                    return []
                if self._can_place_points(geometry_cache, view_key, candidates, guard, view_sizes):
                    return candidates
            raise InsufficientMatchesError(stage_cfg.stage_id if stage_cfg else None, min_extra, 0, raw_sample.get("db_idx"))

        extras_a = _sample_for_view("image1", cfg.min_extra_image1, cfg.max_extra_image1)
        extras_b = _sample_for_view("image2", cfg.min_extra_image2, cfg.max_extra_image2)
        return extras_a, extras_b

    def _build_label_assignment(
        self,
        core_matches: list[dict],
        distractors_a: list[dict],
        distractors_b: list[dict],
        stage_cfg: MatchingStageConfig | None,
    ) -> dict:
        assignment = {
            "stage_id": stage_cfg.stage_id if stage_cfg else None,
            "core": [],
            "distractors_a": [],
            "distractors_b": [],
        }

        num_core = len(core_matches)
        total_a = num_core + len(distractors_a)
        total_b = num_core + len(distractors_b)

        labels_pool_a = [str(i + 1) for i in range(total_a)]
        labels_pool_b = [str(i + 1) for i in range(total_b)]

        core_labels_a = random.sample(labels_pool_a, k=num_core) if num_core else []
        core_labels_b = random.sample(labels_pool_b, k=num_core) if num_core else []

        remaining_labels_a = [lbl for lbl in labels_pool_a if lbl not in core_labels_a]
        remaining_labels_b = [lbl for lbl in labels_pool_b if lbl not in core_labels_b]

        random.shuffle(core_labels_a)
        random.shuffle(core_labels_b)
        random.shuffle(remaining_labels_a)
        random.shuffle(remaining_labels_b)

        shuffle_b = stage_cfg.shuffle_image2_labels if stage_cfg else True
        if shuffle_b and len(core_labels_b) > 1:
            random.shuffle(core_labels_b)

        for idx, match in enumerate(core_matches):
            label_a = core_labels_a[idx] if idx < len(core_labels_a) else str(idx + 1)
            label_b = core_labels_b[idx] if idx < len(core_labels_b) else str(idx + 1)
            assignment["core"].append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "match": match,
                }
            )

        for label, point in zip(remaining_labels_a, distractors_a, strict=True):
            assignment["distractors_a"].append(
                {
                    "label": label,
                    "point": point,
                }
            )

        for label, point in zip(remaining_labels_b, distractors_b, strict=True):
            assignment["distractors_b"].append(
                {
                    "label": label,
                    "point": point,
                }
            )

        return assignment

    @staticmethod
    def _friendly_view_name(view_key: str) -> str:
        return "Image A" if view_key == "image1" else "Image B"

    @staticmethod
    def _should_use_relative_coordinates(raw_sample: dict) -> bool:
        processor = raw_sample.get("processor", None)
        if processor is None:
            return True
        return "Qwen3VLProcessor" in processor.__class__.__name__

    @staticmethod
    def _format_point_coords(
        point: dict,
        width: int,
        height: int,
        scale_mode: Literal["relative", "absolute"],
        precision: int,
    ) -> list[int | float]:
        coords = point["point_2d"]
        is_relative = point.get("is_relative", False)

        if scale_mode == "relative":
            if is_relative:
                x, y = coords
            else:
                x = float(coords[0]) * 1000.0 / max(width, 1)
                y = float(coords[1]) * 1000.0 / max(height, 1)
        else:  # absolute
            if is_relative:
                x = float(coords[0]) * max(width, 1) / 1000.0
                y = float(coords[1]) * max(height, 1) / 1000.0
            else:
                x, y = coords

        if precision <= 0:
            return [int(round(x)), int(round(y))]
        return [round(float(x), precision), round(float(y), precision)]

    def _format_points_section(
        self,
        view_key: str,
        points: list[dict],
        raw_sample: dict,
        rep_cfg: RepresentationConfig,
    ) -> str:
        if not points:
            return ""

        width = int(raw_sample.get("width", 1))
        height = int(raw_sample.get("height", 1))

        default_relative = self._should_use_relative_coordinates(raw_sample)
        scale_mode = rep_cfg.coordinate_scale
        if scale_mode == "auto":
            scale_mode = "relative" if default_relative else "absolute"

        entries = []
        for point in points:
            coords = self._format_point_coords(point, width, height, scale_mode, rep_cfg.text_precision)
            entries.append({"label": point["label"], "point_2d": coords})

        header = rep_cfg.text_header or f"{self._friendly_view_name(view_key)} coordinates"
        if scale_mode == "relative":
            header = f"{header} (0-1000 scale)"
        else:
            header = f"{header} (pixels)"

        if rep_cfg.text_format == "json":
            body = json.dumps(entries, indent=2)
            return f"{header}\n```json\n{body}\n```"

        lines = [header]
        for item in entries:
            x, y = item["point_2d"]
            lines.append(f"- ID {item['label']}: ({x}, {y})")
        return "\n".join(lines)

    def _extract_points_from_assignment(self, assignment: dict, view_key: Literal["image1", "image2"]) -> list[dict]:
        points: list[dict] = []
        x_key = "x1" if view_key == "image1" else "x2"
        y_key = "y1" if view_key == "image1" else "y2"

        for core in assignment["core"]:
            match = core["match"]
            label = core["label_a"] if view_key == "image1" else core["label_b"]
            paired = core["label_b"] if view_key == "image1" else core["label_a"]
            points.append(
                {
                    "label": label,
                    "point_2d": (match[x_key], match[y_key]),
                    "is_relative": match.get("is_relative", False),
                    "source": "core",
                    "paired_label": paired,
                }
            )

        distractor_key = "distractors_a" if view_key == "image1" else "distractors_b"
        for entry in assignment[distractor_key]:
            point = entry["point"]
            is_relative = point.get("is_relative", False)
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            if not is_relative:
                x = float(int(round(x)))
                y = float(int(round(y)))
            points.append(
                {
                    "label": entry["label"],
                    "point_2d": (x, y),
                    "is_relative": is_relative,
                    "source": "distractor",
                    "paired_label": None,
                }
            )

        return points

    def _select_points_subset(
        self,
        points: list[dict],
        ratio: float,
        min_points: int,
        ensure_core: bool,
        exclude_labels: Optional[set[str]] = None,
    ) -> list[dict]:
        if not points:
            return []

        if exclude_labels:
            candidates = [p for p in points if p["label"] not in exclude_labels]
        else:
            candidates = list(points)

        if not candidates:
            return []

        ratio = max(0.0, float(ratio))
        min_points = max(0, int(min_points))
        total = len(candidates)
        target = total if ratio >= 0.999 else min(total, max(min_points, int(math.ceil(total * ratio))))

        random.shuffle(candidates)

        selected: list[dict] = []
        selected_labels: set[str] = set()

        def can_add(candidate: dict) -> bool:
            return candidate["label"] not in selected_labels

        if ensure_core:
            core_candidates = [c for c in candidates if c.get("source") == "core"]
            core_candidates.sort(key=lambda item: item["label"])
        else:
            core_candidates = []

        for candidate in core_candidates:
            if can_add(candidate):
                selected.append(candidate)
                selected_labels.add(candidate["label"])

        for candidate in candidates:
            if len(selected) >= target:
                break
            if candidate["label"] in selected_labels:
                continue
            if can_add(candidate):
                selected.append(candidate)
                selected_labels.add(candidate["label"])

        if ensure_core and core_candidates:
            for candidate in core_candidates:
                if len(selected) >= target:
                    break
                if candidate["label"] in selected_labels:
                    continue
                if can_add(candidate):
                    selected.append(candidate)
                    selected_labels.add(candidate["label"])

        return selected

    def _prepare_view_payload(
        self,
        raw_sample: dict,
        assignment: dict,
        stage_cfg: MatchingStageConfig | None,
        view_key: Literal["image1", "image2"],
    ) -> dict:
        rep_cfg = getattr(stage_cfg.representation, view_key) if stage_cfg is not None else RepresentationConfig()

        points = self._extract_points_from_assignment(assignment, view_key)
        image_points: list[dict] = []
        text_points: list[dict] = []
        annotated_image = None
        text_snippet = ""

        if rep_cfg.render_image and points:
            image_points = self._select_points_subset(
                points,
                rep_cfg.annotate_ratio,
                rep_cfg.min_annotated_points,
                rep_cfg.ensure_core_cover,
            )
            annotate_payload = [
                {"label": p["label"], "point_2d": p["point_2d"], "is_relative": p.get("is_relative", False)}
                for p in image_points
            ]
            src_image = raw_sample["image1"] if view_key == "image1" else raw_sample["image2"]
            annotated_image = annotate_image(src_image, annotate_payload)

        if rep_cfg.include_text and points:
            existing = {p["label"] for p in image_points}
            text_points = self._select_points_subset(
                points,
                rep_cfg.text_ratio,
                rep_cfg.min_text_points,
                rep_cfg.ensure_core_cover,
                exclude_labels=existing,
            )
            text_snippet = self._format_points_section(view_key, text_points, raw_sample, rep_cfg)

        return {
            "view": view_key,
            "image": annotated_image,
            "image_points": image_points,
            "text_points": text_points,
            "text_snippet": text_snippet,
        }

    def _legacy_process_matching(self, raw_sample: dict, filter_params: Optional[dict]) -> tuple[dict, dict, dict]:
        params = filter_params or self.current_matching_params
        matches = self._normalize_matches_to_absolute(raw_sample)
        min_required = max(1, int(params.get("min_points", 1)))
        if len(matches) < min_required:
            raise InsufficientMatchesError("legacy", min_required, len(matches), raw_sample.get("db_idx"))

        safe_min_distance = max(float(params.get("min_distance", 10.0)), self._annotation_guard_px)
        results = draw_img(
            raw_sample,
            filter_crowded_points=True,
            min_point_distance=safe_min_distance,
            max_points_per_image=params["max_points"],
        )

        if len(results["id_mapping"]) < min_required:
            raise InsufficientMatchesError("legacy", min_required, len(results["id_mapping"]), raw_sample.get("db_idx"))

        prompt = {
            "role": "user",
            "content": {
                "text": ANNO_MATCH_MULTI_TEMPLATE,
                "images": [results["annotated_image1"], results["annotated_image2"]],
            },
        }

        answer = results["id_mapping"]
        meta = {
            "stage_id": "legacy",
            "num_core_pairs": len(answer),
            "distractors": {"image1": 0, "image2": 0},
            "representation": {
                "image1": {"render_image": True, "include_text": False},
                "image2": {"render_image": True, "include_text": False},
            },
            "text_labels": {"image1": [], "image2": []},
            "visual_labels": {
                "image1": list(answer.keys()),
                "image2": list(answer.values()),
            },
        }

        return prompt, answer, meta

    def process_matching_task(self, raw_sample: dict, filter_params: Optional[dict] = None) -> tuple[dict, dict, dict]:
        stage_cfg, aggregator_stage_id, variant_name = self._select_current_stage_variant()

        if stage_cfg is None:
            return self._legacy_process_matching(raw_sample, filter_params)

        params = filter_params or self.current_matching_params
        params = self._merge_matching_params(params, stage_cfg.filter_overrides)

        if aggregator_stage_id and stage_cfg.stage_id != aggregator_stage_id:
            variant_stats = self._get_variant_stats(stage_cfg.stage_id)
            variant_payload = {
                "mean_reward": variant_stats.get("mean_reward", 0.0),
                "recent_count": variant_stats.get("recent_count", 0),
                "recent_rewards": variant_stats.get("recent_rewards", []),
            }
            params = self._apply_filter_schedule(params, stage_cfg.filter_schedule, variant_payload)

        matches = self._normalize_matches_to_absolute(raw_sample)
        safe_matches = self._collect_safe_matches(raw_sample, matches)

        min_required = max(1, int(params.get("min_points", 1)))
        available = len(safe_matches)
        if available < min_required:
            raise InsufficientMatchesError(
                stage_cfg.stage_id if stage_cfg else None,
                min_required,
                available,
                raw_sample.get("db_idx"),
            )

        core_matches, remaining_matches = self._select_core_matches(raw_sample, safe_matches, params, stage_cfg)
        effective_min_distance = max(float(params.get("min_distance", 10.0)), self._annotation_guard_px)

        distractors_a, distractors_b = self._sample_stage_distractors(
            raw_sample, core_matches, remaining_matches, stage_cfg, effective_min_distance
        )
        assignment = self._build_label_assignment(core_matches, distractors_a, distractors_b, stage_cfg)

        payload_a = self._prepare_view_payload(raw_sample, assignment, stage_cfg, "image1")
        payload_b = self._prepare_view_payload(raw_sample, assignment, stage_cfg, "image2")

        text_sections = [section for section in [payload_a["text_snippet"], payload_b["text_snippet"]] if section]
        assert len(text_sections) == 0, "Text Points are not supported!"

        # TODO @zhonghao: Check the text points here!
        # prompt_text_sections = [ANNO_MATCH_MULTI_TEMPLATE.strip()]

        # if stage_cfg.prompt_appendix:
        #     prompt_text_sections.append(stage_cfg.prompt_appendix.strip())
        # prompt_text_sections.extend(text_sections)
        # prompt_text = "\n\n".join(prompt_text_sections)

        prompt_text = ANNO_MATCH_MULTI_TEMPLATE.strip()

        images = []
        for payload in (payload_a, payload_b):
            if payload["image"] is not None:
                images.append(payload["image"])

        prompt = {
            "role": "user",
            "content": {
                "images": images,
                "text": prompt_text,
            },
        }

        id_mapping = {core["label_a"]: core["label_b"] for core in assignment["core"]}

        # NOTE @zhonghao: for distractors, do not record in id_mapping
        # NOTE @zhonghao: updated at 1108, modifying the task to look up points from image 1 to image 2
        #                 so distractors in image 1 should have a mapping label "none"

        for distractor in assignment["distractors_a"]:
            id_mapping[distractor["label"]] = "none"

        stage_meta = {
            "stage_id": aggregator_stage_id or stage_cfg.stage_id,
            "variant_stage_id": stage_cfg.stage_id,
            "num_core_pairs": len(assignment["core"]),
            "distractors": {
                "image1": len(assignment["distractors_a"]),
                "image2": len(assignment["distractors_b"]),
            },
            "representation": {
                "image1": asdict(stage_cfg.representation.image1),
                "image2": asdict(stage_cfg.representation.image2),
            },
            "text_labels": {
                "image1": [p["label"] for p in payload_a["text_points"]],
                "image2": [p["label"] for p in payload_b["text_points"]],
            },
            "visual_labels": {
                "image1": [p["label"] for p in payload_a["image_points"]],
                "image2": [p["label"] for p in payload_b["image_points"]],
            },
        }

        return prompt, id_mapping, stage_meta

    def process_grounding_task(
        self, raw_sample: dict, point_params: Optional[dict] = None
    ) -> tuple[dict, list[dict], dict]:
        """Process raw sample for grounding task.

        Extracted from DL3DVG.anno_grounding()

        Args:
            raw_sample: Raw sample from AnnoRawDataset
            point_params: Optional dict overriding point sampling difficulty

        Returns:
            tuple: (prompt_dict, answer_dict)
        """
        params = point_params or self.current_grounding_params
        matches = raw_sample.get("matches", None)
        assert matches, "No matches found in the sample."

        # Sample a subset of matches (curriculum logic)
        upper_bound = min(len(matches), params["max_points"])
        lower_bound = min(params["min_points"], upper_bound)

        if lower_bound == 0:
            lower_bound = 1
        num_points = random.randint(lower_bound, max(lower_bound, upper_bound))
        selected_matches = random.sample(matches, k=num_points)

        img1 = raw_sample["image1"]
        img2 = raw_sample["image2"]

        # Unwrap matches to use a general annotate function
        coords = [
            {"label": str(i + 1), "point_2d": (int(m["x1"]), int(m["y1"])), "is_relative": m.get("is_relative", False)}
            for i, m in enumerate(selected_matches)
        ]

        ref_img = annotate_image(img1, coords)

        answer = [
            {"label": str(i + 1), "point_2d": (int(m["x2"]), int(m["y2"])), "is_relative": m.get("is_relative", False)}
            for i, m in enumerate(selected_matches)
        ]

        prompt = {
            "role": "user",
            "content": {
                "text": ANNO_GROUND_TEMPLATE,
                "images": [ref_img, img2],
            },
        }

        meta = {
            "num_points": len(answer),
        }

        return prompt, answer, meta

    def make_messages(self, prompt: dict, raw_sample: dict) -> list[dict]:
        """Create message list with system prompt and user query.

        Args:
            prompt: User prompt dictionary
            raw_sample: Raw sample containing configuration

        Returns:
            List of message dictionaries
        """
        msgs = [self.sys_msg, prompt]

        # Get processor from raw sample
        # processor = raw_sample["processor"]
        min_pixels = raw_sample["min_pixels"]
        max_pixels = raw_sample["max_pixels"]

        prompt_messages = []
        for msg in msgs:
            if msg["role"] == "user":
                if msg["content"].get("images", None) is not None:
                    text = msg["content"].get("text", None)
                    content = []

                    for img in msg["content"]["images"]:
                        image = {"type": "image"}
                        image["image"] = img
                        image["min_pixels"] = min_pixels
                        image["max_pixels"] = max_pixels
                        content.append(image)

                    content.append({"type": "text", "text": text})
                    prompt_messages.append({"role": "user", "content": content})
                else:
                    raise ValueError("User message must contain an image.")

            elif msg["role"] in ["assistant", "system"]:
                prompt_messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]["text"]}],
                    }
                )
            else:
                raise ValueError(f"Unknown role {msg['role']}")

        return prompt_messages

    def make_openai_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages to OpenAI format.

        Args:
            messages: List of message dictionaries

        Returns:
            OpenAI-formatted messages
        """
        import base64
        from io import BytesIO

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

        openai_messages = []
        for msg in messages:
            if msg["role"] == "user":
                content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        content.append(item)
                    elif item["type"] == "image":
                        base64_image = encode_image(item["image"])
                        content.append(
                            {
                                "type": "image_url",
                                "min_pixels": item["min_pixels"],
                                "max_pixels": item["max_pixels"],
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            }
                        )
                openai_messages.append({"role": "user", "content": content})

            elif msg["role"] in ["assistant", "system"]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                raise ValueError(f"Unknown role {msg['role']}")

        return openai_messages

    def process_raw_sample(
        self,
        raw_sample: dict,
        task: str,
        matching_params: Optional[dict] = None,
        grounding_params: Optional[dict] = None,
    ) -> dict:
        """Process a raw sample into a task-specific training sample.

        Args:
            raw_sample: Raw sample from AnnoRawDataset
            task: Task type ("matching" or "grounding")

        Returns:
            Fully processed sample dict ready for training
        """
        self._normalize_matches_to_absolute(raw_sample)

        stage_info: dict[str, Any] = {}
        # Apply task-specific processing
        if task == "matching":
            params = matching_params or self.current_matching_params
            query, answer, stage_meta = self.process_matching_task(raw_sample, params)
            task_type = "ANNO_MATCH"
            stage_info["matching_stage_id"] = stage_meta.get("stage_id")
            variant_stage_id = stage_meta.get("variant_stage_id")
        else:  # grounding
            params = grounding_params or self.current_grounding_params
            query, answer, stage_meta = self.process_grounding_task(raw_sample, params)
            task_type = "ANNO_GROUND"
            stage_info["grounding_points"] = int(stage_meta.get("num_points", len(answer)))
            variant_stage_id = None

        # Build messages
        prompt_messages = self.make_messages(query, raw_sample)
        # openai_copy = self.make_openai_messages(copy.deepcopy(prompt_messages))

        # Build model inputs
        model_inputs = build_model_inputs(self.raw_dataset, prompt_messages)

        # Add metadata
        model_inputs["reward_model"] = {"ground_truth": json.dumps(answer)}
        # model_inputs["openai"] = openai_copy
        meta_payload = {
            "db_idx": f"{raw_sample['db_idx']:08d}",
            "task_type": task_type,
        }
        if stage_info:
            meta_payload.update(stage_info)
        if variant_stage_id:
            meta_payload["matching_stage_variant"] = variant_stage_id
        if "overlap" in raw_sample:
            try:
                overlap_value = float(raw_sample["overlap"])
            except (TypeError, ValueError):
                overlap_value = raw_sample["overlap"]
            meta_payload["overlap"] = overlap_value
        model_inputs["sample_meta"] = json.dumps(meta_payload)

        # Get processor class name
        # processor = raw_sample["processor"]
        # is_qwen3 = "Qwen3VLProcessor" in processor.__class__.__name__
        is_qwen3 = raw_sample["is_qwen3"]

        data_source = {
            "type": task_type,
            "is_qwen3": is_qwen3,
            "height": raw_sample["height"],
            "width": raw_sample["width"],
        }
        if stage_info.get("matching_stage_id"):
            data_source["matching_stage"] = stage_info["matching_stage_id"]
        if variant_stage_id:
            data_source["matching_stage_variant"] = variant_stage_id
        if "overlap" in raw_sample:
            try:
                data_source["overlap"] = float(raw_sample["overlap"])
            except (TypeError, ValueError):
                data_source["overlap"] = raw_sample["overlap"]
        model_inputs["data_source"] = json.dumps(data_source)

        return model_inputs

    def regenerate_batch(self, batch_dict: dict) -> dict:
        """Regenerate batch samples with dynamically chosen task.

        Args:
            db_indices: List of LMDB indices derived from sample_meta.

        Returns:
            Dictionary containing regenerated samples for the chosen task
        """
        task = self.decide_task()

        regenerated_samples = []
        neighbor_window = max(1, int(getattr(self.config, "resample_neighbor_window", 40)))

        for raw_sample in batch_dict:
            candidate = raw_sample
            processed_sample: Optional[dict] = None

            for attempt in range(max(0, int(self.config.max_resample_attempts)) + 1):
                try:
                    processed_sample = self.process_raw_sample(
                        candidate,
                        task,
                        matching_params=self.current_matching_params,
                        grounding_params=self.current_grounding_params,
                    )
                    break
                except InsufficientMatchesError as exc:
                    if attempt >= self.config.max_resample_attempts:
                        logger.warning("Skipping sample after %d attempts: %s", attempt, exc)
                        if self._buffer_logger is not None:
                            self._buffer_logger.info(
                                json.dumps(
                                    {
                                        "event": "sample_skipped",
                                        "reason": "insufficient_matches",
                                        "details": {
                                            "required": exc.required,
                                            "available": exc.available,
                                            "stage": exc.stage_id,
                                            "db_idx": exc.db_idx,
                                        },
                                    }
                                )
                            )
                        break

                    replacement = self._sample_neighbor_raw(candidate, neighbor_window)
                    if replacement is None:
                        replacement = self._sample_random_raw()
                    if replacement is None:
                        logger.warning("No replacement sample available while handling insufficient matches.")
                        break
                    candidate = replacement

            if processed_sample is not None:
                regenerated_samples.append(processed_sample)

        if not regenerated_samples:
            raise RuntimeError("DynamicTaskBuffer could not produce any valid samples after resampling attempts.")

        return safe_collate_fn(regenerated_samples)

    def process_batch(self, batch_dict: dict) -> dict:
        """Process incoming batch: decide task and regenerate samples.

        This is the main interface called from the training loop.

        Args:
            batch_dict: Original batch from dataloader

        Returns:
            Regenerated batch dict with dynamically chosen task
        """
        if not self.config.enable:
            return batch_dict

        return self.regenerate_batch(batch_dict)

    def get_buffer_info(self) -> dict:
        """Get comprehensive buffer information for logging.

        Returns:
            Dictionary with buffer statistics and task information
        """
        metrics = self.get_task_metrics()

        logger.info(
            f"[Buffer Info] Current Task: {self.current_task}, \n"
            f"[Buffer Info] Task Param: {self.current_grounding_params if self.current_task == 'grounding' else self.current_matching_params}"
        )

        info = {
            "buffer/total_samples": len(self.buffer),
            "buffer/adaptation_ready": int(self._adaptation_ready),
            "buffer/matching_count": metrics["matching"]["count"],
            "buffer/matching_mean_reward": metrics["matching"]["mean_reward"],
            "buffer/grounding_count": metrics["grounding"]["count"],
            "buffer/grounding_mean_reward": metrics["grounding"]["mean_reward"],
            "buffer/num_switches": len(self.task_history),
            # "buffer/current_matching_stage": self.current_matching_stage_id,
        }

        stage_metrics = metrics.get("matching_stages", {})
        for stage_id, stats in stage_metrics.items():
            info[f"buffer/stage/{stage_id}/mean_reward"] = stats.get("mean_reward", 0.0)
            info[f"buffer/stage/{stage_id}/recent_count"] = stats.get("recent_count", 0)
            info[f"buffer/stage/{stage_id}/total_count"] = stats.get("count", 0)

        return info

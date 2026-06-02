"""
Raw LMDB Dataset for Annotation Tasks

This dataset only handles:
1. Reading from LMDB database
2. Decoding images
3. Rescaling coordinates

Task-specific processing (matching vs grounding) is handled by the buffer.
"""

import itertools
import json
import os
import random
from bisect import bisect_right
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .anno_db import AnnoDB
from .utils import MAX_VISUAL_TOKENS, MIN_VISUAL_TOKENS


class AnnoRawDataset(AnnoDB):
    """Raw dataset that only reads and preprocesses data from LMDB.

    This dataset provides raw samples without task-specific formatting.
    Task processing (matching/grounding) is delegated to the buffer or downstream components.

    Attributes:
        datadb: LMDB database handle
        tokenizer: Tokenizer for text processing
        processor: Processor for multimodal inputs
    """

    def post_init(self):
        """Initialize dataset-specific attributes after base class initialization."""
        self.datadb = None
        print("Initialized AnnoRawDataset (raw LMDB reader).")

    def _build_raw_sample(self, sample_data: dict) -> dict:
        """Convert LMDB sample into the raw representation consumed by the buffer."""
        h, w = self.rescale_coordinates(sample_data, to_relative=True)
        sample = {
            "db_idx": sample_data["db_idx"],
            "image1": sample_data["image1"],
            "image2": sample_data["image2"],
            "matches": sample_data["matches"],
            "height": h,
            "width": w,
            "is_qwen3": "Qwen3VLProcessor" in self.processor.__class__.__name__,
            # "processor": self.processor,
            # "tokenizer": self.tokenizer,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "config": self.config,
        }
        if "overlap" in sample_data:
            try:
                sample["overlap"] = float(sample_data["overlap"])
            except (TypeError, ValueError):
                sample["overlap"] = sample_data["overlap"]
        if "meta" in sample_data:
            sample["meta"] = sample_data["meta"]
        return sample

    def __getitem__(self, index):
        """Retrieve and preprocess a raw sample from the database."""
        safe_index = index % self.num_samples
        sample_data = self._get_db_item(safe_index)
        return self._build_raw_sample(sample_data)

    def get_raw_sample(self, db_idx: int | str) -> dict:
        """Fetch a raw sample by LMDB index (already stored in sample_meta)."""
        safe_index = int(db_idx)
        sample_data = self._get_db_item(safe_index)
        return self._build_raw_sample(sample_data)

    def sample_nearby(self, db_idx: int | str, window: int = 40) -> Optional[dict]:
        """Sample another item near the provided LMDB index."""
        try:
            base_idx = int(db_idx)
        except (TypeError, ValueError):
            return None

        start = 0
        stop = start + self.num_samples - 1
        if start > stop:
            return None

        radius = max(1, int(window))
        for _ in range(8):
            delta = random.randint(-radius, radius)
            candidate_idx = max(start, min(stop, base_idx + delta))
            try:
                return self.get_raw_sample(candidate_idx)
            except Exception:
                continue

        try:
            fallback_idx = random.randint(start, stop)
            return self.get_raw_sample(fallback_idx)
        except Exception:
            return None


class MultiRawDataset(Dataset):
    """Container dataset that multiplexes multiple LMDB sources.

    Args:
        data_files: Path to a JSON file containing a list of LMDB directories.
    """

    def __init__(
        self,
        data_files: str,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # @zhonghao, qwen2_5_vl and qwen3_vl have different visual patch size
        self.visual_patch_size = processor.image_processor.patch_size
        self.visual_merge_size = processor.image_processor.merge_size
        self.visual_temporal_patch_size = processor.image_processor.temporal_patch_size

        self.min_pixels = MIN_VISUAL_TOKENS * ((self.visual_patch_size * self.visual_merge_size) ** 2)
        self.max_pixels = MAX_VISUAL_TOKENS * ((self.visual_patch_size * self.visual_merge_size) ** 2)

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        path = Path(data_files)
        if not path.suffix:
            path = path.with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(f"MultiRawDataset expects a JSON file of LMDB paths, got {data_files}")

        with path.open("r", encoding="utf-8") as f:
            lmdb_list = json.load(f)

        if not isinstance(lmdb_list, Sequence):
            raise ValueError("MultiRawDataset JSON must contain a list of LMDB directories.")

        self.datasets: list[AnnoRawDataset] = []
        self.dataset_lengths: list[int] = []
        self.overlap_ranking: list[list[int]] = []

        for entry in lmdb_list:
            lmdb_path = entry["path"] if isinstance(entry, dict) else entry
            dataset = AnnoRawDataset(
                data_files=lmdb_path,
                tokenizer=tokenizer,
                config=config,
                processor=processor,
            )
            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            self.overlap_ranking.append(self._extract_overlap_ranking(dataset))

        self.dataset_start_indices: list[int] = []
        self.dataset_orders: list[list[int]] = []
        for dataset_idx, dataset in enumerate(self.datasets):
            start_idx = int(dataset.meta_data.get("start_index", 0)) if hasattr(dataset, "meta_data") else 0
            order = self._normalize_ranking(self.overlap_ranking[dataset_idx], len(dataset), start_idx)
            self.dataset_start_indices.append(start_idx)
            self.dataset_orders.append(order)

        self.global_order: list[tuple[int, int]] = self._build_global_order(self.dataset_orders)
        self.total_length = sum(self.dataset_lengths)
        self.cumulative_lengths = list(itertools.accumulate(self.dataset_lengths))
        self.dataset_weights = [max(1, length) for length in self.dataset_lengths]
        if self.global_order:
            self.total_length = len(self.global_order)

    @staticmethod
    def _extract_overlap_ranking(dataset: AnnoRawDataset) -> list[int]:
        ranking = []
        meta = getattr(dataset, "meta_data", {}) or {}
        for key in ("overlap_ranks", "overlap_rank", "overlap_sorted_indices", "overlap_indices"):
            if key in meta:
                ranking = meta[key]
                break

        if isinstance(ranking, dict):
            ranking = ranking.get("indices") or ranking.get("values")

        result: list[int] = []
        for value in ranking or []:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue

        if not result:
            # Fallback to sequential ordering covering all samples
            start = 0
            result = list(range(start, start + len(dataset)))

        return result

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> dict:
        if not self.datasets:
            raise IndexError("MultiRawDataset is empty.")
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"Index {index} out of range for MultiRawDataset of length {length}")

        if self.global_order:
            dataset_idx, db_idx = self.global_order[index]
            dataset = self.datasets[dataset_idx]
            sample = dataset.get_raw_sample(db_idx)
            sample["source_dataset"] = dataset_idx
            return sample

        # Fallback to sequential ordering if global_order is unavailable
        dataset_idx = bisect_right(self.cumulative_lengths, index)
        prev_cum = self.cumulative_lengths[dataset_idx - 1] if dataset_idx > 0 else 0
        local_index = index - prev_cum
        sample = self.datasets[dataset_idx][local_index]
        sample["source_dataset"] = dataset_idx
        return sample

    def sample_nearby(
        self, db_idx: int | str, source_dataset: Optional[int] = None, window: int = 40
    ) -> Optional[dict]:
        if not self.datasets:
            return None

        dataset_idx = source_dataset if source_dataset is not None else 0
        dataset_idx = max(0, min(len(self.datasets) - 1, dataset_idx))
        dataset = self.datasets[dataset_idx]
        sample = dataset.sample_nearby(db_idx, window=window)
        if sample is not None:
            sample["source_dataset"] = dataset_idx
        return sample

    @staticmethod
    def _normalize_ranking(ranking: list[int], target_len: int, start_index: int) -> list[int]:
        """Normalize ranking list to a fixed length without duplicates."""
        normalized: list[int] = []
        seen: set[int] = set()
        for value in ranking or []:
            try:
                db_idx = int(value)
            except (TypeError, ValueError):
                continue
            if db_idx in seen:
                continue
            normalized.append(db_idx)
            seen.add(db_idx)
            if len(normalized) == target_len:
                return normalized

        # Fill missing positions with sequential db indices
        candidate = int(start_index)
        while len(normalized) < target_len:
            if candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
            candidate += 1
        return normalized

    @staticmethod
    def _build_global_order(dataset_orders: list[list[int]]) -> list[tuple[int, int]]:
        """Round-robin merge ordered indices from each dataset."""
        if not dataset_orders:
            return []
        max_len = max(len(order) for order in dataset_orders)
        merged: list[tuple[int, int]] = []
        for rank_idx in range(max_len):
            for dataset_idx, order in enumerate(dataset_orders):
                if rank_idx < len(order):
                    merged.append((dataset_idx, order[rank_idx]))
        return merged

    def sample_by_difficulty(self, difficulty: float) -> dict:
        """Legacy difficulty sampling based on overlap rankings."""
        if not self.datasets:
            raise RuntimeError("MultiRawDataset cannot sample because it contains no datasets.")

        difficulty = float(max(0.0, min(1.0, difficulty)))
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.dataset_weights, k=1)[0]
        ranking = self.overlap_ranking[dataset_idx]
        dataset = self.datasets[dataset_idx]

        if ranking:
            target_rank = int(round((1.0 - difficulty) * (len(ranking) - 1)))
            target_rank = max(0, min(len(ranking) - 1, target_rank))
            db_idx = ranking[target_rank]
            sample = dataset.get_raw_sample(db_idx)
            sample["source_dataset"] = dataset_idx
            return sample

        local_idx = random.randint(0, max(len(dataset) - 1, 0))
        sample = dataset[local_idx]
        sample["source_dataset"] = dataset_idx
        return sample

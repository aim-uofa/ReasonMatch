"""
Buffered DataLoader Wrapper for Dynamic Task Switching

This module provides a wrapper around standard PyTorch DataLoader that integrates
with DynamicTaskBuffer to enable dynamic task switching during training.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader

from my_recipe.buffer.dynamic_task_buffer import DynamicTaskBuffer


logger = logging.getLogger(__name__)


@dataclass
class OverlapBinScheduleConfig:
    """Configuration for overlap-based bin staging inside BufferedDataLoader."""

    enable_overlap_bins: bool = False
    overlap_bin_count: int = 4
    promotion_reward_threshold: float = 1.9
    promotion_window: int = 3
    min_batches_per_bin: int = 6
    shuffle_within_bin: bool = True

    @property
    def enabled(self) -> bool:
        return self.enable_overlap_bins

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "OverlapBinScheduleConfig":
        if not cfg:
            return cls()
        return cls(
            enable_overlap_bins=bool(cfg.get("enable_overlap_bins", cfg.get("enable", False))),
            overlap_bin_count=max(1, int(cfg.get("overlap_bin_count", cfg.get("bin_count", cls.overlap_bin_count)))),
            promotion_reward_threshold=float(
                cfg.get("promotion_reward_threshold", cfg.get("reward_threshold", cls.promotion_reward_threshold))
            ),
            promotion_window=max(1, int(cfg.get("promotion_window", cfg.get("window", cls.promotion_window)))),
            min_batches_per_bin=max(1, int(cfg.get("min_batches_per_bin", cfg.get("min_batches", cls.min_batches_per_bin)))),
            shuffle_within_bin=bool(cfg.get("shuffle_within_bin", cls.shuffle_within_bin)),
        )


class OverlapBinManager:
    """Keeps track of overlap bins and reward-based promotion."""

    def __init__(self, dataset_len: int, batch_size: int, config: OverlapBinScheduleConfig):
        self.config = config
        self.dataset_len = int(dataset_len)
        self.batch_size = int(max(1, batch_size))
        self.bin_indices: list[list[int]] = []
        self.bin_offsets: list[int] = []
        self.current_bin: int = 0
        self.batches_in_current_bin: int = 0
        self.reward_window: deque[float] = deque(maxlen=config.promotion_window)
        self.pending_switch_reason: Optional[str] = None
        self.history: list[dict] = []
        self.enabled = bool(config.enabled and self.dataset_len > 0)

        if self.enabled:
            self._build_bins()
            if not self.bin_indices:
                logger.warning("Overlap bin scheduler disabled: could not create at least one valid bin.")
                self.enabled = False

    @property
    def total_bins(self) -> int:
        return len(self.bin_indices)

    def _build_bins(self) -> None:
        if self.dataset_len <= 0:
            return

        desired_bins = max(1, min(self.config.overlap_bin_count, self.dataset_len))
        base_size = self.dataset_len // desired_bins
        remainder = self.dataset_len % desired_bins

        start = 0
        provisional_bins: list[list[int]] = []
        for idx in range(desired_bins):
            extra = 1 if idx < remainder else 0
            stop = start + base_size + extra
            indices = list(range(start, stop))
            if indices:
                provisional_bins.append(indices)
            start = stop

        merged_bins: list[list[int]] = []
        accumulator: list[int] = []
        for indices in provisional_bins:
            accumulator.extend(indices)
            if len(accumulator) >= self.batch_size:
                merged_bins.append(accumulator)
                accumulator = []

        if accumulator:
            if merged_bins:
                merged_bins[-1].extend(accumulator)
            else:
                merged_bins.append(accumulator)

        self.bin_indices = [bin_indices for bin_indices in merged_bins if bin_indices]
        if self.config.shuffle_within_bin:
            for bin_indices in self.bin_indices:
                if len(bin_indices) > 1:
                    random.shuffle(bin_indices)
        self.bin_offsets = [0 for _ in self.bin_indices]
        self.bin_completed = [False for _ in self.bin_indices]

        if not self.bin_indices:
            return

        bin_sizes = ", ".join(str(len(b)) for b in self.bin_indices)
        min_bin = min(len(b) for b in self.bin_indices) if self.bin_indices else 0
        if min_bin < self.batch_size:
            logger.warning(
                "Overlap bin sizes (%s) include bins smaller than batch size (%d). "
                "Samples from those bins will repeat within a batch.",
                bin_sizes,
                self.batch_size,
            )
        logger.info(
            "Initialized overlap bins (count=%d, batch_size=%d): %s",
            len(self.bin_indices),
            self.batch_size,
            bin_sizes,
        )

    def reset_epoch(self) -> None:
        if not self.enabled or not self.bin_indices:
            return
        self.current_bin = 0
        self.batches_in_current_bin = 0
        self.reward_window.clear()
        self.pending_switch_reason = None
        for idx, bin_indices in enumerate(self.bin_indices):
            if self.config.shuffle_within_bin and len(bin_indices) > 1:
                random.shuffle(bin_indices)
            self.bin_offsets[idx] = 0
            self.bin_completed[idx] = False

    def all_bins_completed(self) -> bool:
        if not self.enabled or not self.bin_indices:
            return False
        return all(self.bin_completed)

    def next_batch_indices(self) -> list[int]:
        if not self.enabled or self.current_bin >= self.total_bins:
            return []

        bin_indices = self.bin_indices[self.current_bin]
        if not bin_indices or self.bin_completed[self.current_bin]:
            return []

        offset = self.bin_offsets[self.current_bin]
        if offset >= len(bin_indices):
            self.bin_completed[self.current_bin] = True
            return []

        take = min(self.batch_size, len(bin_indices) - offset)
        selections = bin_indices[offset : offset + take]
        self.bin_offsets[self.current_bin] += take
        if self.bin_offsets[self.current_bin] >= len(bin_indices):
            self.bin_completed[self.current_bin] = True

        if take < self.batch_size:
            pad_source = selections if selections else [bin_indices[-1]]
            pad_idx = 0
            while len(selections) < self.batch_size:
                selections.append(pad_source[pad_idx % len(pad_source)])
                pad_idx += 1

        self.batches_in_current_bin += 1
        return selections

    def record_reward(self, reward_value: float) -> None:
        if not self.enabled or self.current_bin >= self.total_bins:
            return

        reward_float = float(reward_value)
        self.reward_window.append(reward_float)
        window_ready = len(self.reward_window) == self.reward_window.maxlen
        if not window_ready:
            return

        if self.batches_in_current_bin < self.config.min_batches_per_bin:
            return

        avg_reward = sum(self.reward_window) / len(self.reward_window)
        if (
            avg_reward >= self.config.promotion_reward_threshold
            and self.current_bin < self.total_bins - 1
        ):
            self.pending_switch_reason = (
                f"avg_reward={avg_reward:.3f} over last {len(self.reward_window)} batches"
            )
        else:
            self.pending_switch_reason = None

    def maybe_advance_bin(self) -> bool:
        if not self.enabled or self.pending_switch_reason is None:
            return False

        if self.current_bin >= self.total_bins - 1:
            self.pending_switch_reason = None
            return False

        prev_bin = self.current_bin
        self.bin_completed[prev_bin] = True
        self.bin_offsets[prev_bin] = len(self.bin_indices[prev_bin])
        self.current_bin += 1
        self.batches_in_current_bin = 0
        self.reward_window.clear()
        reason = self.pending_switch_reason or "promotion"
        self.pending_switch_reason = None
        self.history.append({"from": prev_bin, "to": self.current_bin, "reason": reason})
        logger.info("BufferedDataLoader promoted overlap bin %d -> %d (%s)", prev_bin, self.current_bin, reason)
        return True

    def skip_empty_bin(self) -> bool:
        if not self.enabled:
            return False
        if self.current_bin >= self.total_bins - 1:
            return False
        prev_bin = self.current_bin
        self.bin_completed[prev_bin] = True
        self.bin_offsets[prev_bin] = len(self.bin_indices[prev_bin])
        self.current_bin += 1
        self.batches_in_current_bin = 0
        self.reward_window.clear()
        self.history.append({"from": prev_bin, "to": self.current_bin, "reason": "empty"})
        logger.warning("BufferedDataLoader skipped empty overlap bin %d", prev_bin)
        return True

    def state_dict(self) -> dict:
        return {
            "current_bin": self.current_bin,
            "bin_offsets": list(self.bin_offsets),
            "bin_completed": list(self.bin_completed),
            "batches_in_current_bin": self.batches_in_current_bin,
            "reward_window": list(self.reward_window),
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict) -> None:
        if not self.enabled:
            return
        self.current_bin = int(state.get("current_bin", self.current_bin))
        offsets = state.get("bin_offsets")
        if offsets and len(offsets) == len(self.bin_offsets):
            self.bin_offsets = [int(v) for v in offsets]
        completed = state.get("bin_completed")
        if completed and len(completed) == len(self.bin_completed):
            self.bin_completed = [bool(v) for v in completed]
        self.batches_in_current_bin = int(state.get("batches_in_current_bin", self.batches_in_current_bin))
        reward_vals = [float(v) for v in state.get("reward_window", [])]
        self.reward_window = deque(reward_vals, maxlen=self.reward_window.maxlen)
        self.history = list(state.get("history", self.history))
        self.pending_switch_reason = None


class BufferedDataLoader:
    """DataLoader wrapper that integrates with DynamicTaskBuffer.

    This wrapper intercepts batches from the dataloader and regenerates them
    with dynamically chosen tasks based on recent performance metrics.

    Attributes:
        base_dataloader: Original dataloader to fetch indices from
        buffer: DynamicTaskBuffer instance for task switching
    """

    def __init__(
        self,
        base_dataloader: DataLoader,
        buffer: DynamicTaskBuffer,
        loader_config: Optional[dict] = None,
    ):
        """Initialize buffered dataloader.

        Args:
            base_dataloader: Original PyTorch DataLoader
            buffer: DynamicTaskBuffer for dynamic task switching
        """
        self.base_dataloader = base_dataloader
        self.buffer = buffer
        self.batch_size = getattr(base_dataloader, "batch_size", None)
        self.max_batches_per_epoch = len(base_dataloader)

        # Store sampler and dataset info for compatibility
        self.sampler = base_dataloader.sampler  # FIXME @zhonghao: seems unused
        self.dataset = base_dataloader.dataset
        self.is_last_step = False
        self._bin_state_loaded = False
        self._bin_config = OverlapBinScheduleConfig.from_dict(loader_config)
        self._bin_manager: Optional[OverlapBinManager] = None
        if self._bin_config.enabled:
            if self.batch_size is None:
                logger.warning("Overlap bin scheduler requires an explicit batch_size; disabling.")
            else:
                manager = OverlapBinManager(len(self.dataset), self.batch_size, self._bin_config)
                if manager.enabled:
                    self._bin_manager = manager
                else:
                    logger.warning("Overlap bin scheduler disabled after initialization.")

    def __iter__(self) -> Iterator[dict]:
        """Iterate through batches with dynamic task switching.

        Yields:
            Batch dictionaries with dynamically chosen tasks
        """
        self.is_last_step = False
        if self._bin_manager is None:
            for batch_dict in self.base_dataloader:
                processed_batch = self.buffer.process_batch(batch_dict)
                yield processed_batch
            return

        if self._bin_state_loaded:
            self._bin_state_loaded = False
        else:
            self._bin_manager.reset_epoch()

        produced = 0
        while produced < self.max_batches_per_epoch:
            self._bin_manager.maybe_advance_bin()
            batch_samples = self._next_overlap_batch()
            if batch_samples is None:
                break
            processed_batch = self.buffer.process_batch(batch_samples)
            produced += 1
            yield processed_batch

    def _next_overlap_batch(self) -> Optional[list[dict]]:
        if self._bin_manager is None or self.batch_size is None:
            return None

        attempts = 0
        while attempts < max(1, self._bin_manager.total_bins):
            indices = self._bin_manager.next_batch_indices()
            if not indices:
                if not self._bin_manager.skip_empty_bin():
                    return None
                attempts += 1
                continue
            batch = [self.dataset[idx] for idx in indices]
            self.is_last_step = self._bin_manager.all_bins_completed()
            return batch
        return None

    def __len__(self) -> int:
        """Return length of base dataloader."""
        return len(self.base_dataloader)

    def state_dict(self):
        """Get state dict for checkpointing."""
        state = {}
        if hasattr(self.base_dataloader, "state_dict"):
            state["base"] = self.base_dataloader.state_dict()
        if self._bin_manager is not None:
            state["bin_manager"] = self._bin_manager.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for checkpointing (delegates to base dataloader)."""
        if not isinstance(state_dict, dict):
            return
        base_state = state_dict.get("base", state_dict)
        if hasattr(self.base_dataloader, "load_state_dict"):
            self.base_dataloader.load_state_dict(base_state)
        if self._bin_manager is not None and "bin_manager" in state_dict:
            self._bin_manager.load_state_dict(state_dict["bin_manager"])
            self._bin_state_loaded = True

    def report_batch_reward(self, rewards) -> None:
        """Report batch-level reward so the bin scheduler can evaluate promotion."""
        if self._bin_manager is None:
            return
        if isinstance(rewards, torch.Tensor):
            if rewards.numel() == 0:
                return
            reward_value = float(rewards.mean().item())
        elif hasattr(rewards, "__iter__"):
            rewards_list = list(rewards)
            if not rewards_list:
                return
            reward_value = float(sum(float(r) for r in rewards_list) / len(rewards_list))
        else:
            reward_value = float(rewards)
        self._bin_manager.record_reward(reward_value)

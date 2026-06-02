# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager

from .rollout_worker import RolloutWorker

logger = logging.getLogger(__name__)

SYS_PROMPT = """You are an expert judge evaluating the reasoning quality of a multimodal model performing cross-view region matching.

## Task Background
The model matches candidate regions between two cross-view images. It outputs:
<thinking>reasoning process</thinking>
<answer>final matching result</answer>

## Scoring Guidelines

Evaluate the overall quality based on THREE criteria:
1. **Reasoning Coherence**: Is the thinking process logical and well-structured?
2. **Visual Feature Analysis**: Does the reasoning analyze actual visual features (shape, appearance, spatial relationships) rather than using shortcuts like positional bias or patterns?
3. **Consistency**: Does the final answer align with the thinking process?

### Score Definitions:
- **Score 0**: Critical failures
  - Thinking is corrupted, repetitive, or nonsensical
  - Clear evidence of reward hacking (always choosing same position, pattern exploitation, no visual analysis)
  - Answer completely contradicts the thinking

- **Score 1**: Partially acceptable
  - Thinking is somewhat reasonable but superficial or contains logical gaps
  - Minimal or generic visual analysis, may rely partly on non-visual heuristics
  - Answer partially aligns with thinking but has noticeable inconsistencies

- **Score 2**: Good quality
  - Thinking is clear, logical, and well-reasoned
  - Demonstrates genuine visual feature analysis with specific details
  - Answer is fully consistent with and supported by the thinking process

## Red Flags for Score 0:
- Generic templates without adaptation to specific images
- Positional bias (e.g., "usually option B is correct")
- Avoiding specific visual descriptions
- Circular reasoning that doesn't analyze features

## Output Format
Return valid JSON between <answer> </answer> tags:

<answer>
{
  "score": <0, 1, or 2>,
  "reasoning": "<brief 1-2 sentence explanation>",
  "reward_hacking_detected": <true or false>
}
</answer>

Be strict in your evaluation. Score 2 should only be given when the model demonstrates clear, feature-based visual reasoning."""

USER_TEM = """
## Model Reasoning Content
{{response}}
## Evaluation Instruction
You can make analysis, but MAKE SURE you output final valid JSON answer between <answer> </answer> tags.
## Your Evaluation
"""

import re


def _parse_consistency_score(raw_text: str, score_key: str, default_score: float) -> float:
    if not raw_text:
        return default_score
    raw_text = raw_text.strip()

    try:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_text, re.DOTALL)
        payload = json.loads(match.group(1))
        score = payload.get(score_key, default_score)
        return float(score)
    except Exception:
        return default_score


async def evaluate_consistency_batch(responses: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate reasoning consistency for a batch of responses.

    Args:
        responses: List of response strings from the actor.
        config: Dict with keys like api_key, base_url, model, timeout,
            max_concurrency, max_retries, default_score, temperature,
            max_tokens, score_key, system_prompt, user_prompt_template.

    Returns:
        List of dicts aligned with responses. Each item includes "score" and "raw_response".
    """
    if not responses:
        return []

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = config.get("base_url")
    model = config.get("model")
    if not api_key:
        raise ValueError("Missing api_key for consistency evaluator")
    if not model:
        raise ValueError("Missing model for consistency evaluator")

    timeout = float(config.get("timeout", 30))
    max_concurrency = int(config.get("max_concurrency", 16))
    max_retries = int(config.get("max_retries", 2))
    default_score = float(config.get("default_score", 0.0))
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 64))
    score_key = str(config.get("score_key", "score"))

    system_prompt = config.get(
        "system_prompt",
        SYS_PROMPT,
    )
    user_prompt_template = config.get("user_prompt_template", USER_TEM)

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_one(response: str) -> dict[str, Any]:
        if response is None or not str(response).strip():
            return {"score": default_score, "raw_response": ""}
        prompt = user_prompt_template.replace("{{response}}", response)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                content = ""
                if completion and completion.choices:
                    content = completion.choices[0].message.content or ""
                score = _parse_consistency_score(content, score_key, default_score)
                return {"score": score, "raw_response": content}
            except Exception as exc:
                if attempt >= max_retries:
                    logger.warning("Consistency eval failed: %s", exc)
                    return {"score": default_score, "raw_response": ""}
                await asyncio.sleep(0.5 * (2**attempt))
        return {"score": default_score, "raw_response": ""}

    tasks = [asyncio.create_task(_score_one(response)) for response in responses]
    try:
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


@register("my_dapo")
class myDAPORewardManager(DAPORewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        rollout_logging: Optional[dict[str, Any]] = None,
        reason_consistency: Optional[dict[str, Any]] = None,
    ) -> None:
        rollout_cfg = rollout_logging or {}
        if isinstance(rollout_cfg, bool):
            rollout_cfg = {"enable": rollout_cfg}

        self.rollout_save_enabled: bool = bool(rollout_cfg.get("enable", False))
        default_dir = os.environ.get("VERL_ROLLOUT_SAVE_DIR")
        rollout_dir = rollout_cfg.get("dir", default_dir or "./rollout_logs")
        self.rollout_save_dir = Path(rollout_dir).expanduser().resolve()
        self.rollout_save_text = bool(rollout_cfg.get("save_text", True))
        self.rollout_save_images = bool(rollout_cfg.get("save_images", True))
        self.rollout_image_format = str(rollout_cfg.get("image_format", "jpg")).lower()
        self.rollout_max_samples_per_step: Optional[int] = rollout_cfg.get("max_samples_per_step")
        self.rollout_overwrite = bool(rollout_cfg.get("overwrite", False))
        self.rollout_record_name = rollout_cfg.get("record_filename", "record.json")
        self.group_size = int(rollout_cfg.get("group_size", 1))

        reason_cfg = reason_consistency or {}
        if isinstance(reason_cfg, bool):
            reason_cfg = {"enable": reason_cfg}
        self.reason_consistency_config = reason_cfg

        # rank_env = next(
        #     (os.environ[key] for key in ("RANK", "WORLD_RANK", "SLURM_PROCID") if key in os.environ),
        #     None,
        # )
        # try:
        #     self.rollout_rank = int(rank_env) if rank_env is not None else None
        # except ValueError:
        #     self.rollout_rank = None
        # if self.rollout_rank is None:
        #     self.rollout_rank = os.getpid()

        self.rollout_rank = 0

        self._fallback_step = 0

        if self.rollout_save_enabled:
            target_dir = self._rank_root_dir()
            target_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
        )
        # breakpoint()

    def prepare(self, data: DataProto, processor, jd_wg: Optional[RolloutWorker] = None):
        """Prepare the data before calling the reward manager."""
        reward_fn = self.compute_score.args[0]
        need_prepare = getattr(reward_fn, "needs_prepare", False)
        if need_prepare:
            reward_fn.prepare(data, tokenizer=self.tokenizer, processor=processor, llm=jd_wg)

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards and optionally dump rollout artifacts to disk."""

        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        raw_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print = 0

        global_step = self._extract_step_id(data.meta_info)

        per_step_saved = 0
        response_strs = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            response_strs.append(response_str)

        reason_consistency_results = self._get_reason_consistency_scores(response_strs)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {}) or {}
            extra_info = self._clone_without_unserializable(extra_info)
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            response_str = response_strs[i]

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            raw_reward = score
            reason_result = reason_consistency_results[i]
            reason_reward = float(reason_result.get("score", 0.0))
            reward = raw_reward + reason_reward
            reward_extra_info["reason_consistency_score"].append(reason_reward)

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            target_index = valid_response_length - 1 if valid_response_length > 0 else 0
            reward_tensor[i, target_index] = reward
            raw_reward_tensor[i, target_index] = raw_reward

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            saved_path = None
            if self.rollout_save_enabled:  # and i % self.group_size == 0:
                if self.rollout_max_samples_per_step is None or per_step_saved < self.rollout_max_samples_per_step:
                    multi_modal = data_item.non_tensor_batch.pop("multi_modal_data")
                    sample_meta_raw = data_item.non_tensor_batch.get("sample_meta")
                    sample_meta = self._maybe_parse_json(sample_meta_raw)
                    score_detail = result if isinstance(result, dict) else {"score": score}
                    score_detail["reason_consistency_score"] = reason_reward
                    score_detail["reason_consistency_raw"] = reason_result.get("raw_response", "")
                    saved_path = self._save_rollout_sample(
                        step_id=global_step,
                        sample_index=i,
                        prompt_text=prompt_str,
                        response_text=response_str,
                        ground_truth=ground_truth,
                        data_source=data_source,
                        reward=reward,
                        raw_score=score,
                        score_detail=score_detail,
                        extra_info=extra_info,
                        sample_meta=sample_meta,
                        rollout_reward_scores=rollout_reward_scores,
                        multi_modal_data=multi_modal,
                    )
                    if saved_path is not None:
                        per_step_saved += 1

            # if saved_path is not None:
            #     rollout_save_paths.append(saved_path)

        # if rollout_save_paths:
        #     reward_extra_info["rollout_paths"].extend(rollout_save_paths)

        reward_extra_info["raw_reward_tensor"] = raw_reward_tensor

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _get_reason_consistency_scores(self, response_strs: list[str]) -> list[dict[str, Any]]:
        if not response_strs:
            return []
        enable = bool(self.reason_consistency_config.get("enable", False))
        default_score = float(self.reason_consistency_config.get("default_score", 0.0))
        if not enable:
            return [{"score": default_score, "raw_response": ""} for _ in response_strs]
        try:
            coro = evaluate_consistency_batch(response_strs, self.reason_consistency_config)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        except Exception as exc:
            logger.warning("Consistency eval failed, fallback to default: %s", exc)
            return [{"score": default_score, "raw_response": ""} for _ in response_strs]

    def _rank_root_dir(self) -> Path:
        return self.rollout_save_dir / f"rank_{self.rollout_rank}"

    @staticmethod
    def _clone_without_unserializable(source: dict) -> dict:
        if not source:
            return {}
        return {key: value for key, value in source.items()}

    def _extract_step_id(self, meta_info: dict[str, Any]) -> int:
        if not isinstance(meta_info, dict):
            self._fallback_step += 1
            return self._fallback_step
        step = meta_info.get("global_steps")
        if isinstance(step, (list, tuple)) and step:
            step = step[0]
        if isinstance(step, (np.integer,)):
            step = int(step)
        if isinstance(step, torch.Tensor):
            if step.numel() > 0:
                step = int(step.view(-1)[0].item())
            else:
                step = None
        if isinstance(step, (int, np.integer)):
            return int(step)
        self._fallback_step += 1
        return self._fallback_step

    def _maybe_parse_json(self, value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        if isinstance(value, dict):
            return value
        return value

    def _save_rollout_sample(
        self,
        step_id: int,
        sample_index: int,
        prompt_text: str,
        response_text: str,
        ground_truth: Any,
        data_source: Any,
        reward: float,
        raw_score: float,
        score_detail: dict[str, Any],
        extra_info: dict[str, Any],
        sample_meta: Any,
        rollout_reward_scores: dict[str, Any],
        multi_modal_data: Any,
    ) -> Optional[str]:
        try:
            step_label = f"step_{step_id:08d}"
            sample_label = f"sample_{sample_index:04d}"
            sample_dir = self._rank_root_dir() / step_label / sample_label

            if sample_dir.exists() and not self.rollout_overwrite:
                logger.debug("Skip saving rollout %s (exists and overwrite disabled)", sample_dir)
                return str(sample_dir / self.rollout_record_name)

            sample_dir.mkdir(parents=True, exist_ok=True)

            images = self._extract_images(multi_modal_data)
            image_rel_paths = []
            if self.rollout_save_images and images:
                for idx, img in enumerate(images):
                    if img is None:
                        continue
                    file_path = sample_dir / f"image_{idx:02d}.{self.rollout_image_format}"
                    img.save(file_path)
                    image_rel_paths.append(file_path.name)

            record = {
                "step": int(step_id),
                "sample_index": sample_index,
                "prompt": prompt_text if self.rollout_save_text else None,
                "response": response_text if self.rollout_save_text else None,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "reward": reward,
                "raw_score": raw_score,
                "score_detail": score_detail,
                "extra_info": extra_info,
                "sample_meta": sample_meta,
                "rollout_reward_scores": rollout_reward_scores,
                "images": image_rel_paths if image_rel_paths else None,
            }

            record = self._sanitize_for_json(record)
            record_path = sample_dir / self.rollout_record_name
            with record_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            if self.rollout_save_text:
                (sample_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
                (sample_dir / "response.txt").write_text(response_text, encoding="utf-8")

            return str(record_path)
        except Exception as exc:
            logger.warning("Failed to save rollout sample %s-%s: %s", step_id, sample_index, exc)
            return None

    def _sanitize_for_json(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [self._sanitize_for_json(v) for v in obj.tolist()]
        if torch.is_tensor(obj):
            return [self._sanitize_for_json(v) for v in obj.detach().cpu().reshape(-1).tolist()]
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, Image.Image):
            return None
        return str(obj)

    def _extract_images(self, multi_modal_data: Any) -> list[Image.Image]:
        if multi_modal_data is None:
            return []
        data = multi_modal_data
        if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
            data = data.item()
        if isinstance(data, dict):
            images = data.get("image")
        else:
            images = None
        if images is None:
            return []
        if isinstance(images, np.ndarray):
            images = images.tolist()
        if not isinstance(images, (list, tuple)):
            images = [images]

        result = []
        for img in images:
            pil = self._ensure_pil_image(img)
            if pil is not None:
                result.append(pil)
        return result

    def _ensure_pil_image(self, img: Any) -> Optional[Image.Image]:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if torch.is_tensor(img):
            tensor = img.detach().cpu()
            arr = tensor.numpy()
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            return None

        arr = np.array(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

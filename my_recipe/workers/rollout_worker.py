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
"""
The pure inference role for flexible usage.
"""

import inspect
import logging
import os
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.profiler import DistProfiler, DistProfilerExtension, GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing
from verl.utils.torch_functional import check_device_is_available, get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs
from typing import Sequence
from vllm import PromptType

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


class SimpleVLLM:
    """Standalone vLLM inference engine for judger/rollout.

    Runs in a SEPARATE Ray actor process from ActorRollout, allowing both
    to use sleep mode on the same GPU pool. They alternate via sleep/wake:
    - When ActorRollout is active (training), Judger sleeps
    - When Judger is active (scoring), ActorRollout sleeps
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """Initialize a standalone vLLM inference engine.

        Args:
            model_path: Path to the model weights
            config: DictConfig containing rollout configuration
            tokenizer: HuggingFace tokenizer
            model_hf_config: HuggingFace model config
            **kwargs: Additional arguments (e.g., trust_remote_code)
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )
        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )
        trust_remote_code = kwargs.get("trust_remote_code", False)
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        # @zhonghao: [feat] multi-instance vLLM inference engine supported
        # TODO: support PP or EP
        # Sleep mode is enabled when free_cache_engine=True
        # This is safe because this worker runs in a SEPARATE Ray actor process
        # from ActorRollout, avoiding the "one instance per process" limitation
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **engine_kwargs,
        )
        # breakpoint()
        # Start in sleep mode to save memory when not in use
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            # skip_special_tokens=False,
        )

        kwargs["detokenize"] = True

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        # breakpoint()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    # @zhonghao, prompt list wrapped in DataProto to fit dispatch policy of veRL
    # FIXME: direct pass str obj may cause error, better pass tokens instead of str.
    @GPUMemoryLogger(role="vllm judger spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts with multi-modal support.

        Args:
            prompts: DataProto, the input prompts
            **kwargs: reserved for future usage

        Supports:
            - Text-only prompts (list of strings or chat templates)
            - Multi-modal prompts (with images, videos, etc.)
        """
        dummy_flag = "<DUMMY>"
        text_prompts = prompts.batch.get("prompts", None)
        multi_modal_data = prompts.non_tensor_batch.get("multi_modal_data", None)
        is_dummy = prompts.non_tensor_batch.get("is_dummy", None)

        assert prompts, 'Judger Rollout must recieve DataProto object containing non-tensor key of "prompts"'

        # Filter out dummy prompts if is_dummy mask is provided
        if is_dummy is not None:
            valid_indices = [i for i, dummy in enumerate(is_dummy) if not dummy]
            # If all prompts are dummy, return empty responses
            if len(valid_indices) == 0:
                prompts.non_tensor_batch["responses"] = np.array([dummy_flag] * len(is_dummy), dtype=object)
                return prompts
        else:
            valid_indices = list(range(len(text_prompts)))

        # Construct vLLM inputs based on whether multi-modal data is present
        if multi_modal_data is not None:
            # Multi-modal mode: create dict inputs with prompt and multi_modal_data
            vllm_inputs: Sequence[PromptType]
            vllm_inputs = []
            # Only process valid (non-dummy) prompts
            for idx in valid_indices:
                if len(multi_modal_data[idx]) > 0:
                    vllm_inputs.append(
                        {
                            "prompt": text_prompts[idx],
                            "multi_modal_data": multi_modal_data[idx],
                        }
                    )
                else:
                    vllm_inputs.append({"prompt": text_prompts[idx]})

        else:
            # Text-only mode: use prompts directly
            vllm_inputs = [text_prompts[i] for i in valid_indices]

        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=True
            )
            valid_responses = [output.outputs[0].text for output in outputs]

        # Reconstruct full response list with dummy placeholders
        responses = [dummy_flag] * len(is_dummy)
        for i, idx in enumerate(valid_indices):
            responses[idx] = valid_responses[i]

        # FIXME: @zhonghao, need a safer way to add an entry to DataProto objects.
        #        the original input has be modified and returned, so be careful about RolloutWorker postprocess.
        prompts.non_tensor_batch["responses"] = np.array(responses,dtype=object)
        return prompts

    # @zhonghao original ActorRolloutRefWorker's generate method.
    def _ori_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# @zhonghao: A vLLM sleep and wake up mgr
class VLLMManager:
    """Manages vLLM engine sleep/wake cycles for memory efficiency.

    Handles:
    - Waking up the engine before inference
    - Sleeping the engine after inference to free GPU memory
    - Managing random states across DP ranks
    """

    @check_device_is_available()
    def __init__(self, inference_engine: LLM, rollout_config: DictConfig, device_mesh: DeviceMesh = None):
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.rollout_config = rollout_config
        self.tp_size = rollout_config.get("tensor_model_parallel_size", 1)
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank() if self.tp_size > 1 else 0
        self.timing = {}  # For tracking timing metrics

        # get a random rng states for DP consistency
        if self.device_mesh is not None and "dp" in self.device_mesh.mesh_dim_names:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            # Store current training random state to restore later
            self.torch_random_states = get_torch_device().get_rng_state()
        else:
            self.gen_random_states = None
            self.torch_random_states = None

    def __enter__(self):
        """Wake up the inference engine and prepare for generation."""
        if self.rollout_config.free_cache_engine:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()
        log_gpu_memory_usage("After waking Judger", logger=logger)
        get_torch_device().empty_cache()
        if (
            self.rollout_config.free_cache_engine
            and "tags" in inspect.signature(self.inference_engine.wake_up).parameters
        ):
            self.inference_engine.wake_up(tags=["kv_cache"])

        # Set generation-specific random states for DP consistency
        if self.gen_random_states is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Put the inference engine to sleep and restore states."""
        if self.rollout_config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.gen_random_states is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]


# TODO: @zhonghao how is a group of tp ranks managed and assigned to the object?
#       currently only dp rank is read, assuming that dp and tp are already managed outside
class RolloutWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        self.profile_option = kwargs.get("profile_option", None)
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print(f"[{role}] Rank: {rank}/{world_size}")

        self.role = role
        # torch.distributed.barrier()
        # breakpoint()

        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self._is_lora = False

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=rank, config=profiler_config, option=self.profile_option)
        )

    def _build_rollout(self, override_model_config, trust_remote_code=False):
        """Build a standalone rollout/judger worker with vLLM.

        This creates a pure inference engine in a separate Ray actor process,
        allowing it to share GPUs with ActorRollout via sleep/wake scheduling.
        """
        from torch.distributed.device_mesh import init_device_mesh
        from transformers import AutoConfig

        from verl.utils.model import update_model_config

        # Setup device mesh for DP and TP
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        self.device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.name
        if rollout_name == "hf":
            raise NotImplementedError("RolloutWorker does not support HF rollout yet.")

            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            self.actor_model_config = AutoConfig.from_pretrained(
                local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
            )
            if getattr(self.actor_model_config, "model_type", None) == "kimi_vl":
                self.actor_model_config.text_config.topk_method = "greedy"
            override_config_kwargs = {
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            override_config_kwargs.update(override_model_config)
            update_model_config(self.actor_model_config, override_config_kwargs=override_config_kwargs)
            if self.rank == 0:
                print(f"Judger model config after override: {self.actor_model_config}")

            # Use SimpleVLLM for standalone inference
            # No LoRA support for judger - it's pure inference
            rollout = SimpleVLLM(
                model_path=local_path,
                config=self.config,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            log_gpu_memory_usage(f"After building {rollout_name} judger", logger=logger)

            # Create manager to handle sleep/wake cycles
            rollout_manager = VLLMManager(
                inference_engine=rollout.inference_engine,
                rollout_config=self.config,
                device_mesh=self.device_mesh,
            )
            log_gpu_memory_usage("After building judger vLLM manager", logger=logger)

        elif rollout_name == "sglang":
            raise NotImplementedError("RolloutWorker does not support SGLang yet.")
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config,
                full_params="hf" in self.config.load_format,
                device_mesh=self.device_mesh,
                offload_param=self._is_offload_param,
                multi_stage_wake_up=self.config.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.name} is not supported")

        return rollout, rollout_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the rollout model and manager.

        For standalone rollout/judger workers, this creates a SimpleVLLM engine
        in a separate Ray actor process from ActorRollout.
        """
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        if self._is_rollout:
            self.rollout, self.rollout_manager = self._build_rollout(
                override_model_config=override_model_config,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )
            # SimpleVLLM doesn't use generation_config - sampling params are set in __init__
            self.generation_config = None
        else:
            raise NotImplementedError("None-Rollout Role not supported by RolloutWorker")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red", role="judger_generate")
    def generate_sequences(self, prompts: DataProto):
        """Generate LLM responses to prompts.non_tensor_batch['prompts']

        Args:
            prompts (DataProto): The requesting batch containing non_tensor_batch key 'prompts'

        Returns:
            DataProto: The input prompts object with an added 'responses' entry.
        """
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_manager:
            log_gpu_memory_usage("After entering judger manager", logger=logger)

            prompts = self.rollout_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After judger generation", logger=logger)

            output = self.rollout_manager.postprocess_data(output)

        timing_generate.update(self.rollout_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

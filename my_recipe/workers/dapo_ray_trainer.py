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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import logging
import os
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, *args, dynamic_buffer_config=None, **kwargs):
        """Initialize trainer with optional dynamic task buffer.

        Args:
            dynamic_buffer_config: Optional DictConfig for dynamic task buffer
            *args, **kwargs: Arguments passed to parent RayPPOTrainer
        """
        # breakpoint()
        super().__init__(*args, **kwargs)

        # Initialize dynamic task buffer if config provided
        self.dynamic_buffer = None
        self.dynamic_buffer_config = dynamic_buffer_config

        if dynamic_buffer_config is not None and dynamic_buffer_config.get("enable", False):
            self._init_dynamic_buffer(dynamic_buffer_config)

    def _init_dynamic_buffer(self, buffer_config):
        """Initialize dynamic task buffer with raw dataset. Note that the buffer dataloader will replace the original self.train_dataloader.

        Args:
            buffer_config: Configuration for dynamic task buffer
        """
        from my_recipe.buffer.dynamic_task_buffer import (
            BufferConfig,
            DynamicTaskBuffer,
            GroundingPointConfig,
            MatchingFilterConfig,
            matching_curriculum_from_dict,
        )
        from my_recipe.mydatasets.anno_raw import AnnoRawDataset, MultiRawDataset

        def _to_container(cfg):
            if cfg is None:
                return {}
            if isinstance(cfg, DictConfig):
                return OmegaConf.to_container(cfg, resolve=True)
            return deepcopy(cfg)

        matching_filter_dict = _to_container(buffer_config.get("matching_filter", None))
        grounding_points_dict = _to_container(buffer_config.get("grounding_points", None))
        curriculum_dict = _to_container(buffer_config.get("matching_curriculum", None))
        loader_config_dict = _to_container(buffer_config.get("buffered_loader", None))

        default_matching_cfg = MatchingFilterConfig()
        default_grounding_cfg = GroundingPointConfig()

        matching_filter = MatchingFilterConfig(
            min_distance=float(matching_filter_dict.get("min_distance", default_matching_cfg.min_distance)),
            min_points=int(matching_filter_dict.get("min_points", default_matching_cfg.min_points)),
            max_points=int(matching_filter_dict.get("max_points", default_matching_cfg.max_points)),
            strategy=str(matching_filter_dict.get("strategy", default_matching_cfg.strategy)),
            schedule=deepcopy(matching_filter_dict.get("schedule", default_matching_cfg.schedule)),
        )
        grounding_points = GroundingPointConfig(
            min_points=int(grounding_points_dict.get("min_points", default_grounding_cfg.min_points)),
            max_points=int(grounding_points_dict.get("max_points", default_grounding_cfg.max_points)),
            strategy=str(grounding_points_dict.get("strategy", default_grounding_cfg.strategy)),
            schedule=deepcopy(grounding_points_dict.get("schedule", default_grounding_cfg.schedule)),
        )

        # Create BufferConfig from DictConfig
        buf_cfg = BufferConfig(
            buffer_size=buffer_config.get("buffer_size", 100),
            task_switch_metric=buffer_config.get("task_switch_metric", "mean_reward"),
            matching_threshold=buffer_config.get("matching_threshold", 0.7),
            grounding_threshold=buffer_config.get("grounding_threshold", 0.3),
            min_samples_for_switch=buffer_config.get("min_samples_for_switch", 20),
            enable=buffer_config.get("enable", True),
            task_mode=buffer_config.get("task_mode", "dynamic"),
            matching_filter=matching_filter,
            matching_curriculum=matching_curriculum_from_dict(curriculum_dict),
            grounding_points=grounding_points,
            log_file=buffer_config.get(
                "log_file", os.path.join(self.config.trainer.default_local_dir, "buffer_info.log")
            ),
        )

        # NOTE: @zhonghao, instead of creating new datasets for the buffer, reuse self.train_dataset

        # Create a single raw dataset that reads from LMDB
        # Task-specific processing is handled by the buffer
        # data_files = self.config.data.train_files

        # raw_dataset = AnnoRawDataset(
        #     data_files=data_files,
        #     tokenizer=self.tokenizer,
        #     config=self.config.data,
        #     processor=self.processor,
        # )

        raw_dataset = self.train_dataset
        assert isinstance(raw_dataset, (AnnoRawDataset, MultiRawDataset)), (
            f"Buffer supports only AnnoRawDataset or MultiRawDataset, got {type(raw_dataset)}"
        )

        # Initialize buffer with raw dataset
        # The buffer will handle task-specific processing (matching/grounding)
        self.dynamic_buffer = DynamicTaskBuffer(config=buf_cfg, raw_dataset=raw_dataset)

        print(f"[DynamicBuffer] Initialized with config: {buf_cfg}")
        print("[DynamicBuffer] task processing integrated in buffer")

        # Wrap dataloader with buffered version
        if self.dynamic_buffer is not None:
            from my_recipe.buffer.buffered_dataloader import BufferedDataLoader

            self.train_dataloader = BufferedDataLoader(
                base_dataloader=self.train_dataloader,
                buffer=self.dynamic_buffer,
                loader_config=loader_config_dict,
            )
            logger.info("[DynamicBuffer] Replace train_dataloader with BufferedDataLoader")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)

        Note: Judger (Role.Rollout) is created in a separate process to avoid
        vLLM sleep mode conflicts with ActorRollout. Both can share GPU pool
        via sleep/wake scheduling.
        """
        self.use_judger = self.config.judger.enable is True
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        self.judger_cls = None  # Store judger separately - will be created in its own process

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # Store judger configuration separately to create in its own process
        if self.use_judger:
            self.judger_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
            self.judger_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Rollout], config=self.config.judger, role="rollout"
            )

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        # Create colocated workers for standard roles (actor_rollout, critic, ref, rm)
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Create judger in a SEPARATE process to avoid vLLM sleep mode conflicts
        # This allows both ActorRollout and Judger to use sleep mode on the same GPUs
        if self.use_judger:
            judger_wg_dict = self.ray_worker_group_cls(
                resource_pool=self.judger_resource_pool,
                ray_cls_with_init=self.judger_cls,
                **wg_kwargs,
            )
            spawn_wg = judger_wg_dict.spawn(prefix_set=["rollout"])
            all_wg["rollout"] = judger_wg_dict

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # @zhonghao: add self.jd_wg for reward_fn calling
        if self.use_judger:
            self.jd_wg = all_wg["rollout"]
            self.jd_wg.init_model()
        else:
            self.jd_wg = None

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                dataloader_last_step = getattr(self.train_dataloader, "is_last_step", False)
                final_epoch = epoch >= self.config.trainer.total_epochs - 1
                is_last_step = self.global_steps >= self.total_training_steps or (dataloader_last_step and final_epoch)
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # TODO: @zhonghao, called right after rollout phase, but ideally should be overlapped with logp computation
                        # for jd_wg, async behavior is defined by self
                        if self.reward_fn is not None:
                            self.reward_fn.prepare(batch, self.processor, self.jd_wg)

                        # judge of a real vLLM Engine or just a client
                        if self.config.reward_model.launch_reward_fn_async:
                            # FIXME: @zhonghao: force re-create the module to avoid serialization failure

                            # future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        record_tensor = None
                        if reward_extra_infos_dict:
                            raw_reward_tensor = reward_extra_infos_dict.pop("raw_reward_tensor", None)
                            if raw_reward_tensor is not None:
                                record_tensor = raw_reward_tensor
                            else:
                                record_tensor = batch.batch["token_level_scores"]

                        # Update dynamic buffer with rewards if enabled
                        if self.dynamic_buffer is not None:
                            # Compute sequence-level rewards (sum over tokens)
                            seq_rewards = record_tensor.sum(dim=-1)
                            self.dynamic_buffer.add_samples(batch, seq_rewards)
                            report_fn = getattr(self.train_dataloader, "report_batch_reward", None)
                            if callable(report_fn):
                                report_fn(seq_rewards)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            reason_scores = reward_extra_infos_dict.get("reason_consistency_score")
                            if reason_scores:
                                metrics["reward/reason_consistency/mean"] = float(np.mean(reason_scores))
                                metrics["reward/reason_consistency/max"] = float(np.max(reason_scores))
                                metrics["reward/reason_consistency/min"] = float(np.min(reason_scores))
                            repetition_scores = reward_extra_infos_dict.get("repetition_penalty")
                            if repetition_scores:
                                metrics["reward/repetition_penalty/mean"] = float(np.mean(repetition_scores))
                                metrics["reward/repetition_penalty/max"] = float(np.max(repetition_scores))
                                metrics["reward/repetition_penalty/min"] = float(np.min(repetition_scores))

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # Add buffer metrics if enabled
                if self.dynamic_buffer is not None:
                    buffer_info = self.dynamic_buffer.get_buffer_info()
                    metrics.update(buffer_info)
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    # def fit(self):
    #     """
    #     The training loop of PPO.
    #     The driver process only need to call the compute functions of the worker group through RPC
    #     to construct the PPO dataflow.
    #     The light-weight advantage computation is done on the driver process.
    #     """
    #     from omegaconf import OmegaConf

    #     from verl.utils.tracking import Tracking

    #     logger = Tracking(
    #         project_name=self.config.trainer.project_name,
    #         experiment_name=self.config.trainer.experiment_name,
    #         default_backend=self.config.trainer.logger,
    #         config=OmegaConf.to_container(self.config, resolve=True),
    #     )

    #     self.global_steps = 0
    #     self.gen_steps = 0

    #     # load checkpoint before doing anything
    #     self._load_checkpoint()

    #     # perform validation before training
    #     # currently, we only support validation using the reward_function.
    #     if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
    #         val_metrics = self._validate()
    #         assert val_metrics, f"{val_metrics=}"
    #         pprint(f"Initial validation metrics: {val_metrics}")
    #         logger.log(data=val_metrics, step=self.global_steps)
    #         if self.config.trainer.get("val_only", False):
    #             return

    #     # add tqdm
    #     progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    #     # we start from step 1
    #     self.global_steps += 1
    #     self.gen_steps += 1
    #     last_val_metrics = None

    #     timing_raw = defaultdict(float)
    #     batch = None
    #     num_prompt_in_batch = 0
    #     num_gen_batches = 0

    #     prev_step_profile = False
    #     curr_step_profile = (
    #         self.global_steps in self.config.global_profiler.steps
    #         if self.config.global_profiler.steps is not None
    #         else False
    #     )
    #     next_step_profile = False

    #     for epoch in range(self.config.trainer.total_epochs):
    #         for batch_dict in self.train_dataloader:
    #             metrics = {}
    #             timing_raw = {}

    #             with marked_timer("start_profile", timing_raw):
    #                 self._start_profiling(
    #                     not prev_step_profile and curr_step_profile
    #                     if self.config.global_profiler.profile_continuous_steps
    #                     else curr_step_profile
    #                 )

    #             new_batch: DataProto = DataProto.from_single_dict(batch_dict)
    #             num_gen_batches += 1

    #             # pop those keys for generation
    #             if "multi_modal_data" in new_batch.non_tensor_batch.keys():
    #                 gen_batch = new_batch.pop(
    #                     batch_keys=["input_ids", "attention_mask", "position_ids"],
    #                     non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
    #                 )
    #             else:
    #                 gen_batch = new_batch.pop(
    #                     batch_keys=["input_ids", "attention_mask", "position_ids"],
    #                     non_tensor_batch_keys=["raw_prompt_ids"],
    #                 )
    #             gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

    #             is_last_step = self.gen_steps >= self.total_training_steps

    #             with marked_timer("step", timing_raw):
    #                 # generate a batch
    #                 with marked_timer("gen", timing_raw, "red"):
    #                     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
    #                     timing_raw.update(gen_batch_output.meta_info["timing"])
    #                     gen_batch_output.meta_info.pop("timing", None)

    #                 if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
    #                     with marked_timer("gen_max", timing_raw, "red"):
    #                         gen_baseline_batch = deepcopy(gen_batch)
    #                         gen_baseline_batch.meta_info["do_sample"] = False
    #                         gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

    #                         new_batch = new_batch.union(gen_baseline_output)
    #                         reward_baseline_tensor = self.reward_fn(new_batch)
    #                         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

    #                         new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

    #                         new_batch.batch["reward_baselines"] = reward_baseline_tensor

    #                         del gen_baseline_batch, gen_baseline_output

    #                 new_batch.non_tensor_batch["uid"] = np.array(
    #                     [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
    #                 )
    #                 # repeat to align with repeated responses in rollout
    #                 new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
    #                 new_batch = new_batch.union(gen_batch_output)

    #                 # TODO: @zhonghao, called right after rollout phase, but ideally should be overlapped with logp computation
    #                 if self.reward_fn is not None:
    #                     self.reward_fn.prepare(new_batch, self.processor, self.jd_wg)

    #                 with marked_timer("reward", timing_raw, "yellow"):
    #                     # compute scores. Support both model and function-based.
    #                     # We first compute the scores using reward model. Then, we call reward_fn to combine
    #                     # the results from reward model and rule-based results.
    #                     if self.use_rm:
    #                         # we first compute reward model score
    #                         reward_tensor = self.rm_wg.compute_rm_score(new_batch)
    #                         new_batch = new_batch.union(reward_tensor)

    #                     # we combine with rule-based rm
    #                     reward_extra_infos_dict: dict[str, list]
    #                     try:
    #                         reward_result = self.reward_fn(new_batch, return_dict=True)
    #                         reward_tensor = reward_result["reward_tensor"]
    #                         reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    #                     except Exception as e:
    #                         print(f"Error in reward_fn: {e}")
    #                         reward_tensor = self.reward_fn(new_batch)
    #                         reward_extra_infos_dict = {}

    #                     new_batch.batch["token_level_scores"] = reward_tensor

    #                     if reward_extra_infos_dict:
    #                         new_batch.non_tensor_batch.update(
    #                             {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
    #                         )

    #                     # compute rewards. apply_kl_penalty if available
    #                     if self.config.algorithm.use_kl_in_reward:
    #                         new_batch, kl_metrics = apply_kl_penalty(
    #                             new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
    #                         )
    #                         metrics.update(
    #                             kl_metrics
    #                         )  # TODO: This will be cleared if we use multiple genenration batches
    #                     else:
    #                         new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

    #                 if not self.config.algorithm.filter_groups.enable:
    #                     batch = new_batch
    #                 else:  # NOTE: When prompts after filtering is less than train batch size,
    #                     # we skip to the next generation batch
    #                     metric_name = self.config.algorithm.filter_groups.metric
    #                     if metric_name == "seq_final_reward":
    #                         # Turn to numpy for easier filtering
    #                         new_batch.non_tensor_batch["seq_final_reward"] = (
    #                             new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
    #                         )
    #                     elif metric_name == "seq_reward":
    #                         new_batch.non_tensor_batch["seq_reward"] = (
    #                             new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
    #                         )

    #                     # Collect the sequence reward for each trajectory
    #                     prompt_uid2metric_vals = defaultdict(list)
    #                     for uid, metric_val in zip(
    #                         new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
    #                     ):
    #                         prompt_uid2metric_vals[uid].append(metric_val)

    #                     prompt_uid2metric_std = {}
    #                     for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
    #                         prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

    #                     kept_prompt_uids = [
    #                         uid
    #                         for uid, std in prompt_uid2metric_std.items()
    #                         if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
    #                     ]
    #                     num_prompt_in_batch += len(kept_prompt_uids)

    #                     kept_traj_idxs = []
    #                     for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
    #                         if traj_from_prompt_uid in kept_prompt_uids:
    #                             kept_traj_idxs.append(idx)

    #                     new_batch = new_batch[kept_traj_idxs]
    #                     batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

    #                     prompt_bsz = self.config.data.train_batch_size
    #                     if num_prompt_in_batch < prompt_bsz:
    #                         print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
    #                         max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    #                         if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
    #                             print(f"{num_gen_batches=}. Keep generating...")
    #                             progress_bar.update(1)
    #                             self.gen_steps += 1
    #                             continue
    #                         else:
    #                             raise ValueError(
    #                                 f"{num_gen_batches=} >= {max_num_gen_batches=}."
    #                                 + " Generated too many. Please check if your data are too difficult."
    #                                 + " You could also try set max_num_gen_batches=0 to enable endless trials."
    #                             )
    #                     else:
    #                         # Align the batch
    #                         traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    #                         batch = batch[:traj_bsz]

    #                 # === Updating ===

    #                 batch.batch["response_mask"] = compute_response_mask(batch)

    #                 # Balance the number of valid tokens across DP ranks.
    #                 # NOTE: This usually changes the order of data in the `batch`,
    #                 # which won't affect the advantage calculation (since it's based on uid),
    #                 # but might affect the loss calculation (due to the change of mini-batching).
    #                 # TODO: Decouple the DP balancing and mini-batching.
    #                 if self.config.trainer.balance_batch:
    #                     self._balance_batch(batch, metrics=metrics)

    #                 # compute global_valid tokens
    #                 batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    #                 # recompute old_log_probs
    #                 with marked_timer("old_log_prob", timing_raw, "blue"):
    #                     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    #                     entropys = old_log_prob.batch["entropys"]
    #                     response_masks = batch.batch["response_mask"]
    #                     loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    #                     entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
    #                     old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
    #                     metrics.update(old_log_prob_metrics)
    #                     old_log_prob.batch.pop("entropys")

    #                     # TODO: @zhonghao, to avoid causing union error, pop old_log_prob since it already exists in batch
    #                     batch.batch.pop("old_log_probs", None)
    #                     batch.meta_info.pop("temperature", None)

    #                     batch = batch.union(old_log_prob)

    #                 if self.use_reference_policy:
    #                     # compute reference log_prob
    #                     with marked_timer("ref", timing_raw, "olive"):
    #                         ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    #                         batch = batch.union(ref_log_prob)

    #                 # compute values
    #                 if self.use_critic:
    #                     with marked_timer("values", timing_raw, "cyan"):
    #                         values = self.critic_wg.compute_values(batch)
    #                         batch = batch.union(values)

    #                 with marked_timer("adv", timing_raw, "brown"):
    #                     # compute advantages, executed on the driver process
    #                     norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
    #                     batch = compute_advantage(
    #                         batch,
    #                         adv_estimator=self.config.algorithm.adv_estimator,
    #                         gamma=self.config.algorithm.gamma,
    #                         lam=self.config.algorithm.lam,
    #                         num_repeat=self.config.actor_rollout_ref.rollout.n,
    #                         norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    #                     )

    #                 # update critic
    #                 if self.use_critic:
    #                     with marked_timer("update_critic", timing_raw, "pink"):
    #                         critic_output = self.critic_wg.update_critic(batch)
    #                     critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    #                     metrics.update(critic_output_metrics)

    #                 # implement critic warmup
    #                 if self.config.trainer.critic_warmup <= self.global_steps:
    #                     # update actor
    #                     with marked_timer("update_actor", timing_raw, "red"):
    #                         actor_output = self.actor_rollout_wg.update_actor(batch)
    #                     actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    #                     metrics.update(actor_output_metrics)

    #                 # validate
    #                 if (
    #                     self.val_reward_fn is not None
    #                     and self.config.trainer.test_freq > 0
    #                     and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
    #                 ):
    #                     with marked_timer("testing", timing_raw, "green"):
    #                         val_metrics: dict = self._validate()
    #                         if is_last_step:
    #                             last_val_metrics = val_metrics
    #                     metrics.update(val_metrics)

    #                 if self.config.trainer.save_freq > 0 and (
    #                     is_last_step or self.global_steps % self.config.trainer.save_freq == 0
    #                 ):
    #                     with marked_timer("save_checkpoint", timing_raw, "green"):
    #                         self._save_checkpoint()

    #             with marked_timer("stop_profile", timing_raw):
    #                 if do_profile:
    #                     self.actor_rollout_wg.stop_profile()
    #                     if self.use_reference_policy:
    #                         self.ref_policy_wg.stop_profile()
    #                     if self.use_critic:
    #                         self.critic_wg.stop_profile()
    #                     if self.use_rm:
    #                         self.rm_wg.stop_profile()

    #             # collect metrics
    #             metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
    #             metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    #             # TODO: implement actual tflpo and theoretical tflpo
    #             n_gpus = self.resource_pool_manager.get_n_gpus()
    #             metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
    #             timing_raw = defaultdict(float)  # clear timing

    #             metrics["train/num_gen_batches"] = num_gen_batches
    #             batch = None
    #             num_prompt_in_batch = 0
    #             num_gen_batches = 0

    #             # TODO: make a canonical logger that supports various backend
    #             logger.log(data=metrics, step=self.global_steps)

    #             if is_last_step:
    #                 pprint(f"Final validation metrics: {last_val_metrics}")
    #                 progress_bar.close()
    #                 return

    #             progress_bar.update(1)
    #             self.global_steps += 1
    #             self.gen_steps += 1

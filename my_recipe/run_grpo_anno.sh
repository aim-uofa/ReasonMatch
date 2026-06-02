#!/usr/bin/env bash
set -euxo pipefail
# Alternate GRPO launcher with a smaller rollout batch.

project_name='reasonmatch_grpo'
exp_name='GRPO-small-rollout'

dataset_name=AnnoRawDataset
# MultiRawDataset
# 'AnnoRawDataset'
# dataset_name='DL3DV'
# dataset_name='PointCDataset'
# custom_reward_function='point_desc_score'

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
max_prompt_length=$((1024 * 5))
max_response_length=$((1024 * 6))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=2

custom_reward_function='anno_score'
val_before_train=False
dataset_path=${DATASET_PATH:?Set DATASET_PATH to a dataset dir or my_recipe/datasets.json}
enable_jd=False

enable_buffer=True
buffer_size=512
task_switch_metric='mean_reward'
min_samples_for_switch=256

# switching task here
task_mode='matching'  # matching / grounding / dynamic


#########################
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"

#########################


adv_estimator=grpo

# entropy_coeff
entropy_coeff=1e-3

use_kl_in_reward=False
kl_coef=0.0

# use_kl_loss set to True for GRPO
use_kl_loss=True
kl_loss_coef=0.005


clip_ratio_low=0.2
clip_ratio_high=0.28



enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"


enable_filter_groups=False
train_prompt_bsz=2
n_resp_per_prompt=4
train_prompt_mini_bsz=1

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1
n_gpus_per_node=2
# ${NNODES:-16}

# Paths
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the base model directory}
TRAIN_FILE=${dataset_path}
TEST_FILE=${dataset_path}
CKPTS_DIR=${CKPTS_DIR:?Set CKPTS_DIR to a checkpoint output directory}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7



    # reward_model.reward_manager=my_dapo \

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- 
python3 -m my_recipe.main_dcrl \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy='fsdp' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.custom_cls.name=${dataset_name} \
    judger.enable=${enable_jd} \
    dynamic_buffer.enable=${enable_buffer} \
    dynamic_buffer.buffer_size=${buffer_size} \
    dynamic_buffer.task_mode=${task_mode} \
    dynamic_buffer.task_switch_metric=${task_switch_metric} \
    dynamic_buffer.min_samples_for_switch=${min_samples_for_switch} \
    dynamic_buffer.buffered_loader.enable_overlap_bins=True \
    dynamic_buffer.buffered_loader.overlap_bin_count=4 \
    dynamic_buffer.buffered_loader.promotion_reward_threshold=1.9 \
    dynamic_buffer.buffered_loader.promotion_window=3 \
    dynamic_buffer.buffered_loader.min_batches_per_bin=6 \
    custom_reward_function.name=${custom_reward_function} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

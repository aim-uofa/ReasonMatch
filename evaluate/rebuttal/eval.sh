#!/bin/bash
log_name=$1

python run_eval.py \
  --testset_root ${TESTSET_ROOT:?Set TESTSET_ROOT to your OOD dataset path} \
  --output_dir ./eval_results \
  --model_name $log_name \
  --runner vllm \
  --model_id auto \
  --temperature 0.6 \
  --max_tokens 8192 \
  --concurrency 32 \
  --base_url ${VLLM_BASE_URL:?Set VLLM_BASE_URL} \
  --api_key EMPTY

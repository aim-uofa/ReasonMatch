#!/bin/bash
log_name=$1

python chk_label_cnt.py \
  --testset_root ${TESTSET_ROOT:?Set TESTSET_ROOT to your OOD dataset path} \
  --output_dir ./eval_visibility_results \
  --model_name $log_name \
  --runner vllm \
  --model_id auto \
  --temperature 0.6 \
  --max_tokens 128 \
  --concurrency 16 \
  --base_url ${VLLM_BASE_URL:-http://localhost:8000/v1} \
  --api_key EMPTY

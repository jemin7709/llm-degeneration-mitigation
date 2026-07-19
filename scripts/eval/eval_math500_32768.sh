#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="3"
export MATH500_PROMPT_SOURCE
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-qwen3-0.6b}"

mkdir -p logs

uv run lighteval vllm \
  "model_name=${MODEL_NAME},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_length=32768,generation_parameters={temperature:0.6,top_p:0.95,top_k:20,max_new_tokens:32768}" \
  "math500_${MATH500_PROMPT_SOURCE}|0" \
  --custom-tasks src/suppress_attack/evals/math500.py \
  --output-dir "outputs/lighteval/${MODEL_TAG}-math500-n4-${MATH500_PROMPT_SOURCE}-32768" \
  --save-details

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU:-3}"

mkdir -p logs

run_model() {
  model_name="$1"
  model_tag="$2"
  output_dir="outputs/ifeval/${model_tag}-16384"
  log_file="logs/${model_tag}-ifeval.log"

  echo "Running ifeval for ${model_tag} on GPU ${CUDA_VISIBLE_DEVICES}"
  echo "  output=${output_dir}"
  echo "  log=${log_file}"

  uv run lighteval vllm \
    "model_name=${model_name},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_length=24576,generation_parameters={max_new_tokens:16384}" \
    "ifeval|0" \
    --output-dir "${output_dir}" \
    --save-details \
    >"${log_file}" 2>&1
}

run_model \
  "outputs/merged_models/qwen3-0.6b-dpo-data_filtering" \
  "qwen3-0.6b-dpo-data_filtering"

run_model \
  "outputs/merged_models/qwen3-0.6b-dpo-math_solution" \
  "qwen3-0.6b-dpo-math_solution"

run_model \
  "outputs/merged_models/qwen3-0.6b-dpo-naive-20260525-231829-naive-g3" \
  "qwen3-0.6b-dpo-naive-20260525-231829-naive-g3"

run_model \
  "Qwen/Qwen3-0.6B" \
  "qwen3-0.6b-base"

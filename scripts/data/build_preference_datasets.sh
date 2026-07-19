#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/preference_datasets}"

build_split() {
  local split="$1"
  shift
  local result_dir="outputs/attack/dpo/qwen3-0.6b_math_${split}_multi_prompt_gcg_b200_s23"
  local math_data="data/math/${split}.jsonl"

  for setting in "$@"; do
    uv run src/suppress_attack/dpo/build_preference_datasets.py \
      "$result_dir" \
      --setting "$setting" \
      --math-data "$math_data" \
      --output-root "$OUTPUT_ROOT"
  done
}

build_split train naive math_solution data_filtering

#!/usr/bin/env bash
set -euo pipefail
export MODEL_NAME="outputs/merged_models/qwen3-0.6b-dpo-math_solution"
export MODEL_TAG="qwen3-0.6b-dpo-math_solution"
scripts/eval/eval_math500_32768.sh

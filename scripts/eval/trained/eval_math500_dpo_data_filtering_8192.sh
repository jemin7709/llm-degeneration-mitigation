#!/usr/bin/env bash
set -euo pipefail
export MODEL_NAME="outputs/merged_models/qwen3-0.6b-dpo-data_filtering"
export MODEL_TAG="qwen3-0.6b-dpo-data_filtering"
scripts/eval/eval_math500_8192.sh

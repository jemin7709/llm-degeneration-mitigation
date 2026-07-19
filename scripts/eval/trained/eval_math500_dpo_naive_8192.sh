#!/usr/bin/env bash
set -euo pipefail
export MODEL_NAME="outputs/merged_models/qwen3-0.6b-dpo-naive"
export MODEL_TAG="qwen3-0.6b-dpo-naive"
scripts/eval/eval_math500_8192.sh

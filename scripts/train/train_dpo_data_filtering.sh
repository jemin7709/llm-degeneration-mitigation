#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SETTING="data_filtering"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
DATASET_DIR="${DATASET_DIR:-data/processed/${SETTING}}"
RUN_NAME="${RUN_NAME:-qwen3-0.6b-dpo-${SETTING}-${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dpo/${RUN_NAME}}"

uv run python src/suppress_attack/dpo/train.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  "$@"

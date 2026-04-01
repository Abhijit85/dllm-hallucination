#!/usr/bin/env bash
set -euo pipefail

WATCH_PID=4148260
ROOT="/mnt/data2/achakr40/dllm-hallucination"
RUN_LOG="/tmp/parade_commonsenseqa_dream7b.log"

while true; do
  if ps -p "$WATCH_PID" > /dev/null 2>&1; then
    echo "$(date -Iseconds) waiting for pid=${WATCH_PID} to exit"
    sleep 60
    continue
  fi

  echo "$(date -Iseconds) pid=${WATCH_PID} finished; launching Dream CommonsenseQA on GPU 7"
  cd "$ROOT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7 \
    .venv/bin/python scripts/run_detection_auroc.py \
    --model_path Dream-org/Dream-v0-Instruct-7B \
    --dataset commonsenseqa \
    --n_samples 500 \
    --n_paths 8 \
    --num_steps 32 \
    --gen_len 32 \
    --full_pipeline \
    --output results/parade_commonsenseqa_dream7b >> "$RUN_LOG" 2>&1
  exit $?
done

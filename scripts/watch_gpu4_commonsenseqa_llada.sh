#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=4
GPU_UUID="GPU-dcbef112-2705-755d-4602-9b9bc68e44ae"
ROOT="/mnt/data2/achakr40/dllm-hallucination"
RUN_LOG="/tmp/parade_commonsenseqa_llada.log"

while true; do
  gpu_line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==4 {print $0}')
  util=$(printf '%s\n' "$gpu_line" | awk -F', ' '{print $2}')
  mem=$(printf '%s\n' "$gpu_line" | awk -F', ' '{print $3}')
  gpu_apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$GPU_UUID" | wc -l || true)

  echo "$(date -Iseconds) gpu${GPU_INDEX} util=${util:-NA} mem=${mem:-NA}MiB compute_apps=${gpu_apps}"

  if [[ -n "${util}" ]] && [[ -n "${mem}" ]] && [[ "${util}" -le 5 ]] && [[ "${mem}" -le 1024 ]] && [[ "${gpu_apps}" -eq 0 ]]; then
    echo "$(date -Iseconds) launching LLaDA CommonsenseQA on GPU ${GPU_INDEX}"
    cd "$ROOT"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU_INDEX} \
      .venv/bin/python scripts/run_detection_auroc.py \
      --model_path GSAI-ML/LLaDA-8B-Instruct \
      --dataset commonsenseqa \
      --n_samples 500 \
      --n_paths 8 \
      --num_steps 32 \
      --gen_len 32 \
      --full_pipeline \
      --output results/parade_commonsenseqa_llada >> "$RUN_LOG" 2>&1
    exit $?
  fi

  sleep 60
done

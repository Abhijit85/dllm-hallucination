#!/bin/bash
# C4 — 3-Seed Variance for Tables 1 & 2
# ======================================
# Paper claims "mean ± std over 3 runs" but Table 1 has no ± values.
# This script runs the full pipeline 3× with different seeds.
#
# Estimated time: ~12-24h total on 4×A100
# Strategy: Run 3 seeds in parallel across GPUs
#
# Usage:
#   bash scripts/c4_three_seeds.sh

set -euo pipefail

cd /mnt/data2/achakr40/dllm-hallucination
source .venv/bin/activate

MODEL_LLADA="/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct"
MODEL_DREAM="DREAM-org/DREAM-v0-Instruct-7B"
SEEDS=(42 137 2024)
DATASETS=("triviaqa" "hotpotqa" "commonsenseqa")

echo "════════════════════════════════════════════════════════"
echo "  C4: 3-Seed Variance Runs"
echo "  Seeds: ${SEEDS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  $(date)"
echo "════════════════════════════════════════════════════════"

# ── LLaDA runs ────────────────────────────────────────────────────────────────
# Run 3 seeds × 3 datasets = 9 jobs
# Spread across GPUs: seed0→GPU0, seed1→GPU1, seed2→GPU2

for si in 0 1 2; do
    SEED=${SEEDS[$si]}
    GPU=$si

    for ds in "${DATASETS[@]}"; do
        OUT="results/c4_seeds/llada_${ds}_seed${SEED}"

        if [ -f "$OUT/raw_results.jsonl" ]; then
            echo "  SKIP: $OUT already exists"
            continue
        fi

        echo ""
        echo "  Launching: LLaDA / $ds / seed=$SEED → GPU $GPU"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$GPU \
        python scripts/run_detection_auroc.py \
            --model_path "$MODEL_LLADA" \
            --dataset "$ds" \
            --n_samples 500 \
            --n_paths 8 \
            --num_steps 32 \
            --gen_len 64 \
            --seed "$SEED" \
            --full_pipeline \
            --output "$OUT" &

        sleep 10
    done

    wait
    echo "  Seed $SEED complete."
done

echo ""
echo "All LLaDA seeds done. Starting Dream..."

# ── Dream runs ────────────────────────────────────────────────────────────────

for si in 0 1 2; do
    SEED=${SEEDS[$si]}
    GPU=$((si + 3))  # GPUs 3,4,5

    for ds in "${DATASETS[@]}"; do
        OUT="results/c4_seeds/dream_${ds}_seed${SEED}"

        if [ -f "$OUT/raw_results.jsonl" ]; then
            echo "  SKIP: $OUT already exists"
            continue
        fi

        echo "  Launching: Dream / $ds / seed=$SEED → GPU $GPU"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$GPU \
        python scripts/run_detection_auroc.py \
            --model_path "$MODEL_DREAM" \
            --dataset "$ds" \
            --n_samples 500 \
            --n_paths 8 \
            --num_steps 32 \
            --gen_len 64 \
            --seed "$SEED" \
            --full_pipeline \
            --output "$OUT" &

        sleep 10
    done

    wait
    echo "  Dream seed $SEED complete."
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All seeds done. Run the aggregation script:"
echo "  python scripts/c4_aggregate_seeds.py"
echo "════════════════════════════════════════════════════════"

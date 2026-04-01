#!/bin/bash
# run_all.sh
# Master pipeline: full experiment + N-paths ablation, run sequentially.
#
# Usage:
#   bash scripts/run_all.sh \
#       --model_path /mnt/shared/.../GSAI-ML--LLaDA-8B-Instruct \
#       --data_path  /mnt/data2/.../external/RAGTruth/dataset \
#       --gpu        6

set -euo pipefail

MODEL_PATH=""
DATA_PATH=""
GPU="0"
FULL_SAMPLES=500
ABL_SAMPLES=100
N_STEPS=64
GEN_LEN=128
TOP_K_PERCENT=0.20
MIN_REFINE_STEPS=16
STEPS_PER_TOKEN=0.5
SEED=42
SKIP_FULL=0
SKIP_ABL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)     MODEL_PATH="$2";     shift 2 ;;
        --data_path)      DATA_PATH="$2";      shift 2 ;;
        --gpu)            GPU="$2";            shift 2 ;;
        --full_samples)   FULL_SAMPLES="$2";   shift 2 ;;
        --abl_samples)    ABL_SAMPLES="$2";    shift 2 ;;
        --n_steps)        N_STEPS="$2";        shift 2 ;;
        --gen_len)        GEN_LEN="$2";        shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --skip_full)      SKIP_FULL=1;         shift   ;;
        --skip_ablation)  SKIP_ABL=1;          shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$DATA_PATH" ]]; then
    echo "Usage: bash scripts/run_all.sh --model_path PATH --data_path PATH [--gpu N]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

exec > >(tee -a "$RESULTS_DIR/master.log") 2>&1

echo "========================================================"
echo "  DLLM Hallucination - Full Pipeline"
echo "  $(date)"
echo "  model : $MODEL_PATH"
echo "  data  : $DATA_PATH"
echo "  gpu   : $GPU  (CUDA_VISIBLE_DEVICES=$GPU)"
echo "  alloc : PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  out   : $RESULTS_DIR"
echo "========================================================"

if [[ $SKIP_FULL -eq 0 ]]; then
    echo ""
    echo "+------------------------------------------------------+"
    echo "|  Stage 1: Full Experiment                            |"
    echo "|  $FULL_SAMPLES samples - 8 paths - $N_STEPS steps                    |"
    echo "+------------------------------------------------------+"
    T0=$SECONDS

    python run_experiment.py \
        --model_id         "$MODEL_PATH" \
        --data_path        "$DATA_PATH" \
        --n_paths          8 \
        --num_steps        "$N_STEPS" \
        --gen_len          "$GEN_LEN" \
        --top_k_percent    "$TOP_K_PERCENT" \
        --steps_per_token  "$STEPS_PER_TOKEN" \
        --min_refine_steps "$MIN_REFINE_STEPS" \
        --max_samples      "$FULL_SAMPLES" \
        --seed             "$SEED" \
        --output_dir       "$RESULTS_DIR/full"

    echo "  Stage 1 done in $(( SECONDS - T0 ))s"
else
    echo "  Stage 1 skipped (--skip_full)"
fi

if [[ $SKIP_ABL -eq 0 ]]; then
    echo ""
    echo "+------------------------------------------------------+"
    echo "|  Stage 2: N-Paths Ablation                           |"
    echo "|  n in {1,2,4,8,16} - $ABL_SAMPLES samples each                |"
    echo "+------------------------------------------------------+"
    T0=$SECONDS

    python scripts/run_npaths_ablation.py \
        --model_id         "$MODEL_PATH" \
        --data_path        "$DATA_PATH" \
        --output_root      "$RESULTS_DIR/ablation" \
        --n_paths_list     1 2 4 8 16 \
        --num_steps        "$N_STEPS" \
        --gen_len          "$GEN_LEN" \
        --top_k_percent    "$TOP_K_PERCENT" \
        --steps_per_token  "$STEPS_PER_TOKEN" \
        --min_refine_steps "$MIN_REFINE_STEPS" \
        --max_samples      "$ABL_SAMPLES" \
        --seed             "$SEED"

    echo "  Stage 2 done in $(( SECONDS - T0 ))s"
else
    echo "  Stage 2 skipped (--skip_ablation)"
fi

echo ""
echo "+------------------------------------------------------+"
echo "|  Stage 3: Summary table                              |"
echo "+------------------------------------------------------+"

python scripts/summarize_results.py --results_dir "$RESULTS_DIR"

echo ""
echo "========================================================"
echo "  Pipeline complete  $(date)"
echo "  Results: $RESULTS_DIR"
echo "========================================================"

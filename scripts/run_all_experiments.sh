#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# OSCAR — Complete Experiment Runner
# Execute experiments in priority order for COLM 2026 submission
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/run_all_experiments.sh [--skip-gpu] [--only STAGE]
#
#   --skip-gpu    Skip GPU-intensive experiments (C3, C4, C5, M1, M4)
#   --only STAGE  Run only one stage: c1|c2|c6|h2|c5|c4|c3|h|m

set -euo pipefail

cd /mnt/data2/achakr40/dllm-hallucination

PYTHON_BIN="${PYTHON_BIN:-/mnt/data2/achakr40/dllm-hallucination/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python interpreter not found: $PYTHON_BIN"
    echo "Set PYTHON_BIN=/path/to/python or create .venv first."
    exit 1
fi

SKIP_GPU=0
ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gpu) SKIP_GPU=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        *) shift ;;
    esac
done

LLADA="/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct"
DATA="/mnt/data2/achakr40/dllm-hallucination/external/RAGTruth/dataset"

echo "════════════════════════════════════════════════════════"
echo "  OSCAR — Complete Experiment Suite"
echo "  $(date)"
echo "  Python: $PYTHON_BIN"
echo "  Skip GPU: $SKIP_GPU"
echo "════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# C1 — Fair AUROC (no GPU, ~2-4h API cost)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "c1" ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  C1: Fair AUROC — Judge eval on ALL baselines        │"
echo "│  Zero GPU. Needs OPENAI_API_KEY set.                 │"
echo "└─────────────────────────────────────────────────────┘"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "  WARNING: OPENAI_API_KEY not set. Skipping C1."
    echo "  Set it with: export OPENAI_API_KEY='sk-...'"
else
    "$PYTHON_BIN" scripts/c1_fair_auroc_judge.py \
        --results_base results/ \
        --output results/c1_fair_auroc/ \
        --model gpt-4o \
        --max_concurrent 20
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# C2 — Majority Vote Baseline (no GPU, ~1h)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "c2" ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  C2: Majority Vote Baseline                          │"
echo "│  Zero GPU. Post-processing existing chains.          │"
echo "└─────────────────────────────────────────────────────┘"

RESULT_DIRS=$(find results/ -name "raw_results.jsonl" -exec dirname {} \; | sort -u)
"$PYTHON_BIN" scripts/c2_majority_vote.py \
    --results_dirs $RESULT_DIRS \
    --output results/c2_majority_vote/
fi

# ─────────────────────────────────────────────────────────────────────────────
# C6 + EQ — Writing Only (no GPU, instant)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "c6" ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  C6 + EQ: Positioning Table & Equations              │"
echo "│  Zero GPU. Pure LaTeX generation.                    │"
echo "└─────────────────────────────────────────────────────┘"

"$PYTHON_BIN" scripts/c6_eq_writing.py --output results/c6_writing/
fi

# ─────────────────────────────────────────────────────────────────────────────
# H2 — Confident-but-Wrong (no GPU, ~1h)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "h2" ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  H2: Confident-but-Wrong Rate                        │"
echo "│  Zero GPU. Post-processing.                          │"
echo "└─────────────────────────────────────────────────────┘"

RESULT_DIRS=$(find results/ -name "raw_results.jsonl" -exec dirname {} \; | sort -u)
"$PYTHON_BIN" scripts/h2_confident_but_wrong.py \
    --results_dirs $RESULT_DIRS \
    --output results/h2_cbw/
fi

# ─────────────────────────────────────────────────────────────────────────────
# C5 — SelfCheckGPT-DLM (GPU, ~4-8h)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "c5" ]]; then
if [[ $SKIP_GPU -eq 0 ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  C5: SelfCheckGPT-DLM Baseline                       │"
echo "│  GPU required. ~4-8h per dataset.                    │"
echo "└─────────────────────────────────────────────────────┘"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
"$PYTHON_BIN" scripts/c5_selfcheck_dlm.py \
    --model_path "$LLADA" \
    --dataset triviaqa \
    --n_samples 500 \
    --n_independent 8 \
    --output results/c5_selfcheck_triviaqa/ &

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 \
"$PYTHON_BIN" scripts/c5_selfcheck_dlm.py \
    --model_path "$LLADA" \
    --dataset hotpotqa \
    --n_samples 500 \
    --n_independent 8 \
    --output results/c5_selfcheck_hotpotqa/ &

wait
echo "  C5 complete."
else
    echo "  SKIPPED C5 (--skip-gpu)"
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# C4 — 3-Seed Variance (GPU, ~12-24h)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "c4" ]]; then
if [[ $SKIP_GPU -eq 0 ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  C4: 3-Seed Variance                                 │"
echo "│  GPU required. ~12-24h total.                        │"
echo "└─────────────────────────────────────────────────────┘"

bash scripts/c4_three_seeds.sh
"$PYTHON_BIN" scripts/c4_aggregate_seeds.py --seeds_dir results/c4_seeds/
else
    echo "  SKIPPED C4 (--skip-gpu)"
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# C3 — Crystallization (GPU, ~4-8h)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "c3" ]]; then
if [[ $SKIP_GPU -eq 0 ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  C3: Hallucination Crystallization                   │"
echo "│  GPU required. ~4-8h.                                │"
echo "└─────────────────────────────────────────────────────┘"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 \
"$PYTHON_BIN" scripts/c3_crystallization.py \
    --model_path "$LLADA" \
    --dataset triviaqa \
    --n_samples 200 \
    --output results/c3_crystallization/
else
    echo "  SKIPPED C3 (--skip-gpu)"
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# H3-H6 — Post-Processing Analyses (no GPU)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "h" ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  H3-H6: Post-Processing Analyses                     │"
echo "│  Zero GPU.                                           │"
echo "└─────────────────────────────────────────────────────┘"

"$PYTHON_BIN" scripts/h3456_analyses.py \
    --results_base results/ \
    --output results/analyses/

"$PYTHON_BIN" scripts/h1_threshold_alpha.py \
    --results_dir results/parade_triviaqa \
    --output results/h1_alpha/
fi

# ─────────────────────────────────────────────────────────────────────────────
# M1, M4, M5 — Medium Priority (mixed GPU/no-GPU)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "m" ]]; then
echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│  M1/M4/M5: Medium Priority                           │"
echo "└─────────────────────────────────────────────────────┘"

"$PYTHON_BIN" scripts/m145_medium.py m5 \
    --results_base results/ \
    --output results/m5_bootstrap/

if [[ $SKIP_GPU -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/m145_medium.py m4 \
        --model_path "$LLADA" \
        --output results/m4_memory/

    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/m145_medium.py m1 \
        --model_path "$LLADA" \
        --data_path "$DATA" \
        --output results/m1_tr/ \
        --n_samples 100
fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo ""
echo "  Results summary:"
echo "    C1: results/c1_fair_auroc/"
echo "    C2: results/c2_majority_vote/"
echo "    C3: results/c3_crystallization/"
echo "    C4: results/c4_seeds/"
echo "    C5: results/c5_selfcheck_*/"
echo "    C6: results/c6_writing/"
echo "    H1: results/h1_alpha/"
echo "    H2: results/h2_cbw/"
echo "    H3-H6: results/analyses/"
echo "    M1: results/m1_tr/"
echo "    M4: results/m4_memory/"
echo "    M5: results/m5_bootstrap/"
echo "════════════════════════════════════════════════════════"

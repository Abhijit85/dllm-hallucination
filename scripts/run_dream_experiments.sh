#!/usr/bin/env bash
# scripts/run_dream_experiments.sh
# ---------------------------------
# Dream-7B PaRaDe DETECTION-ONLY runs using .venv_dream (transformers==4.46.2).
#
# DETECTION-ONLY -- no --full_pipeline, no --abductive
# Dream uses diffusion_generate() as a black box. Step-level remasking and
# abductive verification require direct loop access Dream does not expose.
#
# SETUP (one time)
#   python3 -m venv .venv_dream
#   .venv_dream/bin/pip install transformers==4.46.2 torch accelerate scikit-learn tqdm datasets
#
# USAGE
#   bash scripts/run_dream_experiments.sh smoke   # 5 samples
#   bash scripts/run_dream_experiments.sh full    # 500 samples

set -e
MODE=${1:-smoke}
VENV=".venv_dream/bin/python"
SNAP="/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae"

echo "=== Dream-7B Detection Runner (mode=$MODE) ==="

if [ ! -f "$VENV" ]; then
    echo "ERROR: .venv_dream not found."
    echo "  python3 -m venv .venv_dream"
    echo "  .venv_dream/bin/pip install transformers==4.46.2 torch accelerate scikit-learn tqdm datasets"
    exit 1
fi

TV=$($VENV -c "import transformers; print(transformers.__version__)" 2>/dev/null)
if [[ "$TV" != 4.46* ]]; then
    echo "ERROR: need transformers==4.46.2, found $TV"
    exit 1
fi
echo "transformers $TV OK"
echo ""

if [ "$MODE" = "smoke" ]; then N=5;   STEPS=32; PATHS=4; SUFFIX="smoke"
else                              N=500; STEPS=32; PATHS=8; SUFFIX="500"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

# TriviaQA -- GPU 6
CUDA_VISIBLE_DEVICES=6 $VENV scripts/run_detection_auroc.py \
    --model_path "$SNAP" --dataset triviaqa \
    --n_samples $N --n_paths $PATHS --num_steps $STEPS --gen_len 32 \
    --output "results/triviaqa_dream7b_${SUFFIX}/" &

# HotpotQA -- GPU 7
CUDA_VISIBLE_DEVICES=7 $VENV scripts/run_detection_auroc.py \
    --model_path "$SNAP" --dataset hotpotqa \
    --n_samples $N --n_paths $PATHS --num_steps $STEPS --gen_len 32 \
    --output "results/hotpotqa_dream7b_${SUFFIX}/" &

# Let the first two runs finish before launching the third on GPU 6.
wait

# CommonsenseQA -- GPU 6
CUDA_VISIBLE_DEVICES=6 $VENV scripts/run_detection_auroc.py \
    --model_path "$SNAP" --dataset commonsenseqa \
    --n_samples $N --n_paths $PATHS --num_steps $STEPS --gen_len 20 \
    --output "results/csqa_dream7b_${SUFFIX}/" &

wait
echo ""
echo "=== Dream-7B $MODE runs complete ==="
echo "Check: ls results/*_dream7b_${SUFFIX}/"

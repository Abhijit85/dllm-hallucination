# Reproducibility Guide

This guide describes the minimum information needed to reproduce an experiment
run from this repository.

## Environment

Record:

```bash
git rev-parse HEAD
python --version
pip freeze
```

For GPU runs, also record:

```bash
nvidia-smi
```

## Data

The loader uses the Hugging Face dataset `wanderkid/RAGTruth`. Record the split,
task filter, and any sample cap used in the command.

Example:

```bash
python run_experiment.py \
  --split test \
  --task_type QA \
  --max_samples 500 \
  --output_dir results/ragtruth_qa_500
```

## Model

The default model is:

```text
GSAI-ML/LLaDA-8B-Instruct
```

Record any model change through `--model_id`, plus precision, GPU type, and
whether quantization packages such as `bitsandbytes` were used.

## Randomness

Use `--seed` for path generation. Each denoising path receives a deterministic
offset from the base seed. For full determinism, also report GPU, CUDA, PyTorch,
and driver versions because low-level kernels may still vary.

## Output Files

Keep at least:

```text
raw_results.jsonl
aggregate.json
threshold_ablation.json
```

`raw_results.jsonl` can be large and may contain generated text. Check dataset
and output licensing before redistribution.

## Suggested Main Run

```bash
python run_experiment.py \
  --split test \
  --n_paths 8 \
  --num_steps 64 \
  --entropy_threshold 0.5 \
  --refine_steps 16 \
  --max_samples 500 \
  --ablate_thresholds \
  --seed 42 \
  --output_dir results/main_seed42
```

## Suggested Ablations

| Ablation | Values |
|---|---|
| Paths | `1`, `2`, `4`, `8`, `16` |
| Entropy threshold | `0.3`, `0.5`, `0.7` |
| Refinement steps | `4`, `8`, `16`, `32` |
| Task type | `QA`, `Summary`, `Data2txt` |

Report mean metrics and confidence intervals across seeds when compute budget
allows.

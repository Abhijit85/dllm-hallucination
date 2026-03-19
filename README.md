# DLLM Hallucination Reduction

Experiment code for studying hallucination in diffusion language models with
parallel denoising paths, disagreement-based uncertainty, and targeted
re-masking refinement on RAGTruth.

## Hypothesis Table

| ID | Hypothesis | Validation Signal |
|---|---|---|
| H1 | Path disagreement entropy is higher on hallucinated tokens than grounded tokens. | Mean Spearman rho between entropy and labels is positive. |
| H2 | Re-masking high-entropy tokens improves grounding. | Mean `refinement_delta` is positive on hallucinated samples. |
| H3 | Hybrid or random demasking provides a better exploration/coherence tradeoff than pure learned order. | Compare aggregate metrics across `DemaskingOrder` variants and path mixes. |
| H4 | More parallel paths improve coverage of hallucinated regions up to a diminishing-return point. | Compare threshold ablations and token F1 across `--n_paths`. |

## Setup

```bash
pip install -e .
```

For local development:

```bash
pip install -e ".[dev]"
```

For GPU-backed LLaDA runs:

```bash
pip install -e ".[gpu]"
```

## Experiment Guide

```bash
# Quick QA smoke run
python run_experiment.py \
    --max_samples 20 \
    --task_type QA \
    --n_paths 4 \
    --num_steps 32 \
    --output_dir results/quick_test

# Main run with threshold sweep
python run_experiment.py \
    --n_paths 8 \
    --num_steps 64 \
    --entropy_threshold 0.5 \
    --refine_steps 16 \
    --max_samples 500 \
    --ablate_thresholds \
    --output_dir results/full
```

Key ablations:

| Ablation | Variable | Values |
|---|---|---|
| Number of paths | `--n_paths` | 1, 2, 4, 8, 16 |
| Entropy cutoff | `--entropy_threshold` | 0.3, 0.5, 0.7 |
| Refinement steps | `--refine_steps` | 4, 8, 16, 32 |
| Task slice | `--task_type` | `QA`, `Summary`, `Data2txt` |
| Remask fraction | `remask_fraction` in strategy | 0.25, 0.5, 0.75, 1.0 |

## Repository Layout

```text
dllm-hallucination/
├── .github/workflows/ci.yml
├── .gitignore
├── README.md
├── setup.py
├── requirements.txt
├── run_experiment.py
├── data/
│   ├── __init__.py
│   └── ragtruth_loader.py
├── models/
│   ├── __init__.py
│   └── llada_harness.py
├── strategies/
│   ├── __init__.py
│   └── parallel_remask.py
├── eval/
│   ├── __init__.py
│   └── metrics.py
└── tests/
    ├── __init__.py
    └── test_core.py
```

## Outputs

```text
results/
├── raw_results.jsonl
├── aggregate.json
└── threshold_ablation.json
```

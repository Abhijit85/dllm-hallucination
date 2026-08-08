# DLLM Hallucination Reduction

Research code for studying whether parallel denoising paths can expose and
reduce hallucination in diffusion language models. The current implementation
uses LLaDA-style masked diffusion generation, path-disagreement entropy, and
targeted re-masking refinement on RAGTruth.

This repository is organized as an academic artifact: hypotheses are explicit,
experiment commands are reproducible, outputs are structured for analysis, and
limitations are documented alongside the implementation.

## Abstract

Diffusion language models generate text by iteratively denoising masked tokens.
This project investigates the hypothesis that different demasking orders reveal
uncertainty that is useful for hallucination detection and mitigation. For each
RAGTruth sample, the code runs multiple denoising paths, computes token-level
entropy across final path outputs, flags high-disagreement positions, and
re-denoises those positions under source conditioning. The main validation
signals are token-level hallucination F1, source-overlap grounding proxies,
refinement delta, and Spearman correlation between disagreement entropy and
RAGTruth hallucination labels.

## Research Questions

| ID | Question | Operational Test |
|---|---|---|
| RQ1 | Are hallucinated tokens associated with higher path disagreement than grounded tokens? | Mean Spearman correlation between token entropy and hallucination labels. |
| RQ2 | Does targeted re-masking of high-disagreement tokens improve grounding? | Positive mean `refinement_delta` after source-conditioned refinement. |
| RQ3 | Do random or hybrid demasking orders improve exploration compared with learned-order demasking? | Compare aggregate metrics across `DemaskingOrder` variants and path mixes. |
| RQ4 | How many parallel paths are needed before returns diminish? | Sweep `--n_paths` and compare token F1, flagged-token precision/recall, and runtime. |

## Repository Layout

```text
dllm-hallucination/
|-- README.md
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- LICENSE
|-- setup.py
|-- requirements.txt
|-- run_experiment.py
|-- data/
|   |-- ragtruth_loader.py
|-- models/
|   |-- llada_harness.py
|-- strategies/
|   |-- parallel_remask.py
|-- eval/
|   |-- metrics.py
|-- tests/
|   |-- test_core.py
|-- docs/
|   |-- artifact_checklist.md
|   |-- reproducibility.md
|-- .github/workflows/
|   |-- ci.yml
```

## Installation

Use Python 3.10 or newer. A GPU is recommended for full LLaDA experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

For GPU-backed runs that need `bitsandbytes`:

```bash
pip install -e ".[gpu]"
```

This repo is configured to use only server-local LLaDA checkpoints. The default
instruct checkpoint resolves to:

```text
/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct
```

You can also select a local checkpoint explicitly with `--model_id` or
`LLADA_MODEL_PATH`, but it must be an on-server filesystem path or one of the
built-in local aliases (`llada-8b-instruct`, `llada-8b-base`). Remote model
downloads are disabled.

Temporary files for this project should live under the repo-local `.tmp/`
directory. In CI, `TMPDIR`, `TMP`, and `TEMP` are set to `.tmp/` so temp usage
stays scoped to the repository.

## Quick Start

Run the CPU-safe tests:

```bash
pytest tests/ -m "not gpu"
```

Run a small RAGTruth experiment:

```bash
python run_experiment.py \
  --max_samples 20 \
  --task_type QA \
  --n_paths 4 \
  --num_steps 32 \
  --output_dir results/quick_test
```

Run a larger sweep with threshold ablation:

```bash
python run_experiment.py \
  --n_paths 8 \
  --num_steps 64 \
  --entropy_threshold 0.5 \
  --refine_steps 16 \
  --max_samples 500 \
  --ablate_thresholds \
  --output_dir results/full
```

## Experiment Parameters

| Parameter | Purpose | Common Values |
|---|---|---|
| `--n_paths` | Number of parallel denoising chains. | `1`, `2`, `4`, `8`, `16` |
| `--num_steps` | Denoising steps for initial generation. | `32`, `64` |
| `--entropy_threshold` | Token entropy cutoff for re-masking. | `0.3`, `0.5`, `0.7` |
| `--refine_steps` | Denoising steps for refinement. | `4`, `8`, `16`, `32` |
| `--task_type` | RAGTruth task slice. | `QA`, `Summary`, `Data2txt` |
| `--seed` | Base random seed for path generation. | Any integer |

## Outputs

Each experiment directory contains:

```text
results/
|-- raw_results.jsonl
|-- aggregate.json
|-- threshold_ablation.json
```

`raw_results.jsonl` stores one record per evaluated sample. `aggregate.json`
stores summary metrics used to evaluate the research questions.

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for environment,
hardware, dataset, command, and reporting guidance.

Minimum reporting fields for a run:

| Field | Example |
|---|---|
| Git commit | `git rev-parse HEAD` |
| Python version | `python --version` |
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Dataset | `wanderkid/RAGTruth`, split `test` |
| Hardware | GPU model, VRAM, driver/CUDA version |
| Command | Full `python run_experiment.py ...` invocation |
| Output files | Path to `aggregate.json` and `raw_results.jsonl` |

## Implementation Notes and Limitations

- The full experiment path requires downloading model weights and RAGTruth from
  Hugging Face.
- The current grounding score is a lightweight token n-gram overlap proxy, not a
  full factuality or NLI evaluator.
- Token-level labels are aligned from RAGTruth character spans through tokenizer
  offsets, which can introduce alignment noise.
- `run_experiment.py` currently uses the first 50 characters of the annotated
  response as a query proxy when constructing the prompt. This should be
  replaced with the original query field if the local dataset schema exposes it.
- Generated text and hallucination labels may contain sensitive or dataset-
  licensed content. Do not commit raw experiment outputs unless the dataset
  license permits redistribution.

## Citation

If this repository supports your work, please cite:

[OSCAR: Orchestrated Self-verification and Cross-path Refinement](https://arxiv.org/abs/2604.01624)
by Yash Shah, Abhijit Chakraborty, Naresh Kumar Devulapally, Vishnu Suresh
Lokhande, and Vivek Gupta. [[PDF]](https://arxiv.org/pdf/2604.01624)

```bibtex
@article{Shah2026OSCAROS,
  title = {OSCAR: Orchestrated Self-verification and Cross-path Refinement},
  author = {Yash Shah and Abhijit Chakraborty and Naresh Kumar Devulapally and Vishnu Suresh Lokhande and Vivek Gupta},
  journal = {ArXiv},
  year = {2026},
  volume = {abs/2604.01624},
  url = {https://api.semanticscholar.org/CorpusID:287071996}
}
```

## License

MIT License. See [LICENSE](LICENSE).

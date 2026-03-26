# OSCAR — Evaluation Runner

**O**ur **S**ystem for **C**ontrolled **A**dversarial **R**emasking  
Parallel-path entropy-guided hallucination mitigation in LLaDA-8B.

---

## Table of Contents

1. [Installation](#1-installation)
2. [File Structure](#2-file-structure)
3. [Quick Start](#3-quick-start)
4. [Full Pipeline (all at once)](#4-full-pipeline-all-at-once)
5. [Running Single Components — `--only`](#5-running-single-components----only)
   - [Metric groups](#51-metric-groups)
   - [Tables](#52-tables)
   - [Figures](#53-figures)
   - [Baselines](#54-baselines)
   - [Hyperparameter sweeps](#55-hyperparameter-sweeps)
   - [Combined shortcuts](#56-combined-shortcuts)
6. [All CLI Arguments Reference](#6-all-cli-arguments-reference)
7. [Dataset Options](#7-dataset-options)
8. [OSCAR Hyperparameter Options](#8-oscar-hyperparameter-options)
9. [Sweep Configuration](#9-sweep-configuration)
10. [Output Directory Structure](#10-output-directory-structure)
11. [Standalone Python Usage (no CLI)](#11-standalone-python-usage-no-cli)
    - [Metrics only](#111-metrics-only)
    - [Baselines only](#112-baselines-only)
    - [Figures only](#113-figures-only)
    - [Tables only](#114-tables-only)
12. [Recommended Experiment Order](#12-recommended-experiment-order)
13. [Resuming Interrupted Runs](#13-resuming-interrupted-runs)
14. [Common Errors and Fixes](#14-common-errors-and-fixes)

---

## 1. Installation

```bash
# Clone and enter repo
git clone <your-repo-url>
cd oscar

# Create environment
conda create -n oscar python=3.10 -y
conda activate oscar

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets
pip install rouge-score sacrebleu scipy scikit-learn
pip install matplotlib pandas tabulate

# HuggingFace login (needed for LLaDA-8B and RAGTruth)
huggingface-cli login
```

---

## 2. File Structure

```
oscar/
├── run_oscar_eval.py          ← master runner  (this is the main entry point)
│
├── data/
│   ├── qa_loaders.py          ← TriviaQA, CommonsenseQA, HotpotQA, RAGTruth loaders
│   └── ragtruth_loader.py     ← RAGTruth span-level loader (unchanged)
│
├── models/
│   └── llada_harness.py       ← LLaDA-8B parallel denoising harness (unchanged)
│
├── strategies/
│   └── parallel_remask.py     ← entropy computation + remasking pipeline (unchanged)
│
├── eval/
│   ├── oscar_metrics.py       ← all metric computation (Group A/B/C)
│   ├── oscar_baselines.py     ← 4 baselines + 5 demasking orders + 3 sweeps
│   ├── oscar_figures.py       ← 5 paper figures
│   ├── oscar_tables.py        ← 5 LaTeX + markdown + CSV tables
│   └── metrics.py             ← original RAGTruth metrics (unchanged)
│
├── paper/
│   └── oscar_paper_draft.tex  ← Overleaf draft with placeholders
│
└── results/oscar/             ← auto-created on first run
    ├── raw/                   ← JSONL outputs (all metrics, all methods)
    ├── tables/                ← .tex / .md / .csv for each table
    └── figures/               ← .pdf / .png for each figure
```

---

## 3. Quick Start

**Sanity check on 10 samples (no full GPU run needed):**

```bash
python run_oscar_eval.py \
    --datasets triviaqa \
    --max_samples 10 \
    --n_paths 4 \
    --num_steps 16 \
    --gen_len 64 \
    --refine_steps 4 \
    --output_dir results/smoke_test/
```

**Regenerate all tables from already-cached results (no GPU):**

```bash
python run_oscar_eval.py \
    --only all_tables \
    --skip_model_load \
    --output_dir results/oscar/
```

---

## 4. Full Pipeline (all at once)

Runs everything: OSCAR core pipeline on all 4 datasets, all 4 baselines,
all 5 demasking orders, all 3 hyperparameter sweeps, all 5 tables, all 5 figures.

```bash
python run_oscar_eval.py \
    --datasets triviaqa commonsenseqa hotpotqa ragtruth \
    --n_paths 8 \
    --num_steps 64 \
    --gen_len 128 \
    --entropy_threshold 0.5 \
    --refine_steps 16 \
    --max_samples 500 \
    --split validation \
    --ragtruth_split test \
    --seed 42 \
    --output_dir results/oscar/ \
    --figure_formats pdf png
```

> **Expected runtime on H200:** ~6–8 hours for 500 samples × 4 datasets
> with N=8 paths at 64 steps.  
> Results stream to `results/oscar/raw/*.jsonl` incrementally —
> safe to interrupt and resume.

---

## 5. Running Single Components — `--only`

Every component can be run independently using `--only <target>`.
Components that only read cached JSONL (tables and figures) can be run
with `--skip_model_load` to avoid loading the 8B model.

### 5.1 Metric Groups

Run the OSCAR core pipeline and compute one metric group only.

```bash
# Generation quality only (EM, F1, ROUGE-L, BLEU — all 4 datasets)
python run_oscar_eval.py --only generation

# Hallucination detection only (AUROC, AP, Prec/Rec/F1@τ, Spearman ρ)
# RAGTruth must be in --datasets for this to produce results
python run_oscar_eval.py --only detection --datasets ragtruth

# Refinement analysis only (FactScore Δ, span reduction, intensity Δ)
python run_oscar_eval.py --only refinement --datasets ragtruth
```

### 5.2 Tables

All table commands load from cached `results/oscar/raw/*.jsonl`.
No GPU needed — pass `--skip_model_load`.

```bash
# Table 1 — Generation quality (EM / F1 / ROUGE-L / BLEU, 4 datasets)
python run_oscar_eval.py --only table1 --skip_model_load

# Table 2 — Hallucination detection (AUROC / AP / Prec / Rec / F1 / ρ)
python run_oscar_eval.py --only table2 --skip_model_load

# Table 3 — Refinement analysis (FactScore / span reduction / intensity Δ)
python run_oscar_eval.py --only table3 --skip_model_load

# Table 4 — Baseline comparison (per-dataset sub-columns)
python run_oscar_eval.py --only table4 --skip_model_load

# Table 5 — Demasking order comparison (per-dataset sub-columns)
python run_oscar_eval.py --only table5 --skip_model_load

# All 5 tables at once
python run_oscar_eval.py --only all_tables --skip_model_load
```

Each table saves three files:
- `results/oscar/tables/tableN_<name>.tex`  — paste into Overleaf
- `results/oscar/tables/tableN_<name>.md`   — GitHub/README preview
- `results/oscar/tables/tableN_<name>.csv`  — Excel / pandas

### 5.3 Figures

All figure commands load from cached JSONL. No GPU needed.

```bash
# Figure 1 — Entropy trajectory (hallucinated vs grounded tokens over T steps)
python run_oscar_eval.py --only fig1_trajectory --skip_model_load

# Figure 2 — Precision-Recall curves (per RAGTruth task type)
python run_oscar_eval.py --only fig2_pr_curve --skip_model_load

# Figure 3 — Qualitative example (source | original | refined)
python run_oscar_eval.py --only fig3_qualitative --skip_model_load

# Figure 4 — Hyperparameter ablation (3-panel: N / τ / refine steps)
python run_oscar_eval.py --only fig4_ablation --skip_model_load

# Figure 5 — Demasking order bar chart (F1, ROUGE-L, AUROC per order)
python run_oscar_eval.py --only fig5_demasking --skip_model_load

# All 5 figures at once
python run_oscar_eval.py --only all_figures --skip_model_load

# Save as PNG only (skip PDF)
python run_oscar_eval.py --only all_figures --skip_model_load --figure_formats png

# Save as both PDF and PNG (default)
python run_oscar_eval.py --only all_figures --skip_model_load --figure_formats pdf png
```

Each figure saves to `results/oscar/figures/figN_<name>.[pdf|png]`.

### 5.4 Baselines

Each baseline runs over the datasets in `--datasets` and writes its own JSONL.
If the JSONL already exists it is loaded from cache (no re-run).

```bash
# Vanilla LLaDA — 1 learned path, no parallel sampling, no remasking
python run_oscar_eval.py --only baseline_vanilla

# Extra steps — 1 path at 2×T denoising steps (compute-matched ablation)
python run_oscar_eval.py --only baseline_extra_steps

# Random remask — same number of positions as OSCAR flags, but random selection
# This is the critical ablation that isolates entropy-guided selection
python run_oscar_eval.py --only baseline_random_remask

# SelfCheck-style — binary majority-vote disagreement (no soft entropy)
python run_oscar_eval.py --only baseline_selfcheck
```

After running all baselines, regenerate Table 4:

```bash
python run_oscar_eval.py --only table4 --skip_model_load
```

### 5.5 Hyperparameter Sweeps

Sweeps run on a reduced sample set (100 samples from the first dataset
in `--datasets`) to keep runtime manageable.

```bash
# Sweep number of parallel paths N ∈ {2, 4, 8, 16}
python run_oscar_eval.py --only sweep_n

# Custom N values
python run_oscar_eval.py --only sweep_n --sweep_n_values 2 4 8 12 16

# Sweep entropy threshold τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
python run_oscar_eval.py --only sweep_tau

# Custom τ values
python run_oscar_eval.py --only sweep_tau --sweep_tau_values 0.2 0.3 0.4 0.5 0.6 0.7 0.8

# Sweep refinement steps ∈ {4, 8, 16, 32}
python run_oscar_eval.py --only sweep_steps

# Custom step values
python run_oscar_eval.py --only sweep_steps --sweep_step_values 4 8 16 24 32

# After all sweeps, generate Figure 4
python run_oscar_eval.py --only fig4_ablation --skip_model_load
```

### 5.6 Combined Shortcuts

```bash
# Everything (full pipeline)
python run_oscar_eval.py --only all

# All tables from cache
python run_oscar_eval.py --only all_tables --skip_model_load

# All figures from cache
python run_oscar_eval.py --only all_figures --skip_model_load
```

---

## 6. All CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--datasets` | list | all 4 | Datasets to run: `triviaqa` `commonsenseqa` `hotpotqa` `ragtruth` |
| `--max_samples` | int | 500 | Max samples per dataset |
| `--split` | str | `validation` | HuggingFace split for TriviaQA / CommonsenseQA / HotpotQA |
| `--ragtruth_split` | str | `test` | Split for RAGTruth (labelled split is `test`) |
| `--ragtruth_task_type` | str | None | Filter RAGTruth to `QA`, `Summary`, or `Data2txt` |
| `--no_context` | flag | off | Disable context injection for HotpotQA (ablation) |
| `--n_paths` | int | 8 | Number of parallel denoising paths |
| `--num_steps` | int | 64 | Denoising steps T per path |
| `--gen_len` | int | 128 | Generation length in tokens |
| `--entropy_threshold` | float | 0.5 | τ — entropy threshold for flagging positions |
| `--refine_steps` | int | 16 | Refinement pass denoising steps |
| `--seed` | int | 42 | Base random seed (each path gets seed + path_id) |
| `--sweep_n_values` | list | 2 4 8 16 | N values for `--only sweep_n` |
| `--sweep_tau_values` | list | 0.3…0.7 | τ values for `--only sweep_tau` |
| `--sweep_step_values` | list | 4 8 16 32 | Step values for `--only sweep_steps` |
| `--output_dir` | str | `results/oscar/` | Root directory for all outputs |
| `--model_id` | str | `GSAI-ML/LLaDA-8B-Instruct` | HuggingFace model ID |
| `--figure_formats` | list | `pdf png` | Output formats for figures |
| `--only` | str | `all` | Run only one component (see Section 5) |
| `--skip_model_load` | flag | off | Skip loading LLaDA (use for tables/figures from cache) |

**All valid `--only` values:**

```
Metric groups:   generation  detection  refinement
Tables:          table1  table2  table3  table4  table5  all_tables
Figures:         fig1_trajectory  fig2_pr_curve  fig3_qualitative
                 fig4_ablation  fig5_demasking  all_figures
Baselines:       baseline_vanilla  baseline_extra_steps
                 baseline_random_remask  baseline_selfcheck
Sweeps:          sweep_n  sweep_tau  sweep_steps
Combined:        all
```

---

## 7. Dataset Options

```bash
# TriviaQA only
python run_oscar_eval.py --datasets triviaqa

# CommonsenseQA only
python run_oscar_eval.py --datasets commonsenseqa

# HotpotQA only (with context injection, default)
python run_oscar_eval.py --datasets hotpotqa

# HotpotQA without context injection (ablation)
python run_oscar_eval.py --datasets hotpotqa --no_context

# RAGTruth — all task types
python run_oscar_eval.py --datasets ragtruth

# RAGTruth — QA subset only (shorter responses, faster)
python run_oscar_eval.py --datasets ragtruth --ragtruth_task_type QA

# RAGTruth — Summary subset only
python run_oscar_eval.py --datasets ragtruth --ragtruth_task_type Summary

# RAGTruth — Data2txt subset only
python run_oscar_eval.py --datasets ragtruth --ragtruth_task_type Data2txt

# Two datasets
python run_oscar_eval.py --datasets triviaqa hotpotqa

# All four datasets (default)
python run_oscar_eval.py --datasets triviaqa commonsenseqa hotpotqa ragtruth

# Small dev run — 20 samples from each of two datasets
python run_oscar_eval.py --datasets triviaqa ragtruth --max_samples 20
```

---

## 8. OSCAR Hyperparameter Options

```bash
# Fewer paths (faster, less accurate entropy signal)
python run_oscar_eval.py --n_paths 4

# More paths (better entropy signal, more compute)
python run_oscar_eval.py --n_paths 16

# Lower entropy threshold (flag more tokens, higher recall, lower precision)
python run_oscar_eval.py --entropy_threshold 0.3

# Higher entropy threshold (flag fewer tokens, higher precision, lower recall)
python run_oscar_eval.py --entropy_threshold 0.7

# More denoising steps (better quality, slower)
python run_oscar_eval.py --num_steps 128

# Shorter generation (faster, use for short-answer QA)
python run_oscar_eval.py --gen_len 64

# Longer generation (needed for RAGTruth summaries)
python run_oscar_eval.py --gen_len 256

# More refinement steps
python run_oscar_eval.py --refine_steps 32

# Different seed for reproducibility check
python run_oscar_eval.py --seed 123

# Full custom config
python run_oscar_eval.py \
    --n_paths 8 \
    --num_steps 64 \
    --gen_len 128 \
    --entropy_threshold 0.5 \
    --refine_steps 16 \
    --seed 42
```

---

## 9. Sweep Configuration

```bash
# N sweep with default values {2, 4, 8, 16}
python run_oscar_eval.py --only sweep_n

# N sweep with custom range
python run_oscar_eval.py --only sweep_n --sweep_n_values 1 2 4 8 16 32

# τ sweep with default values {0.3, 0.4, 0.5, 0.6, 0.7}
python run_oscar_eval.py --only sweep_tau

# τ sweep with custom range
python run_oscar_eval.py --only sweep_tau --sweep_tau_values 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

# Steps sweep with default values {4, 8, 16, 32}
python run_oscar_eval.py --only sweep_steps

# Steps sweep with custom range
python run_oscar_eval.py --only sweep_steps --sweep_step_values 2 4 8 16 32 64

# Run all 3 sweeps sequentially with custom ranges
python run_oscar_eval.py --only sweep_n     --sweep_n_values 2 4 8 16
python run_oscar_eval.py --only sweep_tau   --sweep_tau_values 0.3 0.4 0.5 0.6 0.7
python run_oscar_eval.py --only sweep_steps --sweep_step_values 4 8 16 32

# Then generate Figure 4 from cached sweep results
python run_oscar_eval.py --only fig4_ablation --skip_model_load
```

> **Note:** Sweeps run on 100 samples from the first dataset in `--datasets`
> to keep runtime manageable. To change this, edit the sweep sample slice
> in `run_oscar_eval.py` line ~765: `sweep_samples = all_samples[ds][:min(100, ...)]`

---

## 10. Output Directory Structure

```
results/oscar/                        ← set by --output_dir
│
├── raw/                              ← JSONL, one line per sample
│   ├── triviaqa_raw.jsonl            ← generation metrics per sample
│   ├── commonsenseqa_raw.jsonl
│   ├── hotpotqa_raw.jsonl
│   ├── ragtruth_raw.jsonl            ← + token_entropy, token_labels,
│   │                                    spearman_rho, fact_score_*, span_*
│   ├── baseline_baseline_vanilla.jsonl
│   ├── baseline_baseline_extra_steps.jsonl
│   ├── baseline_baseline_random_remask.jsonl
│   ├── baseline_baseline_selfcheck.jsonl
│   ├── order_All_learned.jsonl
│   ├── order_All_random.jsonl
│   ├── order_Hybrid_(50_50).jsonl
│   ├── order_Entropy-ordered.jsonl
│   ├── sweep_n.jsonl                 ← all N-sweep samples (param_value field added)
│   ├── sweep_tau.jsonl
│   └── sweep_steps.jsonl
│
├── tables/
│   ├── table1_generation.tex         ← booktabs LaTeX, paste into Overleaf
│   ├── table1_generation.md          ← markdown preview
│   ├── table1_generation.csv         ← raw numbers
│   ├── table2_detection.tex / .md / .csv
│   ├── table3_refinement.tex / .md / .csv
│   ├── table4_baselines.tex / .md / .csv
│   └── table5_demasking.tex / .md / .csv
│
├── figures/
│   ├── fig1_entropy_trajectory.pdf / .png
│   ├── fig2_pr_curve.pdf / .png
│   ├── fig3_qualitative.pdf / .png
│   ├── fig4_ablation.pdf / .png
│   └── fig5_demasking_order.pdf / .png
│
└── aggregate.json                    ← all DatasetGenMetrics serialised
```

---

## 11. Standalone Python Usage (no CLI)

Import any module directly in a notebook or script without going through
the runner. Useful for custom analysis or integration into other pipelines.

### 11.1 Metrics only

```python
from eval.oscar_metrics import (
    evaluate_generation_sample,
    aggregate_generation,
    compute_generation_metrics,
    compute_detection_metrics,
    compute_refinement_metrics,
    build_ref_result,
)

# --- Generation metrics for one sample ---
result = evaluate_generation_sample(
    sample_id="my_sample_001",
    dataset="triviaqa",
    task="open",
    pred_before="The answer is Paris.",
    pred_after="Paris, France.",
    ground_truths=["Paris", "Paris, France", "the city of Paris"],
)
print(result.f1_after)     # token F1 after refinement
print(result.em_after)     # exact match after refinement

# --- Aggregate across a list of samples ---
dm = aggregate_generation(results_list, dataset_name="triviaqa")
print(dm.f1_delta)         # mean F1 improvement

# --- Full multi-dataset aggregation ---
gen_metrics = compute_generation_metrics({
    "triviaqa":     triviaqa_results,
    "hotpotqa":     hotpotqa_results,
    "ragtruth":     ragtruth_results,
})
macro = gen_metrics["macro_all"]

# --- Detection metrics (RAGTruth only) ---
import numpy as np
pairs = [(np.array(entropy_arr), np.array(label_arr))]
tags  = ["QA"]
det = compute_detection_metrics(pairs, tags, tau=0.5)
print(det["QA"].auroc)
print(det["QA"].spearman_rho)

# --- Refinement metrics ---
ref_result = build_ref_result(
    sample_id="rgt_001",
    subset="QA",
    original_text=pred_before,
    refined_text=pred_after,
    source_info=source_passage,
    original_tokens=refinement.original_tokens,
    refined_tokens=refinement.refined_tokens,
    prompt_len=prompt_len,
    hall_spans=sample.hall_spans,
    remasked_positions=refinement.remasked_positions,
)
ref_metrics = compute_refinement_metrics([ref_result])
print(ref_metrics["QA"].fact_score_delta)
```

### 11.2 Baselines only

```python
from eval.oscar_baselines import (
    run_baseline,
    run_all_baselines,
    run_all_demasking_orders,
    sweep_n_paths,
    sweep_tau,
    sweep_refine_steps,
    METHOD_VANILLA,
    METHOD_RANDOM_REMASK,
    METHOD_SELFCHECK,
    ORDER_OSCAR,
    ORDER_ALL_RANDOM,
)

# Run one baseline
results = run_baseline(METHOD_VANILLA, harness, samples, args)

# Run all baselines at once
all_results = run_all_baselines(harness, samples, args)

# Run specific baselines only
results = run_all_baselines(
    harness, samples, args,
    methods=[METHOD_VANILLA, METHOD_RANDOM_REMASK]
)

# Run all demasking order configs
order_results = run_all_demasking_orders(harness, samples, args)

# Run one demasking order
results = run_all_demasking_orders(
    harness, samples, args,
    orders=[ORDER_OSCAR, ORDER_ALL_RANDOM]
)

# Hyperparameter sweeps
n_sweep    = sweep_n_paths(harness, samples, args, n_values=[2, 4, 8, 16])
tau_sweep  = sweep_tau(harness, samples, args, tau_values=[0.3, 0.5, 0.7])
step_sweep = sweep_refine_steps(harness, samples, args, step_values=[4, 16, 32])
# Each returns {param_value: [BaselineRunResult, ...]}
```

### 11.3 Figures only

```python
from eval.oscar_figures import (
    entropy_trajectory_plot, TrajectoryData,
    pr_curve_plot, build_pr_data,
    qualitative_example_plot, QualitativeExample,
    ablation_plots, AblationData,
    demasking_order_bar, DemaskingBarData,
    save_figure,
)
import numpy as np

# Figure 1 — Entropy trajectory
data = TrajectoryData(
    steps=np.arange(64),
    mean_entropy={
        "hallucinated": np.random.rand(64),   # replace with real values
        "grounded":     np.random.rand(64) * 0.5,
    },
    std_entropy={
        "hallucinated": np.random.rand(64) * 0.05,
        "grounded":     np.random.rand(64) * 0.05,
    },
)
fig = entropy_trajectory_plot(data)
save_figure(fig, "figures/fig1_entropy_trajectory", formats=["pdf", "png"])

# Figure 2 — PR curves (built from detection metrics)
pr_data = build_pr_data(detection_metrics_dict)
fig = pr_curve_plot(pr_data)
save_figure(fig, "figures/fig2_pr_curve")

# Figure 3 — Qualitative example
ex = QualitativeExample(
    source_text="The Eiffel Tower was completed in 1889...",
    original_gen="The Eiffel Tower was built in 1901.",     # hallucinated date
    refined_gen="The Eiffel Tower was completed in 1889.",  # corrected
    high_entropy_spans=[(38, 42)],   # char span of "1901"
    changed_spans=[(38, 42)],        # char span of "1889"
    sample_id="rgt_042",
    dataset="ragtruth",
    task_type="QA",
)
fig = qualitative_example_plot(ex)
save_figure(fig, "figures/fig3_qualitative")

# Figure 4 — Ablation (3 panels)
ablations = [
    AblationData("N paths",            [2,4,8,16],           [0.48,0.51,0.54,0.55], [...], [...], 8),
    AblationData("τ (entropy thresh.)", [0.3,0.4,0.5,0.6,0.7],[...],               [...], [...], 0.5),
    AblationData("Refine steps",        [4,8,16,32],          [...],                [...], [...], 16),
]
fig = ablation_plots(ablations)
save_figure(fig, "figures/fig4_ablation")

# Figure 5 — Demasking order bar chart
data = DemaskingBarData(
    orders=["All learned", "All random", "Hybrid (50/50)", "Entropy-ordered", "OSCAR (1L + N-1R)"],
    f1_vals=[0.48, 0.51, 0.52, 0.50, 0.54],
    rougeL_vals=[0.45, 0.48, 0.49, 0.47, 0.51],
    auroc_vals=[float("nan"), 0.68, 0.70, 0.71, 0.72],
    oscar_idx=4,
)
fig = demasking_order_bar(data)
save_figure(fig, "figures/fig5_demasking_order")
```

### 11.4 Tables only

```python
from eval.oscar_tables import (
    render_table1,
    render_table2,
    render_table3,
    render_table4,
    render_table5,
)

# Table 1 — Generation quality
latex, md, df = render_table1(
    gen_metrics,          # dict[str, DatasetGenMetrics] from compute_generation_metrics()
    n_paths=8,
    tau=0.5,
    refine_steps=16,
)
print(latex)              # paste into Overleaf
df.to_csv("table1.csv")

# Table 2 — Detection quality
latex, md, df = render_table2(
    det_metrics,          # dict[str, DetectionMetrics] from compute_detection_metrics()
    tau=0.5,
)

# Table 3 — Refinement analysis
latex, md, df = render_table3(
    ref_metrics,          # dict[str, SubsetRefMetrics] from compute_refinement_metrics()
)

# Table 4 — Baseline comparison (per-dataset sub-columns)
latex, md, df = render_table4(
    baseline_gen_by_ds,   # dict[method_name → dict[dataset → DatasetGenMetrics]]
    baseline_det,         # dict[method_name → DetectionMetrics]
)

# Table 5 — Demasking order comparison (per-dataset sub-columns)
latex, md, df = render_table5(
    order_gen_by_ds,      # dict[order_name → dict[dataset → DatasetGenMetrics]]
    order_det,            # dict[order_name → DetectionMetrics]
)
```

---

## 12. Recommended Experiment Order

Run these in sequence to build up cached results incrementally:

```bash
# Step 1 — OSCAR core on all datasets (most expensive, ~4–6 hrs)
python run_oscar_eval.py \
    --only generation \
    --datasets triviaqa commonsenseqa hotpotqa ragtruth \
    --max_samples 500

# Step 2 — Detection and refinement (reads same JSONL, no extra GPU time)
python run_oscar_eval.py --only detection --datasets ragtruth --skip_model_load
python run_oscar_eval.py --only refinement --datasets ragtruth --skip_model_load

# Step 3 — Tables 1, 2, 3 (from core results)
python run_oscar_eval.py --only table1 --skip_model_load
python run_oscar_eval.py --only table2 --skip_model_load
python run_oscar_eval.py --only table3 --skip_model_load

# Step 4 — Baselines (~2 hrs total)
python run_oscar_eval.py --only baseline_vanilla
python run_oscar_eval.py --only baseline_extra_steps
python run_oscar_eval.py --only baseline_random_remask
python run_oscar_eval.py --only baseline_selfcheck

# Step 5 — Table 4 (from baseline results)
python run_oscar_eval.py --only table4 --skip_model_load

# Step 6 — Demasking order sweep (~1.5 hrs)
# (runs on hotpotqa by default — override with --datasets if needed)
python run_oscar_eval.py \
    --datasets hotpotqa ragtruth \
    --only all \
    --max_samples 200

# Step 7 — Table 5 (from order results)
python run_oscar_eval.py --only table5 --skip_model_load

# Step 8 — Hyperparameter sweeps (~1 hr each, 100 samples)
python run_oscar_eval.py --only sweep_n
python run_oscar_eval.py --only sweep_tau
python run_oscar_eval.py --only sweep_steps

# Step 9 — All figures (from all cached results)
python run_oscar_eval.py --only all_figures --skip_model_load
```

---

## 13. Resuming Interrupted Runs

All results stream to JSONL incrementally — if a run is interrupted,
restart with the exact same command. The runner appends to existing
JSONL files and skips samples already present.

```bash
# Safe to re-run — will append new samples to existing JSONL
python run_oscar_eval.py --only generation --datasets triviaqa

# To start fresh (wipe cache for one dataset)
rm results/oscar/raw/triviaqa_raw.jsonl
python run_oscar_eval.py --only generation --datasets triviaqa

# To start completely fresh
rm -rf results/oscar/
python run_oscar_eval.py
```

> **Warning:** The JSONL appends unconditionally — if you re-run after
> changing hyperparameters without deleting the old file, you will have
> mixed results in the same file. Always delete the relevant JSONL
> before changing `--n_paths`, `--entropy_threshold`, or `--num_steps`.

---

## 14. Common Errors and Fixes

**`ModuleNotFoundError: No module named 'rouge_score'`**
```bash
pip install rouge-score
```

**`ModuleNotFoundError: No module named 'sacrebleu'`**
```bash
pip install sacrebleu
```

**`OSError: GSAI-ML/LLaDA-8B-Instruct does not exist`**
```bash
huggingface-cli login
# or pass a local path:
python run_oscar_eval.py --model_id /path/to/llada-8b-local
```

**`CUDA out of memory`**
```bash
# Reduce batch size by lowering n_paths or gen_len
python run_oscar_eval.py --n_paths 4 --gen_len 64
```

**`datasets.exceptions.DatasetNotFoundError: wanderkid/RAGTruth`**
```bash
# Ensure HuggingFace login and dataset access
huggingface-cli login
python -c "from datasets import load_dataset; load_dataset('wanderkid/RAGTruth', split='test')"
```

**Tables render but all cells show `—`**  
The cached JSONL exists but is empty or corrupted. Delete it and re-run:
```bash
rm results/oscar/raw/<dataset>_raw.jsonl
python run_oscar_eval.py --only generation --datasets <dataset>
```

**`KeyError: 'token_entropy'` when running `--only detection`**  
Detection requires `track_trajectory=True` which is only set when running
the full pipeline or `--only detection` directly. Re-run the RAGTruth core:
```bash
rm results/oscar/raw/ragtruth_raw.jsonl
python run_oscar_eval.py --only detection --datasets ragtruth
```

**Figure 4 shows only 1 or 2 panels**  
Not all sweep JSONL files exist. Run all three sweeps first:
```bash
python run_oscar_eval.py --only sweep_n
python run_oscar_eval.py --only sweep_tau
python run_oscar_eval.py --only sweep_steps
python run_oscar_eval.py --only fig4_ablation --skip_model_load
```
